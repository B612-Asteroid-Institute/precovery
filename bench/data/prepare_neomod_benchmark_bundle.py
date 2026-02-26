from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import json

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from adam_core.orbits import Orbits
from precovery.healpix_geom import radec_to_healpixel
from precovery.search.backends.duckdb_parquet import DuckDbParquetBackend
from precovery.search.covariance_psd import repair_orbits_covariance_psd_for_sampling
from precovery.search.time_key import mjd_to_time_key_us

from bench.benchmarks.runner import _load_truth_table
from bench.benchmarks.workload import month_bounds_mjd_utc
from .neomod_synthetic_covariance import (
    SyntheticCovarianceModel,
    build_synthetic_covariance_model,
    fetch_sbdb_neo_calibration_table,
    generate_covariance_matrices_for_batch,
    write_model_json,
)


@dataclass(frozen=True)
class InputUris:
    quad_truth: str
    noise_100: str
    neomod_orbits: str


@dataclass(frozen=True)
class BuildPaths:
    root: Path
    detections_dir: Path
    truth_detections: Path
    truth_unresolved: Path
    orbit_ids_with_truth: Path
    orbits_with_truth: Path
    orbit_ids_fast: Path
    orbits_fast: Path
    orbits_synth: Path
    object_orbit_mapping: Path
    sbdb_calibration: Path
    synthetic_cov_model_json: Path
    manifest_json: Path


@dataclass(frozen=True)
class BuildResult:
    paths: BuildPaths
    n_detections_total: int
    n_truth_detections: int
    n_truth_unresolved_rows: int
    n_orbits_full: int
    n_orbits_with_truth: int
    n_orbits_fast: int


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _mjd_from_time_struct(time_col: pa.ChunkedArray | pa.Array) -> np.ndarray:
    days = pc.cast(pc.struct_field(time_col, "days"), pa.int64()).to_numpy(zero_copy_only=False)
    nanos = pc.cast(pc.struct_field(time_col, "nanos"), pa.int64()).to_numpy(zero_copy_only=False)
    return np.asarray(days, dtype=np.float64) + np.asarray(nanos, dtype=np.float64) / 86_400_000_000_000.0


def _build_paths(*, out_dir: Path, month: str, obscode: str, fast_k: int) -> BuildPaths:
    root = Path(out_dir)
    return BuildPaths(
        root=root,
        detections_dir=root / "detections_parquet",
        truth_detections=root / "truth_detections.parquet",
        truth_unresolved=root / "truth_unresolved_designations.parquet",
        orbit_ids_with_truth=root / "orbit_ids_with_truth.parquet",
        orbits_with_truth=root / "orbits_with_truth.parquet",
        orbit_ids_fast=root / f"orbit_ids_fast{int(fast_k)}.parquet",
        orbits_fast=root / f"orbits_fast{int(fast_k)}.parquet",
        orbits_synth=root / "orbits_synth_cov.parquet",
        object_orbit_mapping=root / "object_orbit_mapping.parquet",
        sbdb_calibration=root / "sbdb_neo_calibration.parquet",
        synthetic_cov_model_json=root / "synthetic_cov_model.json",
        manifest_json=root / "manifest.json",
    )


def _normalize_detection_batch(
    *,
    batch: pa.Table,
    start_mjd: float,
    end_mjd: float,
    obscode: str,
    healpix_nside: int,
) -> pa.Table:
    mjd = _mjd_from_time_struct(batch["time"])
    obs = np.asarray(pc.cast(batch["observatory_code"], pa.large_string()).to_pylist(), dtype=object)
    keep = (mjd >= float(start_mjd)) & (mjd < float(end_mjd)) & (obs == str(obscode))
    idx = np.nonzero(keep)[0]
    if idx.size == 0:
        return pa.table(
            {
                "obscode": pa.array([], type=pa.large_string()),
                "exposure_mjd_mid_utc": pa.array([], type=pa.float64()),
                "exposure_mjd_mid_key_us": pa.array([], type=pa.int64()),
                "filter": pa.array([], type=pa.large_string()),
                "healpixel": pa.array([], type=pa.int64()),
                "observation_id": pa.array([], type=pa.large_string()),
                "obstime_mjd_utc": pa.array([], type=pa.float64()),
                "ra_deg": pa.array([], type=pa.float64()),
                "dec_deg": pa.array([], type=pa.float64()),
                "ra_sigma_deg": pa.array([], type=pa.float64()),
                "dec_sigma_deg": pa.array([], type=pa.float64()),
                "mag": pa.array([], type=pa.float64()),
                "mag_sigma": pa.array([], type=pa.float64()),
            }
        )

    idx_pa = pa.array(idx.astype(np.int64, copy=False), type=pa.int64())
    t = batch.take(idx_pa)

    mjd_keep = mjd[idx]
    key_us = mjd_to_time_key_us(mjd_keep)

    ra = np.asarray(pc.cast(t["ra"], pa.float64()).to_numpy(zero_copy_only=False), dtype=np.float64)
    dec = np.asarray(pc.cast(t["dec"], pa.float64()).to_numpy(zero_copy_only=False), dtype=np.float64)
    hp = np.asarray(radec_to_healpixel(ra=ra, dec=dec, nside=int(healpix_nside)), dtype=np.int64)

    return pa.table(
        {
            "obscode": pc.cast(t["observatory_code"], pa.large_string()),
            "exposure_mjd_mid_utc": pa.array(mjd_keep, type=pa.float64()),
            "exposure_mjd_mid_key_us": pa.array(key_us, type=pa.int64()),
            "filter": pc.cast(t["filter"], pa.large_string()),
            "healpixel": pa.array(hp, type=pa.int64()),
            "observation_id": pc.cast(t["id"], pa.large_string()),
            "obstime_mjd_utc": pa.array(mjd_keep, type=pa.float64()),
            "ra_deg": pc.cast(t["ra"], pa.float64()),
            "dec_deg": pc.cast(t["dec"], pa.float64()),
            "ra_sigma_deg": pc.cast(t["ra_sigma"], pa.float64()),
            "dec_sigma_deg": pc.cast(t["dec_sigma"], pa.float64()),
            "mag": pc.cast(t["mag"], pa.float64()),
            "mag_sigma": pc.cast(t["mag_sigma"], pa.float64()),
        }
    )


def _write_detection_dataset(
    *,
    sources: tuple[str, ...],
    out_dir: Path,
    month: str,
    obscode: str,
    healpix_nside: int,
    batch_size: int,
) -> int:
    start_mjd, end_mjd = month_bounds_mjd_utc(month)
    out_dir.mkdir(parents=True, exist_ok=True)

    part = 0
    n_rows = 0

    for src in sources:
        pf = pq.ParquetFile(src)
        for b in pf.iter_batches(
            batch_size=int(batch_size),
            columns=[
                "id",
                "time",
                "ra",
                "dec",
                "ra_sigma",
                "dec_sigma",
                "mag",
                "mag_sigma",
                "filter",
                "observatory_code",
            ],
        ):
            t = pa.Table.from_batches([b])
            norm = _normalize_detection_batch(
                batch=t,
                start_mjd=float(start_mjd),
                end_mjd=float(end_mjd),
                obscode=str(obscode),
                healpix_nside=int(healpix_nside),
            )
            if norm.num_rows <= 0:
                continue
            out = out_dir / f"part-{part:06d}.parquet"
            pq.write_table(norm, str(out))
            part += 1
            n_rows += int(norm.num_rows)

    return int(n_rows)


def _build_object_orbit_mapping(*, orbits_parquet: str, out_path: Path) -> tuple[pa.Table, int]:
    t = pq.read_table(str(orbits_parquet), columns=["object_id", "orbit_id"])
    m = pc.and_(pc.is_valid(t["object_id"]), pc.is_valid(t["orbit_id"]))
    t = t.filter(m)
    t = t.select(["object_id", "orbit_id"])

    # Detect multiplicity and collapse deterministically when needed.
    g = t.group_by(["object_id"]).aggregate([("orbit_id", "count_distinct")])
    bad = g.filter(pc.greater(pc.cast(g["orbit_id_count_distinct"], pa.int64()), 1))
    n_dup = int(bad.num_rows)

    t = t.group_by(["object_id"]).aggregate([("orbit_id", "min")]).rename_columns(
        ["object_id", "orbit_id"]
    )
    pq.write_table(t, str(out_path))
    return t, int(n_dup)


def _build_truth_detections(
    *,
    truth_source: str,
    object_orbit_mapping: pa.Table,
    out_truth: Path,
    out_unresolved: Path,
    month: str,
    obscode: str,
) -> tuple[int, int]:
    start_mjd, end_mjd = month_bounds_mjd_utc(month)

    t = pq.read_table(
        str(truth_source),
        columns=["id", "object_id", "observatory_code", "time"],
    )

    mjd = _mjd_from_time_struct(t["time"])
    obs = np.asarray(pc.cast(t["observatory_code"], pa.large_string()).to_pylist(), dtype=object)
    obj = pc.cast(t["object_id"], pa.large_string())

    keep = (mjd >= float(start_mjd)) & (mjd < float(end_mjd)) & (obs == str(obscode))
    keep = keep & np.asarray(pc.is_valid(obj).to_numpy(zero_copy_only=False), dtype=bool)
    idx = np.nonzero(keep)[0]
    if idx.size == 0:
        empty_truth = pa.table(
            {
                "orbit_id": pa.array([], type=pa.large_string()),
                "observation_id": pa.array([], type=pa.large_string()),
                "obscode": pa.array([], type=pa.large_string()),
                "truth_time_mjd_utc": pa.array([], type=pa.float64()),
                "exposure_mjd_mid_key_us": pa.array([], type=pa.int64()),
            }
        )
        pq.write_table(empty_truth, str(out_truth))
        pq.write_table(pa.table({"object_id": pa.array([], type=pa.large_string())}), str(out_unresolved))
        return 0, 0

    idx_pa = pa.array(idx.astype(np.int64, copy=False), type=pa.int64())
    tf = t.take(idx_pa)
    mjd_k = mjd[idx]
    key_k = mjd_to_time_key_us(mjd_k)

    tf = pa.table(
        {
            "object_id": pc.cast(tf["object_id"], pa.large_string()),
            "observation_id": pc.cast(tf["id"], pa.large_string()),
            "obscode": pc.cast(tf["observatory_code"], pa.large_string()),
            "truth_time_mjd_utc": pa.array(mjd_k, type=pa.float64()),
            "exposure_mjd_mid_key_us": pa.array(key_k, type=pa.int64()),
        }
    )

    joined = tf.join(object_orbit_mapping, keys=["object_id"], join_type="left outer")
    unresolved = joined.filter(pc.is_null(joined["orbit_id"]))
    resolved = joined.filter(pc.is_valid(joined["orbit_id"]))

    truth = resolved.select(
        ["orbit_id", "observation_id", "obscode", "truth_time_mjd_utc", "exposure_mjd_mid_key_us"]
    )
    pq.write_table(truth, str(out_truth))

    unresolved_out = (
        unresolved.select(["object_id", "observation_id", "obscode", "truth_time_mjd_utc"])
        if unresolved.num_rows > 0
        else pa.table(
            {
                "object_id": pa.array([], type=pa.large_string()),
                "observation_id": pa.array([], type=pa.large_string()),
                "obscode": pa.array([], type=pa.large_string()),
                "truth_time_mjd_utc": pa.array([], type=pa.float64()),
            }
        )
    )
    pq.write_table(unresolved_out, str(out_unresolved))

    return int(truth.num_rows), int(unresolved_out.num_rows)


def _build_synthetic_orbits(
    *,
    source_orbits_parquet: str,
    out_orbits_parquet: Path,
    model: SyntheticCovarianceModel,
    seed: int,
    batch_size: int,
) -> int:
    pf = pq.ParquetFile(str(source_orbits_parquet))
    out_orbits_parquet.parent.mkdir(parents=True, exist_ok=True)

    src_schema = pf.schema_arrow
    writer: pq.ParquetWriter | None = None
    rng = np.random.default_rng(int(seed))
    n_rows = 0

    for b in pf.iter_batches(batch_size=int(batch_size)):
        t = pa.Table.from_batches([b])
        n = int(t.num_rows)
        if n <= 0:
            continue

        cov = generate_covariance_matrices_for_batch(n_rows=n, model=model, rng=rng)
        flat = cov.reshape(n, 36)

        vals = pa.array(flat.reshape(-1), type=pa.float64())
        offsets = pa.array(np.arange(0, (n + 1) * 36, 36, dtype=np.int64), type=pa.int64())
        cov_values = pa.LargeListArray.from_arrays(offsets, vals)
        cov_struct = pa.StructArray.from_arrays([cov_values], names=["values"])

        coords = t["coordinates"].combine_chunks()
        x = pc.struct_field(coords, "x")
        y = pc.struct_field(coords, "y")
        z = pc.struct_field(coords, "z")
        vx = pc.struct_field(coords, "vx")
        vy = pc.struct_field(coords, "vy")
        vz = pc.struct_field(coords, "vz")
        tm = pc.struct_field(coords, "time")
        org = pc.struct_field(coords, "origin")

        coords_new = pa.StructArray.from_arrays(
            [x, y, z, vx, vy, vz, tm, cov_struct, org],
            names=["x", "y", "z", "vx", "vy", "vz", "time", "covariance", "origin"],
        )

        cols: dict[str, pa.ChunkedArray | pa.Array] = {}
        for name in t.column_names:
            if name == "coordinates":
                cols[name] = pa.chunked_array([coords_new])
            else:
                cols[name] = t[name]

        out_t = pa.table(cols)
        out_t = out_t.replace_schema_metadata(src_schema.metadata)

        if writer is None:
            writer = pq.ParquetWriter(str(out_orbits_parquet), out_t.schema)
        writer.write_table(out_t)
        n_rows += n

    if writer is not None:
        writer.close()

    return int(n_rows)


def _write_orbit_id_table(*, orbit_ids: list[str], out_path: Path) -> None:
    arr = pa.array([str(x) for x in orbit_ids], type=pa.large_string())
    pq.write_table(pa.table({"orbit_id": arr}), str(out_path))


def _filter_orbits_to_ids(*, orbits_parquet: Path, orbit_ids: list[str]) -> Orbits:
    if not orbit_ids:
        return Orbits.empty()
    dset = ds.dataset(str(orbits_parquet), format="parquet")
    filt = ds.field("orbit_id").isin([str(x) for x in orbit_ids])
    t = dset.to_table(filter=filt)
    if "physical_parameters" not in t.column_names:
        pp_type = Orbits.empty().table.schema.field("physical_parameters").type
        t = t.append_column("physical_parameters", pa.nulls(t.num_rows, type=pp_type))
    return Orbits.from_pyarrow(t)


def _pick_fast_orbits(*, truth_parquet: Path, k: int) -> list[str]:
    t = pq.read_table(str(truth_parquet), columns=["orbit_id"])
    if t.num_rows <= 0:
        return []
    g = t.group_by(["orbit_id"]).aggregate([("orbit_id", "count")]).rename_columns(
        ["orbit_id", "n_truth_detections"]
    )
    g = g.sort_by([("n_truth_detections", "descending"), ("orbit_id", "ascending")])
    ids = [str(x) for x in g["orbit_id"].to_pylist()]
    return ids[: int(max(0, k))]


def _validate_detections_schema(*, detections_dir: Path) -> None:
    files = sorted(detections_dir.glob("part-*.parquet"))
    if not files:
        raise RuntimeError(f"No detection parquet parts under {detections_dir}")
    t = pq.read_table(str(files[0]))

    need = {
        "obscode",
        "exposure_mjd_mid_utc",
        "exposure_mjd_mid_key_us",
        "filter",
        "healpixel",
        "observation_id",
        "obstime_mjd_utc",
        "ra_deg",
        "dec_deg",
        "ra_sigma_deg",
        "dec_sigma_deg",
        "mag",
        "mag_sigma",
    }
    miss = sorted(need - set(t.column_names))
    if miss:
        raise RuntimeError(f"Detection parquet missing required columns: {miss}")


def _validate_truth_loader(*, truth_path: Path) -> None:
    t = _load_truth_table(truth_path)
    if t.num_rows <= 0:
        raise RuntimeError("Truth table is empty after loader normalization")


def prepare_neomod_benchmark_bundle(
    *,
    out_dir: Path,
    month: str,
    obscode: str,
    fast_orbits_k: int,
    healpix_nside: int,
    seed: int,
    detection_batch_size: int,
    orbits_batch_size: int,
    sbdb_max_rows: int,
    sbdb_page_size: int,
    sbdb_timeout_sec: float,
    sbdb_max_attempts: int,
    sbdb_force_refresh: bool,
    fail_on_ambiguous_object_mapping: bool,
    inputs: InputUris,
) -> BuildResult:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = _build_paths(out_dir=out_dir, month=month, obscode=obscode, fast_k=fast_orbits_k)
    paths.detections_dir.mkdir(parents=True, exist_ok=True)

    # 1) SBDB calibration + synthetic covariance model
    sbdb_table = fetch_sbdb_neo_calibration_table(
        source_orbits_parquet=str(inputs.neomod_orbits),
        cache_parquet=paths.sbdb_calibration,
        max_rows=int(sbdb_max_rows),
        page_size=int(sbdb_page_size),
        timeout_sec=float(sbdb_timeout_sec),
        max_attempts=int(sbdb_max_attempts),
        force_refresh=bool(sbdb_force_refresh),
        seed=int(seed),
    )
    model = build_synthetic_covariance_model(sbdb_table=sbdb_table, seed=int(seed))
    write_model_json(model=model, out_path=paths.synthetic_cov_model_json)

    # 2) Object->orbit mapping (from NEOMOD source)
    mapping, n_object_id_multi_mapping = _build_object_orbit_mapping(
        orbits_parquet=str(inputs.neomod_orbits),
        out_path=paths.object_orbit_mapping,
    )
    if bool(fail_on_ambiguous_object_mapping) and int(n_object_id_multi_mapping) > 0:
        raise RuntimeError(
            "Ambiguous object_id->orbit_id mapping detected in source orbits: "
            f"{int(n_object_id_multi_mapping)} object_ids map to multiple orbit_ids. "
            "This can invalidate truth mapping. "
            "Either disambiguate upstream or pass --allow-ambiguous-object-mapping to proceed."
        )

    # 3) Detections dataset (explicit union: quads + noise100)
    n_det = _write_detection_dataset(
        sources=(str(inputs.quad_truth), str(inputs.noise_100)),
        out_dir=paths.detections_dir,
        month=str(month),
        obscode=str(obscode),
        healpix_nside=int(healpix_nside),
        batch_size=int(detection_batch_size),
    )

    # 4) Truth artifacts for canonical month
    n_truth, n_unresolved = _build_truth_detections(
        truth_source=str(inputs.quad_truth),
        object_orbit_mapping=mapping,
        out_truth=paths.truth_detections,
        out_unresolved=paths.truth_unresolved,
        month=str(month),
        obscode=str(obscode),
    )

    truth_tbl = pq.read_table(str(paths.truth_detections), columns=["orbit_id"]) if n_truth > 0 else pa.table(
        {"orbit_id": pa.array([], type=pa.large_string())}
    )
    orbit_ids_with_truth = (
        sorted({str(x) for x in truth_tbl["orbit_id"].to_pylist() if x is not None}) if n_truth > 0 else []
    )
    _write_orbit_id_table(orbit_ids=orbit_ids_with_truth, out_path=paths.orbit_ids_with_truth)

    # 5) Full NEOMOD synthetic-covariance orbit table
    n_orbits_full = _build_synthetic_orbits(
        source_orbits_parquet=str(inputs.neomod_orbits),
        out_orbits_parquet=paths.orbits_synth,
        model=model,
        seed=int(seed),
        batch_size=int(orbits_batch_size),
    )

    # 6) Subset orbits (truth-bearing + fast)
    orbits_with_truth = _filter_orbits_to_ids(orbits_parquet=paths.orbits_synth, orbit_ids=orbit_ids_with_truth)
    orbits_with_truth.to_parquet(paths.orbits_with_truth)

    fast_ids = _pick_fast_orbits(truth_parquet=paths.truth_detections, k=int(fast_orbits_k))
    _write_orbit_id_table(orbit_ids=fast_ids, out_path=paths.orbit_ids_fast)
    orbits_fast = _filter_orbits_to_ids(orbits_parquet=paths.orbits_synth, orbit_ids=fast_ids)
    orbits_fast.to_parquet(paths.orbits_fast)

    # 7) Validations
    _validate_detections_schema(detections_dir=paths.detections_dir)
    _validate_truth_loader(truth_path=paths.truth_detections)
    repaired_fast, _n_rep_fast, n_rej_fast = repair_orbits_covariance_psd_for_sampling(orbits_fast)
    if int(n_rej_fast) > 0 or int(len(repaired_fast)) != int(len(orbits_fast)):
        raise RuntimeError(
            "Synthetic covariance validation failed on fast subset: "
            f"n_rejected={int(n_rej_fast)} len_fast={int(len(orbits_fast))}"
        )

    # Backend smoke checks (schema/path-level). This avoids a full benchmark run in the builder.
    backend = DuckDbParquetBackend(parquet_path=paths.detections_dir)
    start_mjd, end_mjd = month_bounds_mjd_utc(month)
    n_frames = backend.count_total_window_frames(
        start_mjd_utc=float(start_mjd), end_mjd_utc=float(end_mjd), obscodes=(str(obscode),)
    )
    if int(n_frames) <= 0:
        raise RuntimeError("Detection dataset appears empty for configured month/obscode")

    sbdb_condition_bin_counts: dict[str, int] = {}
    sbdb_arc_bin_counts: dict[str, int] = {}
    sbdb_uncertainty_decile_counts: dict[str, int] = {}
    if "condition_bin" in sbdb_table.column_names:
        t_cb = sbdb_table.select(["condition_bin"]).group_by(["condition_bin"]).aggregate([("condition_bin", "count")])
        for i in range(int(t_cb.num_rows)):
            sbdb_condition_bin_counts[str(int(t_cb["condition_bin"][i].as_py()))] = int(
                t_cb["condition_bin_count"][i].as_py()
            )
    if "arc_bin" in sbdb_table.column_names:
        t_ab = sbdb_table.select(["arc_bin"]).group_by(["arc_bin"]).aggregate([("arc_bin", "count")])
        for i in range(int(t_ab.num_rows)):
            sbdb_arc_bin_counts[str(int(t_ab["arc_bin"][i].as_py()))] = int(
                t_ab["arc_bin_count"][i].as_py()
            )
    if "uncertainty_decile" in sbdb_table.column_names:
        t_ud = (
            sbdb_table.select(["uncertainty_decile"])
            .group_by(["uncertainty_decile"])
            .aggregate([("uncertainty_decile", "count")])
        )
        for i in range(int(t_ud.num_rows)):
            sbdb_uncertainty_decile_counts[str(int(t_ud["uncertainty_decile"][i].as_py()))] = int(
                t_ud["uncertainty_decile_count"][i].as_py()
            )

    manifest = {
        "generated_at_utc": _utc_now_iso(),
        "month": str(month),
        "obscode": str(obscode),
        "seed": int(seed),
        "healpix_nside": int(healpix_nside),
        "sources": {
            "quad_truth": str(inputs.quad_truth),
            "noise_100": str(inputs.noise_100),
            "neomod_orbits": str(inputs.neomod_orbits),
        },
        "sbdb_calibration": {
            "table_parquet": str(paths.sbdb_calibration),
            "model_json": str(paths.synthetic_cov_model_json),
            "rows": int(sbdb_table.num_rows),
            "max_rows_requested": int(sbdb_max_rows),
            "page_size": int(sbdb_page_size),
            "max_attempts": int(sbdb_max_attempts),
            "sbdb_filter": {"sb-group": "neo", "sb-kind": "a"},
            "selection_method": "stratified(condition_bin,arc_bin)+uncertainty_tails",
            "condition_bin_counts": dict(sorted(sbdb_condition_bin_counts.items())),
            "arc_bin_counts": dict(sorted(sbdb_arc_bin_counts.items())),
            "uncertainty_decile_counts": dict(sorted(sbdb_uncertainty_decile_counts.items())),
        },
        "outputs": {
            "detections_parquet": str(paths.detections_dir),
            "truth_detections_parquet": str(paths.truth_detections),
            "truth_unresolved_parquet": str(paths.truth_unresolved),
            "orbits_parquet": str(paths.orbits_synth),
            "orbit_ids_with_truth_parquet": str(paths.orbit_ids_with_truth),
            "orbits_with_truth_parquet": str(paths.orbits_with_truth),
            "orbit_ids_fast_parquet": str(paths.orbit_ids_fast),
            "orbits_fast_parquet": str(paths.orbits_fast),
            "object_orbit_mapping_parquet": str(paths.object_orbit_mapping),
        },
        "counts": {
            "n_detections_total": int(n_det),
            "n_truth_detections": int(n_truth),
            "n_truth_unresolved_rows": int(n_unresolved),
            "n_object_id_multi_mapping": int(n_object_id_multi_mapping),
            "n_orbits_full": int(n_orbits_full),
            "n_orbits_with_truth": int(len(orbits_with_truth)),
            "n_orbits_fast": int(len(orbits_fast)),
            "n_window_frames": int(n_frames),
        },
        "fast_orbits": {
            "k": int(fast_orbits_k),
            "orbit_ids": [str(x) for x in fast_ids],
        },
        "defaults": {
            "noise_model": "noise_100.000",
            "canonical_month": str(month),
            "observatory_scope": str(obscode),
            "covariance_calibration_source": (
                "SBDB real NEO asteroid sample (sb-group=neo, sb-kind=a) "
                "queried via query_sbdb_new for covariance scales"
            ),
            "fail_on_ambiguous_object_mapping": bool(fail_on_ambiguous_object_mapping),
        },
    }
    paths.manifest_json.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    return BuildResult(
        paths=paths,
        n_detections_total=int(n_det),
        n_truth_detections=int(n_truth),
        n_truth_unresolved_rows=int(n_unresolved),
        n_orbits_full=int(n_orbits_full),
        n_orbits_with_truth=int(len(orbits_with_truth)),
        n_orbits_fast=int(len(orbits_fast)),
    )


def _default_inputs() -> InputUris:
    root = "gs://asteroid-institute-data/simulations/20251002"
    return InputUris(
        quad_truth=f"{root}/X05-quad_X05_NEOMOD.parquet",
        noise_100=f"{root}/X05-quad_X05_NEOMOD_noise_100.000.parquet",
        neomod_orbits=f"{root}/neomod_orbits_20251002.parquet",
    )


def _parse_args() -> argparse.Namespace:
    d = _default_inputs()
    p = argparse.ArgumentParser(
        description="Prepare NEOMOD benchmark bundle (quads + noise100) using existing benchmark schemas."
    )
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--month", type=str, default="2026-01")
    p.add_argument("--obscode", type=str, default="X05")
    p.add_argument("--fast-orbits-k", type=int, default=64)
    p.add_argument("--seed", type=int, default=20260225)
    p.add_argument("--healpix-nside", type=int, default=32)

    p.add_argument("--quad-truth-uri", type=str, default=d.quad_truth)
    p.add_argument("--noise100-uri", type=str, default=d.noise_100)
    p.add_argument("--neomod-orbits-uri", type=str, default=d.neomod_orbits)

    p.add_argument("--detection-batch-size", type=int, default=200_000)
    p.add_argument("--orbits-batch-size", type=int, default=100_000)

    p.add_argument("--sbdb-max-rows", type=int, default=4_096)
    p.add_argument("--sbdb-page-size", type=int, default=256)
    p.add_argument("--sbdb-timeout-sec", type=float, default=60.0)
    p.add_argument("--sbdb-max-attempts", type=int, default=5)
    p.add_argument("--sbdb-force-refresh", action="store_true")
    p.add_argument(
        "--allow-ambiguous-object-mapping",
        action="store_true",
        help="Proceed even when source orbits contain object_id values mapping to multiple orbit_ids.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    result = prepare_neomod_benchmark_bundle(
        out_dir=Path(args.out_dir),
        month=str(args.month),
        obscode=str(args.obscode),
        fast_orbits_k=int(args.fast_orbits_k),
        healpix_nside=int(args.healpix_nside),
        seed=int(args.seed),
        detection_batch_size=int(args.detection_batch_size),
        orbits_batch_size=int(args.orbits_batch_size),
        sbdb_max_rows=int(args.sbdb_max_rows),
        sbdb_page_size=int(args.sbdb_page_size),
        sbdb_timeout_sec=float(args.sbdb_timeout_sec),
        sbdb_max_attempts=int(args.sbdb_max_attempts),
        sbdb_force_refresh=bool(args.sbdb_force_refresh),
        fail_on_ambiguous_object_mapping=not bool(args.allow_ambiguous_object_mapping),
        inputs=InputUris(
            quad_truth=str(args.quad_truth_uri),
            noise_100=str(args.noise100_uri),
            neomod_orbits=str(args.neomod_orbits_uri),
        ),
    )

    summary = {
        "out_dir": str(result.paths.root),
        "manifest": str(result.paths.manifest_json),
        "n_detections_total": int(result.n_detections_total),
        "n_truth_detections": int(result.n_truth_detections),
        "n_truth_unresolved_rows": int(result.n_truth_unresolved_rows),
        "n_orbits_full": int(result.n_orbits_full),
        "n_orbits_with_truth": int(result.n_orbits_with_truth),
        "n_orbits_fast": int(result.n_orbits_fast),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
