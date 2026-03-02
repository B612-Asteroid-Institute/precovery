from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import json
import zlib

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.fs as pafs
import pyarrow.parquet as pq
import quivr as qv
import ray

from adam_core.dynamics.propagation import propagate_2body
from adam_core.observations.detections import PointSourceDetections
from adam_core.observations.exposures import Exposures
from adam_core.orbits import Orbits
from adam_core.photometry import estimate_absolute_magnitude_v_from_detections_grouped
from adam_core.ray_cluster import initialize_use_ray
from adam_core.time import Timestamp
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
    orbit_photometry: Path
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
    n_orbits_with_photometry: int
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
        orbit_photometry=root / "orbit_photometry.parquet",
        object_orbit_mapping=root / "object_orbit_mapping.parquet",
        sbdb_calibration=root / "sbdb_neo_calibration.parquet",
        synthetic_cov_model_json=root / "synthetic_cov_model.json",
        manifest_json=root / "manifest.json",
    )


def _looks_like_uri(path_or_uri: str) -> bool:
    s = str(path_or_uri)
    return "://" in s


def _download_uri_to_local_cache(*, uri: str, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    src_name = Path(str(uri).split("://", 1)[-1]).name or "source.parquet"
    digest = hashlib.sha1(str(uri).encode("utf-8")).hexdigest()[:12]
    out_path = cache_dir / f"{digest}_{src_name}"
    if out_path.exists():
        return out_path

    fs, fs_path = pafs.FileSystem.from_uri(str(uri))
    tmp_path = out_path.with_suffix(out_path.suffix + ".partial")
    with fs.open_input_file(str(fs_path)) as src, open(tmp_path, "wb") as dst:
        while True:
            chunk = src.read(8 * 1024 * 1024)
            if not chunk:
                break
            dst.write(chunk)
    tmp_path.replace(out_path)
    return out_path


def _materialize_input_uris_local(
    *,
    inputs: InputUris,
    cache_dir: Path,
) -> InputUris:
    def _resolve_one(path_or_uri: str) -> str:
        p = Path(str(path_or_uri))
        if p.exists():
            return str(p.resolve())
        if not _looks_like_uri(str(path_or_uri)):
            raise FileNotFoundError(f"Input file does not exist: {path_or_uri}")
        return str(_download_uri_to_local_cache(uri=str(path_or_uri), cache_dir=cache_dir).resolve())

    return InputUris(
        quad_truth=_resolve_one(inputs.quad_truth),
        noise_100=_resolve_one(inputs.noise_100),
        neomod_orbits=_resolve_one(inputs.neomod_orbits),
    )


def _normalize_detection_batch(
    *,
    batch: pa.Table,
    start_mjd: float,
    end_mjd: float,
    obscode: str,
    healpix_nside: int,
    detection_sigma_floor_mas: float | None = None,
    detection_sigma_cap_mas: float | None = None,
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
    # Sorcha-provided astrometric sigmas in this NEOMOD simulation are in arcseconds.
    # Precovery detection schema expects degrees.
    ra_sigma_deg_np = (
        np.asarray(pc.cast(t["ra_sigma"], pa.float64()).to_numpy(zero_copy_only=False), dtype=np.float64) / 3600.0
    )
    dec_sigma_deg_np = (
        np.asarray(pc.cast(t["dec_sigma"], pa.float64()).to_numpy(zero_copy_only=False), dtype=np.float64) / 3600.0
    )

    sigma_floor_mas = (
        None if detection_sigma_floor_mas is None else float(detection_sigma_floor_mas)
    )
    sigma_cap_mas = None if detection_sigma_cap_mas is None else float(detection_sigma_cap_mas)
    if sigma_floor_mas is not None and (not np.isfinite(sigma_floor_mas) or sigma_floor_mas < 0.0):
        sigma_floor_mas = None
    if sigma_cap_mas is not None and (not np.isfinite(sigma_cap_mas) or sigma_cap_mas <= 0.0):
        sigma_cap_mas = None
    if sigma_floor_mas is not None and sigma_cap_mas is not None and sigma_cap_mas < sigma_floor_mas:
        raise ValueError(
            f"detection_sigma_cap_mas ({sigma_cap_mas}) must be >= detection_sigma_floor_mas ({sigma_floor_mas})"
        )

    if sigma_floor_mas is not None:
        floor_deg = float(sigma_floor_mas) / 3_600_000.0
        ra_sigma_deg_np = np.maximum(ra_sigma_deg_np, floor_deg)
        dec_sigma_deg_np = np.maximum(dec_sigma_deg_np, floor_deg)
    if sigma_cap_mas is not None:
        cap_deg = float(sigma_cap_mas) / 3_600_000.0
        ra_sigma_deg_np = np.minimum(ra_sigma_deg_np, cap_deg)
        dec_sigma_deg_np = np.minimum(dec_sigma_deg_np, cap_deg)

    ra_sigma_deg = pa.array(ra_sigma_deg_np, type=pa.float64())
    dec_sigma_deg = pa.array(dec_sigma_deg_np, type=pa.float64())

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
            "ra_sigma_deg": ra_sigma_deg,
            "dec_sigma_deg": dec_sigma_deg,
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
    detection_sigma_floor_mas: float | None,
    detection_sigma_cap_mas: float | None,
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
                detection_sigma_floor_mas=(
                    None if detection_sigma_floor_mas is None else float(detection_sigma_floor_mas)
                ),
                detection_sigma_cap_mas=(
                    None if detection_sigma_cap_mas is None else float(detection_sigma_cap_mas)
                ),
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


def _empty_orbit_photometry_table() -> pa.Table:
    return pa.table(
        {
            "orbit_id": pa.array([], type=pa.large_string()),
            "H_v": pa.array([], type=pa.float64()),
            "H_v_sigma": pa.array([], type=pa.float64()),
            "G": pa.array([], type=pa.float64()),
            "G_sigma": pa.array([], type=pa.float64()),
            "sigma_eff": pa.array([], type=pa.float64()),
            "chi2_red": pa.array([], type=pa.float64()),
            "n_fit_detections": pa.array([], type=pa.int64()),
        }
    )


def _orbit_photometry_bucket_index(*, orbit_id: str, n_buckets: int) -> int:
    return int(zlib.crc32(str(orbit_id).encode("utf-8")) % max(1, int(n_buckets)))


def _write_orbit_photometry_bucket_inputs(
    *,
    truth_source: str,
    object_orbit_mapping: pa.Table,
    month: str,
    obscode: str,
    bucket_dir: Path,
    n_buckets: int,
    batch_size: int,
) -> list[Path]:
    start_mjd, end_mjd = month_bounds_mjd_utc(month)
    bucket_dir.mkdir(parents=True, exist_ok=True)

    mapping_obj = pc.cast(object_orbit_mapping["object_id"], pa.large_string())
    mapping_orbit = pc.cast(object_orbit_mapping["orbit_id"], pa.large_string())

    writers: dict[int, pq.ParquetWriter] = {}
    bucket_paths: dict[int, Path] = {}

    pf = pq.ParquetFile(str(truth_source))
    for b in pf.iter_batches(
        batch_size=int(max(1, batch_size)),
        columns=[
            "id",
            "object_id",
            "observatory_code",
            "time",
            "ra",
            "dec",
            "mag",
            "mag_sigma",
            "filter",
        ],
    ):
        t = pa.Table.from_batches([b])
        if t.num_rows <= 0:
            continue

        mjd = _mjd_from_time_struct(t["time"])
        obs = np.asarray(pc.cast(t["observatory_code"], pa.large_string()).to_pylist(), dtype=object)
        obj = pc.cast(t["object_id"], pa.large_string())
        mag = pc.cast(t["mag"], pa.float64())

        keep = (mjd >= float(start_mjd)) & (mjd < float(end_mjd)) & (obs == str(obscode))
        keep = keep & np.asarray(pc.is_valid(obj).to_numpy(zero_copy_only=False), dtype=bool)
        keep = keep & np.asarray(
            pc.fill_null(pc.is_finite(mag), False).to_numpy(zero_copy_only=False), dtype=bool
        )
        idx = np.nonzero(keep)[0]
        if idx.size <= 0:
            continue

        tf = t.take(pa.array(idx.astype(np.int64, copy=False), type=pa.int64()))
        tf = pa.table(
            {
                "object_id": pc.cast(tf["object_id"], pa.large_string()),
                "observation_id": pc.cast(tf["id"], pa.large_string()),
                "obscode": pc.cast(tf["observatory_code"], pa.large_string()),
                "time_mjd_utc": pa.array(mjd[idx], type=pa.float64()),
                "ra": pc.cast(tf["ra"], pa.float64()),
                "dec": pc.cast(tf["dec"], pa.float64()),
                "mag": pc.cast(tf["mag"], pa.float64()),
                "mag_sigma": pc.cast(tf["mag_sigma"], pa.float64()),
                "filter": pc.cast(tf["filter"], pa.large_string()),
            }
        )

        idx_map = pc.fill_null(pc.index_in(tf["object_id"], value_set=mapping_obj), -1)
        idx_map_np = np.asarray(idx_map.to_numpy(zero_copy_only=False), dtype=np.int64)
        valid = idx_map_np >= 0
        if not np.any(valid):
            continue

        keep_idx = np.nonzero(valid)[0].astype(np.int64, copy=False)
        tf = tf.take(pa.array(keep_idx, type=pa.int64()))
        orbit_ids = pc.take(mapping_orbit, pa.array(idx_map_np[valid], type=pa.int64()))
        tf = tf.append_column("orbit_id", pc.cast(orbit_ids, pa.large_string()))

        orbit_np = np.asarray(tf["orbit_id"].to_numpy(zero_copy_only=False), dtype=object)
        bucket_np = np.fromiter(
            (_orbit_photometry_bucket_index(orbit_id=str(oid), n_buckets=int(n_buckets)) for oid in orbit_np),
            dtype=np.int64,
            count=int(orbit_np.size),
        )
        if bucket_np.size <= 0:
            continue

        for bidx in np.unique(bucket_np):
            sel = np.nonzero(bucket_np == int(bidx))[0]
            if sel.size <= 0:
                continue
            part = tf.take(pa.array(sel.astype(np.int64, copy=False), type=pa.int64()))
            p = bucket_dir / f"bucket-{int(bidx):03d}.parquet"
            if int(bidx) not in writers:
                writers[int(bidx)] = pq.ParquetWriter(str(p), part.schema)
                bucket_paths[int(bidx)] = p
            writers[int(bidx)].write_table(part)

    for w in writers.values():
        w.close()

    return [bucket_paths[k] for k in sorted(bucket_paths)]


def _build_orbit_photometry_bucket(
    *,
    bucket_parquet: str,
    source_orbits_parquet: str,
    default_G: float,
    composition: str,
) -> pa.Table:
    t = pq.read_table(str(bucket_parquet))
    if t.num_rows <= 0:
        return _empty_orbit_photometry_table()

    joined = t.sort_by([("orbit_id", "ascending"), ("time_mjd_utc", "ascending"), ("observation_id", "ascending")])
    orbit_ids = [str(x) for x in joined["orbit_id"].to_pylist()]
    unique_orbits = sorted(set(orbit_ids))
    if not unique_orbits:
        return _empty_orbit_photometry_table()

    orbits_tbl = ds.dataset(str(source_orbits_parquet), format="parquet").to_table(
        filter=ds.field("orbit_id").isin(unique_orbits)
    )
    if orbits_tbl.num_rows <= 0:
        return _empty_orbit_photometry_table()
    if "physical_parameters" not in orbits_tbl.column_names:
        pp_type = Orbits.empty().table.schema.field("physical_parameters").type
        orbits_tbl = orbits_tbl.append_column("physical_parameters", pa.nulls(orbits_tbl.num_rows, type=pp_type))
    orbits = Orbits.from_pyarrow(orbits_tbl)
    orbit_row_idx = {str(x): i for i, x in enumerate(orbits.orbit_id.to_pylist())}

    obs_id_np = np.asarray(pc.cast(joined["observation_id"], pa.large_string()).to_numpy(zero_copy_only=False), dtype=object)
    obscode_np = np.asarray(pc.cast(joined["obscode"], pa.large_string()).to_numpy(zero_copy_only=False), dtype=object)
    filter_np = np.asarray(pc.cast(joined["filter"], pa.large_string()).to_numpy(zero_copy_only=False), dtype=object)
    time_np = np.asarray(pc.cast(joined["time_mjd_utc"], pa.float64()).to_numpy(zero_copy_only=False), dtype=np.float64)
    ra_np = np.asarray(pc.cast(joined["ra"], pa.float64()).to_numpy(zero_copy_only=False), dtype=np.float64)
    dec_np = np.asarray(pc.cast(joined["dec"], pa.float64()).to_numpy(zero_copy_only=False), dtype=np.float64)
    mag_np = np.asarray(pc.cast(joined["mag"], pa.float64()).to_numpy(zero_copy_only=False), dtype=np.float64)
    mag_sigma_np = np.asarray(pc.cast(joined["mag_sigma"], pa.float64()).to_numpy(zero_copy_only=False), dtype=np.float64)
    orbit_idx_np = np.asarray([orbit_row_idx.get(str(oid), -1) for oid in orbit_ids], dtype=np.int64)

    keep = orbit_idx_np >= 0
    if not np.any(keep):
        return _empty_orbit_photometry_table()

    orbit_idx_np = orbit_idx_np[keep]
    orbit_ids_np = np.asarray(orbit_ids, dtype=object)[keep]
    obs_id_np = obs_id_np[keep]
    obscode_np = obscode_np[keep]
    filter_np = filter_np[keep]
    time_np = time_np[keep]
    ra_np = ra_np[keep]
    dec_np = dec_np[keep]
    mag_np = mag_np[keep]
    mag_sigma_np = mag_sigma_np[keep]

    ts = Timestamp.from_mjd(pa.array(time_np, type=pa.float64()), scale="utc")
    det = PointSourceDetections.from_kwargs(
        id=obs_id_np.tolist(),
        exposure_id=obs_id_np.tolist(),
        time=ts,
        ra=ra_np.tolist(),
        dec=dec_np.tolist(),
        mag=mag_np.tolist(),
        mag_sigma=mag_sigma_np.tolist(),
    )
    exp = Exposures.from_kwargs(
        id=obs_id_np.tolist(),
        start_time=ts,
        duration=[0.0] * int(len(obs_id_np)),
        filter=filter_np.tolist(),
        observatory_code=obscode_np.tolist(),
        seeing=[None] * int(len(obs_id_np)),
        depth_5sigma=[None] * int(len(obs_id_np)),
    )

    # Row-aligned propagation: propagate unique (orbit, time) pairs without creating
    # an NxM Cartesian product.
    unique_times, inv_time = np.unique(time_np, return_inverse=True)
    coords_parts = []
    idx_parts = []
    for tidx, mjd_t in enumerate(unique_times.tolist()):
        row_idx = np.nonzero(inv_time == int(tidx))[0]
        if row_idx.size <= 0:
            continue
        orbit_idx_sub = orbit_idx_np[row_idx]
        uniq_orbit_idx, inv_orbit = np.unique(orbit_idx_sub, return_inverse=True)
        ts_t = Timestamp.from_mjd(pa.array([float(mjd_t)], type=pa.float64()), scale="utc")
        prop_t = propagate_2body(orbits.take(uniq_orbit_idx.tolist()), ts_t, max_processes=1)
        coords_sub = prop_t.coordinates.take(inv_orbit.tolist())
        coords_parts.append(coords_sub)
        idx_parts.append(row_idx.astype(np.int64, copy=False))
    if len(coords_parts) <= 0:
        return _empty_orbit_photometry_table()
    coords_cat = qv.concatenate(coords_parts)
    idx_cat = np.concatenate(idx_parts).astype(np.int64, copy=False)
    order = np.argsort(idx_cat, kind="mergesort")
    coords_row_aligned = coords_cat.take(order.tolist())

    grouped = estimate_absolute_magnitude_v_from_detections_grouped(
        det,
        exp,
        coords_row_aligned,
        orbit_ids_np.tolist(),
        composition=str(composition),
        G=float(default_G),
        strict_band_mapping=False,
    )
    if len(grouped) <= 0:
        return _empty_orbit_photometry_table()

    pp = grouped.physical_parameters
    return pa.table(
        {
            "orbit_id": grouped.object_id,
            "H_v": pp.H_v,
            "H_v_sigma": pp.H_v_sigma,
            "G": pp.G,
            "G_sigma": pp.G_sigma,
            "sigma_eff": pp.sigma_eff,
            "chi2_red": pp.chi2_red,
            "n_fit_detections": grouped.n_fit_detections,
        }
    )


def _build_orbit_photometry_bucket_remote(
    *,
    bucket_parquet: str,
    source_orbits_parquet: str,
    default_G: float,
    composition: str,
) -> pa.Table:
    return _build_orbit_photometry_bucket(
        bucket_parquet=str(bucket_parquet),
        source_orbits_parquet=str(source_orbits_parquet),
        default_G=float(default_G),
        composition=str(composition),
    )


def _build_orbit_photometry_table(
    *,
    truth_source: str,
    source_orbits_parquet: str,
    object_orbit_mapping: pa.Table,
    month: str,
    obscode: str,
    out_path: Path,
    default_G: float,
    composition: str,
) -> pa.Table:
    work_root = out_path.parent / "_orbit_photometry_work"
    bucket_inputs_dir = work_root / "bucket_inputs"
    bucket_inputs_dir.mkdir(parents=True, exist_ok=True)
    bucket_paths = _write_orbit_photometry_bucket_inputs(
        truth_source=str(truth_source),
        object_orbit_mapping=object_orbit_mapping.select(["object_id", "orbit_id"]),
        month=str(month),
        obscode=str(obscode),
        bucket_dir=bucket_inputs_dir,
        n_buckets=128,
        batch_size=200_000,
    )
    if len(bucket_paths) <= 0:
        out = _empty_orbit_photometry_table()
        pq.write_table(out, str(out_path))
        return out

    workers = int(max(1, min(8, len(bucket_paths), int(os.cpu_count() or 1))))
    max_inflight = int(max(2, min(16, workers * 2)))

    grouped_parts: list[pa.Table] = []
    use_ray = bool(workers > 1 and initialize_use_ray(num_cpus=int(workers)))
    if use_ray:
        remote_fn = ray.remote(num_cpus=1)(_build_orbit_photometry_bucket_remote)
        pending: list[ray.ObjectRef] = []
        for p in bucket_paths:
            while len(pending) >= int(max_inflight):
                n_wait = int(max(1, min(len(pending), len(pending) // 2)))
                ready, pending = ray.wait(pending, num_returns=n_wait)
                for tbl in ray.get(ready):
                    if int(tbl.num_rows) > 0:
                        grouped_parts.append(tbl)
            pending.append(
                remote_fn.remote(
                    bucket_parquet=str(p),
                    source_orbits_parquet=str(source_orbits_parquet),
                    default_G=float(default_G),
                    composition=str(composition),
                )
            )
        while pending:
            n_wait = int(max(1, min(len(pending), len(pending) // 2)))
            ready, pending = ray.wait(pending, num_returns=n_wait)
            for tbl in ray.get(ready):
                if int(tbl.num_rows) > 0:
                    grouped_parts.append(tbl)
    else:
        for p in bucket_paths:
            tbl = _build_orbit_photometry_bucket(
                bucket_parquet=str(p),
                source_orbits_parquet=str(source_orbits_parquet),
                default_G=float(default_G),
                composition=str(composition),
            )
            if int(tbl.num_rows) > 0:
                grouped_parts.append(tbl)

    if len(grouped_parts) <= 0:
        out = _empty_orbit_photometry_table()
    else:
        out = (
            grouped_parts[0]
            if len(grouped_parts) == 1
            else pa.concat_tables(grouped_parts, promote_options="default")
        )
        out = out.sort_by([("orbit_id", "ascending")])

    pq.write_table(out, str(out_path))
    return out


def _build_synthetic_orbits(
    *,
    source_orbits_parquet: str,
    out_orbits_parquet: Path,
    model: SyntheticCovarianceModel,
    orbit_photometry: pa.Table | None,
    default_G: float,
    seed: int,
    batch_size: int,
) -> int:
    pf = pq.ParquetFile(str(source_orbits_parquet))
    out_orbits_parquet.parent.mkdir(parents=True, exist_ok=True)

    src_schema = pf.schema_arrow
    writer: pq.ParquetWriter | None = None
    rng = np.random.default_rng(int(seed))
    n_rows = 0
    pp_type = Orbits.empty().table.schema.field("physical_parameters").type
    pp_fields = ["H_v", "H_v_sigma", "G", "G_sigma", "sigma_eff", "chi2_red"]

    phot_by_orbit: dict[str, tuple[float | None, float | None, float | None, float | None, float | None, float | None]] = {}
    if orbit_photometry is not None and orbit_photometry.num_rows > 0:
        for rec in orbit_photometry.to_pylist():
            oid = rec.get("orbit_id")
            if oid is None:
                continue
            phot_by_orbit[str(oid)] = (
                None if rec.get("H_v") is None else float(rec["H_v"]),
                None if rec.get("H_v_sigma") is None else float(rec["H_v_sigma"]),
                None if rec.get("G") is None else float(rec["G"]),
                None if rec.get("G_sigma") is None else float(rec["G_sigma"]),
                None if rec.get("sigma_eff") is None else float(rec["sigma_eff"]),
                None if rec.get("chi2_red") is None else float(rec["chi2_red"]),
            )

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
        orbit_ids = [str(x) for x in pc.cast(out_t["orbit_id"], pa.large_string()).to_pylist()]

        base_vals: dict[str, list[float | None]] = {k: [None] * n for k in pp_fields}
        if "physical_parameters" in out_t.column_names:
            pp_src = out_t["physical_parameters"].combine_chunks()
            for k in pp_fields:
                arr = pc.cast(pc.struct_field(pp_src, k), pa.float64())
                vals = arr.to_pylist()
                base_vals[k] = [None if v is None else float(v) for v in vals]

        for i, oid in enumerate(orbit_ids):
            est = phot_by_orbit.get(str(oid))
            if est is None:
                continue
            base_vals["H_v"][i] = est[0]
            base_vals["H_v_sigma"][i] = est[1]
            base_vals["G"][i] = est[2]
            base_vals["G_sigma"][i] = est[3]
            base_vals["sigma_eff"][i] = est[4]
            base_vals["chi2_red"][i] = est[5]

        for i in range(n):
            if base_vals["H_v"][i] is None:
                continue
            g_i = base_vals["G"][i]
            if g_i is None or not np.isfinite(float(g_i)):
                base_vals["G"][i] = float(default_G)

        pp_struct = pa.StructArray.from_arrays(
            [pa.array(base_vals[k], type=pa.float64()) for k in pp_fields],
            names=pp_fields,
        )
        if "physical_parameters" in out_t.column_names:
            out_t = out_t.set_column(
                out_t.schema.get_field_index("physical_parameters"),
                "physical_parameters",
                pa.chunked_array([pp_struct]),
            )
        else:
            out_t = out_t.append_column(
                "physical_parameters",
                pa.chunked_array([pp_struct], type=pp_type),
            )

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
    photometry_default_G: float,
    photometry_composition: str,
    detection_sigma_floor_mas: float | None,
    detection_sigma_cap_mas: float | None,
    inputs: InputUris,
) -> BuildResult:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = _build_paths(out_dir=out_dir, month=month, obscode=obscode, fast_k=fast_orbits_k)
    paths.detections_dir.mkdir(parents=True, exist_ok=True)
    local_inputs = _materialize_input_uris_local(
        inputs=inputs,
        cache_dir=paths.root / "source_cache",
    )

    # 1) SBDB calibration + synthetic covariance model
    sbdb_table = fetch_sbdb_neo_calibration_table(
        source_orbits_parquet=str(local_inputs.neomod_orbits),
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
        orbits_parquet=str(local_inputs.neomod_orbits),
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
        sources=(str(local_inputs.quad_truth), str(local_inputs.noise_100)),
        out_dir=paths.detections_dir,
        month=str(month),
        obscode=str(obscode),
        healpix_nside=int(healpix_nside),
        batch_size=int(detection_batch_size),
        detection_sigma_floor_mas=(
            None if detection_sigma_floor_mas is None else float(detection_sigma_floor_mas)
        ),
        detection_sigma_cap_mas=(None if detection_sigma_cap_mas is None else float(detection_sigma_cap_mas)),
    )

    # 4) Truth artifacts for canonical month
    n_truth, n_unresolved = _build_truth_detections(
        truth_source=str(local_inputs.quad_truth),
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

    # 5) Orbit photometry fit from truth detections (for non-null pred_mag in benchmark runs).
    orbit_photometry = _build_orbit_photometry_table(
        truth_source=str(local_inputs.quad_truth),
        source_orbits_parquet=str(local_inputs.neomod_orbits),
        object_orbit_mapping=mapping,
        month=str(month),
        obscode=str(obscode),
        out_path=paths.orbit_photometry,
        default_G=float(photometry_default_G),
        composition=str(photometry_composition),
    )

    # 6) Full NEOMOD synthetic-covariance orbit table + physical parameters.
    n_orbits_full = _build_synthetic_orbits(
        source_orbits_parquet=str(local_inputs.neomod_orbits),
        out_orbits_parquet=paths.orbits_synth,
        model=model,
        orbit_photometry=orbit_photometry,
        default_G=float(photometry_default_G),
        seed=int(seed),
        batch_size=int(orbits_batch_size),
    )

    # 7) Subset orbits (truth-bearing + fast)
    orbits_with_truth = _filter_orbits_to_ids(orbits_parquet=paths.orbits_synth, orbit_ids=orbit_ids_with_truth)
    orbits_with_truth.to_parquet(paths.orbits_with_truth)

    fast_ids = _pick_fast_orbits(truth_parquet=paths.truth_detections, k=int(fast_orbits_k))
    _write_orbit_id_table(orbit_ids=fast_ids, out_path=paths.orbit_ids_fast)
    orbits_fast = _filter_orbits_to_ids(orbits_parquet=paths.orbits_synth, orbit_ids=fast_ids)
    orbits_fast.to_parquet(paths.orbits_fast)

    # 8) Validations
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
        "sources_local": {
            "quad_truth": str(local_inputs.quad_truth),
            "noise_100": str(local_inputs.noise_100),
            "neomod_orbits": str(local_inputs.neomod_orbits),
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
            "orbit_photometry_parquet": str(paths.orbit_photometry),
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
            "n_orbits_with_photometry": int(orbit_photometry.num_rows),
        },
        "fast_orbits": {
            "k": int(fast_orbits_k),
            "orbit_ids": [str(x) for x in fast_ids],
        },
        "defaults": {
            "noise_model": "noise_100.000",
            "canonical_month": str(month),
            "observatory_scope": str(obscode),
            "detection_sigma_floor_mas": (
                None if detection_sigma_floor_mas is None else float(detection_sigma_floor_mas)
            ),
            "detection_sigma_cap_mas": (
                None if detection_sigma_cap_mas is None else float(detection_sigma_cap_mas)
            ),
            "photometry_default_G": float(photometry_default_G),
            "photometry_composition": str(photometry_composition),
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
        n_orbits_with_photometry=int(orbit_photometry.num_rows),
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
    p.add_argument(
        "--detection-sigma-floor-mas",
        type=float,
        default=None,
        help="Optional astrometric sigma floor applied after unit conversion (mas).",
    )
    p.add_argument(
        "--detection-sigma-cap-mas",
        type=float,
        default=None,
        help="Optional astrometric sigma cap to shrink sigmas toward LSST-like minima (mas).",
    )
    p.add_argument("--photometry-default-G", type=float, default=0.15)
    p.add_argument("--photometry-composition", type=str, default="NEO")

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
        photometry_default_G=float(args.photometry_default_G),
        photometry_composition=str(args.photometry_composition),
        detection_sigma_floor_mas=(
            None if args.detection_sigma_floor_mas is None else float(args.detection_sigma_floor_mas)
        ),
        detection_sigma_cap_mas=(
            None if args.detection_sigma_cap_mas is None else float(args.detection_sigma_cap_mas)
        ),
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
        "n_orbits_with_photometry": int(result.n_orbits_with_photometry),
        "n_orbits_with_truth": int(result.n_orbits_with_truth),
        "n_orbits_fast": int(result.n_orbits_fast),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
