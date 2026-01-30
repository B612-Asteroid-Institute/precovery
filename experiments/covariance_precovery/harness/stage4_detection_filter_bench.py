from __future__ import annotations

import json
import sqlite3
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import quivr as qv

from adam_core.orbits import Orbits
from adam_core.orbits.ephemeris import Ephemeris
from adam_core.orbits.variants import VariantEphemeris

from precovery.frame_db import HealpixFrame
from precovery.observation import ObservationsTable
from precovery.precovery_db import (
    PrecoveryDatabase,
    find_observation_matches_covariance,
    generate_ephem_for_per_obs_timestamps,
)

from ..methods.detection_filtering import FilterMetrics
from ..methods.footprints import CorridorFootprint, EllipseFootprint, Footprint, SamplePerimeterPolygonFootprint
from .stage3_healpixel_bench import (
    _collapse_variant_ephemeris_group,
    _designation_from_object_id,
    _ephem_key_table,
    _filter_ephem_to_truth,
    _group_slices_by_orbit_target,
    _map_ephem_to_target_idx_by_time,
    _map_times_to_target_idx_by_obscode,
    _predicted_pixels_from_mean_row,
    _predicted_pixels_from_samples,
    _read_stage2_targets,
)


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, obj: dict[str, object]) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class TruthMatches(qv.Table):
    """
    One row per truth observation that was crossmatched to a detection in the subset.
    """

    orbit_id = qv.LargeStringColumn()
    target_idx = qv.Int64Column()
    match_observation_id = qv.LargeStringColumn()


class TruthKeys(qv.Table):
    orbit_id = qv.LargeStringColumn()
    target_idx = qv.Int64Column()


class Stage4Metrics(qv.Table):
    stage2_run_dir = qv.LargeStringColumn()
    subset_dir = qv.LargeStringColumn()
    strategy = qv.LargeStringColumn()
    variant_kind = qv.LargeStringColumn(nullable=True)
    footprint = qv.LargeStringColumn()
    detection_filter = qv.LargeStringColumn()
    healpix_nside = qv.Int64Column()

    n_orbit_targets = qv.Int64Column()
    n_frames_loaded = qv.Int64Column()
    n_observations_loaded = qv.Int64Column()
    n_after_prefilter = qv.Int64Column(nullable=True)
    n_after_chi2 = qv.Int64Column()

    io_sec = qv.Float64Column()
    filter_sec = qv.Float64Column()
    chi2_sec = qv.Float64Column()
    runtime_total_sec = qv.Float64Column()

    n_errors = qv.Int64Column(nullable=True)
    error = qv.LargeStringColumn(nullable=True)


class Stage4Coverage(qv.Table):
    stage2_run_dir = qv.LargeStringColumn()
    subset_dir = qv.LargeStringColumn()
    strategy = qv.LargeStringColumn()
    variant_kind = qv.LargeStringColumn(nullable=True)
    footprint = qv.LargeStringColumn()
    detection_filter = qv.LargeStringColumn()
    healpix_nside = qv.Int64Column()

    n_truth_matched = qv.Int64Column()
    n_recovered = qv.Int64Column()
    recall = qv.Float64Column()


_ObsCacheKey = tuple[str, int, int]


@dataclass
class _FilterAgg:
    # counts
    n_orbit_targets: int = 0
    n_frames_loaded: int = 0
    n_observations_loaded: int = 0
    n_after_prefilter: int | None = 0
    n_after_chi2: int = 0

    # timings
    io_sec: float = 0.0
    filter_sec: float = 0.0
    chi2_sec: float = 0.0

    # errors
    n_errors: int = 0
    first_error: str | None = None

    # Recovered truth pairs, stored explicitly so memory is bounded by truth size
    recovered_truth_pairs: set[tuple[str, str]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.recovered_truth_pairs is None:
            self.recovered_truth_pairs = set()

    def record_error(self, e: BaseException) -> None:
        self.n_errors += 1
        if self.first_error is None:
            self.first_error = f"{type(e).__name__}: {e}"


def _decode_obs_ids(obs: ObservationsTable) -> list[str]:
    out: list[str] = []
    for x in obs.id.to_pylist():
        if isinstance(x, (bytes, bytearray)):
            out.append(x.decode("utf8"))
        else:
            out.append(str(x))
    return out


def _truth_matches_empty() -> pa.Table:
    return pa.table(
        {
            "orbit_id": pa.array([], pa.large_string()),
            "target_idx": pa.array([], pa.int64()),
            "match_observation_id": pa.array([], pa.large_string()),
        }
    )


def _read_truth_matches_table(*, subset_dir: Path, targets: pa.Table) -> pa.Table:
    """
    Return matched truth detections aligned to Stage 2 targets as:
      (orbit_id, target_idx, match_observation_id)
    """
    artifacts_dir = subset_dir / "artifacts"
    truth_path = artifacts_dir / "truth_precovery_crossmatch.parquet"
    orbits_path = artifacts_dir / "orbits_selected_sbdb.parquet"
    if not truth_path.exists():
        raise FileNotFoundError(f"Missing truth crossmatch parquet: {truth_path}")
    if not orbits_path.exists():
        raise FileNotFoundError(f"Missing SBDB orbits parquet: {orbits_path}")

    truth = pq.read_table(
        str(truth_path),
        columns=["matched", "designation", "obscode", "match_time_mjd_utc", "match_observation_id"],
    )
    truth = truth.filter(pc.equal(truth["matched"], True))
    if truth.num_rows == 0:
        return _truth_matches_empty()

    # Map designation -> orbit_id using orbits_selected_sbdb.parquet (robust to historical non-unique orbit_id).
    orbits = pq.read_table(str(orbits_path), columns=["orbit_id", "object_id"])
    orbit_map = pa.table(
        {
            "designation": pa.array(
                [_designation_from_object_id(str(x)) for x in orbits["object_id"].to_pylist()],
                pa.large_string(),
            ),
            "orbit_id": orbits["orbit_id"],
        }
    )
    truth = truth.select(["designation", "obscode", "match_time_mjd_utc", "match_observation_id"])
    truth = truth.join(orbit_map, keys=["designation"], join_type="inner")
    if truth.num_rows == 0:
        return _truth_matches_empty()

    # Map match_time_mjd_utc -> nearest target_idx per obscode (within 60s).
    dt_days = float(60.0) / 86400.0
    t_obscode = np.asarray(truth["obscode"].to_pylist(), dtype=object)
    t_time = np.asarray(truth["match_time_mjd_utc"].to_numpy(zero_copy_only=False), dtype=np.float64)
    target_idx = _map_times_to_target_idx_by_obscode(
        obscode=t_obscode, time_mjd=t_time, targets=targets, dt_days=dt_days
    )
    ok = target_idx >= 0
    if not ok.any():
        return _truth_matches_empty()

    obsid = truth["match_observation_id"].to_pylist()
    orbit_id = truth["orbit_id"].to_pylist()
    # Ensure observation_id is always a string (truth writer uses string, but be defensive).
    obsid_s = ["" if x is None else str(x) for x in obsid]

    out = TruthMatches.from_kwargs(
        orbit_id=[str(x) for x in np.asarray(orbit_id, dtype=object)[ok].tolist()],
        target_idx=[int(x) for x in target_idx[ok].tolist()],
        match_observation_id=[str(x) for x in np.asarray(obsid_s, dtype=object)[ok].tolist()],
    )
    # Drop any rows where match_observation_id is empty (shouldn't happen for matched==True).
    m = pc.not_equal(out.table["match_observation_id"], pa.scalar(""))
    return out.table.filter(m)


def _truth_keys_from_truth_matches(truth_matches: pa.Table) -> pa.Table:
    if truth_matches.num_rows == 0:
        return TruthKeys.empty().table
    qt = TruthKeys.from_pyarrow(truth_matches.select(["orbit_id", "target_idx"]))
    return qt.drop_duplicates(subset=["orbit_id", "target_idx"]).table


def _filter_truth_matches_to_orbit_ids(truth_matches: pa.Table, orbit_ids: set[str]) -> pa.Table:
    if truth_matches.num_rows == 0 or not orbit_ids:
        return _truth_matches_empty()
    mask = pc.is_in(
        truth_matches["orbit_id"],
        value_set=pa.array(sorted(orbit_ids), type=pa.large_string()),
    )
    return truth_matches.filter(mask)


def _allowed_orbit_ids(*, orbit_ids_in_strategy: set[str], max_orbits: int | None) -> set[str]:
    if max_orbits is None:
        return orbit_ids_in_strategy
    # Deterministic truncation for benchmarking.
    return set(sorted(orbit_ids_in_strategy)[: int(max_orbits)])


def _query_frames_for_pixels(
    *,
    conn: sqlite3.Connection,
    obscode: str,
    exposure_mjd_mid: float,
    healpixels: np.ndarray,
    mjd_tol_days: float = 1e-7,
    max_in_clause: int = 900,
) -> list[dict[str, object]]:
    if healpixels.size == 0:
        return []
    hpix = np.unique(np.asarray(healpixels, dtype=np.int64))

    q0 = float(exposure_mjd_mid) - float(mjd_tol_days)
    q1 = float(exposure_mjd_mid) + float(mjd_tol_days)

    rows: list[tuple[object, ...]] = []
    for i0 in range(0, int(hpix.size), int(max_in_clause)):
        chunk = hpix[i0 : i0 + int(max_in_clause)]
        q_marks = ",".join(["?"] * int(len(chunk)))
        rows.extend(
            conn.execute(
                f"""
                SELECT
                  dataset_id, obscode, exposure_id, filter,
                  exposure_mjd_start, exposure_mjd_mid, exposure_duration,
                  healpixel, data_uri, data_offset, data_length
                FROM frames
                WHERE obscode = ?
                  AND exposure_mjd_mid >= ?
                  AND exposure_mjd_mid <= ?
                  AND healpixel IN ({q_marks})
                """,
                (str(obscode), float(q0), float(q1), *[int(x) for x in chunk.tolist()]),
            ).fetchall()
        )

    out: list[dict[str, object]] = []
    for r in rows:
        out.append(
            dict(
                dataset_id=str(r[0]),
                obscode=str(r[1]),
                exposure_id=str(r[2]),
                filter=str(r[3]),
                exposure_mjd_start=float(r[4]),
                exposure_mjd_mid=float(r[5]),
                exposure_duration=float(r[6]),
                healpixel=int(r[7]),
                data_uri=str(r[8]),
                data_offset=int(r[9]),
                data_length=int(r[10]),
            )
        )
    return out


def _load_observations_for_frames(
    db: PrecoveryDatabase,
    frames: list[dict[str, object]],
    *,
    obs_cache: OrderedDict[_ObsCacheKey, ObservationsTable] | None = None,
    obs_cache_max_frames: int = 0,
) -> ObservationsTable:
    if not frames:
        return ObservationsTable.empty()

    out_list: list[ObservationsTable] = []
    for fr in frames:
        key = (str(fr["data_uri"]), int(fr["data_offset"]), int(fr["data_length"]))
        if obs_cache is not None and key in obs_cache:
            obs = obs_cache[key]
            obs_cache.move_to_end(key)
            out_list.append(obs)
            continue

        hf = HealpixFrame.from_kwargs(
            dataset_id=[str(fr["dataset_id"])],
            obscode=[str(fr["obscode"])],
            exposure_id=[str(fr["exposure_id"])],
            filter=[str(fr["filter"])],
            exposure_mjd_start=[float(fr["exposure_mjd_start"])],
            exposure_mjd_mid=[float(fr["exposure_mjd_mid"])],
            exposure_duration=[float(fr["exposure_duration"])],
            healpixel=[int(fr["healpixel"])],
            data_uri=[str(fr["data_uri"])],
            data_offset=[int(fr["data_offset"])],
            data_length=[int(fr["data_length"])],
        )
        obs = db.frames.get_observations(hf)
        out_list.append(obs)

        if obs_cache is not None and int(obs_cache_max_frames) > 0:
            obs_cache[key] = obs
            obs_cache.move_to_end(key)
            while len(obs_cache) > int(obs_cache_max_frames):
                obs_cache.popitem(last=False)

    if not out_list:
        return ObservationsTable.empty()
    return qv.concatenate(out_list)


def _repeat_one_row_ephem(ephem_row: Ephemeris, n: int) -> Ephemeris:
    if n <= 0:
        return Ephemeris.empty()
    idx = np.zeros(int(n), dtype=int)
    return Ephemeris.from_pyarrow(ephem_row.table.take(idx))


def _filter_chi2_only(
    *,
    observations: ObservationsTable,
    ephem: Ephemeris,
    n_sigma: float,
) -> tuple[ObservationsTable, Ephemeris, FilterMetrics, float, float]:
    t0 = time.perf_counter()
    obs2, eph2 = find_observation_matches_covariance(observations, ephem, n_sigma=float(n_sigma))
    chi2_sec = time.perf_counter() - t0
    return (
        obs2,
        eph2,
        FilterMetrics(n_in=len(observations), n_after_footprint=None, n_after_chi2=len(obs2)),
        0.0,
        float(chi2_sec),
    )


def _filter_footprint_then_chi2(
    *,
    observations: ObservationsTable,
    ephem: Ephemeris,
    footprint: Footprint | None,
    n_sigma: float,
) -> tuple[ObservationsTable, Ephemeris, FilterMetrics, float, float]:
    if footprint is None:
        return _filter_chi2_only(observations=observations, ephem=ephem, n_sigma=n_sigma)

    t0 = time.perf_counter()
    lon = observations.ra.to_numpy(zero_copy_only=False).astype(np.float64)
    lat = observations.dec.to_numpy(zero_copy_only=False).astype(np.float64)
    keep = footprint.contains(lon, lat)
    filter_sec = time.perf_counter() - t0

    obs_fp = observations.apply_mask(pa.array(keep))
    eph_fp = ephem.apply_mask(pa.array(keep))
    if len(obs_fp) == 0:
        empty = ObservationsTable.empty()
        return (
            empty,
            ephem.take(np.array([], dtype=np.int64)),
            FilterMetrics(n_in=len(observations), n_after_footprint=0, n_after_chi2=0),
            float(filter_sec),
            0.0,
        )

    t0 = time.perf_counter()
    obs2, eph2 = find_observation_matches_covariance(obs_fp, eph_fp, n_sigma=float(n_sigma))
    chi2_sec = time.perf_counter() - t0
    return (
        obs2,
        eph2,
        FilterMetrics(n_in=len(observations), n_after_footprint=len(obs_fp), n_after_chi2=len(obs2)),
        float(filter_sec),
        float(chi2_sec),
    )


def _filter_fn(name: str):
    if name == "chi2_only":
        return _filter_chi2_only
    if name == "footprint_then_chi2":
        return _filter_footprint_then_chi2
    raise ValueError(f"Unknown detection filter: {name}")

def _truth_pairs(truth_matches: pa.Table) -> set[tuple[str, str]]:
    if truth_matches.num_rows == 0:
        return set()
    oid = [str(x) for x in truth_matches["orbit_id"].to_pylist()]
    obsid = [str(x) for x in truth_matches["match_observation_id"].to_pylist()]
    return set(zip(oid, obsid))


def run_stage4_detection_filter_bench(
    *,
    subset_dir: Path,
    stage2_run_dir: Path,
    healpix_nside: int,
    n_sigma: float = 3.0,
    polygon_vertices: int = 32,
    cov_mc_num_samples: int = 64,
    cov_mc_seed: int = 0,
    corridor_radius_arcsec: float = 30.0,
    corridor_step_arcsec: float = 30.0,
    out_dir: Path | None = None,
    strategies: list[str] | None = None,
    filters: list[str] | None = None,
    only_truth: bool = False,
    max_orbits: int | None = None,
    max_targets: int | None = None,
    fallback_per_obs_ephem: bool = False,
    obs_cache_max_frames: int = 0,
) -> Path:
    """
    Stage 4 (atomic): benchmark detection-level filtering variants after loading observations.

    Outputs:
      - metrics.parquet: runtime + counts by (strategy, footprint, detection_filter)
      - coverage.parquet: truth-detection recall by (strategy, footprint, detection_filter)
    """
    if out_dir is None:
        out_dir = subset_dir / "artifacts" / "stage4"
    _ensure_dir(out_dir)
    run_dir = out_dir / stage2_run_dir.name
    _ensure_dir(run_dir)

    targets_tbl = _read_stage2_targets(stage2_run_dir)
    truth_all = _read_truth_matches_table(subset_dir=subset_dir, targets=targets_tbl)

    db = PrecoveryDatabase.from_dir(str(subset_dir), allow_version_mismatch=True)

    # Targets arrays for quick lookup by target_idx.
    targ_obscode = np.asarray(targets_tbl["obscode"].to_pylist(), dtype=object)
    targ_mjd = np.asarray(targets_tbl["exposure_mjd_mid"].to_numpy(zero_copy_only=False), dtype=np.float64)

    index_db = subset_dir / "index.db"
    if not index_db.exists():
        raise FileNotFoundError(f"Missing subset index.db: {index_db}")

    # Detection filters.
    if filters is None:
        filters = ["chi2_only", "footprint_then_chi2"]
    filters = [str(x).strip() for x in filters if str(x).strip()]
    for f in filters:
        _ = _filter_fn(f)  # validate

    # Optional fallback: per-observation ephemeris generation requires orbits and a propagator.
    orbits: Orbits | None = None
    orbit_row_by_designation: dict[str, int] | None = None
    propagator = None
    if bool(fallback_per_obs_ephem):
        try:
            from adam_assist import ASSISTPropagator  # type: ignore

            meta_path = stage2_run_dir / "meta.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
                orbits_path = Path(str(meta.get("orbits_parquet_used", "")))
            else:
                orbits_path = subset_dir / "artifacts" / "orbits_selected_sbdb.parquet"

            if not orbits_path.exists():
                raise FileNotFoundError(f"Missing orbits parquet for fallback: {orbits_path}")
            orbits = Orbits.from_parquet(str(orbits_path))
            obj = orbits.object_id.to_pylist() if getattr(orbits, "object_id", None) is not None else None
            if obj is None:
                raise ValueError("Fallback requires Orbits.object_id to build designation mapping.")
            orbit_row_by_designation = {
                _designation_from_object_id(str(object_id)): int(i) for i, object_id in enumerate(obj)
            }
            propagator = ASSISTPropagator()
        except BaseException as e:  # noqa: BLE001
            raise RuntimeError(
                "fallback_per_obs_ephem requested, but required dependencies/data are unavailable"
            ) from e

    metrics_rows: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []

    strategies_root = stage2_run_dir / "strategies"
    if not strategies_root.exists():
        raise FileNotFoundError(f"Missing Stage 2 strategies dir: {strategies_root}")

    def _enabled(name: str) -> bool:
        return strategies is None or name in strategies

    def _enabled_variant(strategy: str, variant_kind: str) -> bool:
        key = f"{strategy}:{variant_kind}"
        return strategies is None or key in strategies or strategy in strategies

    # Reuse a single sqlite connection for all frame queries.
    conn = sqlite3.connect(str(index_db))
    try:
        obs_cache: OrderedDict[_ObsCacheKey, ObservationsTable] | None = (
            OrderedDict() if int(obs_cache_max_frames) > 0 else None
        )

        # Mean strategies (point/covariance-derived footprints).
        for strat_dir in sorted(strategies_root.glob("*")):
            if not strat_dir.is_dir():
                continue
            name = strat_dir.name
            if name in {"assist_variants", "assist_window_then_2body_variants"}:
                continue
            if not _enabled(name):
                continue
            mean_dir = strat_dir / "mean_ephemeris"
            if not mean_dir.exists():
                continue
            part_files = sorted(mean_dir.glob("part-*.parquet"))
            if not part_files:
                continue

            orbit_ids_in_strategy: set[str]
            try:
                ep_first = Ephemeris.from_parquet(str(part_files[0]))
                orbit_ids_in_strategy = set(_ephem_key_table(ep_first)["orbit_id"].to_pylist())
            except Exception:  # noqa: BLE001
                orbit_ids_in_strategy = set()

            allowed_orbits = _allowed_orbit_ids(
                orbit_ids_in_strategy=orbit_ids_in_strategy, max_orbits=max_orbits
            )
            truth_strategy = _filter_truth_matches_to_orbit_ids(truth_all, allowed_orbits)
            if max_targets is not None and truth_strategy.num_rows > 0:
                truth_strategy = truth_strategy.filter(
                    pc.less(truth_strategy["target_idx"], pa.scalar(int(max_targets)))
                )
            truth_keys_strategy = _truth_keys_from_truth_matches(truth_strategy) if truth_strategy.num_rows > 0 else TruthKeys.empty().table
            truth_pairs = _truth_pairs(truth_strategy)
            n_truth = int(len(truth_pairs))

            # Determine whether covariance is present for this strategy.
            try:
                ep0 = Ephemeris.from_parquet(str(part_files[0]))
                has_cov = (
                    ep0.coordinates.covariance is not None
                    and (not ep0.coordinates.covariance.is_all_nan())
                )
            except Exception:  # noqa: BLE001
                has_cov = False

            mean_footprints = ["point"] + ([] if not has_cov else ["cov_disc", "cov_polygon", "cov_mc", "cov_polygon_moc"])
            needs_time_map = name == "assist_window_then_2body"

            for footprint in mean_footprints:
                aggs: dict[str, _FilterAgg] = {f: _FilterAgg() for f in filters}

                for pf in part_files:
                    ephem = Ephemeris.from_parquet(str(pf))
                    if len(ephem) == 0:
                        continue

                    # Map target_idx for this part.
                    if needs_time_map:
                        target_idx = _map_ephem_to_target_idx_by_time(
                            ephem=ephem, targets=targets_tbl, dt_days=float(60.0) / 86400.0
                        )
                        keep = target_idx >= 0
                        if not keep.any():
                            continue
                        if not keep.all():
                            hit = np.nonzero(keep)[0]
                            ephem = ephem.take(hit.tolist())
                            target_idx = target_idx[hit]
                    else:
                        meta = json.loads((strat_dir / "meta.json").read_text())
                        n_orbits = int(meta["n_orbits"])
                        chunk = int(meta.get("time_chunk_size", 1024))
                        n_targets = int(meta.get("n_time_targets", len(targets_tbl)))
                        part_idx = int(pf.stem.split("-")[-1])
                        start = part_idx * chunk
                        chunk_len = int(min(chunk, n_targets - start))
                        if chunk_len <= 0:
                            continue
                        target_idx = np.tile(
                            (np.arange(chunk_len, dtype=np.int64) + start), int(n_orbits)
                        )
                        target_idx = target_idx[: int(len(ephem))]

                    if max_targets is not None:
                        m = target_idx < int(max_targets)
                        if not m.any():
                            continue
                        if not m.all():
                            hit = np.nonzero(m)[0]
                            ephem = ephem.take(hit.tolist())
                            target_idx = target_idx[hit]

                    if bool(only_truth):
                        mask = _filter_ephem_to_truth(
                            ephem=ephem, target_idx=target_idx, truth_keys=truth_keys_strategy
                        )
                        hit = np.nonzero(mask)[0]
                        if hit.size == 0:
                            continue
                        ephem = ephem.take(hit.tolist())
                        target_idx = target_idx[hit]

                    if len(ephem) == 0:
                        continue

                    orbit_id = np.asarray(_ephem_key_table(ephem)["orbit_id"].to_pylist(), dtype=object)
                    lon = ephem.coordinates.lon.to_numpy(zero_copy_only=False).astype(np.float64)
                    lat = ephem.coordinates.lat.to_numpy(zero_copy_only=False).astype(np.float64)

                    cov_ll: np.ndarray | None = None
                    if ephem.coordinates.covariance is not None and (not ephem.coordinates.covariance.is_all_nan()):
                        cov6 = ephem.coordinates.covariance.to_matrix().astype(np.float64)
                        cov_ll = cov6[:, 1:3, 1:3]

                    for i in range(len(ephem)):
                        oid = str(orbit_id[i])
                        if allowed_orbits and oid not in allowed_orbits:
                            continue

                        tidx = int(target_idx[i])
                        if tidx < 0 or tidx >= len(targ_mjd):
                            continue

                        # Predicted healpixels used to select candidate frames.
                        cov_row = None if cov_ll is None else cov_ll[i]
                        try:
                            pred_pix = _predicted_pixels_from_mean_row(
                                lon_deg=float(lon[i]),
                                lat_deg=float(lat[i]),
                                cov_ll_deg2=cov_row,
                                nside=int(healpix_nside),
                                footprint=str(footprint),
                                n_sigma=float(n_sigma),
                                polygon_vertices=int(polygon_vertices),
                                mc_num_samples=int(cov_mc_num_samples),
                                mc_seed=int(cov_mc_seed),
                            )
                        except BaseException as e:  # noqa: BLE001
                            for agg in aggs.values():
                                agg.record_error(e)
                            continue

                        # Footprint object for prefilter (if applicable).
                        fp_obj: Footprint | None = None
                        if footprint != "point" and cov_row is not None:
                            fp_obj = EllipseFootprint(
                                lon0_deg=float(lon[i]),
                                lat0_deg=float(lat[i]),
                                cov_ll_deg2=np.asarray(cov_row, dtype=np.float64),
                                n_sigma=float(n_sigma),
                                polygon_vertices=int(polygon_vertices),
                                raster_mode=("disc" if footprint == "cov_disc" else "polygon"),
                            )

                        # Load frames + observations (I/O).
                        obscode = str(targ_obscode[tidx])
                        mjd_mid = float(targ_mjd[tidx])
                        t_io0 = time.perf_counter()
                        frames = _query_frames_for_pixels(
                            conn=conn,
                            obscode=obscode,
                            exposure_mjd_mid=mjd_mid,
                            healpixels=pred_pix,
                        )
                        obs = _load_observations_for_frames(
                            db, frames, obs_cache=obs_cache, obs_cache_max_frames=int(obs_cache_max_frames)
                        )
                        io_sec = time.perf_counter() - t_io0

                        # Ephemeris row for this (orbit,target) (must include covariance for chi2).
                        ephem_row = ephem.take([int(i)])

                        # If fallback per-observation ephemeris is requested and observation times differ,
                        # build a per-observation ephemeris using the orbit state.
                        rep = _repeat_one_row_ephem(ephem_row, len(obs))
                        if bool(fallback_per_obs_ephem) and len(obs) > 0:
                            try:
                                ok = pc.all(
                                    rep.coordinates.time.rescale("utc").equals(
                                        obs.time.rescale("utc"), precision="ms"
                                    )
                                ).as_py()
                            except Exception:  # noqa: BLE001
                                ok = False
                            if not ok:
                                try:
                                    if orbits is None or orbit_row_by_designation is None or propagator is None:
                                        raise RuntimeError("Fallback requested but not initialized.")
                                    j = orbit_row_by_designation.get(oid)
                                    if j is None:
                                        raise KeyError(f"orbit_id not found in orbits parquet: {oid}")
                                    orb1 = orbits.take([int(j)])
                                    rep = generate_ephem_for_per_obs_timestamps(  # type: ignore[name-defined]
                                        orb1, obs, obscode, propagator
                                    )
                                except BaseException as e:  # noqa: BLE001
                                    for agg in aggs.values():
                                        agg.record_error(e)
                                    continue

                        # Update common counts/timing per filter (same I/O for all filters).
                        for agg in aggs.values():
                            agg.n_orbit_targets += 1
                            agg.n_frames_loaded += int(len(frames))
                            agg.n_observations_loaded += int(len(obs))
                            agg.io_sec += float(io_sec)

                        if len(obs) == 0:
                            continue

                        # Apply each detection filter variant.
                        for filt_name, agg in aggs.items():
                            try:
                                fn = _filter_fn(filt_name)
                                if filt_name == "chi2_only":
                                    obs2, _, met, filt_sec, chi2_sec = fn(
                                        observations=obs, ephem=rep, n_sigma=float(n_sigma)
                                    )
                                else:
                                    obs2, _, met, filt_sec, chi2_sec = fn(
                                        observations=obs,
                                        ephem=rep,
                                        footprint=fp_obj,
                                        n_sigma=float(n_sigma),
                                    )
                            except BaseException as e:  # noqa: BLE001
                                if isinstance(e, (KeyboardInterrupt, SystemExit)):
                                    raise
                                agg.record_error(e)
                                continue

                            # Aggregate counts/timings.
                            if met.n_after_footprint is None:
                                agg.n_after_prefilter = None
                            else:
                                if agg.n_after_prefilter is not None:
                                    agg.n_after_prefilter += int(met.n_after_footprint)
                            agg.n_after_chi2 += int(met.n_after_chi2)
                            agg.filter_sec += float(filt_sec)
                            agg.chi2_sec += float(chi2_sec)

                            # Accepted detections (for recall attribution).
                            if len(obs2) > 0:
                                ids = _decode_obs_ids(obs2)
                                for obs_id in ids:
                                    pair = (oid, obs_id)
                                    if pair in truth_pairs:
                                        agg.recovered_truth_pairs.add(pair)

                # Emit metrics + coverage rows per detection filter.
                for filt_name, agg in aggs.items():
                    runtime_total = float(agg.io_sec + agg.filter_sec + agg.chi2_sec)
                    metrics_rows.append(
                        dict(
                            stage2_run_dir=str(stage2_run_dir),
                            subset_dir=str(subset_dir),
                            strategy=name,
                            variant_kind=None,
                            footprint=str(footprint),
                            detection_filter=str(filt_name),
                            healpix_nside=int(healpix_nside),
                            n_orbit_targets=int(agg.n_orbit_targets),
                            n_frames_loaded=int(agg.n_frames_loaded),
                            n_observations_loaded=int(agg.n_observations_loaded),
                            n_after_prefilter=(None if agg.n_after_prefilter is None else int(agg.n_after_prefilter)),
                            n_after_chi2=int(agg.n_after_chi2),
                            io_sec=float(agg.io_sec),
                            filter_sec=float(agg.filter_sec),
                            chi2_sec=float(agg.chi2_sec),
                            runtime_total_sec=float(runtime_total),
                            n_errors=(None if agg.n_errors == 0 else int(agg.n_errors)),
                            error=agg.first_error,
                        )
                    )

                    n_rec = int(len(agg.recovered_truth_pairs))
                    coverage_rows.append(
                        dict(
                            stage2_run_dir=str(stage2_run_dir),
                            subset_dir=str(subset_dir),
                            strategy=name,
                            variant_kind=None,
                            footprint=str(footprint),
                            detection_filter=str(filt_name),
                            healpix_nside=int(healpix_nside),
                            n_truth_matched=int(n_truth),
                            n_recovered=int(n_rec),
                            recall=(0.0 if n_truth == 0 else float(n_rec) / float(n_truth)),
                        )
                    )

        # Variant strategies (sample-derived and reconstructed-covariance footprints).
        for variant_root_name in ["assist_variants", "assist_window_then_2body_variants"]:
            root = strategies_root / variant_root_name
            if not root.exists():
                continue
            for strat_dir in sorted(root.glob("*")):
                if not strat_dir.is_dir():
                    continue
                variant_kind = strat_dir.name
                if not _enabled_variant(variant_root_name, variant_kind):
                    continue
                ephem_dir = strat_dir / "variants_ephemeris"
                if not ephem_dir.exists():
                    continue
                part_files = sorted(ephem_dir.glob("part-*.parquet"))
                if not part_files:
                    continue

                orbit_ids_in_strategy: set[str]
                try:
                    ep_first = VariantEphemeris.from_parquet(str(part_files[0]))
                    orbit_ids_in_strategy = set(_ephem_key_table(ep_first)["orbit_id"].to_pylist())
                except Exception:  # noqa: BLE001
                    orbit_ids_in_strategy = set()

                allowed_orbits = _allowed_orbit_ids(
                    orbit_ids_in_strategy=orbit_ids_in_strategy, max_orbits=max_orbits
                )
                truth_strategy = _filter_truth_matches_to_orbit_ids(truth_all, allowed_orbits)
                if max_targets is not None and truth_strategy.num_rows > 0:
                    truth_strategy = truth_strategy.filter(
                        pc.less(truth_strategy["target_idx"], pa.scalar(int(max_targets)))
                    )
                truth_keys_strategy = _truth_keys_from_truth_matches(truth_strategy) if truth_strategy.num_rows > 0 else TruthKeys.empty().table
                truth_pairs = _truth_pairs(truth_strategy)
                n_truth = int(len(truth_pairs))

                variant_footprints = [
                    ("sample_direct", None),
                    ("sample_polygon", "angle_sort"),
                    ("sample_polygon", "convex_hull"),
                    ("sample_polygon_moc", "angle_sort"),
                    ("sample_polygon_moc", "convex_hull"),
                    ("sample_corridor", None),
                    ("cov_disc_reconstructed", None),
                    ("cov_polygon_reconstructed", None),
                    ("cov_polygon_reconstructed_moc", None),
                    ("cov_mc_reconstructed", None),
                ]

                for footprint, polygon_mode in variant_footprints:
                    out_fp = (
                        f"{footprint}:{polygon_mode}"
                        if (footprint in {"sample_polygon", "sample_polygon_moc"} and polygon_mode is not None)
                        else str(footprint)
                    )
                    aggs: dict[str, _FilterAgg] = {f: _FilterAgg() for f in filters}

                    for pf in part_files:
                        ephem = VariantEphemeris.from_parquet(str(pf))
                        if len(ephem) == 0:
                            continue

                        target_idx = _map_ephem_to_target_idx_by_time(
                            ephem=ephem, targets=targets_tbl, dt_days=float(60.0) / 86400.0
                        )
                        keep = target_idx >= 0
                        if not keep.any():
                            continue
                        if not keep.all():
                            hit = np.nonzero(keep)[0]
                            ephem = ephem.take(hit.tolist())
                            target_idx = target_idx[hit]

                        if max_targets is not None:
                            m = target_idx < int(max_targets)
                            if not m.any():
                                continue
                            if not m.all():
                                hit = np.nonzero(m)[0]
                                ephem = ephem.take(hit.tolist())
                                target_idx = target_idx[hit]

                        if bool(only_truth):
                            mask = _filter_ephem_to_truth(
                                ephem=ephem, target_idx=target_idx, truth_keys=truth_keys_strategy
                            )
                            hit = np.nonzero(mask)[0]
                            if hit.size == 0:
                                continue
                            ephem = ephem.take(hit.tolist())
                            target_idx = target_idx[hit]

                        if len(ephem) == 0:
                            continue

                        orbit_id = np.asarray(_ephem_key_table(ephem)["orbit_id"].to_pylist(), dtype=object)
                        lon = ephem.coordinates.lon.to_numpy(zero_copy_only=False).astype(np.float64)
                        lat = ephem.coordinates.lat.to_numpy(zero_copy_only=False).astype(np.float64)
                        order, starts = _group_slices_by_orbit_target(orbit_id, target_idx.astype(np.int64))
                        if starts.size == 0:
                            continue
                        ends = np.concatenate([starts[1:], np.array([len(order)], dtype=np.int64)])

                        for s, e in zip(starts.tolist(), ends.tolist()):
                            idx = order[s:e]
                            if idx.size == 0:
                                continue
                            oid = str(orbit_id[idx[0]])
                            if allowed_orbits and oid not in allowed_orbits:
                                continue
                            tidx = int(target_idx[idx[0]])
                            if tidx < 0 or tidx >= len(targ_mjd):
                                continue

                            lon_g = lon[idx]
                            lat_g = lat[idx]

                            # For any chi2-based filter we need a covariance-bearing mean ephemeris row.
                            try:
                                collapsed = _collapse_variant_ephemeris_group(variants=ephem.take(idx.tolist()))
                                lon0 = float(collapsed.coordinates.lon[0].as_py())
                                lat0 = float(collapsed.coordinates.lat[0].as_py())
                                cov6 = collapsed.coordinates.covariance.to_matrix()[0].astype(np.float64)
                                cov_ll = cov6[1:3, 1:3]
                            except BaseException as e:  # noqa: BLE001
                                for agg in aggs.values():
                                    agg.record_error(e)
                                continue

                            # Compute predicted pixels for frame selection.
                            try:
                                if "_reconstructed" in str(footprint):
                                    if str(footprint).endswith("_reconstructed_moc"):
                                        base = str(footprint).replace("_reconstructed_moc", "_moc")
                                    else:
                                        base = str(footprint).replace("_reconstructed", "")
                                    pred_pix = _predicted_pixels_from_mean_row(
                                        lon_deg=float(lon0),
                                        lat_deg=float(lat0),
                                        cov_ll_deg2=cov_ll,
                                        nside=int(healpix_nside),
                                        footprint=str(base),
                                        n_sigma=float(n_sigma),
                                        polygon_vertices=int(polygon_vertices),
                                        mc_num_samples=int(cov_mc_num_samples),
                                        mc_seed=int(cov_mc_seed),
                                    )
                                else:
                                    pred_pix = _predicted_pixels_from_samples(
                                        lon_deg=lon_g,
                                        lat_deg=lat_g,
                                        nside=int(healpix_nside),
                                        footprint=str(footprint),
                                        polygon_mode=("angle_sort" if polygon_mode is None else str(polygon_mode)),
                                        corridor_radius_arcsec=float(corridor_radius_arcsec),
                                        corridor_step_arcsec=float(corridor_step_arcsec),
                                    )
                            except BaseException as e:  # noqa: BLE001
                                for agg in aggs.values():
                                    agg.record_error(e)
                                continue

                            # Footprint object for prefilter (where meaningful).
                            fp_obj: Footprint | None = None
                            if "_reconstructed" in str(footprint):
                                fp_obj = EllipseFootprint(
                                    lon0_deg=float(lon0),
                                    lat0_deg=float(lat0),
                                    cov_ll_deg2=cov_ll.astype(np.float64, copy=False),
                                    n_sigma=float(n_sigma),
                                    polygon_vertices=int(polygon_vertices),
                                    raster_mode=("disc" if str(footprint).startswith("cov_disc") else "polygon"),
                                )
                            elif str(footprint) in {"sample_polygon", "sample_polygon_moc"}:
                                fp_obj = SamplePerimeterPolygonFootprint(
                                    lon0_deg=float(lon_g[0]),
                                    lat0_deg=float(lat_g[0]),
                                    sample_lon_deg=lon_g.astype(np.float64, copy=False),
                                    sample_lat_deg=lat_g.astype(np.float64, copy=False),
                                    polygon_mode=("angle_sort" if str(polygon_mode) == "angle_sort" else "convex_hull"),
                                    buffer_arcsec=0.0,
                                )
                            elif str(footprint) == "sample_corridor":
                                fp_obj = CorridorFootprint(
                                    lon0_deg=float(lon_g[0]),
                                    lat0_deg=float(lat_g[0]),
                                    path_lon_deg=lon_g.astype(np.float64, copy=False),
                                    path_lat_deg=lat_g.astype(np.float64, copy=False),
                                    radius_arcsec=float(corridor_radius_arcsec),
                                    raster_step_arcsec=float(corridor_step_arcsec),
                                )

                            obscode = str(targ_obscode[tidx])
                            mjd_mid = float(targ_mjd[tidx])
                            t_io0 = time.perf_counter()
                            frames = _query_frames_for_pixels(
                                conn=conn,
                                obscode=obscode,
                                exposure_mjd_mid=mjd_mid,
                                healpixels=pred_pix,
                            )
                            obs = _load_observations_for_frames(
                                db, frames, obs_cache=obs_cache, obs_cache_max_frames=int(obs_cache_max_frames)
                            )
                            io_sec = time.perf_counter() - t_io0

                            # Repeat collapsed ephemeris row to match observation count.
                            rep = _repeat_one_row_ephem(collapsed, len(obs))
                            if bool(fallback_per_obs_ephem) and len(obs) > 0:
                                try:
                                    ok = pc.all(
                                        rep.coordinates.time.rescale("utc").equals(
                                            obs.time.rescale("utc"), precision="ms"
                                        )
                                    ).as_py()
                                except Exception:  # noqa: BLE001
                                    ok = False
                                if not ok:
                                    try:
                                        if orbits is None or orbit_row_by_designation is None or propagator is None:
                                            raise RuntimeError("Fallback requested but not initialized.")
                                        j = orbit_row_by_designation.get(oid)
                                        if j is None:
                                            raise KeyError(f"orbit_id not found in orbits parquet: {oid}")
                                        orb1 = orbits.take([int(j)])
                                        rep = generate_ephem_for_per_obs_timestamps(  # type: ignore[name-defined]
                                            orb1, obs, obscode, propagator
                                        )
                                    except BaseException as e:  # noqa: BLE001
                                        for agg in aggs.values():
                                            agg.record_error(e)
                                        continue

                            for agg in aggs.values():
                                agg.n_orbit_targets += 1
                                agg.n_frames_loaded += int(len(frames))
                                agg.n_observations_loaded += int(len(obs))
                                agg.io_sec += float(io_sec)

                            if len(obs) == 0:
                                continue

                            for filt_name, agg in aggs.items():
                                try:
                                    fn = _filter_fn(filt_name)
                                    if filt_name == "chi2_only":
                                        obs2, _, met, filt_sec, chi2_sec = fn(
                                            observations=obs, ephem=rep, n_sigma=float(n_sigma)
                                        )
                                    else:
                                        obs2, _, met, filt_sec, chi2_sec = fn(
                                            observations=obs,
                                            ephem=rep,
                                            footprint=fp_obj,
                                            n_sigma=float(n_sigma),
                                        )
                                except BaseException as e:  # noqa: BLE001
                                    if isinstance(e, (KeyboardInterrupt, SystemExit)):
                                        raise
                                    agg.record_error(e)
                                    continue

                                if met.n_after_footprint is None:
                                    agg.n_after_prefilter = None
                                else:
                                    if agg.n_after_prefilter is not None:
                                        agg.n_after_prefilter += int(met.n_after_footprint)
                                agg.n_after_chi2 += int(met.n_after_chi2)
                                agg.filter_sec += float(filt_sec)
                                agg.chi2_sec += float(chi2_sec)

                                if len(obs2) > 0:
                                    ids = _decode_obs_ids(obs2)
                                    for obs_id in ids:
                                        pair = (oid, obs_id)
                                        if pair in truth_pairs:
                                            agg.recovered_truth_pairs.add(pair)

                    for filt_name, agg in aggs.items():
                        runtime_total = float(agg.io_sec + agg.filter_sec + agg.chi2_sec)
                        metrics_rows.append(
                            dict(
                                stage2_run_dir=str(stage2_run_dir),
                                subset_dir=str(subset_dir),
                                strategy=variant_root_name,
                                variant_kind=str(variant_kind),
                                footprint=str(out_fp),
                                detection_filter=str(filt_name),
                                healpix_nside=int(healpix_nside),
                                n_orbit_targets=int(agg.n_orbit_targets),
                                n_frames_loaded=int(agg.n_frames_loaded),
                                n_observations_loaded=int(agg.n_observations_loaded),
                                n_after_prefilter=(None if agg.n_after_prefilter is None else int(agg.n_after_prefilter)),
                                n_after_chi2=int(agg.n_after_chi2),
                                io_sec=float(agg.io_sec),
                                filter_sec=float(agg.filter_sec),
                                chi2_sec=float(agg.chi2_sec),
                                runtime_total_sec=float(runtime_total),
                                n_errors=(None if agg.n_errors == 0 else int(agg.n_errors)),
                                error=agg.first_error,
                            )
                        )

                        n_rec = int(len(agg.recovered_truth_pairs))
                        coverage_rows.append(
                            dict(
                                stage2_run_dir=str(stage2_run_dir),
                                subset_dir=str(subset_dir),
                                strategy=variant_root_name,
                                variant_kind=str(variant_kind),
                                footprint=str(out_fp),
                                detection_filter=str(filt_name),
                                healpix_nside=int(healpix_nside),
                                n_truth_matched=int(n_truth),
                                n_recovered=int(n_rec),
                                recall=(0.0 if n_truth == 0 else float(n_rec) / float(n_truth)),
                            )
                        )

    finally:
        conn.close()

    metrics = Stage4Metrics.from_pyarrow(pa.Table.from_pylist(metrics_rows))
    coverage = Stage4Coverage.from_pyarrow(pa.Table.from_pylist(coverage_rows))
    metrics.to_parquet(str(run_dir / "metrics.parquet"))
    coverage.to_parquet(str(run_dir / "coverage.parquet"))

    out_meta = dict(
        subset_dir=str(subset_dir),
        stage2_run_dir=str(stage2_run_dir),
        healpix_nside=int(healpix_nside),
        n_sigma=float(n_sigma),
        polygon_vertices=int(polygon_vertices),
        cov_mc_num_samples=int(cov_mc_num_samples),
        cov_mc_seed=int(cov_mc_seed),
        corridor_radius_arcsec=float(corridor_radius_arcsec),
        corridor_step_arcsec=float(corridor_step_arcsec),
        filters=list(filters),
        strategies_requested=None if strategies is None else list(strategies),
        only_truth=bool(only_truth),
        max_orbits=None if max_orbits is None else int(max_orbits),
        max_targets=None if max_targets is None else int(max_targets),
        fallback_per_obs_ephem=bool(fallback_per_obs_ephem),
        obs_cache_max_frames=int(obs_cache_max_frames),
        generated_at_utc=_now_utc(),
    )
    _write_json(run_dir / "meta.json", out_meta)
    return run_dir


def _parse_csv_arg(arg: str | None) -> list[str] | None:
    if arg is None:
        return None
    items = [s.strip() for s in str(arg).split(",")]
    items = [s for s in items if s]
    return None if not items else items


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(
        description="Stage 4: atomic detection filtering benchmark runner (loads observations; evaluates chi2 + prefilters)."
    )
    p.add_argument("--subset-dir", type=str, required=True)
    p.add_argument("--stage2-run-dir", type=str, required=True)
    p.add_argument("--healpix-nside", type=int, required=True)
    p.add_argument("--n-sigma", type=float, default=3.0)
    p.add_argument("--polygon-vertices", type=int, default=32)
    p.add_argument("--cov-mc-num-samples", type=int, default=64)
    p.add_argument("--cov-mc-seed", type=int, default=0)
    p.add_argument("--corridor-radius-arcsec", type=float, default=30.0)
    p.add_argument("--corridor-step-arcsec", type=float, default=30.0)
    p.add_argument(
        "--strategies",
        type=str,
        default=None,
        help=(
            "Comma-separated list of Stage2 strategy folder names to include "
            "(for variants, supports 'assist_variants:sigma_points' style)."
        ),
    )
    p.add_argument(
        "--filters",
        type=str,
        default=None,
        help="Comma-separated list of detection filters (default: chi2_only,footprint_then_chi2).",
    )
    p.add_argument(
        "--only-truth",
        action="store_true",
        help="Only evaluate orbit/targets present in truth crossmatch (much faster for full runs).",
    )
    p.add_argument("--max-orbits", type=int, default=None)
    p.add_argument("--max-targets", type=int, default=None)
    p.add_argument(
        "--obs-cache-max-frames",
        type=int,
        default=0,
        help=(
            "Cache ObservationsTable by (data_uri, offset, length). "
            "Greatly reduces repeated disk reads across footprints. 0 disables."
        ),
    )
    p.add_argument(
        "--fallback-per-obs-ephem",
        action="store_true",
        help=(
            "If observation timestamps do not match Stage 2 target midpoints, generate a per-observation ephemeris "
            "using the orbit state (requires orbits parquet + ASSIST). Off by default."
        ),
    )
    args = p.parse_args()

    run_dir = run_stage4_detection_filter_bench(
        subset_dir=Path(args.subset_dir),
        stage2_run_dir=Path(args.stage2_run_dir),
        healpix_nside=int(args.healpix_nside),
        n_sigma=float(args.n_sigma),
        polygon_vertices=int(args.polygon_vertices),
        cov_mc_num_samples=int(args.cov_mc_num_samples),
        cov_mc_seed=int(args.cov_mc_seed),
        corridor_radius_arcsec=float(args.corridor_radius_arcsec),
        corridor_step_arcsec=float(args.corridor_step_arcsec),
        strategies=_parse_csv_arg(args.strategies),
        filters=_parse_csv_arg(args.filters),
        only_truth=bool(args.only_truth),
        max_orbits=args.max_orbits,
        max_targets=args.max_targets,
        obs_cache_max_frames=int(args.obs_cache_max_frames),
        fallback_per_obs_ephem=bool(args.fallback_per_obs_ephem),
    )
    print(f"run_dir={run_dir}")
    print(f"metrics_parquet={run_dir / 'metrics.parquet'}")
    print(f"coverage_parquet={run_dir / 'coverage.parquet'}")


if __name__ == "__main__":
    main()

