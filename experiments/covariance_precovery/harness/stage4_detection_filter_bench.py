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

from adam_core.orbits.ephemeris import Ephemeris
from adam_core.orbits.variants import VariantEphemeris
from adam_core.time import Timestamp

from precovery.frame_db import HealpixFrame
from precovery.observation import ObservationsTable
from precovery.precovery_db import PrecoveryDatabase
from ..methods.footprints import (
    EllipseFootprint,
    FixedPolygonFootprint,
    ellipse_boundary_vertices_lonlat_deg_from_cov,
    perimeter_polygon_from_samples,
)
from .stage3_healpixel_bench import (
    _collapse_variant_ephemeris_group,
    _designation_from_object_id,
    _ephem_key_table,
    _filter_ephem_to_truth,
    _group_slices_by_orbit_target,
    _map_ephem_to_target_idx_by_time,
    _map_times_to_target_idx_by_obscode,
    _read_stage2_targets,
)


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, obj: dict[str, object]) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class TruthMatchedObs(qv.Table):
    """
    One row per truth observation that was crossmatched to a detection in the subset,
    aligned to Stage 2 targets.
    """

    orbit_id = qv.LargeStringColumn()
    target_idx = qv.Int64Column()
    truth_obsid = qv.LargeStringColumn()
    truth_time_mjd_utc = qv.Float64Column()
    truth_ra_deg = qv.Float64Column()
    truth_dec_deg = qv.Float64Column()


class TruthKeys(qv.Table):
    orbit_id = qv.LargeStringColumn()
    target_idx = qv.Int64Column()


class TruthFrameKeys(qv.Table):
    orbit_id = qv.LargeStringColumn()
    target_idx = qv.Int64Column()
    healpixel = qv.Int64Column()


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

    io_sec = qv.Float64Column()
    prep_sec = qv.Float64Column()
    filter_sec = qv.Float64Column()
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


class Stage4PerTarget(qv.Table):
    """
    One row per (orbit_id, target_idx, filter) processed in Stage 4.

    This is used to compute per-target distributions (percentiles) for loaded frames,
    evaluated detections, and accepted detections.
    """

    stage2_run_dir = qv.LargeStringColumn()
    subset_dir = qv.LargeStringColumn()
    strategy = qv.LargeStringColumn()
    variant_kind = qv.LargeStringColumn(nullable=True)
    footprint = qv.LargeStringColumn()
    detection_filter = qv.LargeStringColumn()
    healpix_nside = qv.Int64Column()

    orbit_id = qv.LargeStringColumn()
    target_idx = qv.Int64Column()
    obscode = qv.LargeStringColumn()
    exposure_mjd_mid = qv.Float64Column()

    n_frames_loaded = qv.Int64Column()
    n_observations_loaded = qv.Int64Column()
    n_accepted = qv.Int64Column()

    io_sec = qv.Float64Column()
    prep_sec = qv.Float64Column()
    filter_sec = qv.Float64Column()


_ObsCacheKey = tuple[str, int, int]

# Per-dataset astrometric uncertainty floors for innovation gating (arcsec).
# These are *lower bounds* used when per-detection sigmas are missing or too small.
#
# Values are chosen to be below (or comparable to) typical uncertainties in the
# current ATLAS/ZTF/NSC subset. In this subset, ZTF astrometric sigmas are NaN
# in the stored observation records, so the floor is used for ZTF most often.
_DET_SIGMA_FLOOR_ARCSEC_BY_DATASET: dict[str, float] = {
    "atlas": 0.30,
    "ztf": 0.10,
    "nsc": 0.07,
}
_DET_SIGMA_FLOOR_ARCSEC_DEFAULT = 0.10


@dataclass
class _FilterAgg:
    # counts
    n_orbit_targets: int = 0
    n_frames_loaded: int = 0
    n_observations_loaded: int = 0
    n_after_prefilter: int | None = 0

    # timings
    io_sec: float = 0.0
    prep_sec: float = 0.0
    filter_sec: float = 0.0

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
            "truth_obsid": pa.array([], pa.large_string()),
            "truth_time_mjd_utc": pa.array([], pa.float64()),
            "truth_ra_deg": pa.array([], pa.float64()),
            "truth_dec_deg": pa.array([], pa.float64()),
        }
    )


def _read_truth_matches_table(*, subset_dir: Path, targets: pa.Table) -> pa.Table:
    """
    Return matched truth detections aligned to Stage 2 targets as:
      (orbit_id, target_idx, truth_obsid, truth_time_mjd_utc, truth_ra_deg, truth_dec_deg)

    IMPORTANT: We intentionally do NOT match by `match_observation_id` for recall, since
    observation IDs can differ across sources/ingests. Stage 4 recall is evaluated in the
    same crossmatch sense: (time, sky position) within the truth tolerances.
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
        columns=["matched", "designation", "obscode", "truth_obsid", "truth_time_mjd_utc", "truth_ra_deg", "truth_dec_deg"],
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
    truth = truth.select(
        ["designation", "obscode", "truth_obsid", "truth_time_mjd_utc", "truth_ra_deg", "truth_dec_deg"]
    )
    truth = truth.join(orbit_map, keys=["designation"], join_type="inner")
    if truth.num_rows == 0:
        return _truth_matches_empty()

    t_obscode = np.asarray(truth["obscode"].to_pylist(), dtype=object)
    t_time = np.asarray(truth["truth_time_mjd_utc"].to_numpy(zero_copy_only=False), dtype=np.float64)
    t_time_utc = Timestamp.from_mjd(t_time.tolist(), scale="utc")
    target_idx = _map_times_to_target_idx_by_obscode(
        obscode=t_obscode,
        time_utc=t_time_utc,
        targets=targets,
        dt_sec=60.0,
        precision="us",
    )
    ok = target_idx >= 0
    if not ok.any():
        return _truth_matches_empty()

    orbit_id = truth["orbit_id"].to_pylist()
    truth_obsid = truth["truth_obsid"].to_pylist()
    truth_mjd = truth["truth_time_mjd_utc"].to_numpy(zero_copy_only=False).tolist()
    truth_ra = truth["truth_ra_deg"].to_numpy(zero_copy_only=False).tolist()
    truth_dec = truth["truth_dec_deg"].to_numpy(zero_copy_only=False).tolist()

    out = TruthMatchedObs.from_kwargs(
        orbit_id=[str(x) for x in np.asarray(orbit_id, dtype=object)[ok].tolist()],
        target_idx=[int(x) for x in target_idx[ok].tolist()],
        truth_obsid=[str(x) for x in np.asarray(truth_obsid, dtype=object)[ok].tolist()],
        truth_time_mjd_utc=[float(x) for x in np.asarray(truth_mjd, dtype=object)[ok].tolist()],
        truth_ra_deg=[float(x) for x in np.asarray(truth_ra, dtype=object)[ok].tolist()],
        truth_dec_deg=[float(x) for x in np.asarray(truth_dec, dtype=object)[ok].tolist()],
    )
    # Drop any rows where truth_obsid is empty (shouldn't happen).
    m = pc.not_equal(out.table["truth_obsid"], pa.scalar(""))
    return out.table.filter(m)


def _read_truth_frame_keys_table(*, subset_dir: Path, targets: pa.Table) -> pa.Table:
    """
    Return truth-matched frame keys aligned to Stage 2 targets as:
      (orbit_id, target_idx, healpixel)

    This is used for an optional Stage 4 speed-up where we only load observations from frames
    that are known (from truth crossmatch) to contain a true detection for that orbit/target.
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
        columns=["matched", "designation", "obscode", "match_time_mjd_utc", "healpixel"],
    )
    truth = truth.filter(pc.equal(truth["matched"], True))
    if truth.num_rows == 0:
        return pa.table(
            {
                "orbit_id": pa.array([], pa.large_string()),
                "target_idx": pa.array([], pa.int64()),
                "healpixel": pa.array([], pa.int64()),
            }
        )

    # Map designation -> orbit_id using orbits_selected_sbdb.parquet.
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
    truth = truth.select(["designation", "obscode", "match_time_mjd_utc", "healpixel"])
    truth = truth.join(orbit_map, keys=["designation"], join_type="inner")
    if truth.num_rows == 0:
        return pa.table(
            {
                "orbit_id": pa.array([], pa.large_string()),
                "target_idx": pa.array([], pa.int64()),
                "healpixel": pa.array([], pa.int64()),
            }
        )

    t_obscode = np.asarray(truth["obscode"].to_pylist(), dtype=object)
    t_time = np.asarray(truth["match_time_mjd_utc"].to_numpy(zero_copy_only=False), dtype=np.float64)
    t_time_utc = Timestamp.from_mjd(t_time.tolist(), scale="utc")
    target_idx = _map_times_to_target_idx_by_obscode(
        obscode=t_obscode,
        time_utc=t_time_utc,
        targets=targets,
        dt_sec=60.0,
        precision="us",
    )
    ok = target_idx >= 0
    if not ok.any():
        return pa.table(
            {
                "orbit_id": pa.array([], pa.large_string()),
                "target_idx": pa.array([], pa.int64()),
                "healpixel": pa.array([], pa.int64()),
            }
        )

    orbit_id = np.asarray(truth["orbit_id"].to_pylist(), dtype=object)[ok].tolist()
    hpix = np.asarray(truth["healpixel"].to_numpy(zero_copy_only=False), dtype=np.int64)[ok].tolist()
    tidx = target_idx[ok].tolist()
    out = pa.table(
        {
            "orbit_id": pa.array([str(x) for x in orbit_id], pa.large_string()),
            "target_idx": pa.array([int(x) for x in tidx], pa.int64()),
            "healpixel": pa.array([int(x) for x in hpix], pa.int64()),
        }
    )
    # Deduplicate to keep mapping compact.
    qt = TruthFrameKeys.from_pyarrow(out).drop_duplicates(subset=["orbit_id", "target_idx", "healpixel"])
    return qt.table


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
                (str(obscode), float(q0), float(q1), *chunk.astype(np.int64, copy=False).tolist()),
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


def _det_sigma_floor_arcsec_for_frames(*, frames: list[dict[str, object]]) -> float:
    """
    Return a conservative (max) sigma floor to apply for a loaded set of frames.

    We do not have dataset_id stored per-detection in `ObservationsTable`, so for
    mixed-dataset loads we apply the *largest* floor among frames. This only matters
    when per-detection sigmas are missing/NaN.
    """
    if not frames:
        return float(_DET_SIGMA_FLOOR_ARCSEC_DEFAULT)
    floors: list[float] = []
    for fr in frames:
        ds = str(fr.get("dataset_id", "")).strip().lower()
        floors.append(float(_DET_SIGMA_FLOOR_ARCSEC_BY_DATASET.get(ds, _DET_SIGMA_FLOOR_ARCSEC_DEFAULT)))
    return float(max(floors)) if floors else float(_DET_SIGMA_FLOOR_ARCSEC_DEFAULT)


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


def _sqrtm_2x2_psd(m: np.ndarray) -> np.ndarray:
    """
    Safe-ish square-root for a 2x2 PSD-ish matrix using an eigen decomposition.
    """
    m = np.asarray(m, dtype=np.float64)
    m = 0.5 * (m + m.T)
    w, v = np.linalg.eigh(m)
    w = np.maximum(w, 0.0)
    return (v * np.sqrt(w)) @ v.T


def _cov_xy_from_cov_ll(*, cov_ll_deg2: np.ndarray, lat0_deg: float) -> tuple[np.ndarray, float]:
    """
    Convert lon/lat covariance (deg^2) into local tangent-plane (x=lon*cos(lat), y=lat).
    Returns (cov_xy_deg2, cos_lat).
    """
    cos_lat = float(np.cos(np.deg2rad(float(lat0_deg))))
    cos_lat = cos_lat if np.isfinite(cos_lat) and abs(cos_lat) > 1e-12 else 1e-12
    A = np.array([[cos_lat, 0.0], [0.0, 1.0]], dtype=np.float64)
    cov_xy = A @ np.asarray(cov_ll_deg2, dtype=np.float64) @ A.T
    cov_xy = 0.5 * (cov_xy + cov_xy.T)
    return cov_xy, cos_lat


def _sigma_major_deg_from_cov_ll(*, cov_ll_deg2: np.ndarray, lat0_deg: float) -> float:
    cov_xy, _ = _cov_xy_from_cov_ll(cov_ll_deg2=cov_ll_deg2, lat0_deg=lat0_deg)
    w = np.linalg.eigvalsh(cov_xy)
    w = np.maximum(w, 0.0)
    return float(np.sqrt(float(np.max(w))))


def _draw_cov_samples_lonlat_deg(
    *,
    lon0_deg: float,
    lat0_deg: float,
    cov_ll_deg2: np.ndarray,
    n_sigma: float,
    num_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Draw lon/lat samples (deg) from N(0, cov_ll) in a local tangent plane.
    """
    if int(num_samples) <= 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    cov_xy, cos_lat = _cov_xy_from_cov_ll(cov_ll_deg2=cov_ll_deg2, lat0_deg=lat0_deg)
    S = _sqrtm_2x2_psd(cov_xy)
    rng = np.random.default_rng(int(seed))
    z = rng.standard_normal((int(num_samples), 2))
    dxy = (z @ S.T) * float(n_sigma)
    dlon = dxy[:, 0] / float(cos_lat)
    dlat = dxy[:, 1]

    lon_s = (float(lon0_deg) + dlon) % 360.0
    lat_s = np.clip(float(lat0_deg) + dlat, -89.999999, 89.999999)
    return lon_s.astype(np.float64, copy=False), lat_s.astype(np.float64, copy=False)


_DetectionGeometryName = str


def _parse_detection_geometry_spec(spec: str) -> tuple[str, float | None]:
    """
    Parse a detection-geometry spec with an optional sigma override.

    Examples:
      - "cov_ellipse" -> ("cov_ellipse", None)
      - "cov_ellipse@1" -> ("cov_ellipse", 1.0)
      - "cov_disc@3" -> ("cov_disc", 3.0)
    """
    s = str(spec).strip()
    if "@" not in s:
        return s, None
    base, rhs = s.split("@", 1)
    base = base.strip()
    rhs = rhs.strip()
    if rhs == "":
        raise ValueError(f"Invalid detection geometry spec (empty sigma): {spec!r}")
    try:
        sig = float(rhs)
    except ValueError as e:
        raise ValueError(f"Invalid detection geometry spec (bad sigma): {spec!r}") from e
    if not np.isfinite(sig) or sig <= 0:
        raise ValueError(f"Invalid detection geometry spec (sigma must be > 0): {spec!r}")
    return base, float(sig)


def _validate_detection_geometry(spec: str) -> tuple[str, float | None]:
    key, sigma_override = _parse_detection_geometry_spec(spec)
    allowed = {
        # point-derived
        "point_disc",
        # covariance-derived (Gaussian)
        "cov_disc",
        "cov_ellipse",  # Mahalanobis ellipse using predicted covariance only
        "innov_ellipse",  # Mahalanobis ellipse using (predicted + observational) covariance
        "cov_polygon_moc",  # ellipse boundary polygon membership (MOC naming; membership is polygon)
        "cov_mc_polygon_moc",  # perimeter polygon from covariance MC samples OR provided samples (convex hull)
        # sample-derived (variant ephemeris cloud)
        "sample_perimeter_polygon_moc",  # convex-hull perimeter polygon from provided samples
    }
    if key not in allowed:
        raise ValueError(f"Unknown detection geometry: {spec!r}. Allowed: {sorted(allowed)}")
    return key, sigma_override


def _is_monte_carlo_variant_kind(kind: str) -> bool:
    """
    Heuristic: Stage 2 variant kinds are named like "mc_256", "mc_1024", etc.
    """
    k = str(kind).strip().lower()
    return ("mc" in k) and ("sigma" not in k)


@dataclass(frozen=True)
class _ObsPrep:
    """
    Shared per-(orbit_id,target_idx) arrays extracted from loaded observations.

    This is the shared-product for Stage 4: all detection geometries operate on these
    arrays rather than re-extracting from the Quivr table.
    """

    mjd_utc: np.ndarray  # (N,) float64
    ra_deg: np.ndarray  # (N,) float64
    dec_deg: np.ndarray  # (N,) float64
    ra_sigma_deg: np.ndarray  # (N,) float64
    dec_sigma_deg: np.ndarray  # (N,) float64
    ra_rad: np.ndarray  # (N,) float64
    dec_rad: np.ndarray  # (N,) float64
    sin_dec: np.ndarray  # (N,) float64
    cos_dec: np.ndarray  # (N,) float64


def _prepare_observation_arrays(obs: ObservationsTable) -> _ObsPrep:
    """
    Extract numpy arrays needed for all detection-geometry masks.
    """
    mjd = obs.time.mjd().to_numpy(zero_copy_only=False).astype(np.float64)
    ra_deg = obs.ra.to_numpy(zero_copy_only=False).astype(np.float64)
    dec_deg = obs.dec.to_numpy(zero_copy_only=False).astype(np.float64)
    ra_sigma_deg = obs.ra_sigma.to_numpy(zero_copy_only=False).astype(np.float64)
    dec_sigma_deg = obs.dec_sigma.to_numpy(zero_copy_only=False).astype(np.float64)
    ra_rad = np.deg2rad(ra_deg).astype(np.float64, copy=False)
    dec_rad = np.deg2rad(dec_deg).astype(np.float64, copy=False)
    sin_dec = np.sin(dec_rad)
    cos_dec = np.cos(dec_rad)
    return _ObsPrep(
        mjd_utc=mjd,
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        ra_sigma_deg=ra_sigma_deg,
        dec_sigma_deg=dec_sigma_deg,
        ra_rad=ra_rad,
        dec_rad=dec_rad,
        sin_dec=sin_dec,
        cos_dec=cos_dec,
    )


def _disc_keep_mask(*, prep: _ObsPrep, lon0_deg: float, lat0_deg: float, radius_deg: float) -> np.ndarray:
    """
    Great-circle disc membership using cosine separation threshold.
    """
    lon0_rad = float(np.deg2rad(float(lon0_deg)))
    lat0_rad = float(np.deg2rad(float(lat0_deg)))
    sin0 = float(np.sin(lat0_rad))
    cos0 = float(np.cos(lat0_rad))
    cos_tol = float(np.cos(np.deg2rad(float(radius_deg))))
    cos_dlon = np.cos(prep.ra_rad - lon0_rad)
    cos_sep = sin0 * prep.sin_dec + cos0 * prep.cos_dec * cos_dlon
    cos_sep = np.clip(cos_sep, -1.0, 1.0)
    return cos_sep >= cos_tol


def _detection_geometry_keep_mask(
    *,
    prep: _ObsPrep,
    geometry: _DetectionGeometryName,
    lon0_deg: float,
    lat0_deg: float,
    cov_ll_deg2: np.ndarray | None,
    n_sigma: float,
    polygon_vertices: int,
    point_radius_arcsec: float,
    cov_mc_num_samples: int,
    cov_mc_seed: int,
    det_sigma_floor_arcsec: float = float(_DET_SIGMA_FLOOR_ARCSEC_DEFAULT),
    sample_lon_deg: np.ndarray | None = None,
    sample_lat_deg: np.ndarray | None = None,
) -> np.ndarray:
    """
    Return a boolean mask over `prep` selecting accepted observations.

    NOTE: This intentionally does not mutate/construct Quivr tables; it returns masks
    so the caller can share the prep work and time each geometry independently.
    """
    geom, sigma_override = _validate_detection_geometry(geometry)
    sig = float(n_sigma if sigma_override is None else sigma_override)

    if prep.ra_deg.size == 0:
        return np.zeros(0, dtype=bool)

    if geom == "point_disc":
        r_deg = float(point_radius_arcsec) / 3600.0
        return _disc_keep_mask(prep=prep, lon0_deg=float(lon0_deg), lat0_deg=float(lat0_deg), radius_deg=float(r_deg))

    if geom == "sample_perimeter_polygon_moc":
        if sample_lon_deg is None or sample_lat_deg is None or len(sample_lon_deg) == 0:
            raise ValueError("sample_perimeter_polygon_moc requires non-empty sample_lon_deg/sample_lat_deg")
        poly = perimeter_polygon_from_samples(
            lon0_deg=float(lon0_deg),
            lat0_deg=float(lat0_deg),
            lon_deg=np.asarray(sample_lon_deg, dtype=np.float64),
            lat_deg=np.asarray(sample_lat_deg, dtype=np.float64),
            mode="convex_hull",
        )
        fp = FixedPolygonFootprint(
            lon0_deg=float(lon0_deg),
            lat0_deg=float(lat0_deg),
            vertex_lon_deg=np.asarray(poly[:, 0], dtype=np.float64),
            vertex_lat_deg=np.asarray(poly[:, 1], dtype=np.float64),
            buffer_arcsec=0.0,
        )
        return fp.contains(prep.ra_deg, prep.dec_deg)

    # Remaining geometries require covariance.
    if cov_ll_deg2 is None:
        raise ValueError(f"Geometry {geom!r} requires covariance, but none was available.")

    if geom == "cov_disc":
        r_deg = float(sig) * _sigma_major_deg_from_cov_ll(cov_ll_deg2=cov_ll_deg2, lat0_deg=float(lat0_deg))
        return _disc_keep_mask(prep=prep, lon0_deg=float(lon0_deg), lat0_deg=float(lat0_deg), radius_deg=float(r_deg))

    if geom == "innov_ellipse":
        # Innovation (observed + predicted) covariance gate in local tangent plane.
        #
        # Predicted covariance is provided as (lon,lat) deg^2. Convert to tangent-plane xy.
        cov_xy, cos_lat = _cov_xy_from_cov_ll(
            cov_ll_deg2=np.asarray(cov_ll_deg2, dtype=np.float64),
            lat0_deg=float(lat0_deg),
        )
        a_p = float(cov_xy[0, 0])
        b_p = float(cov_xy[0, 1])
        d_p = float(cov_xy[1, 1])

        # Observational 1-sigma uncertainties in degrees (diagonal-only).
        floor_deg = float(det_sigma_floor_arcsec) / 3600.0
        ra_sig = np.asarray(prep.ra_sigma_deg, dtype=np.float64)
        dec_sig = np.asarray(prep.dec_sigma_deg, dtype=np.float64)
        ra_sig = np.where(np.isfinite(ra_sig) & (ra_sig > 0.0), ra_sig, 0.0)
        dec_sig = np.where(np.isfinite(dec_sig) & (dec_sig > 0.0), dec_sig, 0.0)
        ra_sig = np.maximum(ra_sig, float(floor_deg))
        dec_sig = np.maximum(dec_sig, float(floor_deg))

        # Convert obs sigmas to the same tangent-plane x/y basis:
        #   x = Δlon*cos(lat0), y = Δlat.
        sig_x = ra_sig * float(cos_lat)
        sig_y = dec_sig
        var_x = sig_x * sig_x
        var_y = sig_y * sig_y

        # Total covariance per detection: [[a,b],[b,d]] where obs contributes only to diagonal.
        a = a_p + var_x
        b = np.full_like(a, b_p, dtype=np.float64)
        d = d_p + var_y

        # Residuals in the same tangent plane (degrees).
        dlon = (prep.ra_deg - float(lon0_deg) + 180.0) % 360.0 - 180.0
        x = dlon * float(cos_lat)
        y = prep.dec_deg - float(lat0_deg)

        # Fast 2x2 inverse chi^2: chi2 = [x y] Σ^{-1} [x y]^T.
        det = a * d - b * b
        det = np.where(np.isfinite(det) & (det > 0.0), det, np.inf)
        chi2 = (x * x * d + y * y * a - 2.0 * x * y * b) / det
        chi2 = np.where(np.isfinite(chi2), chi2, np.inf)
        return chi2 <= float(sig) ** 2

    if geom == "cov_ellipse":
        fp = EllipseFootprint(
            lon0_deg=float(lon0_deg),
            lat0_deg=float(lat0_deg),
            cov_ll_deg2=np.asarray(cov_ll_deg2, dtype=np.float64),
            n_sigma=float(sig),
            polygon_vertices=int(polygon_vertices),
            raster_mode="polygon",
        )
        return fp.contains(prep.ra_deg, prep.dec_deg)

    if geom == "cov_polygon_moc":
        lonv, latv = ellipse_boundary_vertices_lonlat_deg_from_cov(
            lon0_deg=float(lon0_deg),
            lat0_deg=float(lat0_deg),
            cov_ll_deg2=np.asarray(cov_ll_deg2, dtype=np.float64),
            n_sigma=float(sig),
            num_vertices=int(polygon_vertices),
        )
        fp = FixedPolygonFootprint(
            lon0_deg=float(lon0_deg),
            lat0_deg=float(lat0_deg),
            vertex_lon_deg=np.asarray(lonv, dtype=np.float64),
            vertex_lat_deg=np.asarray(latv, dtype=np.float64),
            buffer_arcsec=0.0,
        )
        return fp.contains(prep.ra_deg, prep.dec_deg)

    if geom == "cov_mc_polygon_moc":
        if sample_lon_deg is not None and sample_lat_deg is not None and len(sample_lon_deg) > 0:
            lon_s = np.asarray(sample_lon_deg, dtype=np.float64)
            lat_s = np.asarray(sample_lat_deg, dtype=np.float64)
        else:
            lon_s, lat_s = _draw_cov_samples_lonlat_deg(
                lon0_deg=float(lon0_deg),
                lat0_deg=float(lat0_deg),
                cov_ll_deg2=np.asarray(cov_ll_deg2, dtype=np.float64),
                n_sigma=float(sig),
                num_samples=int(cov_mc_num_samples),
                seed=int(cov_mc_seed),
            )
        poly = perimeter_polygon_from_samples(
            lon0_deg=float(lon0_deg),
            lat0_deg=float(lat0_deg),
            lon_deg=np.asarray(lon_s, dtype=np.float64),
            lat_deg=np.asarray(lat_s, dtype=np.float64),
            mode="convex_hull",
        )
        fp = FixedPolygonFootprint(
            lon0_deg=float(lon0_deg),
            lat0_deg=float(lat0_deg),
            vertex_lon_deg=np.asarray(poly[:, 0], dtype=np.float64),
            vertex_lat_deg=np.asarray(poly[:, 1], dtype=np.float64),
            buffer_arcsec=0.0,
        )
        return fp.contains(prep.ra_deg, prep.dec_deg)

    raise AssertionError(f"Unhandled geometry: {geom}")

#
# NOTE: We intentionally removed the older `_apply_detection_geometry` entrypoint (which
# re-extracted arrays and built filtered tables per-geometry). Stage 4 now uses shared
# numpy arrays + boolean masks via `_prepare_observation_arrays` and
# `_detection_geometry_keep_mask`.

def _truth_index_by_orbit_target(
    truth_matches: pa.Table,
) -> tuple[dict[tuple[str, int], list[tuple[str, float, float, float]]], set[tuple[str, str]]]:
    """
    Return:
      - dict[(orbit_id, target_idx)] -> list of (truth_obsid, truth_mjd, truth_ra_deg, truth_dec_deg)
      - set of unique (orbit_id, truth_obsid) pairs (denominator for recall)
    """
    if truth_matches.num_rows == 0:
        return {}, set()
    oid = truth_matches["orbit_id"].to_pylist()
    tidx = np.asarray(truth_matches["target_idx"].to_numpy(zero_copy_only=False), dtype=np.int64)
    tobs = truth_matches["truth_obsid"].to_pylist()
    tmjd = np.asarray(truth_matches["truth_time_mjd_utc"].to_numpy(zero_copy_only=False), dtype=np.float64)
    tra = np.asarray(truth_matches["truth_ra_deg"].to_numpy(zero_copy_only=False), dtype=np.float64)
    tdec = np.asarray(truth_matches["truth_dec_deg"].to_numpy(zero_copy_only=False), dtype=np.float64)

    by_key: dict[tuple[str, int], list[tuple[str, float, float, float]]] = {}
    pairs: set[tuple[str, str]] = set()
    for o, t, obsid, mjd, ra, dec in zip(oid, tidx, tobs, tmjd, tra, tdec):
        key = (str(o), int(t))
        by_key.setdefault(key, []).append((str(obsid), float(mjd), float(ra), float(dec)))
        pairs.add((str(o), str(obsid)))
    return by_key, pairs


def _recover_truth_obsids_for_orbit_target(
    *,
    truth_entries: list[tuple[str, float, float, float]],
    accepted_obs: ObservationsTable,
    time_tol_sec: float,
    dist_tol_arcsec: float,
) -> set[str]:
    """
    Return truth_obsid values recovered by accepted detections, using the same crossmatch sense:
      |Δt| <= time_tol_sec and great-circle distance <= dist_tol_arcsec
    """
    if not truth_entries or len(accepted_obs) == 0:
        return set()

    obs_mjd = accepted_obs.time.mjd().to_numpy(zero_copy_only=False).astype(np.float64)
    obs_ra_deg = accepted_obs.ra.to_numpy(zero_copy_only=False).astype(np.float64)
    obs_dec_deg = accepted_obs.dec.to_numpy(zero_copy_only=False).astype(np.float64)

    # Convert truth entries to numpy arrays (truth multiplicity per orbit/target is usually small).
    n_truth = int(len(truth_entries))
    truth_obsid = np.empty(n_truth, dtype=object)
    truth_mjd = np.empty(n_truth, dtype=np.float64)
    truth_ra_deg = np.empty(n_truth, dtype=np.float64)
    truth_dec_deg = np.empty(n_truth, dtype=np.float64)
    for i, (obsid, mjd, ra, dec) in enumerate(truth_entries):
        truth_obsid[i] = str(obsid)
        truth_mjd[i] = float(mjd)
        truth_ra_deg[i] = float(ra)
        truth_dec_deg[i] = float(dec)

    # Time gating first (broadcast): (N_obs, N_truth)
    dt_ok = (np.abs(obs_mjd[:, None] - truth_mjd[None, :]) * 86400.0) <= float(time_tol_sec)
    if not bool(np.any(dt_ok)):
        return set()

    # Great-circle distance test using cosine threshold (avoids arcsin/atan and is fully vectorized).
    # sep <= dist_tol  <=>  cos(sep) >= cos(dist_tol)
    dist_tol_rad = np.deg2rad(float(dist_tol_arcsec) / 3600.0)
    cos_tol = float(np.cos(dist_tol_rad))

    obs_ra = np.deg2rad(obs_ra_deg).astype(np.float64, copy=False)[:, None]
    obs_dec = np.deg2rad(obs_dec_deg).astype(np.float64, copy=False)[:, None]
    tru_ra = np.deg2rad(truth_ra_deg).astype(np.float64, copy=False)[None, :]
    tru_dec = np.deg2rad(truth_dec_deg).astype(np.float64, copy=False)[None, :]

    sin_obs_dec = np.sin(obs_dec)
    cos_obs_dec = np.cos(obs_dec)
    sin_tru_dec = np.sin(tru_dec)
    cos_tru_dec = np.cos(tru_dec)
    cos_dra = np.cos(obs_ra - tru_ra)
    cos_sep = sin_obs_dec * sin_tru_dec + cos_obs_dec * cos_tru_dec * cos_dra

    # Numerical safety: cos_sep may drift slightly outside [-1, 1].
    cos_sep = np.clip(cos_sep, -1.0, 1.0)
    match = dt_ok & (cos_sep >= cos_tol)
    if not bool(np.any(match)):
        return set()

    recovered_mask = np.any(match, axis=0)
    return {str(x) for x in truth_obsid[recovered_mask].tolist()}


def _recover_truth_obsids_for_orbit_target_masked(
    *,
    truth_entries: list[tuple[str, float, float, float]],
    prep: _ObsPrep,
    accepted_mask: np.ndarray,
    time_tol_sec: float,
    dist_tol_arcsec: float,
) -> set[str]:
    """
    Vectorized truth recovery using a boolean accepted-mask over shared observation arrays.
    """
    if not truth_entries or prep.ra_deg.size == 0:
        return set()
    if accepted_mask.size == 0 or (not bool(np.any(accepted_mask))):
        return set()

    obs_mjd = prep.mjd_utc[accepted_mask]
    obs_ra_rad = prep.ra_rad[accepted_mask]
    obs_dec_rad = prep.dec_rad[accepted_mask]

    n_truth = int(len(truth_entries))
    truth_obsid = np.empty(n_truth, dtype=object)
    truth_mjd = np.empty(n_truth, dtype=np.float64)
    truth_ra_deg = np.empty(n_truth, dtype=np.float64)
    truth_dec_deg = np.empty(n_truth, dtype=np.float64)
    for i, (obsid, mjd, ra, dec) in enumerate(truth_entries):
        truth_obsid[i] = str(obsid)
        truth_mjd[i] = float(mjd)
        truth_ra_deg[i] = float(ra)
        truth_dec_deg[i] = float(dec)

    dt_ok = (np.abs(obs_mjd[:, None] - truth_mjd[None, :]) * 86400.0) <= float(time_tol_sec)
    if not bool(np.any(dt_ok)):
        return set()

    dist_tol_rad = np.deg2rad(float(dist_tol_arcsec) / 3600.0)
    cos_tol = float(np.cos(dist_tol_rad))

    obs_ra = obs_ra_rad[:, None]
    obs_dec = obs_dec_rad[:, None]
    tru_ra = np.deg2rad(truth_ra_deg).astype(np.float64, copy=False)[None, :]
    tru_dec = np.deg2rad(truth_dec_deg).astype(np.float64, copy=False)[None, :]

    sin_obs_dec = np.sin(obs_dec)
    cos_obs_dec = np.cos(obs_dec)
    sin_tru_dec = np.sin(tru_dec)
    cos_tru_dec = np.cos(tru_dec)
    cos_dra = np.cos(obs_ra - tru_ra)
    cos_sep = sin_obs_dec * sin_tru_dec + cos_obs_dec * cos_tru_dec * cos_dra
    cos_sep = np.clip(cos_sep, -1.0, 1.0)

    match = dt_ok & (cos_sep >= cos_tol)
    if not bool(np.any(match)):
        return set()
    recovered_mask = np.any(match, axis=0)
    return {str(x) for x in truth_obsid[recovered_mask].tolist()}


def run_stage4_detection_filter_bench(
    *,
    subset_dir: Path,
    stage2_run_dir: Path,
    stage3_run_dir: Path | None = None,
    truth_frames_only: bool = False,
    healpix_nside: int,
    n_sigma: float = 3.0,
    polygon_vertices: int = 32,
    cov_mc_num_samples: int = 64,
    cov_mc_seed: int = 0,
    point_radius_arcsec: float = 5.0,
    corridor_radius_arcsec: float = 30.0,
    corridor_step_arcsec: float = 30.0,
    out_dir: Path | None = None,
    strategies: list[str] | None = None,
    filters: list[str] | None = None,
    footprints: list[str] | None = None,
    footprint_set: str | None = None,
    only_truth: bool = False,
    max_orbits: int | None = None,
    max_targets: int | None = None,
    obs_cache_max_frames: int = 0,
    write_per_target_metrics: bool = False,
) -> Path:
    """
    Stage 4 (atomic): benchmark detection-level filtering variants after loading observations.

    Outputs:
      - metrics.parquet: runtime + counts by (strategy, footprint, detection_filter)
      - coverage.parquet: truth-detection recall by (strategy, footprint, detection_filter)
      - per_target.parquet (optional): per-(orbit,target,filter) counts + timings
    """
    if out_dir is None:
        out_dir = subset_dir / "artifacts" / "stage4"
    _ensure_dir(out_dir)
    run_dir = out_dir / stage2_run_dir.name
    _ensure_dir(run_dir)

    def _normalize_fp_list(items: list[str]) -> set[str]:
        return {str(x).strip() for x in items if str(x).strip()}

    def _footprints_for_set(name: str) -> set[str]:
        """
        Curated footprint subset for Stage 4 benchmarks.

        Notes:
        - Prefer MOC variants of polygons to avoid `healpy.query_polygon` fragility.
        - Non-MOC polygon rasterizers are intentionally not supported.
        """
        key = str(name).strip().lower()
        if key in {"diverse", "diverse_footprints"}:
            return {
                # mean outputs
                "point",
                "cov_disc",
                "cov_mc",
                "cov_polygon_moc",
                # variants (sample-driven)
                "sample_direct",
                "sample_corridor",
                "sample_polygon_moc:convex_hull",
                # variants (reconstructed covariance)
                "cov_disc_reconstructed",
                "cov_polygon_reconstructed_moc",
                "cov_mc_reconstructed",
            }
        if key in {"moc_only", "moc"}:
            return {
                "point",
                "cov_polygon_moc",
                "sample_polygon_moc:convex_hull",
                "cov_polygon_reconstructed_moc",
            }
        if key in {"fast"}:
            return {
                "point",
                "cov_disc",
                "sample_direct",
                "cov_disc_reconstructed",
            }
        raise ValueError(f"Unknown footprint_set: {name!r}")

    selected_footprints: set[str] | None = None
    if footprints is not None and footprint_set is not None:
        raise ValueError("Specify at most one of footprints or footprint_set")
    if footprints is not None:
        selected_footprints = _normalize_fp_list(footprints)
    if footprint_set is not None:
        selected_footprints = _footprints_for_set(footprint_set)

    targets_tbl = _read_stage2_targets(stage2_run_dir)
    truth_all = _read_truth_matches_table(subset_dir=subset_dir, targets=targets_tbl)
    truth_frame_keys_tbl = _read_truth_frame_keys_table(subset_dir=subset_dir, targets=targets_tbl)

    # Truth tolerances (match sense): keep consistent with the truth crossmatch writer.
    time_tol_sec = 60.0
    dist_tol_arcsec = 5.0
    meta_path = subset_dir / "artifacts" / "truth_precovery_crossmatch_meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            time_tol_sec = float(meta.get("time_tol_sec", time_tol_sec))
            dist_tol_arcsec = float(meta.get("dist_tol_arcsec", dist_tol_arcsec))
        except Exception:  # noqa: BLE001
            pass

    db = PrecoveryDatabase.from_dir(str(subset_dir), allow_version_mismatch=True)

    # Targets arrays for quick lookup by target_idx.
    targ_obscode = np.asarray(targets_tbl["obscode"].to_pylist(), dtype=object)
    t = targets_tbl.column("time").combine_chunks()
    targ_time_utc = Timestamp.from_kwargs(days=t.field("days"), nanos=t.field("nanos"), scale="utc")
    # index.db stores exposure midpoints as float MJD UTC.
    targ_mjd = np.asarray(targ_time_utc.mjd().to_numpy(zero_copy_only=False), dtype=np.float64)

    index_db = subset_dir / "index.db"
    if not index_db.exists():
        raise FileNotFoundError(f"Missing subset index.db: {index_db}")

    # Optional Stage 3 selected-keys mode: reuse Stage 3’s (orbit_id, target_idx, healpixel)
    # intersections and avoid recomputing footprint healpixels in Stage 4.
    if stage3_run_dir is None:
        stage3_run_dir = subset_dir / "artifacts" / "stage3" / stage2_run_dir.name
    stage3_run_dir = Path(stage3_run_dir)

    stage3_meta_path = stage3_run_dir / "meta.json"
    if stage3_meta_path.exists():
        try:
            stage3_meta = json.loads(stage3_meta_path.read_text())
            stage3_nside = int(stage3_meta.get("healpix_nside", int(healpix_nside)))
            if stage3_nside != int(healpix_nside):
                raise ValueError(
                    f"healpix_nside mismatch: stage4={int(healpix_nside)} vs stage3={stage3_nside} "
                    f"(stage3 meta: {stage3_meta_path})"
                )
        except Exception:  # noqa: BLE001
            # If meta is malformed, proceed; selected_keys are already computed at some nside.
            pass

    def _stage3_keys_path(*, strategy: str, variant_kind: str | None, footprint: str) -> Path:
        strat_key = str(strategy) if variant_kind is None else f"{strategy}:{variant_kind}"
        return stage3_run_dir / "selected_keys" / strat_key / str(footprint) / "selected_keys_unique.parquet"

    def _keys_map_from_selected_keys_table(tbl: pa.Table) -> dict[tuple[str, int], np.ndarray]:
        """
        Build a map (orbit_id, target_idx) -> np.ndarray[healpixel].
        """
        if tbl.num_rows == 0:
            return {}
        oid = tbl["orbit_id"].to_pylist()
        tidx = np.asarray(tbl["target_idx"].to_numpy(zero_copy_only=False), dtype=np.int64)
        hpix = np.asarray(tbl["healpixel"].to_numpy(zero_copy_only=False), dtype=np.int64)
        out: dict[tuple[str, int], list[int]] = {}
        for o, t, h in zip(oid, tidx, hpix):
            k = (str(o), int(t))
            out.setdefault(k, []).append(int(h))
        return {k: np.unique(np.asarray(v, dtype=np.int64)) for k, v in out.items()}

    truth_frames_map: dict[tuple[str, int], np.ndarray] | None = None
    if bool(truth_frames_only):
        truth_frames_map = _keys_map_from_selected_keys_table(truth_frame_keys_tbl)

    # Stage 4 detection filtering geometries.
    if filters is None:
        # Defaults are geometry-only acceptance tests.
        # These are intentionally independent of how Stage 3 selected candidate frames.
        filters = [
            "cov_ellipse@1",
            "cov_ellipse@2",
            "cov_ellipse@3",
            "innov_ellipse@1",
            "innov_ellipse@2",
            "innov_ellipse@3",
            "cov_polygon_moc",
            "cov_disc",
        ]
    filters = [str(x).strip() for x in filters if str(x).strip()]
    for f in filters:
        _ = _validate_detection_geometry(f)  # validate

    metrics_rows: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []
    per_target_rows: list[dict[str, object]] = []

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

        def _run_mean_strategy(*, strat_dir: Path, name: str, part_files: list[Path]) -> None:
            """
            Run Stage 4 for one mean-ephemeris Stage 2 strategy directory.

            Appends to outer-scope metrics_rows/coverage_rows.
            """
            orbit_ids_in_strategy: set[str]
            try:
                ep_first = Ephemeris.from_parquet(str(part_files[0]))
                orbit_ids_in_strategy = set(_ephem_key_table(ep_first)["orbit_id"].to_pylist())
            except Exception:  # noqa: BLE001
                orbit_ids_in_strategy = set()

            allowed_orbits = _allowed_orbit_ids(orbit_ids_in_strategy=orbit_ids_in_strategy, max_orbits=max_orbits)
            truth_strategy = _filter_truth_matches_to_orbit_ids(truth_all, allowed_orbits)
            if max_targets is not None and truth_strategy.num_rows > 0:
                truth_strategy = truth_strategy.filter(pc.less(truth_strategy["target_idx"], pa.scalar(int(max_targets))))
            truth_keys_strategy = (
                _truth_keys_from_truth_matches(truth_strategy)
                if truth_strategy.num_rows > 0
                else TruthKeys.empty().table
            )
            truth_by_orbit_target, truth_pair_set = _truth_index_by_orbit_target(truth_strategy)
            n_truth = int(len(truth_pair_set))

            try:
                ep0 = Ephemeris.from_parquet(str(part_files[0]))
                has_cov = (ep0.coordinates.covariance is not None) and (not ep0.coordinates.covariance.is_all_nan())
            except Exception:  # noqa: BLE001
                has_cov = False

            mean_footprints = ["point"] + ([] if not has_cov else ["cov_disc", "cov_polygon", "cov_mc", "cov_polygon_moc"])
            # NOTE: We intentionally do not support `cov_polygon` (non-MOC) here. It was
            # evaluated historically but is too fragile; prefer `cov_polygon_moc`.
            mean_footprints = ["point"] + ([] if not has_cov else ["cov_disc", "cov_mc", "cov_polygon_moc"])
            needs_time_map = name == "assist_window_then_2body"
            dt_days = float(60.0) / 86400.0

            meta: dict[str, object] | None = None
            n_orbits = 0
            chunk = 0
            n_targets = 0
            if not needs_time_map:
                meta = json.loads((strat_dir / "meta.json").read_text())
                n_orbits = int(meta["n_orbits"])
                chunk = int(meta.get("time_chunk_size", 1024))
                n_targets = int(meta.get("n_time_targets", len(targets_tbl)))

            for footprint in mean_footprints:
                if selected_footprints is not None and str(footprint) not in selected_footprints:
                    continue

                active_geoms = (filters if bool(has_cov) else ["point_disc"])
                aggs: dict[str, _FilterAgg] = {f: _FilterAgg() for f in active_geoms}

                p_keys = _stage3_keys_path(strategy=name, variant_kind=None, footprint=str(footprint))
                if not p_keys.exists():
                    raise FileNotFoundError(
                        f"Stage 3 selected_keys missing for {name}/{footprint}: {p_keys} "
                        f"(did Stage 3 run with --compute-extra-frames?)"
                    )
                keys_tbl = pq.read_table(str(p_keys), columns=["orbit_id", "target_idx", "healpixel"])
                keys_map = _keys_map_from_selected_keys_table(keys_tbl)

                for pf in part_files:
                    ephem = Ephemeris.from_parquet(str(pf))
                    if len(ephem) == 0:
                        continue

                    if needs_time_map:
                        target_idx = _map_ephem_to_target_idx_by_time(ephem=ephem, targets=targets_tbl, dt_days=dt_days)
                        keep = target_idx >= 0
                        if not keep.any():
                            continue
                        if not keep.all():
                            hit = np.nonzero(keep)[0]
                            ephem = ephem.take(hit.tolist())
                            target_idx = target_idx[hit]
                    else:
                        part_idx = int(pf.stem.split("-")[-1])
                        start = part_idx * chunk
                        chunk_len = int(min(chunk, n_targets - start))
                        if chunk_len <= 0:
                            continue
                        target_idx = np.tile((np.arange(chunk_len, dtype=np.int64) + start), int(n_orbits))
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
                        mask = _filter_ephem_to_truth(ephem=ephem, target_idx=target_idx, truth_keys=truth_keys_strategy)
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

                        cov_row = None if cov_ll is None else cov_ll[i]
                        pred_pix = keys_map.get((oid, int(tidx)), np.array([], dtype=np.int64))
                        if pred_pix.size == 0:
                            continue

                        if truth_frames_map is not None:
                            hp_truth = truth_frames_map.get((oid, int(tidx)), np.array([], dtype=np.int64))
                            if hp_truth.size == 0:
                                continue
                            pred_pix = np.intersect1d(pred_pix, hp_truth, assume_unique=False)
                            if pred_pix.size == 0:
                                continue

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
                        det_sigma_floor_arcsec = _det_sigma_floor_arcsec_for_frames(frames=frames)

                        for agg in aggs.values():
                            agg.n_orbit_targets += 1
                            agg.n_frames_loaded += int(len(frames))
                            agg.n_observations_loaded += int(len(obs))
                            agg.io_sec += float(io_sec)

                        if len(obs) == 0:
                            if bool(write_per_target_metrics):
                                for filt_name in aggs.keys():
                                    per_target_rows.append(
                                        dict(
                                            stage2_run_dir=str(stage2_run_dir),
                                            subset_dir=str(subset_dir),
                                            strategy=name,
                                            variant_kind=None,
                                            footprint=str(footprint),
                                            detection_filter=str(filt_name),
                                            healpix_nside=int(healpix_nside),
                                            orbit_id=str(oid),
                                            target_idx=int(tidx),
                                            obscode=str(obscode),
                                            exposure_mjd_mid=float(mjd_mid),
                                            n_frames_loaded=int(len(frames)),
                                            n_observations_loaded=0,
                                            n_accepted=0,
                                            io_sec=float(io_sec),
                                            prep_sec=0.0,
                                            filter_sec=0.0,
                                        )
                                    )
                            continue

                        t_prep0 = time.perf_counter()
                        prep = _prepare_observation_arrays(obs)
                        prep_sec = time.perf_counter() - t_prep0
                        for agg in aggs.values():
                            agg.prep_sec += float(prep_sec)

                        for filt_name, agg in aggs.items():
                            try:
                                t_f0 = time.perf_counter()
                                keep = _detection_geometry_keep_mask(
                                    prep=prep,
                                    geometry=str(filt_name),
                                    lon0_deg=float(lon[i]),
                                    lat0_deg=float(lat[i]),
                                    cov_ll_deg2=(None if cov_row is None else np.asarray(cov_row, dtype=np.float64)),
                                    n_sigma=float(n_sigma),
                                    polygon_vertices=int(polygon_vertices),
                                    point_radius_arcsec=float(point_radius_arcsec),
                                    cov_mc_num_samples=int(cov_mc_num_samples),
                                    cov_mc_seed=int(cov_mc_seed),
                                    det_sigma_floor_arcsec=float(det_sigma_floor_arcsec),
                                    sample_lon_deg=None,
                                    sample_lat_deg=None,
                                )
                                filt_sec = float(time.perf_counter() - t_f0)
                            except BaseException as e:  # noqa: BLE001
                                if isinstance(e, (KeyboardInterrupt, SystemExit)):
                                    raise
                                agg.record_error(e)
                                continue

                            n_after = int(np.count_nonzero(keep))
                            if agg.n_after_prefilter is not None:
                                agg.n_after_prefilter += int(n_after)
                            agg.filter_sec += float(filt_sec)
                            # no separate chi2 step (geometry-only)

                            if bool(write_per_target_metrics):
                                per_target_rows.append(
                                    dict(
                                        stage2_run_dir=str(stage2_run_dir),
                                        subset_dir=str(subset_dir),
                                        strategy=name,
                                        variant_kind=None,
                                        footprint=str(footprint),
                                        detection_filter=str(filt_name),
                                        healpix_nside=int(healpix_nside),
                                        orbit_id=str(oid),
                                        target_idx=int(tidx),
                                        obscode=str(obscode),
                                        exposure_mjd_mid=float(mjd_mid),
                                        n_frames_loaded=int(len(frames)),
                                        n_observations_loaded=int(len(obs)),
                                        n_accepted=int(n_after),
                                        io_sec=float(io_sec),
                                        prep_sec=float(prep_sec),
                                        filter_sec=float(filt_sec),
                                    )
                                )

                            if n_after > 0:
                                truth_entries = truth_by_orbit_target.get((oid, int(tidx)), [])
                                rec = _recover_truth_obsids_for_orbit_target_masked(
                                    truth_entries=truth_entries,
                                    prep=prep,
                                    accepted_mask=keep,
                                    time_tol_sec=float(time_tol_sec),
                                    dist_tol_arcsec=float(dist_tol_arcsec),
                                )
                                for truth_obsid in rec:
                                    agg.recovered_truth_pairs.add((oid, str(truth_obsid)))

                for filt_name, agg in aggs.items():
                    runtime_total = float(agg.io_sec + agg.prep_sec + agg.filter_sec)
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
                            io_sec=float(agg.io_sec),
                            prep_sec=float(agg.prep_sec),
                            filter_sec=float(agg.filter_sec),
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

        def _run_variant_kind(
            *,
            variant_root_name: str,
            strat_dir: Path,
            variant_kind: str,
            part_files: list[Path],
        ) -> None:
            """Run Stage 4 for one variant-ephemeris (strategy, variant_kind)."""
            orbit_ids_in_strategy: set[str] = set()
            # IMPORTANT: For windowed strategies, part-000000 may not include all orbits, so
            # use variants_orbits.parquet when present.
            orbits_path = Path(strat_dir) / "variants_orbits.parquet"
            if orbits_path.exists():
                try:
                    orbits_tbl = pq.read_table(str(orbits_path), columns=["orbit_id", "object_id"])
                    obj = orbits_tbl.column("object_id").to_pylist()
                    orb = orbits_tbl.column("orbit_id").to_pylist()
                    keys: list[str] = []
                    for i, x in enumerate(obj):
                        s = "" if x is None else str(x).strip()
                        if s:
                            keys.append(_designation_from_object_id(s))
                        else:
                            keys.append(str(orb[i]) if i < len(orb) else "")
                    orbit_ids_in_strategy = set([k for k in keys if str(k).strip() != ""])
                except Exception:  # noqa: BLE001
                    orbit_ids_in_strategy = set()
            if not orbit_ids_in_strategy:
                try:
                    ep_first = VariantEphemeris.from_parquet(str(part_files[0]))
                    orbit_ids_in_strategy = set(_ephem_key_table(ep_first)["orbit_id"].to_pylist())
                except Exception:  # noqa: BLE001
                    orbit_ids_in_strategy = set()

            allowed_orbits = _allowed_orbit_ids(orbit_ids_in_strategy=orbit_ids_in_strategy, max_orbits=max_orbits)
            truth_strategy = _filter_truth_matches_to_orbit_ids(truth_all, allowed_orbits)
            if max_targets is not None and truth_strategy.num_rows > 0:
                truth_strategy = truth_strategy.filter(pc.less(truth_strategy["target_idx"], pa.scalar(int(max_targets))))
            truth_keys_strategy = (
                _truth_keys_from_truth_matches(truth_strategy)
                if truth_strategy.num_rows > 0
                else TruthKeys.empty().table
            )
            truth_by_orbit_target, truth_pair_set = _truth_index_by_orbit_target(truth_strategy)
            n_truth = int(len(truth_pair_set))

            variant_footprints = [
                ("sample_direct", None),
                ("sample_polygon_moc", "angle_sort"),
                ("sample_polygon_moc", "convex_hull"),
                ("sample_corridor", None),
                ("cov_disc_reconstructed", None),
                ("cov_polygon_reconstructed_moc", None),
                ("cov_mc_reconstructed", None),
            ]

            dt_days = float(60.0) / 86400.0
            is_mc = _is_monte_carlo_variant_kind(str(variant_kind))

            for footprint, polygon_mode in variant_footprints:
                out_fp = (
                    f"{footprint}:{polygon_mode}"
                    if (footprint in {"sample_polygon_moc"} and polygon_mode is not None)
                    else str(footprint)
                )
                if selected_footprints is not None and str(out_fp) not in selected_footprints:
                    continue

                active_geoms = list(filters)
                if not is_mc:
                    active_geoms = [
                        g
                        for g in active_geoms
                        if not str(g).startswith("sample_perimeter_polygon_moc")
                        and not str(g).startswith("cov_mc_polygon_moc")
                    ]
                aggs: dict[str, _FilterAgg] = {f: _FilterAgg() for f in active_geoms}

                p_keys = _stage3_keys_path(
                    strategy=str(variant_root_name),
                    variant_kind=str(variant_kind),
                    footprint=str(out_fp),
                )
                if not p_keys.exists():
                    raise FileNotFoundError(
                        f"Stage 3 selected_keys missing for {variant_root_name}:{variant_kind}/{out_fp}: {p_keys} "
                        f"(did Stage 3 run with --compute-extra-frames?)"
                    )
                keys_tbl = pq.read_table(str(p_keys), columns=["orbit_id", "target_idx", "healpixel"])
                keys_map = _keys_map_from_selected_keys_table(keys_tbl)

                for pf in part_files:
                    ephem = VariantEphemeris.from_parquet(str(pf))
                    if len(ephem) == 0:
                        continue

                    target_idx = _map_ephem_to_target_idx_by_time(ephem=ephem, targets=targets_tbl, dt_days=dt_days)
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
                        mask = _filter_ephem_to_truth(ephem=ephem, target_idx=target_idx, truth_keys=truth_keys_strategy)
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

                        if keys_map is not None:
                            pred_pix = keys_map.get((oid, int(tidx)), np.array([], dtype=np.int64))
                            if pred_pix.size == 0:
                                continue

                        if truth_frames_map is not None:
                            hp_truth = truth_frames_map.get((oid, int(tidx)), np.array([], dtype=np.int64))
                            if hp_truth.size == 0:
                                continue
                            pred_pix = np.intersect1d(pred_pix, hp_truth, assume_unique=False)
                            if pred_pix.size == 0:
                                continue

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
                        det_sigma_floor_arcsec = _det_sigma_floor_arcsec_for_frames(frames=frames)

                        for agg in aggs.values():
                            agg.n_orbit_targets += 1
                            agg.n_frames_loaded += int(len(frames))
                            agg.n_observations_loaded += int(len(obs))
                            agg.io_sec += float(io_sec)

                        if len(obs) == 0:
                            continue

                        t_prep0 = time.perf_counter()
                        prep = _prepare_observation_arrays(obs)
                        prep_sec = time.perf_counter() - t_prep0
                        for agg in aggs.values():
                            agg.prep_sec += float(prep_sec)

                        for filt_name, agg in aggs.items():
                            try:
                                t_f0 = time.perf_counter()
                                keep = _detection_geometry_keep_mask(
                                    prep=prep,
                                    geometry=str(filt_name),
                                    lon0_deg=float(lon0),
                                    lat0_deg=float(lat0),
                                    cov_ll_deg2=np.asarray(cov_ll, dtype=np.float64),
                                    n_sigma=float(n_sigma),
                                    polygon_vertices=int(polygon_vertices),
                                    point_radius_arcsec=float(point_radius_arcsec),
                                    cov_mc_num_samples=int(cov_mc_num_samples),
                                    cov_mc_seed=int(cov_mc_seed),
                                    det_sigma_floor_arcsec=float(det_sigma_floor_arcsec),
                                    sample_lon_deg=lon_g.astype(np.float64, copy=False),
                                    sample_lat_deg=lat_g.astype(np.float64, copy=False),
                                )
                                filt_sec = float(time.perf_counter() - t_f0)
                            except BaseException as e:  # noqa: BLE001
                                if isinstance(e, (KeyboardInterrupt, SystemExit)):
                                    raise
                                agg.record_error(e)
                                continue

                            n_after = int(np.count_nonzero(keep))
                            if agg.n_after_prefilter is not None:
                                agg.n_after_prefilter += int(n_after)
                            agg.filter_sec += float(filt_sec)

                            if n_after > 0:
                                truth_entries = truth_by_orbit_target.get((oid, int(tidx)), [])
                                rec = _recover_truth_obsids_for_orbit_target_masked(
                                    truth_entries=truth_entries,
                                    prep=prep,
                                    accepted_mask=keep,
                                    time_tol_sec=float(time_tol_sec),
                                    dist_tol_arcsec=float(dist_tol_arcsec),
                                )
                                for truth_obsid in rec:
                                    agg.recovered_truth_pairs.add((oid, str(truth_obsid)))

                for filt_name, agg in aggs.items():
                    runtime_total = float(agg.io_sec + agg.prep_sec + agg.filter_sec)
                    metrics_rows.append(
                        dict(
                            stage2_run_dir=str(stage2_run_dir),
                            subset_dir=str(subset_dir),
                            strategy=str(variant_root_name),
                            variant_kind=str(variant_kind),
                            footprint=str(out_fp),
                            detection_filter=str(filt_name),
                            healpix_nside=int(healpix_nside),
                            n_orbit_targets=int(agg.n_orbit_targets),
                            n_frames_loaded=int(agg.n_frames_loaded),
                            n_observations_loaded=int(agg.n_observations_loaded),
                            n_after_prefilter=(None if agg.n_after_prefilter is None else int(agg.n_after_prefilter)),
                            io_sec=float(agg.io_sec),
                            prep_sec=float(agg.prep_sec),
                            filter_sec=float(agg.filter_sec),
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
                            strategy=str(variant_root_name),
                            variant_kind=str(variant_kind),
                            footprint=str(out_fp),
                            detection_filter=str(filt_name),
                            healpix_nside=int(healpix_nside),
                            n_truth_matched=int(n_truth),
                            n_recovered=int(n_rec),
                            recall=(0.0 if n_truth == 0 else float(n_rec) / float(n_truth)),
                        )
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
            _run_mean_strategy(strat_dir=strat_dir, name=name, part_files=part_files)

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

                _run_variant_kind(
                    variant_root_name=str(variant_root_name),
                    strat_dir=strat_dir,
                    variant_kind=str(variant_kind),
                    part_files=part_files,
                )

    finally:
        conn.close()

    if not metrics_rows:
        metrics = Stage4Metrics.empty()
    else:
        metrics = Stage4Metrics.from_pyarrow(pa.Table.from_pylist(metrics_rows))
    if not coverage_rows:
        coverage = Stage4Coverage.empty()
    else:
        coverage = Stage4Coverage.from_pyarrow(pa.Table.from_pylist(coverage_rows))
    metrics.to_parquet(str(run_dir / "metrics.parquet"))
    coverage.to_parquet(str(run_dir / "coverage.parquet"))

    if bool(write_per_target_metrics):
        per_target = (
            Stage4PerTarget.empty()
            if not per_target_rows
            else Stage4PerTarget.from_pyarrow(pa.Table.from_pylist(per_target_rows))
        )
        per_target.to_parquet(str(run_dir / "per_target.parquet"))

    out_meta = dict(
        subset_dir=str(subset_dir),
        stage2_run_dir=str(stage2_run_dir),
        stage3_run_dir=str(stage3_run_dir),
        truth_frames_only=bool(truth_frames_only),
        healpix_nside=int(healpix_nside),
        n_sigma=float(n_sigma),
        polygon_vertices=int(polygon_vertices),
        cov_mc_num_samples=int(cov_mc_num_samples),
        cov_mc_seed=int(cov_mc_seed),
        point_radius_arcsec=float(point_radius_arcsec),
        corridor_radius_arcsec=float(corridor_radius_arcsec),
        corridor_step_arcsec=float(corridor_step_arcsec),
        detection_geometries=list(filters),
        strategies_requested=None if strategies is None else list(strategies),
        footprints_requested=None if footprints is None else list(footprints),
        footprint_set=None if footprint_set is None else str(footprint_set),
        only_truth=bool(only_truth),
        max_orbits=None if max_orbits is None else int(max_orbits),
        max_targets=None if max_targets is None else int(max_targets),
        obs_cache_max_frames=int(obs_cache_max_frames),
        write_per_target_metrics=bool(write_per_target_metrics),
        det_sigma_floor_arcsec_by_dataset=dict(_DET_SIGMA_FLOOR_ARCSEC_BY_DATASET),
        det_sigma_floor_arcsec_default=float(_DET_SIGMA_FLOOR_ARCSEC_DEFAULT),
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
        description="Stage 4: atomic geometry-based detection filtering benchmark runner (loads observations; evaluates acceptance geometries)."
    )
    p.add_argument("--subset-dir", type=str, required=True)
    p.add_argument("--stage2-run-dir", type=str, required=True)
    p.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help=(
            "Output root directory (default: <subset_dir>/artifacts/stage4). "
            "Run directory will be <out_dir>/<stage2_run_dir.name>."
        ),
    )
    p.add_argument(
        "--stage3-run-dir",
        type=str,
        default=None,
        help=(
            "Stage 3 run dir for selected_keys reuse (default: <subset_dir>/artifacts/stage3/<stage2_run_dir.name>)."
        ),
    )
    p.add_argument(
        "--truth-frames-only",
        action="store_true",
        help=(
            "Speed-up: only load observations from frames/healpixels that are known (from truth crossmatch) "
            "to contain a true detection for that orbit/target."
        ),
    )
    p.add_argument("--healpix-nside", type=int, required=True)
    p.add_argument("--n-sigma", type=float, default=3.0)
    p.add_argument("--polygon-vertices", type=int, default=32)
    p.add_argument("--cov-mc-num-samples", type=int, default=64)
    p.add_argument("--cov-mc-seed", type=int, default=0)
    p.add_argument(
        "--point-radius-arcsec",
        type=float,
        default=5.0,
        help="Radius (arcsec) for point_disc detection geometry (default: 5).",
    )
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
        help=(
            "Comma-separated list of Stage 4 detection geometries (acceptance tests). "
            "Defaults: cov_ellipse@1,cov_ellipse@2,cov_ellipse@3,innov_ellipse@1,innov_ellipse@2,innov_ellipse@3,"
            "cov_polygon_moc,cov_disc. "
            "Available: point_disc,cov_disc,cov_ellipse,innov_ellipse,cov_polygon_moc,cov_mc_polygon_moc,"
            "sample_perimeter_polygon_moc. "
            "You can override sigma per-geometry using '@', e.g. cov_ellipse@1. "
            "Note: perimeter-based geometries (sample_perimeter_polygon_moc and cov_mc_polygon_moc) are only "
            "enabled for Monte Carlo variant kinds (e.g. mc_256), not sigma points."
        ),
    )
    p.add_argument(
        "--footprints",
        type=str,
        default=None,
        help=(
            "Comma-separated list of footprint output names to include (e.g. 'point,cov_disc,cov_polygon_moc,"
            "sample_direct,sample_polygon_moc:convex_hull,cov_polygon_reconstructed_moc')."
        ),
    )
    p.add_argument(
        "--footprint-set",
        type=str,
        default=None,
        help="Named curated footprint subset (e.g. 'diverse', 'moc_only', 'fast').",
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
        "--write-per-target-metrics",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Write per-(orbit,target,filter) metrics to per_target.parquet so we can compute "
            "per-target distributions (percentiles) for loaded frames/detections/accepted detections."
        ),
    )
    args = p.parse_args()

    run_dir = run_stage4_detection_filter_bench(
        subset_dir=Path(args.subset_dir),
        stage2_run_dir=Path(args.stage2_run_dir),
        stage3_run_dir=None if args.stage3_run_dir is None else Path(args.stage3_run_dir),
        truth_frames_only=bool(args.truth_frames_only),
        out_dir=None if args.out_dir is None else Path(args.out_dir),
        healpix_nside=int(args.healpix_nside),
        n_sigma=float(args.n_sigma),
        polygon_vertices=int(args.polygon_vertices),
        cov_mc_num_samples=int(args.cov_mc_num_samples),
        cov_mc_seed=int(args.cov_mc_seed),
        point_radius_arcsec=float(args.point_radius_arcsec),
        corridor_radius_arcsec=float(args.corridor_radius_arcsec),
        corridor_step_arcsec=float(args.corridor_step_arcsec),
        strategies=_parse_csv_arg(args.strategies),
        filters=_parse_csv_arg(args.filters),
        footprints=_parse_csv_arg(args.footprints),
        footprint_set=(None if args.footprint_set is None else str(args.footprint_set)),
        only_truth=bool(args.only_truth),
        max_orbits=args.max_orbits,
        max_targets=args.max_targets,
        obs_cache_max_frames=int(args.obs_cache_max_frames),
        write_per_target_metrics=bool(args.write_per_target_metrics),
    )
    print(f"run_dir={run_dir}")
    print(f"metrics_parquet={run_dir / 'metrics.parquet'}")
    print(f"coverage_parquet={run_dir / 'coverage.parquet'}")


if __name__ == "__main__":
    main()

