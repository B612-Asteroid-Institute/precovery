from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import healpy as hp
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import quivr as qv

from adam_core.coordinates.cartesian import CartesianCoordinates
from adam_core.coordinates.spherical import SphericalCoordinates
from adam_core.orbits.ephemeris import Ephemeris
from adam_core.orbits.variants import VariantEphemeris
from adam_core.time import Timestamp

from ..methods.footprint_geometry_artifacts import write_geometry_artifacts
from ..methods.footprints import (
    corridor_path_lonlat_deg_from_samples,
    corridor_pixels_from_samples,
    disc_pixels_from_cov,
    ellipse_polygon_pixels_from_cov_moc,
    ellipse_boundary_vertices_lonlat_deg_from_cov,
    mc_pixels_from_cov,
    perimeter_polygon_from_samples,
    sample_pixels_direct,
    sample_perimeter_polygon_pixels_moc,
)
from ..selection.designation_normalization import normalize_designation
from ..selection.subset_designations import read_subset_window


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, obj: dict[str, object]) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _timestamp_to_utc(t: Timestamp) -> Timestamp:
    # Guard: normalize timestamps before any joins.
    return t.rescale("utc")


def _ephem_to_utc(ephem: Ephemeris) -> Ephemeris:
    # Guard: normalize timestamps before any joins.
    try:
        ephem = ephem.set_column("coordinates.time", _timestamp_to_utc(ephem.coordinates.time))
    except Exception:  # noqa: BLE001
        pass
    try:
        ephem = ephem.set_column(
            "aberrated_coordinates.time", _timestamp_to_utc(ephem.aberrated_coordinates.time)
        )
    except Exception:  # noqa: BLE001
        pass
    return ephem


class FrameTimeTargets(qv.Table):
    obscode = qv.LargeStringColumn()
    time = Timestamp.as_column()


class Stage3Metrics(qv.Table):
    stage2_run_dir = qv.LargeStringColumn()
    subset_dir = qv.LargeStringColumn()
    strategy = qv.LargeStringColumn()
    variant_kind = qv.LargeStringColumn(nullable=True)
    footprint = qv.LargeStringColumn()
    healpix_nside = qv.Int64Column()

    n_rows_ephem = qv.Int64Column()
    n_groups = qv.Int64Column(nullable=True)
    n_rows_with_cov = qv.Int64Column(nullable=True)

    sum_pred_pixels = qv.Int64Column()
    sum_frame_pixels = qv.Int64Column()
    sum_intersection = qv.Int64Column()
    runtime_sec = qv.Float64Column()
    n_errors = qv.Int64Column(nullable=True)
    error = qv.LargeStringColumn(nullable=True)


class Stage3Coverage(qv.Table):
    stage2_run_dir = qv.LargeStringColumn()
    subset_dir = qv.LargeStringColumn()
    strategy = qv.LargeStringColumn()
    variant_kind = qv.LargeStringColumn(nullable=True)
    footprint = qv.LargeStringColumn()
    healpix_nside = qv.Int64Column()

    n_truth = qv.Int64Column()
    n_covered = qv.Int64Column()
    coverage = qv.Float64Column()
    n_selected = qv.Int64Column(nullable=True)
    n_extra_frames = qv.Int64Column(nullable=True)

def _designation_from_object_id(object_id: str) -> str:
    return normalize_designation(str(object_id))


def _read_stage2_targets(stage2_run_dir: Path) -> pa.Table:
    """
    Table with columns:
      - target_idx (int64): 0..n_targets-1
      - obscode (large_string)
      - time (struct<days: int64, nanos: int64>): UTC exposure midpoint timestamp
    """
    targets = FrameTimeTargets.from_parquet(
        str(stage2_run_dir / "inputs" / "frame_time_targets.parquet")
    )
    idx = np.arange(len(targets), dtype=np.int64)
    return pa.Table.from_arrays(
        [
            pa.array(idx, type=pa.int64()),
            pa.array(targets.obscode.to_pylist(), type=pa.large_string()),
            targets.table.column("time").combine_chunks(),
        ],
        names=["target_idx", "obscode", "time"],
    )


class FramesPixels(qv.Table):
    target_idx = qv.Int64Column()
    healpixel = qv.Int64Column()

class TruthKeys(qv.Table):
    orbit_id = qv.LargeStringColumn()
    target_idx = qv.Int64Column()

class TruthObservations(qv.Table):
    """
    One row per truth observation that was crossmatched to the subset.

    This is the correct denominator for "coverage of possible crossmatches".
    """
    orbit_id = qv.LargeStringColumn()
    target_idx = qv.Int64Column()
    healpixel = qv.Int64Column()


def _read_frames_pixels_table(
    *,
    subset_dir: Path,
    targets: pa.Table,
) -> pa.Table:
    """
    Return distinct (target_idx, healpixel) for the Stage 2 target set.

    This is the vectorized join key-space Stage 3 uses (no python dict lookups).
    """
    index_db = subset_dir / "index.db"
    if not index_db.exists():
        raise FileNotFoundError(f"Missing subset index.db: {index_db}")

    n_targets = int(targets.num_rows)
    if n_targets == 0:
        return FramesPixels.empty().table

    # Stage 3 should *always* be restricted to the Stage 2 target set.
    # We implement this as an exact join (targets JOIN frames) so we never
    # accidentally scan extra frames outside the Stage 2 workload.
    import sqlite3

    codes = targets["obscode"].to_pylist()
    # `frames.exposure_mjd_mid` is stored as float UTC MJD in index.db, so we compute the
    # float join key on demand from the Stage 2 target timestamps (UTC).
    t = targets.column("time").combine_chunks()
    t_utc = Timestamp.from_kwargs(days=t.field("days"), nanos=t.field("nanos"), scale="utc")
    mjd = t_utc.mjd().to_numpy(zero_copy_only=False).astype(np.float64)
    tidx = pc.cast(targets["target_idx"], pa.int64()).to_numpy(zero_copy_only=False)

    conn = sqlite3.connect(str(index_db))
    try:
        conn.execute(
            "CREATE TEMP TABLE stage3_targets (obscode TEXT, exposure_mjd_mid REAL, target_idx INTEGER)"
        )
        # Batch inserts to avoid huge single executemany payloads on large target sets.
        batch = 100000
        n = int(len(codes))
        for i0 in range(0, n, batch):
            i1 = min(i0 + batch, n)
            conn.executemany(
                "INSERT INTO stage3_targets (obscode, exposure_mjd_mid, target_idx) VALUES (?, ?, ?)",
                [
                    (str(o), float(m), int(i))
                    for o, m, i in zip(
                        codes[i0:i1],
                        mjd[i0:i1].tolist(),
                        tidx[i0:i1].tolist(),
                    )
                ],
            )
        conn.execute(
            "CREATE INDEX stage3_targets_idx ON stage3_targets (obscode, exposure_mjd_mid)"
        )
        rows = conn.execute(
            """
            SELECT t.target_idx, f.healpixel
            FROM frames f
            INNER JOIN stage3_targets t
              ON f.obscode = t.obscode
             AND f.exposure_mjd_mid = t.exposure_mjd_mid
            """
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        return FramesPixels.empty().table

    out = pa.table(
        {
            "target_idx": pa.array([int(r[0]) for r in rows], pa.int64()),
            "healpixel": pa.array([int(r[1]) for r in rows], pa.int64()),
        }
    ).combine_chunks()
    qt = FramesPixels.from_pyarrow(out).drop_duplicates(subset=["target_idx", "healpixel"])
    return qt.table


def _map_times_to_target_idx_by_obscode(
    *,
    obscode: np.ndarray,
    time_utc: Timestamp,
    targets: pa.Table,
    dt_sec: float,
    precision: str = "us",
) -> np.ndarray:
    """
    Map each (obscode, time_utc) row to the nearest Stage 2 target_idx for that obscode
    within tolerance, using integer time keys (no float MJD matching).

    Notes
    -----
    - This is a *nearest-neighbor* lookup (per obscode) within `dt_sec`. We intentionally
      avoid float MJD comparisons and also avoid exact (days,nanos) equality; ns-level
      offsets (e.g. -16 ns) are naturally handled because the nearest target time is still
      the same exposure midpoint.
    - `precision` is accepted for API compatibility, but this routine does not require
      precision-bucket rounding to be robust for the current use case (dt_sec ~ 60s).
    """
    targ_obscode = np.asarray(targets["obscode"].to_pylist(), dtype=object)
    t = targets.column("time").combine_chunks()
    targ_time = Timestamp.from_kwargs(days=t.field("days"), nanos=t.field("nanos"), scale="utc")
    targ_idx = np.asarray(targets["target_idx"].to_numpy(zero_copy_only=False), dtype=np.int64)

    # Convert UTC timestamps to a single integer nanosecond key for fast sorting/search.
    # This is safe because `Timestamp.nanos` is already "nanos since start of day".
    day_ns = 86_400 * 1_000_000_000
    tt = _timestamp_to_utc(targ_time)
    t0 = _timestamp_to_utc(time_utc)
    targ_key = (
        tt.days.to_numpy(zero_copy_only=False).astype(np.int64, copy=False) * day_ns
        + tt.nanos.to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    )
    time_key = (
        t0.days.to_numpy(zero_copy_only=False).astype(np.int64, copy=False) * day_ns
        + t0.nanos.to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    )

    out = np.full(len(time_key), -1, dtype=np.int64)
    tol_ns = int(float(dt_sec) * 1_000_000_000)
    for code in sorted(set(obscode.tolist())):
        mt = obscode == code
        if not mt.any():
            continue
        mg = targ_obscode == code
        if not mg.any():
            continue
        key = targ_key[mg]
        idx = targ_idx[mg]
        order = np.argsort(key)
        key_s = key[order]
        idx_s = idx[order]
        times = time_key[mt]
        j = np.searchsorted(key_s, times)
        j0 = np.clip(j - 1, 0, len(key_s) - 1)
        j1 = np.clip(j, 0, len(key_s) - 1)
        d0 = np.abs(key_s[j0] - times)
        d1 = np.abs(key_s[j1] - times)
        use1 = d1 < d0
        jj = np.where(use1, j1, j0)
        dd = np.where(use1, d1, d0)
        ok = dd <= int(tol_ns)
        out[np.nonzero(mt)[0][ok]] = idx_s[jj[ok]]
    return out


def _read_truth_observations_table(
    *, subset_dir: Path, targets: pa.Table, inputs_artifacts_dir: Path | None = None
) -> pa.Table:
    """
    Return matched truth observations aligned to Stage 2 targets as:
      (orbit_id, target_idx, healpixel)

    - orbit_id is the stable designation (SBDB-derived; see fetch step)
    - target_idx is Stage 2's exposure-time index
    - healpixel is the healpixel of the truth observation (nside = subset frames nside)
    """
    artifacts_dir = (
        Path(inputs_artifacts_dir).expanduser().resolve()
        if inputs_artifacts_dir is not None
        else (subset_dir / "artifacts")
    )
    truth_path = artifacts_dir / "truth_precovery_crossmatch.parquet"
    orbits_path = artifacts_dir / "orbits_selected_sbdb.parquet"
    if not truth_path.exists():
        raise FileNotFoundError(f"Missing truth crossmatch parquet: {truth_path}")
    if not orbits_path.exists():
        raise FileNotFoundError(f"Missing SBDB orbits parquet: {orbits_path}")

    truth = pq.read_table(
        str(truth_path),
        columns=[
            "matched",
            "designation",
            "obscode",
            "match_dataset_id",
            "match_exposure_id",
            "healpixel",
        ],
    )
    truth = truth.filter(pc.equal(truth["matched"], True))
    if truth.num_rows == 0:
        return TruthObservations.empty().table

    # Keep only rows with exposure identifiers so we can align to exposure midpoints.
    m_has = pc.and_(
        pc.invert(pc.is_null(truth["match_exposure_id"])),
        pc.invert(pc.is_null(truth["match_dataset_id"])),
    )
    truth = truth.filter(m_has)
    if truth.num_rows == 0:
        return TruthObservations.empty().table

    orbits = pq.read_table(str(orbits_path), columns=["orbit_id", "object_id"])
    # orbit_id is already normalized designation now, but be robust.
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
        ["designation", "obscode", "match_dataset_id", "match_exposure_id", "healpixel"]
    )
    truth = truth.join(orbit_map, keys=["designation"], join_type="inner")
    if truth.num_rows == 0:
        return TruthObservations.empty().table

    # Align truth to Stage 2 targets by exposure midpoint, not `match_time_mjd_utc`.
    # `match_time_mjd_utc` can be significantly offset from `frames.exposure_mjd_mid` for
    # some data sources; Stage 3/4 use exposure midpoints as the key.
    win = read_subset_window(subset_dir)
    index_db = Path(win.index_db)
    if not index_db.exists():
        raise FileNotFoundError(f"Missing subset index.db: {index_db}")

    ds = [str(x) for x in truth["match_dataset_id"].to_pylist()]
    code = [str(x) for x in truth["obscode"].to_pylist()]
    exp = [str(x) for x in truth["match_exposure_id"].to_pylist()]
    keys = list(zip(ds, code, exp))
    uniq = sorted(set(keys))
    if not uniq:
        return TruthObservations.empty().table

    import sqlite3

    conn = sqlite3.connect(str(index_db))
    try:
        conn.execute(
            "CREATE INDEX IF NOT EXISTS frames_truth_join_idx "
            "ON frames(dataset_id, obscode, exposure_id)"
        )
        conn.execute(
            "CREATE TEMP TABLE truth_keys (dataset_id TEXT, obscode TEXT, exposure_id TEXT)"
        )
        conn.executemany(
            "INSERT INTO truth_keys (dataset_id, obscode, exposure_id) VALUES (?, ?, ?)",
            uniq,
        )
        conn.execute("CREATE INDEX truth_keys_idx ON truth_keys (dataset_id, obscode, exposure_id)")
        rows = conn.execute(
            """
            SELECT t.dataset_id, t.obscode, t.exposure_id, f.exposure_mjd_mid
            FROM frames f
            INNER JOIN truth_keys t
              ON f.dataset_id = t.dataset_id
             AND f.obscode = t.obscode
             AND f.exposure_id = t.exposure_id
            """
        ).fetchall()
    finally:
        conn.close()

    mid_by_key: dict[tuple[str, str, str], float] = {
        (str(r[0]), str(r[1]), str(r[2])): float(r[3]) for r in rows
    }
    mjd_mid = [mid_by_key.get((ds, oc, ex)) for ds, oc, ex in keys]
    ok_mid = np.asarray([x is not None for x in mjd_mid], dtype=bool)
    if not ok_mid.any():
        return TruthObservations.empty().table

    mjd_mid_f = np.asarray([float(x) for x in np.asarray(mjd_mid, dtype=object)[ok_mid].tolist()], dtype=np.float64)
    t_obscode = np.asarray(truth["obscode"].to_pylist(), dtype=object)[ok_mid]
    t_orbit = np.asarray(truth["orbit_id"].to_pylist(), dtype=object)[ok_mid]
    t_hpix = np.asarray(truth["healpixel"].to_numpy(zero_copy_only=False), dtype=np.int64)[ok_mid]

    t_time_utc = Timestamp.from_mjd(mjd_mid_f.tolist(), scale="utc")
    target_idx = _map_times_to_target_idx_by_obscode(
        obscode=t_obscode,
        time_utc=t_time_utc,
        targets=targets,
        dt_sec=0.1,
        precision="us",
    )
    ok = target_idx >= 0
    if not ok.any():
        return TruthObservations.empty().table

    return TruthObservations.from_kwargs(
        orbit_id=[str(x) for x in t_orbit[ok].tolist()],
        target_idx=[int(x) for x in target_idx[ok].tolist()],
        healpixel=[int(x) for x in t_hpix[ok].tolist()],
    ).table


def _filter_truth_to_orbit_ids(truth_obs: pa.Table, orbit_ids: set[str]) -> pa.Table:
    """
    Restrict truth observations to a strategy's orbit_id set.

    This is critical for smoke tests where Stage 2 is run on a subset of orbits: the
    Stage 3 coverage denominator should only include truth rows relevant to those orbits.
    """
    if truth_obs.num_rows == 0 or not orbit_ids:
        return TruthObservations.empty().table
    mask = pc.is_in(
        truth_obs["orbit_id"],
        value_set=pa.array(sorted(orbit_ids), type=pa.large_string()),
    )
    return truth_obs.filter(mask)

def _ephem_key_table(ephem: Ephemeris) -> pa.Table:
    # Use `object_id` (SBDB name) when present; stage2's `orbit_id` historically was not unique.
    orbit = ephem.orbit_id.to_pylist()
    obj_col = getattr(ephem, "object_id", None)
    obj = obj_col.to_pylist() if obj_col is not None else None
    if obj is None:
        keys = [str(x) for x in orbit]
    else:
        # Be robust to Ephemeris tables that have an `object_id` column but contain nulls:
        # fall back to orbit_id on a per-row basis.
        keys: list[str] = []
        for i, x in enumerate(obj):
            s = "" if x is None else str(x).strip()
            if s:
                keys.append(_designation_from_object_id(s))
            else:
                keys.append(str(orbit[i]) if i < len(orbit) else "")
    orbit_id = pa.array(keys, type=pa.large_string())
    return pa.table({"orbit_id": orbit_id})


def _filter_ephem_to_truth(
    *, ephem: Ephemeris, target_idx: np.ndarray, truth_keys: pa.Table
) -> np.ndarray:
    """
    Return a boolean mask selecting ephemeris rows present in truth_keys.
    Vectorized via pyarrow join.
    """
    if len(truth_keys) == 0 or len(ephem) == 0:
        return np.zeros(len(ephem), dtype=bool)
    keys = _ephem_key_table(ephem)
    keys = keys.append_column("target_idx", pa.array(target_idx.astype(np.int64), pa.int64()))
    keys = keys.append_column("row_idx", pa.array(np.arange(len(ephem), dtype=np.int64)))
    # Older pyarrow versions may not support semi-joins; emulate via inner join on keys.
    joined = keys.join(
        truth_keys,
        keys=["orbit_id", "target_idx"],
        join_type="inner",
    )
    hit_idx = np.asarray(joined["row_idx"].to_numpy(zero_copy_only=False), dtype=np.int64)
    if hit_idx.size > 0:
        hit_idx = np.unique(hit_idx)
    m = np.zeros(len(ephem), dtype=bool)
    m[hit_idx] = True
    return m


def _map_ephem_to_target_idx_by_time(
    *,
    ephem: Ephemeris,
    targets: pa.Table,
    dt_days: float,
) -> np.ndarray:
    """
    Map each ephemeris row to the nearest Stage 2 target_idx by matching (obscode, time).

    This is used for strategies whose output ordering is not a simple (orbit-major × time-chunk)
    cross product of Stage 2 targets (e.g. assist_window_then_2body).
    """
    # Normalize ephem time to UTC and match using Timestamp keys (no float MJD matching).
    t_utc = _timestamp_to_utc(ephem.coordinates.time)
    code = np.asarray(ephem.coordinates.origin.code.to_pylist(), dtype=object)
    return _map_times_to_target_idx_by_obscode(
        obscode=code,
        time_utc=t_utc,
        targets=targets,
        dt_sec=float(dt_days) * 86400.0,
        precision="us",
    )


def _eval_mean_point(
    *,
    ephem: Ephemeris,
    frames_pixels: pa.Table,
    nside: int,
    target_idx: np.ndarray,
) -> tuple[int, int, int]:
    """
    Vectorized point-pixel intersection count via join:
      predicted_pixel rows JOIN frame_pixels rows on (obscode, days, nanos, healpixel)
    Returns (n_rows, sum_pred_pixels, sum_intersection)
    """
    lon = ephem.coordinates.lon.to_numpy(zero_copy_only=False).astype(np.float64)
    lat = ephem.coordinates.lat.to_numpy(zero_copy_only=False).astype(np.float64)
    pix = hp.ang2pix(int(nside), lon, lat, lonlat=True, nest=True).astype(np.int64)
    n_rows = int(len(ephem))

    pred = pa.table(
        {
            "target_idx": pa.array(target_idx.astype(np.int64), pa.int64()),
            "healpixel": pa.array(pix, pa.int64()),
        }
    )
    joined = pred.join(frames_pixels, keys=["target_idx", "healpixel"], join_type="inner")
    return n_rows, n_rows, int(len(joined))

def _truth_keys_from_truth_observations(truth_obs: pa.Table) -> pa.Table:
    if truth_obs.num_rows == 0:
        return TruthKeys.empty().table
    qt = TruthKeys.from_pyarrow(truth_obs.select(["orbit_id", "target_idx"]))
    qt = qt.drop_duplicates(subset=["orbit_id", "target_idx"])
    return qt.table


def _predicted_pixels_from_mean_row(
    *,
    lon_deg: float,
    lat_deg: float,
    cov_ll_deg2: np.ndarray | None,
    nside: int,
    footprint: str,
    n_sigma: float,
    polygon_vertices: int,
    mc_num_samples: int,
    mc_seed: int,
) -> np.ndarray:
    if footprint == "point":
        return np.unique(np.asarray([hp.ang2pix(int(nside), float(lon_deg), float(lat_deg), lonlat=True, nest=True)], dtype=np.int64))
    if cov_ll_deg2 is None:
        # No covariance: fall back to exact pixel.
        return np.unique(np.asarray([hp.ang2pix(int(nside), float(lon_deg), float(lat_deg), lonlat=True, nest=True)], dtype=np.int64))
    if footprint == "cov_disc":
        return disc_pixels_from_cov(lon0_deg=float(lon_deg), lat0_deg=float(lat_deg), cov_ll_deg2=cov_ll_deg2, nside=int(nside), n_sigma=float(n_sigma))
    if footprint == "cov_polygon_moc":
        return ellipse_polygon_pixels_from_cov_moc(
            lon0_deg=float(lon_deg),
            lat0_deg=float(lat_deg),
            cov_ll_deg2=cov_ll_deg2,
            nside=int(nside),
            n_sigma=float(n_sigma),
            num_vertices=int(polygon_vertices),
        )
    if footprint == "cov_mc":
        return mc_pixels_from_cov(lon0_deg=float(lon_deg), lat0_deg=float(lat_deg), cov_ll_deg2=cov_ll_deg2, nside=int(nside), n_sigma=float(n_sigma), num_samples=int(mc_num_samples), seed=int(mc_seed))
    raise ValueError(f"Unknown footprint for mean ephemeris: {footprint}")


def _predicted_pixels_from_samples(
    *,
    lon_deg: np.ndarray,
    lat_deg: np.ndarray,
    nside: int,
    footprint: str,
    polygon_mode: str,
    corridor_radius_arcsec: float,
    corridor_step_arcsec: float,
) -> np.ndarray:
    if lon_deg.size == 0:
        return np.array([], dtype=np.int64)
    if footprint == "sample_direct":
        return sample_pixels_direct(lon_deg=lon_deg, lat_deg=lat_deg, nside=int(nside))
    if footprint == "sample_polygon_moc":
        return sample_perimeter_polygon_pixels_moc(
            lon0_deg=float(lon_deg[0]),
            lat0_deg=float(lat_deg[0]),
            lon_deg=lon_deg.astype(np.float64, copy=False),
            lat_deg=lat_deg.astype(np.float64, copy=False),
            nside=int(nside),
            mode=("convex_hull" if str(polygon_mode) == "convex_hull" else "angle_sort"),
            include_center_disc=True,
        )
    if footprint == "sample_corridor":
        return corridor_pixels_from_samples(
            lon0_deg=float(lon_deg[0]),
            lat0_deg=float(lat_deg[0]),
            lon_deg=lon_deg,
            lat_deg=lat_deg,
            nside=int(nside),
            radius_arcsec=float(corridor_radius_arcsec),
            step_arcsec=float(corridor_step_arcsec),
        )
    raise ValueError(f"Unknown footprint for sample ephemeris: {footprint}")


def _collapse_variant_ephemeris_group(
    *,
    variants: VariantEphemeris,
) -> Ephemeris:
    """
    Collapse a single grouped `VariantEphemeris` into one covariance-bearing `Ephemeris` row.

    Newer `adam_core` provides `VariantEphemeris.collapse_by_object_id()`, which groups by
    (object_id, time, origin code) and reconstructs mean + covariance.

    For compatibility with older `adam_core` versions, we fall back to:
    - computing a mean ephemeris row for this group (using `weights` when available), then
    - calling `VariantEphemeris.collapse(mean_ephemeris)` to attach covariances.
    """
    if hasattr(variants, "collapse_by_object_id"):
        collapsed = variants.collapse_by_object_id()
    else:
        if len(variants) == 0:
            return Ephemeris.empty()

        # Compute a weighted mean in spherical coordinates.
        vals = variants.coordinates.values.astype(np.float64, copy=False)  # (N, 6)
        w = variants.weights.to_numpy(zero_copy_only=False).astype(np.float64)
        w = np.where(np.isfinite(w), w, 0.0)
        s = float(np.sum(w))
        if (not np.isfinite(s)) or s <= 0.0:
            w = np.full(len(variants), 1.0 / float(len(variants)), dtype=np.float64)
        else:
            w = w / s

        mean = np.sum(vals * w[:, None], axis=0)
        coords = SphericalCoordinates.from_kwargs(
            rho=[float(mean[0])],
            lon=[float(mean[1])],
            lat=[float(mean[2])],
            vrho=[float(mean[3])],
            vlon=[float(mean[4])],
            vlat=[float(mean[5])],
            time=variants.coordinates.time[:1],
            origin=variants.coordinates.origin[:1],
            frame=variants.coordinates.frame,
        )

        # If aberrated coords exist on variants, carry a compatible mean ephemeris row too.
        try:
            has_aberrated = not pc.all(pc.is_null(variants.aberrated_coordinates.x)).as_py()
        except Exception:  # noqa: BLE001
            has_aberrated = False
        if has_aberrated:
            ab_vals = variants.aberrated_coordinates.values.astype(np.float64, copy=False)
            mean_ab = np.sum(ab_vals * w[:, None], axis=0)
            ab = CartesianCoordinates.from_kwargs(
                x=[float(mean_ab[0])],
                y=[float(mean_ab[1])],
                z=[float(mean_ab[2])],
                vx=[float(mean_ab[3])],
                vy=[float(mean_ab[4])],
                vz=[float(mean_ab[5])],
                time=variants.aberrated_coordinates.time[:1],
                origin=variants.aberrated_coordinates.origin[:1],
                frame=variants.aberrated_coordinates.frame,
            )
            ephem_mean = Ephemeris.from_kwargs(
                orbit_id=[str(variants.orbit_id[0].as_py())],
                coordinates=coords,
                aberrated_coordinates=ab,
            )
        else:
            ephem_mean = Ephemeris.from_kwargs(
                orbit_id=[str(variants.orbit_id[0].as_py())],
                coordinates=coords,
            )
        collapsed = variants.collapse(ephem_mean)
    if len(collapsed) != 1:
        raise ValueError(
            "Expected exactly one collapsed ephemeris row for a grouped VariantEphemeris; "
            f"got {len(collapsed)}"
        )
    return collapsed


def _group_slices_by_orbit_target(
    orbit_id: np.ndarray, target_idx: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (order, group_starts) where:
    - order sorts rows by (orbit_id, target_idx)
    - group_starts are start indices into order for each group
    """
    enc = pa.array(orbit_id.tolist(), type=pa.large_string()).dictionary_encode()
    orbit_code = np.asarray(enc.indices.to_numpy(zero_copy_only=False), dtype=np.int64)
    # Include original row index as the last tie-breaker so within-group ordering is stable.
    order = np.lexsort((np.arange(len(target_idx), dtype=np.int64), target_idx.astype(np.int64), orbit_code))
    oc = orbit_code[order]
    ti = target_idx[order].astype(np.int64)
    if len(order) == 0:
        return order, np.array([], dtype=np.int64)
    boundaries = np.nonzero((oc[1:] != oc[:-1]) | (ti[1:] != ti[:-1]))[0] + 1
    starts = np.concatenate([np.array([0], dtype=np.int64), boundaries.astype(np.int64)])
    return order, starts


def run_stage3_healpixel_bench(
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
    persist_geometry: bool = True,
    out_dir: Path | None = None,
    inputs_artifacts_dir: Path | None = None,
    strategies: list[str] | None = None,
    footprints: list[str] | None = None,
    only_truth: bool = False,
    compute_extra_frames: bool = False,
) -> Path:
    """
    Stage 3 (atomic): consume cached Stage 2 ephemerides and compute healpixel intersections.

    Outputs:
      - metrics.parquet: runtime + row counts by (strategy, footprint)
      - coverage.parquet: truth healpixel coverage by (strategy, footprint)
      - footprint_geometry/**: reusable footprint geometry artifacts (optional)
    """
    if out_dir is None:
        out_dir = subset_dir / "artifacts" / "stage3"
    _ensure_dir(out_dir)
    run_dir = out_dir / stage2_run_dir.name
    _ensure_dir(run_dir)

    targets_tbl = _read_stage2_targets(stage2_run_dir)
    frames_pixels = _read_frames_pixels_table(subset_dir=subset_dir, targets=targets_tbl)
    truth_obs_tbl = _read_truth_observations_table(
        subset_dir=subset_dir, targets=targets_tbl, inputs_artifacts_dir=inputs_artifacts_dir
    )
    truth_keys_tbl = _truth_keys_from_truth_observations(truth_obs_tbl)
    # NOTE: truth_unique tables are computed per-strategy after filtering to the
    # orbits present in that strategy's Stage 2 outputs.

    metrics_rows: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []

    strategies_root = stage2_run_dir / "strategies"
    if not strategies_root.exists():
        raise FileNotFoundError(f"Missing Stage 2 strategies dir: {strategies_root}")

    def _normalize_fp_list(lst: list[str]) -> set[str]:
        out: set[str] = set()
        for x in lst:
            s = str(x).strip()
            if s:
                out.add(s)
        return out

    selected_footprints: set[str] | None = None
    if footprints is not None:
        selected_footprints = _normalize_fp_list(footprints)

    def _enabled(name: str) -> bool:
        return strategies is None or name in strategies

    def _enabled_variant(strategy: str, variant_kind: str) -> bool:
        key = f"{strategy}:{variant_kind}"
        return strategies is None or key in strategies or strategy in strategies

    def _run_mean_strategy(*, strat_dir: Path, name: str) -> None:
        """
        Evaluate all mean-ephemeris footprints for a single Stage 2 strategy directory.

        Appends to outer-scope metrics_rows/coverage_rows and writes per-footprint artifacts
        into run_dir as needed.
        """
        mean_dir = strat_dir / "mean_ephemeris"
        if not mean_dir.exists():
            return
        part_files = sorted(mean_dir.glob("part-*.parquet"))
        if not part_files:
            return

        orbit_ids_in_strategy: set[str] = set()
        try:
            ep_first = Ephemeris.from_parquet(str(part_files[0]))
            orbit_ids_in_strategy = set(_ephem_key_table(ep_first)["orbit_id"].to_pylist())
        except Exception:  # noqa: BLE001
            orbit_ids_in_strategy = set()

        truth_obs_tbl_strategy = _filter_truth_to_orbit_ids(truth_obs_tbl, orbit_ids_in_strategy)
        truth_keys_tbl_strategy = _truth_keys_from_truth_observations(truth_obs_tbl_strategy)
        truth_unique_tbl_strategy: pa.Table | None = None
        if bool(compute_extra_frames) and truth_obs_tbl_strategy.num_rows > 0:
            truth_unique_tbl_strategy = TruthObservations.from_pyarrow(truth_obs_tbl_strategy).drop_duplicates(
                subset=["orbit_id", "target_idx", "healpixel"]
            ).table

        try:
            ep0 = _ephem_to_utc(Ephemeris.from_parquet(str(part_files[0])))
            has_cov = (
                ep0.coordinates.covariance is not None and (not ep0.coordinates.covariance.is_all_nan())
            )
        except Exception:  # noqa: BLE001
            has_cov = False

        # NOTE: We intentionally avoid the non-MOC polygon rasterizers (`cov_polygon`) here.
        #
        # We previously evaluated both `cov_polygon` and `cov_polygon_moc`. In practice,
        # `healpy.query_polygon` is extremely sensitive to convexity/degeneracy and can
        # hard-abort in C++; even with our Python prechecks, it produced high error rates
        # (e.g. in truth-only runs over subset_2019-08_*).
        #
        # The MOC-named variants (`*_moc`) implement polygon membership/rasterization in a
        # more robust way and are the recommended default for coverage/recall evaluation.
        active_footprints = ["point"] + ([] if not has_cov else ["cov_disc", "cov_mc", "cov_polygon_moc"])
        if not active_footprints:
            return

        meta = json.loads((strat_dir / "meta.json").read_text())
        n_orbits = int(meta["n_orbits"])
        chunk = int(meta.get("time_chunk_size", 1024))
        n_targets = int(meta.get("n_time_targets", len(targets_tbl)))
        needs_time_map = (name == "assist_window_then_2body")
        dt_days = float(60.0) / 86400.0

        truth_obs = truth_obs_tbl_strategy
        truth_unique = (
            truth_unique_tbl_strategy
            if truth_unique_tbl_strategy is not None
            else pa.table(
                {
                    "orbit_id": pa.array([], pa.large_string()),
                    "target_idx": pa.array([], pa.int64()),
                    "healpixel": pa.array([], pa.int64()),
                }
            )
        )

        for footprint in active_footprints:
            if selected_footprints is not None and str(footprint) not in selected_footprints:
                continue
            t0 = time.perf_counter()
            n_rows_total = 0
            n_rows_used = 0
            n_rows_with_cov = 0
            sum_pred_pixels = 0
            sum_intersection = 0
            covered_total = 0
            n_errors = 0
            first_error: str | None = None
            selected_parts: list[pa.Table] = []
            geom_rows: list[dict[str, object]] = []
            geom_points: list[dict[str, object]] = []
            seen_geom: set[tuple[str, int]] = set()

            for pf in part_files:
                ephem = Ephemeris.from_parquet(str(pf))
                n_rows_total += int(len(ephem))
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

                if bool(only_truth):
                    mask = _filter_ephem_to_truth(ephem=ephem, target_idx=target_idx, truth_keys=truth_keys_tbl_strategy)
                    hit = np.nonzero(mask)[0]
                    if hit.size == 0:
                        continue
                    ephem = ephem.take(hit.tolist())
                    target_idx = target_idx[hit]

                n_rows_used += int(len(ephem))
                if len(ephem) == 0:
                    continue

                orbit_id_arr = _ephem_key_table(ephem)["orbit_id"].to_pylist()
                lon = ephem.coordinates.lon.to_numpy(zero_copy_only=False).astype(np.float64)
                lat = ephem.coordinates.lat.to_numpy(zero_copy_only=False).astype(np.float64)

                if footprint == "point":
                    pix = hp.ang2pix(int(healpix_nside), lon, lat, lonlat=True, nest=True).astype(np.int64)
                    sum_pred_pixels += int(len(pix))
                    pred = pa.table(
                        {
                            "orbit_id": pa.array(orbit_id_arr, pa.large_string()),
                            "target_idx": pa.array(target_idx.astype(np.int64), pa.int64()),
                            "healpixel": pa.array(pix, pa.int64()),
                        }
                    )
                else:
                    if ephem.coordinates.covariance is None or ephem.coordinates.covariance.is_all_nan():
                        continue
                    cov6 = ephem.coordinates.covariance.to_matrix()
                    cov_ll = cov6[:, 1:3, 1:3].astype(np.float64)
                    n_rows_with_cov += int(len(ephem))

                    pix_list: list[np.ndarray] = []
                    lens = np.empty(int(len(ephem)), dtype=np.int64)
                    for i in range(int(len(ephem))):
                        try:
                            pix_i = _predicted_pixels_from_mean_row(
                                lon_deg=float(lon[i]),
                                lat_deg=float(lat[i]),
                                cov_ll_deg2=cov_ll[i],
                                nside=int(healpix_nside),
                                footprint=str(footprint),
                                n_sigma=float(n_sigma),
                                polygon_vertices=int(polygon_vertices),
                                mc_num_samples=int(cov_mc_num_samples),
                                mc_seed=int(cov_mc_seed),
                            )
                        except BaseException as e:  # noqa: BLE001
                            if isinstance(e, (KeyboardInterrupt, SystemExit)):
                                raise
                            n_errors += 1
                            if first_error is None:
                                first_error = f"{type(e).__name__}: {e}"
                            pix_i = np.array([], dtype=np.int64)
                        pix_list.append(pix_i)
                        lens[i] = int(len(pix_i))

                        if bool(persist_geometry):
                            k = (str(orbit_id_arr[i]), int(target_idx[i]))
                            if k not in seen_geom:
                                seen_geom.add(k)
                                geom_id = f"{k[0]}|{k[1]}"
                                fp = str(footprint)
                                if fp in {"cov_disc", "cov_mc"}:
                                    geom_rows.append(
                                        dict(
                                            strategy=str(name),
                                            variant_kind=None,
                                            footprint=str(footprint),
                                            orbit_id=str(k[0]),
                                            target_idx=int(k[1]),
                                            geometry_kind="ellipse_cov",
                                            lon0_deg=float(lon[i]),
                                            lat0_deg=float(lat[i]),
                                            cov_ll_00=float(cov_ll[i][0, 0]),
                                            cov_ll_01=float(cov_ll[i][0, 1]),
                                            cov_ll_10=float(cov_ll[i][1, 0]),
                                            cov_ll_11=float(cov_ll[i][1, 1]),
                                            n_sigma=float(n_sigma),
                                            polygon_vertices=None,
                                            polygon_mode=None,
                                            corridor_radius_arcsec=None,
                                            corridor_step_arcsec=None,
                                            buffer_arcsec=None,
                                            geom_id=None,
                                        )
                                    )
                                elif fp in {"cov_polygon_moc"}:
                                    lonv, latv = ellipse_boundary_vertices_lonlat_deg_from_cov(
                                        lon0_deg=float(lon[i]),
                                        lat0_deg=float(lat[i]),
                                        cov_ll_deg2=np.asarray(cov_ll[i], dtype=np.float64),
                                        n_sigma=float(n_sigma),
                                        num_vertices=int(polygon_vertices),
                                    )
                                    geom_rows.append(
                                        dict(
                                            strategy=str(name),
                                            variant_kind=None,
                                            footprint=str(footprint),
                                            orbit_id=str(k[0]),
                                            target_idx=int(k[1]),
                                            geometry_kind="polygon_vertices",
                                            lon0_deg=float(lon[i]),
                                            lat0_deg=float(lat[i]),
                                            cov_ll_00=float(cov_ll[i][0, 0]),
                                            cov_ll_01=float(cov_ll[i][0, 1]),
                                            cov_ll_10=float(cov_ll[i][1, 0]),
                                            cov_ll_11=float(cov_ll[i][1, 1]),
                                            n_sigma=float(n_sigma),
                                            polygon_vertices=int(polygon_vertices),
                                            polygon_mode=None,
                                            corridor_radius_arcsec=None,
                                            corridor_step_arcsec=None,
                                            buffer_arcsec=0.0,
                                            geom_id=str(geom_id),
                                        )
                                    )
                                    for j in range(len(lonv)):
                                        geom_points.append(
                                            dict(
                                                geom_id=str(geom_id),
                                                kind="polygon_vertex",
                                                idx=int(j),
                                                lon_deg=float(lonv[j]),
                                                lat_deg=float(latv[j]),
                                            )
                                        )

                    if not pix_list:
                        continue
                    pix_cat = np.concatenate(pix_list) if len(pix_list) > 1 else pix_list[0]
                    sum_pred_pixels += int(len(pix_cat))
                    rep_orbit = np.repeat(np.asarray(orbit_id_arr, dtype=object), lens)
                    rep_tidx = np.repeat(target_idx.astype(np.int64), lens)
                    pred = pa.table(
                        {
                            "orbit_id": pa.array(rep_orbit.tolist(), pa.large_string()),
                            "target_idx": pa.array(rep_tidx, pa.int64()),
                            "healpixel": pa.array(pix_cat.astype(np.int64), pa.int64()),
                        }
                    )

                selected = pred.join(frames_pixels, keys=["target_idx", "healpixel"], join_type="inner")
                sum_intersection += int(selected.num_rows)
                if truth_obs.num_rows > 0 and selected.num_rows > 0:
                    covered_total += int(
                        truth_obs.join(selected, keys=["orbit_id", "target_idx", "healpixel"], join_type="inner").num_rows
                    )
                if bool(compute_extra_frames) and selected.num_rows > 0:
                    selected_parts.append(selected.select(["orbit_id", "target_idx", "healpixel"]))

            dt = time.perf_counter() - t0
            metrics_rows.append(
                dict(
                    stage2_run_dir=str(stage2_run_dir),
                    subset_dir=str(subset_dir),
                    strategy=str(name),
                    variant_kind=None,
                    footprint=str(footprint),
                    healpix_nside=int(healpix_nside),
                    n_rows_ephem=int(n_rows_used),
                    n_groups=None,
                    n_rows_with_cov=(None if footprint == "point" else int(n_rows_with_cov)),
                    sum_pred_pixels=int(sum_pred_pixels),
                    sum_frame_pixels=0,
                    sum_intersection=int(sum_intersection),
                    runtime_sec=float(dt),
                    n_errors=(None if n_errors == 0 else int(n_errors)),
                    error=first_error,
                )
            )

            tt = int(truth_obs_tbl_strategy.num_rows)
            th = int(covered_total)
            n_selected = None
            n_extra = None
            if bool(compute_extra_frames) and selected_parts:
                sel_all = pa.concat_tables(selected_parts).combine_chunks()
                sel_unique = TruthObservations.from_pyarrow(sel_all).drop_duplicates(
                    subset=["orbit_id", "target_idx", "healpixel"]
                ).table
                n_selected = int(sel_unique.num_rows)
                n_in_truth = int(
                    sel_unique.join(truth_unique, keys=["orbit_id", "target_idx", "healpixel"], join_type="inner").num_rows
                )
                n_extra = int(n_selected - n_in_truth)
                keys_dir = run_dir / "selected_keys" / str(name) / str(footprint)
                _ensure_dir(keys_dir)
                TruthObservations.from_pyarrow(sel_unique).to_parquet(str(keys_dir / "selected_keys_unique.parquet"))

            if bool(compute_extra_frames) and bool(persist_geometry):
                write_geometry_artifacts(
                    run_dir=run_dir,
                    strategy=str(name),
                    variant_kind=None,
                    footprint=str(footprint),
                    geometry_rows=geom_rows,
                    points_rows=geom_points,
                )

            coverage_rows.append(
                dict(
                    stage2_run_dir=str(stage2_run_dir),
                    subset_dir=str(subset_dir),
                    strategy=str(name),
                    variant_kind=None,
                    footprint=str(footprint),
                    healpix_nside=int(healpix_nside),
                    n_truth=tt,
                    n_covered=th,
                    coverage=(0.0 if tt == 0 else float(th) / float(tt)),
                    n_selected=n_selected,
                    n_extra_frames=n_extra,
                )
            )

    def _run_variant_kind(
        *,
        variant_root_name: str,
        strat_dir: Path,
        variant_kind: str,
        part_files: list[Path],
    ) -> None:
        """
        Evaluate all variant-ephemeris footprints for one (strategy, variant_kind).

        Appends to outer-scope metrics_rows/coverage_rows and writes per-footprint artifacts
        into run_dir as needed.
        """
        orbit_ids_in_strategy: set[str] = set()
        # IMPORTANT: For some strategies (notably windowed propagation), the ephemeris parts
        # are not orbit-major, so part-000000 may contain only a subset of orbits. When
        # present, prefer `variants_orbits.parquet` as the authoritative orbit-id set.
        orbits_path = strat_dir / "variants_orbits.parquet"
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

        truth_obs_tbl_strategy = _filter_truth_to_orbit_ids(truth_obs_tbl, orbit_ids_in_strategy)
        truth_keys_tbl_strategy = _truth_keys_from_truth_observations(truth_obs_tbl_strategy)
        truth_unique_tbl_strategy: pa.Table | None = None
        if bool(compute_extra_frames) and truth_obs_tbl_strategy.num_rows > 0:
            truth_unique_tbl_strategy = TruthObservations.from_pyarrow(truth_obs_tbl_strategy).drop_duplicates(
                subset=["orbit_id", "target_idx", "healpixel"]
            ).table

        # Always map target_idx by (obscode,time) for variant outputs (parts can be irregular).
        dt_days = float(60.0) / 86400.0
        truth_obs = truth_obs_tbl_strategy
        truth_unique = (
            truth_unique_tbl_strategy
            if truth_unique_tbl_strategy is not None
            else pa.table(
                {
                    "orbit_id": pa.array([], pa.large_string()),
                    "target_idx": pa.array([], pa.int64()),
                    "healpixel": pa.array([], pa.int64()),
                }
            )
        )

        # NOTE: We intentionally avoid the non-MOC sample-perimeter and covariance-polygon
        # rasterizers (`sample_polygon:*` and `cov_polygon_reconstructed`) here.
        #
        # They were evaluated early on, but in real subsets they frequently fail convexity
        # prechecks (and historically could hard-crash healpy in C++). This was especially
        # apparent for sigma-point variant clouds (13-point stencils), where a “perimeter”
        # is not a meaningful uncertainty envelope.
        #
        # Keep the MOC variants + corridor/direct pixels + reconstructed covariance methods,
        # which are robust and sufficient for recovery-rate comparisons.
        variant_footprints = [
            ("sample_direct", None),
            ("sample_polygon_moc", "angle_sort"),
            ("sample_polygon_moc", "convex_hull"),
            ("sample_corridor", None),
            ("cov_disc_reconstructed", None),
            ("cov_polygon_reconstructed_moc", None),
            ("cov_mc_reconstructed", None),
        ]

        for footprint, polygon_mode in variant_footprints:
            out_fp = (
                f"{footprint}:{polygon_mode}"
                if (footprint in {"sample_polygon_moc"} and polygon_mode is not None)
                else str(footprint)
            )
            if selected_footprints is not None and str(out_fp) not in selected_footprints:
                continue
            t0 = time.perf_counter()
            n_rows_total = 0
            n_rows_used = 0
            n_groups = 0
            sum_pred_pixels = 0
            sum_intersection = 0
            covered_total = 0
            n_errors = 0
            first_error: str | None = None
            selected_parts: list[pa.Table] = []
            geom_rows: list[dict[str, object]] = []
            geom_points: list[dict[str, object]] = []
            seen_geom: set[tuple[str, int]] = set()

            for pf in part_files:
                ephem = VariantEphemeris.from_parquet(str(pf))
                n_rows_total += int(len(ephem))
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

                if bool(only_truth):
                    mask = _filter_ephem_to_truth(ephem=ephem, target_idx=target_idx, truth_keys=truth_keys_tbl_strategy)
                    hit = np.nonzero(mask)[0]
                    if hit.size == 0:
                        continue
                    ephem = ephem.take(hit.tolist())
                    target_idx = target_idx[hit]

                if len(ephem) == 0:
                    continue
                n_rows_used += int(len(ephem))

                # Fast-path for reconstructed-covariance footprints:
                # collapse the *entire* VariantEphemeris at once (vectorized in adam_core),
                # rather than collapsing 13-point groups in a Python loop.
                if ("_reconstructed" in str(footprint)) and hasattr(ephem, "collapse_by_object_id"):
                    try:
                        collapsed = ephem.collapse_by_object_id()
                    except BaseException as e:  # noqa: BLE001
                        if isinstance(e, (KeyboardInterrupt, SystemExit)):
                            raise
                        n_errors += 1
                        if first_error is None:
                            first_error = f"{type(e).__name__}: {e}"
                        continue

                    if len(collapsed) == 0:
                        continue
                    target_idx_c = _map_ephem_to_target_idx_by_time(
                        ephem=collapsed, targets=targets_tbl, dt_days=dt_days
                    )
                    keep_c = target_idx_c >= 0
                    if not keep_c.any():
                        continue
                    if not keep_c.all():
                        hit = np.nonzero(keep_c)[0]
                        collapsed = collapsed.take(hit.tolist())
                        target_idx_c = target_idx_c[hit]

                    if bool(only_truth):
                        mask = _filter_ephem_to_truth(
                            ephem=collapsed, target_idx=target_idx_c, truth_keys=truth_keys_tbl_strategy
                        )
                        hit = np.nonzero(mask)[0]
                        if hit.size == 0:
                            continue
                        collapsed = collapsed.take(hit.tolist())
                        target_idx_c = target_idx_c[hit]

                    if len(collapsed) == 0:
                        continue
                    n_groups += int(len(collapsed))

                    orbit_id_c = np.asarray(_ephem_key_table(collapsed)["orbit_id"].to_pylist(), dtype=object)
                    lon0 = collapsed.coordinates.lon.to_numpy(zero_copy_only=False).astype(np.float64)
                    lat0 = collapsed.coordinates.lat.to_numpy(zero_copy_only=False).astype(np.float64)

                    cov6 = collapsed.coordinates.covariance.to_matrix().astype(np.float64)
                    cov_ll = cov6[:, 1:3, 1:3]

                    if str(footprint).endswith("_reconstructed_moc"):
                        base = str(footprint).replace("_reconstructed_moc", "_moc")
                    else:
                        base = str(footprint).replace("_reconstructed", "")

                    pred_orbit: list[str] = []
                    pred_tidx: list[int] = []
                    pred_hpix: list[int] = []
                    for i in range(int(len(collapsed))):
                        oid = str(orbit_id_c[i])
                        tidx = int(target_idx_c[i])
                        cov_ll_i = np.asarray(cov_ll[i], dtype=np.float64)

                        if bool(persist_geometry):
                            k = (str(oid), int(tidx))
                            if k not in seen_geom:
                                seen_geom.add(k)
                                geom_id = f"{k[0]}|{k[1]}"
                                if str(footprint) in {"cov_polygon_reconstructed_moc"}:
                                    lonv, latv = ellipse_boundary_vertices_lonlat_deg_from_cov(
                                        lon0_deg=float(lon0[i]),
                                        lat0_deg=float(lat0[i]),
                                        cov_ll_deg2=cov_ll_i,
                                        n_sigma=float(n_sigma),
                                        num_vertices=int(polygon_vertices),
                                    )
                                    geom_rows.append(
                                        dict(
                                            strategy=str(variant_root_name),
                                            variant_kind=str(variant_kind),
                                            footprint=str(out_fp),
                                            orbit_id=str(k[0]),
                                            target_idx=int(k[1]),
                                            geometry_kind="polygon_vertices",
                                            lon0_deg=float(lon0[i]),
                                            lat0_deg=float(lat0[i]),
                                            cov_ll_00=float(cov_ll_i[0, 0]),
                                            cov_ll_01=float(cov_ll_i[0, 1]),
                                            cov_ll_10=float(cov_ll_i[1, 0]),
                                            cov_ll_11=float(cov_ll_i[1, 1]),
                                            n_sigma=float(n_sigma),
                                            polygon_vertices=int(polygon_vertices),
                                            polygon_mode=None,
                                            corridor_radius_arcsec=None,
                                            corridor_step_arcsec=None,
                                            buffer_arcsec=0.0,
                                            geom_id=str(geom_id),
                                        )
                                    )
                                    for j in range(len(lonv)):
                                        geom_points.append(
                                            dict(
                                                geom_id=str(geom_id),
                                                kind="polygon_vertex",
                                                idx=int(j),
                                                lon_deg=float(lonv[j]),
                                                lat_deg=float(latv[j]),
                                            )
                                        )
                                else:
                                    geom_rows.append(
                                        dict(
                                            strategy=str(variant_root_name),
                                            variant_kind=str(variant_kind),
                                            footprint=str(out_fp),
                                            orbit_id=str(k[0]),
                                            target_idx=int(k[1]),
                                            geometry_kind="ellipse_cov",
                                            lon0_deg=float(lon0[i]),
                                            lat0_deg=float(lat0[i]),
                                            cov_ll_00=float(cov_ll_i[0, 0]),
                                            cov_ll_01=float(cov_ll_i[0, 1]),
                                            cov_ll_10=float(cov_ll_i[1, 0]),
                                            cov_ll_11=float(cov_ll_i[1, 1]),
                                            n_sigma=float(n_sigma),
                                            polygon_vertices=None,
                                            polygon_mode=None,
                                            corridor_radius_arcsec=None,
                                            corridor_step_arcsec=None,
                                            buffer_arcsec=None,
                                            geom_id=None,
                                        )
                                    )

                        try:
                            pix = _predicted_pixels_from_mean_row(
                                lon_deg=float(lon0[i]),
                                lat_deg=float(lat0[i]),
                                cov_ll_deg2=cov_ll_i,
                                nside=int(healpix_nside),
                                footprint=str(base),
                                n_sigma=float(n_sigma),
                                polygon_vertices=int(polygon_vertices),
                                mc_num_samples=int(cov_mc_num_samples),
                                mc_seed=int(cov_mc_seed),
                            )
                        except BaseException as e:  # noqa: BLE001
                            if isinstance(e, (KeyboardInterrupt, SystemExit)):
                                raise
                            n_errors += 1
                            if first_error is None:
                                first_error = f"{type(e).__name__}: {e}"
                            pix = np.array([], dtype=np.int64)

                        if pix.size == 0:
                            continue
                        sum_pred_pixels += int(pix.size)
                        pred_orbit.extend([oid] * int(pix.size))
                        pred_tidx.extend([tidx] * int(pix.size))
                        pred_hpix.extend([int(x) for x in pix.tolist()])

                    if not pred_orbit:
                        continue
                    pred = pa.table(
                        {
                            "orbit_id": pa.array(pred_orbit, pa.large_string()),
                            "target_idx": pa.array(np.asarray(pred_tidx, dtype=np.int64), pa.int64()),
                            "healpixel": pa.array(np.asarray(pred_hpix, dtype=np.int64), pa.int64()),
                        }
                    )
                    selected = pred.join(frames_pixels, keys=["target_idx", "healpixel"], join_type="inner")
                    sum_intersection += int(selected.num_rows)
                    if truth_obs.num_rows > 0 and selected.num_rows > 0:
                        covered_total += int(
                            truth_obs.join(selected, keys=["orbit_id", "target_idx", "healpixel"], join_type="inner").num_rows
                        )
                    if bool(compute_extra_frames) and selected.num_rows > 0:
                        selected_parts.append(selected.select(["orbit_id", "target_idx", "healpixel"]))
                    continue

                orbit_id = np.asarray(_ephem_key_table(ephem)["orbit_id"].to_pylist(), dtype=object)
                lon = ephem.coordinates.lon.to_numpy(zero_copy_only=False).astype(np.float64)
                lat = ephem.coordinates.lat.to_numpy(zero_copy_only=False).astype(np.float64)
                order, starts = _group_slices_by_orbit_target(orbit_id, target_idx.astype(np.int64))
                if starts.size == 0:
                    continue
                ends = np.concatenate([starts[1:], np.array([len(order)], dtype=np.int64)])

                pred_orbit: list[str] = []
                pred_tidx: list[int] = []
                pred_hpix: list[int] = []
                for s, e in zip(starts.tolist(), ends.tolist()):
                    idx = order[s:e]
                    if idx.size == 0:
                        continue
                    n_groups += 1
                    oid = str(orbit_id[idx[0]])
                    tidx = int(target_idx[idx[0]])
                    lon_g = lon[idx]
                    lat_g = lat[idx]

                    if "_reconstructed" in footprint:
                        vsub = ephem.take(idx.tolist())
                        try:
                            collapsed = _collapse_variant_ephemeris_group(variants=vsub)
                            lon0 = float(collapsed.coordinates.lon[0].as_py())
                            lat0 = float(collapsed.coordinates.lat[0].as_py())
                            cov6 = collapsed.coordinates.covariance.to_matrix()[0].astype(np.float64)
                            cov_ll = cov6[1:3, 1:3]
                            if bool(persist_geometry):
                                k = (str(oid), int(tidx))
                                if k not in seen_geom:
                                    seen_geom.add(k)
                                    geom_id = f"{k[0]}|{k[1]}"
                                    if str(footprint) in {"cov_polygon_reconstructed_moc"}:
                                        lonv, latv = ellipse_boundary_vertices_lonlat_deg_from_cov(
                                            lon0_deg=float(lon0),
                                            lat0_deg=float(lat0),
                                            cov_ll_deg2=np.asarray(cov_ll, dtype=np.float64),
                                            n_sigma=float(n_sigma),
                                            num_vertices=int(polygon_vertices),
                                        )
                                        geom_rows.append(
                                            dict(
                                                strategy=str(variant_root_name),
                                                variant_kind=str(variant_kind),
                                                footprint=str(out_fp),
                                                orbit_id=str(k[0]),
                                                target_idx=int(k[1]),
                                                geometry_kind="polygon_vertices",
                                                lon0_deg=float(lon0),
                                                lat0_deg=float(lat0),
                                                cov_ll_00=float(cov_ll[0, 0]),
                                                cov_ll_01=float(cov_ll[0, 1]),
                                                cov_ll_10=float(cov_ll[1, 0]),
                                                cov_ll_11=float(cov_ll[1, 1]),
                                                n_sigma=float(n_sigma),
                                                polygon_vertices=int(polygon_vertices),
                                                polygon_mode=None,
                                                corridor_radius_arcsec=None,
                                                corridor_step_arcsec=None,
                                                buffer_arcsec=0.0,
                                                geom_id=str(geom_id),
                                            )
                                        )
                                        for j in range(len(lonv)):
                                            geom_points.append(
                                                dict(
                                                    geom_id=str(geom_id),
                                                    kind="polygon_vertex",
                                                    idx=int(j),
                                                    lon_deg=float(lonv[j]),
                                                    lat_deg=float(latv[j]),
                                                )
                                            )
                                    else:
                                        geom_rows.append(
                                            dict(
                                                strategy=str(variant_root_name),
                                                variant_kind=str(variant_kind),
                                                footprint=str(out_fp),
                                                orbit_id=str(k[0]),
                                                target_idx=int(k[1]),
                                                geometry_kind="ellipse_cov",
                                                lon0_deg=float(lon0),
                                                lat0_deg=float(lat0),
                                                cov_ll_00=float(cov_ll[0, 0]),
                                                cov_ll_01=float(cov_ll[0, 1]),
                                                cov_ll_10=float(cov_ll[1, 0]),
                                                cov_ll_11=float(cov_ll[1, 1]),
                                                n_sigma=float(n_sigma),
                                                polygon_vertices=None,
                                                polygon_mode=None,
                                                corridor_radius_arcsec=None,
                                                corridor_step_arcsec=None,
                                                buffer_arcsec=None,
                                                geom_id=None,
                                            )
                                        )
                            if footprint.endswith("_reconstructed_moc"):
                                base = footprint.replace("_reconstructed_moc", "_moc")
                            else:
                                base = footprint.replace("_reconstructed", "")
                            pix = _predicted_pixels_from_mean_row(
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
                        except BaseException as e:  # noqa: BLE001
                            if isinstance(e, (KeyboardInterrupt, SystemExit)):
                                raise
                            n_errors += 1
                            if first_error is None:
                                first_error = f"{type(e).__name__}: {e}"
                            pix = np.array([], dtype=np.int64)
                    else:
                        try:
                            if bool(persist_geometry):
                                k = (str(oid), int(tidx))
                                if k not in seen_geom:
                                    seen_geom.add(k)
                                    geom_id = f"{k[0]}|{k[1]}"
                                    if str(footprint) in {"sample_polygon_moc"}:
                                        poly = perimeter_polygon_from_samples(
                                            lon0_deg=float(lon_g[0]),
                                            lat0_deg=float(lat_g[0]),
                                            lon_deg=lon_g.astype(np.float64, copy=False),
                                            lat_deg=lat_g.astype(np.float64, copy=False),
                                            mode=("convex_hull" if polygon_mode == "convex_hull" else "angle_sort"),
                                        )
                                        geom_rows.append(
                                            dict(
                                                strategy=str(variant_root_name),
                                                variant_kind=str(variant_kind),
                                                footprint=str(out_fp),
                                                orbit_id=str(k[0]),
                                                target_idx=int(k[1]),
                                                geometry_kind="polygon_vertices",
                                                lon0_deg=float(lon_g[0]),
                                                lat0_deg=float(lat_g[0]),
                                                cov_ll_00=None,
                                                cov_ll_01=None,
                                                cov_ll_10=None,
                                                cov_ll_11=None,
                                                n_sigma=None,
                                                polygon_vertices=int(len(poly)),
                                                polygon_mode=str(polygon_mode),
                                                corridor_radius_arcsec=None,
                                                corridor_step_arcsec=None,
                                                buffer_arcsec=0.0,
                                                geom_id=str(geom_id),
                                            )
                                        )
                                        for j in range(len(poly)):
                                            geom_points.append(
                                                dict(
                                                    geom_id=str(geom_id),
                                                    kind="polygon_vertex",
                                                    idx=int(j),
                                                    lon_deg=float(poly[j, 0]),
                                                    lat_deg=float(poly[j, 1]),
                                                )
                                            )
                                    elif str(footprint) == "sample_corridor":
                                        lonp, latp = corridor_path_lonlat_deg_from_samples(
                                            lon0_deg=float(lon_g[0]),
                                            lat0_deg=float(lat_g[0]),
                                            lon_deg=lon_g.astype(np.float64, copy=False),
                                            lat_deg=lat_g.astype(np.float64, copy=False),
                                        )
                                        geom_rows.append(
                                            dict(
                                                strategy=str(variant_root_name),
                                                variant_kind=str(variant_kind),
                                                footprint=str(out_fp),
                                                orbit_id=str(k[0]),
                                                target_idx=int(k[1]),
                                                geometry_kind="corridor_polyline",
                                                lon0_deg=float(lon_g[0]),
                                                lat0_deg=float(lat_g[0]),
                                                cov_ll_00=None,
                                                cov_ll_01=None,
                                                cov_ll_10=None,
                                                cov_ll_11=None,
                                                n_sigma=None,
                                                polygon_vertices=None,
                                                polygon_mode=None,
                                                corridor_radius_arcsec=float(corridor_radius_arcsec),
                                                corridor_step_arcsec=float(corridor_step_arcsec),
                                                buffer_arcsec=None,
                                                geom_id=str(geom_id),
                                            )
                                        )
                                        for j in range(len(lonp)):
                                            geom_points.append(
                                                dict(
                                                    geom_id=str(geom_id),
                                                    kind="corridor_path",
                                                    idx=int(j),
                                                    lon_deg=float(lonp[j]),
                                                    lat_deg=float(latp[j]),
                                                )
                                            )
                            pix = _predicted_pixels_from_samples(
                                lon_deg=lon_g,
                                lat_deg=lat_g,
                                nside=int(healpix_nside),
                                footprint=str(footprint),
                                polygon_mode=("convex_hull" if polygon_mode == "convex_hull" else "angle_sort"),
                                corridor_radius_arcsec=float(corridor_radius_arcsec),
                                corridor_step_arcsec=float(corridor_step_arcsec),
                            )
                        except BaseException as e:  # noqa: BLE001
                            if isinstance(e, (KeyboardInterrupt, SystemExit)):
                                raise
                            n_errors += 1
                            if first_error is None:
                                first_error = f"{type(e).__name__}: {e}"
                            pix = np.array([], dtype=np.int64)

                    if pix.size == 0:
                        continue
                    sum_pred_pixels += int(pix.size)
                    pred_orbit.extend([oid] * int(pix.size))
                    pred_tidx.extend([tidx] * int(pix.size))
                    pred_hpix.extend([int(x) for x in pix.tolist()])

                if not pred_orbit:
                    continue
                pred = pa.table(
                    {
                        "orbit_id": pa.array(pred_orbit, pa.large_string()),
                        "target_idx": pa.array(np.asarray(pred_tidx, dtype=np.int64), pa.int64()),
                        "healpixel": pa.array(np.asarray(pred_hpix, dtype=np.int64), pa.int64()),
                    }
                )
                selected = pred.join(frames_pixels, keys=["target_idx", "healpixel"], join_type="inner")
                sum_intersection += int(selected.num_rows)
                if truth_obs.num_rows > 0 and selected.num_rows > 0:
                    covered_total += int(
                        truth_obs.join(selected, keys=["orbit_id", "target_idx", "healpixel"], join_type="inner").num_rows
                    )
                if bool(compute_extra_frames) and selected.num_rows > 0:
                    selected_parts.append(selected.select(["orbit_id", "target_idx", "healpixel"]))

            dt = time.perf_counter() - t0
            metrics_rows.append(
                dict(
                    stage2_run_dir=str(stage2_run_dir),
                    subset_dir=str(subset_dir),
                    strategy=variant_root_name,
                    variant_kind=str(variant_kind),
                    footprint=str(out_fp),
                    healpix_nside=int(healpix_nside),
                    n_rows_ephem=int(n_rows_used),
                    n_groups=int(n_groups),
                    n_rows_with_cov=0,
                    sum_pred_pixels=int(sum_pred_pixels),
                    sum_frame_pixels=0,
                    sum_intersection=int(sum_intersection),
                    runtime_sec=float(dt),
                    n_errors=(None if n_errors == 0 else int(n_errors)),
                    error=first_error,
                )
            )
            tt = int(truth_obs_tbl_strategy.num_rows)
            th = int(covered_total)
            n_selected = None
            n_extra = None
            if bool(compute_extra_frames) and selected_parts:
                sel_all = pa.concat_tables(selected_parts).combine_chunks()
                sel_unique = TruthObservations.from_pyarrow(sel_all).drop_duplicates(
                    subset=["orbit_id", "target_idx", "healpixel"]
                ).table
                n_selected = int(sel_unique.num_rows)
                n_in_truth = int(
                    sel_unique.join(truth_unique, keys=["orbit_id", "target_idx", "healpixel"], join_type="inner").num_rows
                )
                n_extra = int(n_selected - n_in_truth)
                keys_dir = run_dir / "selected_keys" / f"{variant_root_name}:{variant_kind}" / out_fp
                _ensure_dir(keys_dir)
                TruthObservations.from_pyarrow(sel_unique).to_parquet(str(keys_dir / "selected_keys_unique.parquet"))
            if bool(compute_extra_frames) and bool(persist_geometry):
                write_geometry_artifacts(
                    run_dir=run_dir,
                    strategy=str(variant_root_name),
                    variant_kind=str(variant_kind),
                    footprint=str(out_fp),
                    geometry_rows=geom_rows,
                    points_rows=geom_points,
                )
            coverage_rows.append(
                dict(
                    stage2_run_dir=str(stage2_run_dir),
                    subset_dir=str(subset_dir),
                    strategy=variant_root_name,
                    variant_kind=str(variant_kind),
                    footprint=str(out_fp),
                    healpix_nside=int(healpix_nside),
                    n_truth=tt,
                    n_covered=th,
                    coverage=(0.0 if tt == 0 else float(th) / float(tt)),
                    n_selected=n_selected,
                    n_extra_frames=n_extra,
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
        _run_mean_strategy(strat_dir=strat_dir, name=name)

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

    metrics = Stage3Metrics.from_pyarrow(pa.Table.from_pylist(metrics_rows))
    coverage = Stage3Coverage.from_pyarrow(pa.Table.from_pylist(coverage_rows))
    metrics.to_parquet(str(run_dir / "metrics.parquet"))
    coverage.to_parquet(str(run_dir / "coverage.parquet"))

    out_meta = dict(
        subset_dir=str(subset_dir),
        stage2_run_dir=str(stage2_run_dir),
        inputs_artifacts_dir=str(
            (
                Path(inputs_artifacts_dir).expanduser().resolve()
                if inputs_artifacts_dir is not None
                else (Path(subset_dir) / "artifacts")
            )
        ),
        healpix_nside=int(healpix_nside),
        n_sigma=float(n_sigma),
        polygon_vertices=int(polygon_vertices),
        cov_mc_num_samples=int(cov_mc_num_samples),
        cov_mc_seed=int(cov_mc_seed),
        corridor_radius_arcsec=float(corridor_radius_arcsec),
        corridor_step_arcsec=float(corridor_step_arcsec),
        only_truth=bool(only_truth),
        compute_extra_frames=bool(compute_extra_frames),
        footprints=(None if selected_footprints is None else sorted(selected_footprints)),
        n_truth_keys=int(len(truth_keys_tbl)),
        generated_at_utc=_now_utc(),
    )
    _write_json(run_dir / "meta.json", out_meta)
    return run_dir


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Stage 3: atomic healpixel intersection runner (consumes Stage 2 ephemerides).")
    p.add_argument("--subset-dir", type=str, required=True)
    p.add_argument("--stage2-run-dir", type=str, required=True)
    p.add_argument(
        "--inputs-artifacts-dir",
        type=str,
        default=None,
        help="Directory containing truth_precovery_crossmatch.parquet and orbits_selected_sbdb.parquet (default: <subset_dir>/artifacts).",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help=(
            "Output root directory (default: <subset_dir>/artifacts/stage3). "
            "Run directory will be <out_dir>/<stage2_run_dir.name>."
        ),
    )
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
        help="Comma-separated list of Stage2 strategy folder names to include (e.g. '2body_with_covariance,assist_mean').",
    )
    p.add_argument(
        "--footprints",
        type=str,
        default=None,
        help=(
            "Comma-separated list of footprint output names to include "
            "(e.g. 'cov_polygon_moc,cov_polygon_reconstructed_moc'). If omitted, runs all."
        ),
    )
    p.add_argument(
        "--only-truth",
        action="store_true",
        help="Only evaluate keys present in truth crossmatch (much faster for full runs).",
    )
    p.add_argument(
        "--compute-extra-frames",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Compute extra-frames metrics by materializing and de-duplicating selected frame keys "
            "(default: enabled)."
        ),
    )
    p.add_argument(
        "--persist-geometry",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Persist footprint geometry artifacts for Stage 4 reuse (default: enabled).",
    )
    args = p.parse_args()

    strategies = None if args.strategies is None else [s.strip() for s in str(args.strategies).split(",") if s.strip()]
    footprints = None if args.footprints is None else [s.strip() for s in str(args.footprints).split(",") if s.strip()]
    run_dir = run_stage3_healpixel_bench(
        subset_dir=Path(args.subset_dir),
        stage2_run_dir=Path(args.stage2_run_dir),
        out_dir=None if args.out_dir is None else Path(args.out_dir),
        healpix_nside=int(args.healpix_nside),
        inputs_artifacts_dir=None if args.inputs_artifacts_dir is None else Path(args.inputs_artifacts_dir),
        n_sigma=float(args.n_sigma),
        polygon_vertices=int(args.polygon_vertices),
        cov_mc_num_samples=int(args.cov_mc_num_samples),
        cov_mc_seed=int(args.cov_mc_seed),
        corridor_radius_arcsec=float(args.corridor_radius_arcsec),
        corridor_step_arcsec=float(args.corridor_step_arcsec),
        strategies=strategies,
        footprints=footprints,
        only_truth=bool(args.only_truth),
        compute_extra_frames=bool(args.compute_extra_frames),
        persist_geometry=bool(args.persist_geometry),
    )
    print(f"run_dir={run_dir}")
    print(f"metrics_parquet={run_dir / 'metrics.parquet'}")
    print(f"coverage_parquet={run_dir / 'coverage.parquet'}")


if __name__ == "__main__":
    main()

