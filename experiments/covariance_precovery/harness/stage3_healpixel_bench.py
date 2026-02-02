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
    SamplePerimeterPolygonFootprint,
    corridor_path_lonlat_deg_from_samples,
    corridor_pixels_from_samples,
    disc_pixels_from_cov,
    ellipse_boundary_vertices_lonlat_deg_from_cov,
    ellipse_polygon_pixels_from_cov,
    ellipse_polygon_pixels_from_cov_moc,
    mc_pixels_from_cov,
    perimeter_polygon_from_samples,
    sample_pixels_direct,
    sample_perimeter_polygon_pixels_moc,
)


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
    n_groups = qv.Int64Column()
    n_rows_with_cov = qv.Int64Column()

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
    s = str(object_id).strip()
    if s.startswith("(") and s.endswith(")") and len(s) >= 3:
        # Provisional designation stored as "(2019 NY21)".
        return s[1:-1].strip()
    # Otherwise first token is the numeric designation (e.g. "191305 (2003 HQ20)").
    return s.split()[0].strip()


def _read_stage2_targets(stage2_run_dir: Path) -> pa.Table:
    """
    Table with columns:
      - target_idx (int64): 0..n_targets-1
      - obscode (large_string)
      - exposure_mjd_mid (float64): UTC MJD midpoint as stored by Stage 2 inputs
    """
    targets = FrameTimeTargets.from_parquet(
        str(stage2_run_dir / "inputs" / "frame_time_targets.parquet")
    )
    mjd = targets.time.mjd().to_numpy(zero_copy_only=False).astype(np.float64)
    idx = np.arange(len(targets), dtype=np.int64)
    return pa.Table.from_arrays(
        [
            pa.array(idx, type=pa.int64()),
            pa.array(targets.obscode.to_pylist(), type=pa.large_string()),
            pa.array(mjd, type=pa.float64()),
        ],
        names=["target_idx", "obscode", "exposure_mjd_mid"],
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
    Return distinct (target_idx, healpixel) for the subset window.

    This is the vectorized join key-space Stage 3 uses (no python dict lookups).
    """
    index_db = subset_dir / "index.db"
    if not index_db.exists():
        raise FileNotFoundError(f"Missing subset index.db: {index_db}")

    # Query only obscodes and MJD range present in targets (the index is already trimmed).
    codes = sorted({str(x) for x in pc.unique(targets["obscode"]).to_pylist()})
    if not codes:
        return FramesPixels.empty().table
    q_marks = ",".join(["?"] * len(codes))
    mjd = pc.cast(targets["exposure_mjd_mid"], pa.float64()).to_numpy(zero_copy_only=False)
    q0 = float(np.min(mjd)) - 1e-9
    q1 = float(np.max(mjd)) + 1e-9

    import sqlite3

    conn = sqlite3.connect(str(index_db))
    try:
        rows = conn.execute(
            f"""
            SELECT obscode, exposure_mjd_mid, healpixel
            FROM frames
            WHERE obscode IN ({q_marks})
              AND exposure_mjd_mid >= ?
              AND exposure_mjd_mid <= ?
            """,
            (*codes, float(q0), float(q1)),
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        return FramesPixels.empty().table

    obsc = pa.array([r[0] for r in rows], type=pa.large_string())
    mjd_mid = pa.array([float(r[1]) for r in rows], type=pa.float64())
    hpix = pa.array([int(r[2]) for r in rows], type=pa.int64())
    frames = pa.table({"obscode": obsc, "exposure_mjd_mid": mjd_mid, "healpixel": hpix})

    # Join frames -> targets to map exposure_mjd_mid to target_idx.
    joined = frames.join(
        targets.select(["obscode", "exposure_mjd_mid", "target_idx"]),
        keys=["obscode", "exposure_mjd_mid"],
        join_type="inner",
    )
    out = joined.select(["target_idx", "healpixel"]).combine_chunks()
    qt = FramesPixels.from_pyarrow(out).drop_duplicates(subset=["target_idx", "healpixel"])
    return qt.table


def _map_times_to_target_idx_by_obscode(
    *,
    obscode: np.ndarray,
    time_mjd: np.ndarray,
    targets: pa.Table,
    dt_days: float,
) -> np.ndarray:
    """
    Map each (obscode, time_mjd) row to the nearest Stage 2 target_idx for that obscode
    within tolerance.
    """
    targ_obscode = np.asarray(targets["obscode"].to_pylist(), dtype=object)
    targ_mjd = np.asarray(targets["exposure_mjd_mid"].to_numpy(zero_copy_only=False), dtype=np.float64)
    targ_idx = np.asarray(targets["target_idx"].to_numpy(zero_copy_only=False), dtype=np.int64)

    out = np.full(len(time_mjd), -1, dtype=np.int64)
    for code in sorted(set(obscode.tolist())):
        mt = obscode == code
        if not mt.any():
            continue
        mg = targ_obscode == code
        if not mg.any():
            continue
        mjd = targ_mjd[mg]
        idx = targ_idx[mg]
        order = np.argsort(mjd)
        mjd_s = mjd[order]
        idx_s = idx[order]
        times = time_mjd[mt]
        j = np.searchsorted(mjd_s, times)
        j0 = np.clip(j - 1, 0, len(mjd_s) - 1)
        j1 = np.clip(j, 0, len(mjd_s) - 1)
        d0 = np.abs(mjd_s[j0] - times)
        d1 = np.abs(mjd_s[j1] - times)
        use1 = d1 < d0
        jj = np.where(use1, j1, j0)
        dd = np.where(use1, d1, d0)
        ok = dd <= float(dt_days)
        out[np.nonzero(mt)[0][ok]] = idx_s[jj[ok]]
    return out


def _read_truth_observations_table(*, subset_dir: Path, targets: pa.Table) -> pa.Table:
    """
    Return matched truth observations aligned to Stage 2 targets as:
      (orbit_id, target_idx, healpixel)

    - orbit_id is the stable designation (SBDB-derived; see fetch step)
    - target_idx is Stage 2's exposure-time index
    - healpixel is the healpixel of the truth observation (nside = subset frames nside)
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

    truth = truth.select(["designation", "obscode", "match_time_mjd_utc", "healpixel"])
    truth = truth.join(orbit_map, keys=["designation"], join_type="inner")
    if truth.num_rows == 0:
        return TruthObservations.empty().table

    # Map match_time_mjd_utc -> nearest target_idx per obscode (within 60s).
    dt_days = float(60.0) / 86400.0

    t_obscode = np.asarray(truth["obscode"].to_pylist(), dtype=object)
    t_time = np.asarray(truth["match_time_mjd_utc"].to_numpy(zero_copy_only=False), dtype=np.float64)
    t_orbit = np.asarray(truth["orbit_id"].to_pylist(), dtype=object)
    t_hpix = np.asarray(truth["healpixel"].to_numpy(zero_copy_only=False), dtype=np.int64)

    target_idx = _map_times_to_target_idx_by_obscode(
        obscode=t_obscode, time_mjd=t_time, targets=targets, dt_days=dt_days
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
    orbit = [str(x) for x in ephem.orbit_id.to_pylist()]
    obj_col = getattr(ephem, "object_id", None)
    obj = obj_col.to_pylist() if obj_col is not None else None
    if obj is None:
        keys = orbit
    else:
        # Be robust to Ephemeris tables that have an `object_id` column but contain nulls:
        # fall back to orbit_id on a per-row basis.
        keys: list[str] = []
        for i, x in enumerate(obj):
            s = "" if x is None else str(x).strip()
            if s:
                keys.append(_designation_from_object_id(s))
            else:
                keys.append(orbit[i] if i < len(orbit) else "")
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
    # Normalize ephem time to UTC MJD for matching to target exposure midpoints.
    t_mjd = ephem.coordinates.time.rescale("utc").mjd().to_numpy(zero_copy_only=False).astype(np.float64)
    code = np.asarray(ephem.coordinates.origin.code.to_pylist(), dtype=object)
    return _map_times_to_target_idx_by_obscode(
        obscode=code, time_mjd=t_mjd, targets=targets, dt_days=float(dt_days)
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
    if footprint == "cov_polygon":
        return ellipse_polygon_pixels_from_cov(lon0_deg=float(lon_deg), lat0_deg=float(lat_deg), cov_ll_deg2=cov_ll_deg2, nside=int(nside), n_sigma=float(n_sigma), num_vertices=int(polygon_vertices))
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
    if footprint == "sample_polygon":
        lon0 = float(lon_deg[0])
        lat0 = float(lat_deg[0])
        fp = SamplePerimeterPolygonFootprint(
            lon0_deg=lon0,
            lat0_deg=lat0,
            sample_lon_deg=lon_deg.astype(np.float64),
            sample_lat_deg=lat_deg.astype(np.float64),
            polygon_mode=("angle_sort" if str(polygon_mode) == "angle_sort" else "convex_hull"),
            buffer_arcsec=0.0,
        )
        return fp.pixels(int(nside))
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
    strategies: list[str] | None = None,
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
    truth_obs_tbl = _read_truth_observations_table(subset_dir=subset_dir, targets=targets_tbl)
    truth_keys_tbl = _truth_keys_from_truth_observations(truth_obs_tbl)
    # NOTE: truth_unique tables are computed per-strategy after filtering to the
    # orbits present in that strategy's Stage 2 outputs.

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

    # Mean strategies (point/covariance-derived footprints).
    for strat_dir in sorted(strategies_root.glob("*")):
        if not strat_dir.is_dir():
            continue
        name = strat_dir.name
        if name == "assist_variants":
            continue
        if name == "assist_window_then_2body_variants":
            continue
        if not _enabled(name):
            continue
        mean_dir = strat_dir / "mean_ephemeris"
        if not mean_dir.exists():
            continue
        part_files = sorted(mean_dir.glob("part-*.parquet"))
        if not part_files:
            continue

        # Filter truth to the orbits actually present in this Stage 2 strategy output.
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

        # Determine whether covariance is present for this strategy.
        try:
            ep0 = _ephem_to_utc(Ephemeris.from_parquet(str(part_files[0])))
            has_cov = (
                ep0.coordinates.covariance is not None
                and (not ep0.coordinates.covariance.is_all_nan())
            )
        except Exception:  # noqa: BLE001
            has_cov = False

        active_footprints = ["point"] + ([] if not has_cov else ["cov_disc", "cov_polygon", "cov_mc"])
        # Optional robust polygon rasterization via MOC.
        if has_cov:
            active_footprints.append("cov_polygon_moc")
        if not active_footprints:
            continue

        # Evaluate footprints one technique at a time (atomic).
        for footprint in active_footprints:
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
            meta = json.loads((strat_dir / "meta.json").read_text())
            n_orbits = int(meta["n_orbits"])
            chunk = int(meta.get("time_chunk_size", 1024))
            n_targets = int(meta.get("n_time_targets", len(targets_tbl)))
            # Strategy-specific mapping:
            # - 2body_* are orbit-major cross products over fixed time chunks
            # - assist_window_then_2body writes variable-sized parts; we map by (obscode,time)->target_idx
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
            for pf in part_files:
                ephem = Ephemeris.from_parquet(str(pf))
                n_rows_total += int(len(ephem))
                if needs_time_map:
                    target_idx = _map_ephem_to_target_idx_by_time(
                        ephem=ephem, targets=targets_tbl, dt_days=dt_days
                    )
                    keep = target_idx >= 0
                    if not keep.any():
                        continue
                    # If we couldn't map some rows, drop them (cannot join to frames).
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
                    target_idx = np.tile(
                        (np.arange(chunk_len, dtype=np.int64) + start), int(n_orbits)
                    )
                    # Safety: some parts may be truncated; align to ephem length.
                    target_idx = target_idx[: int(len(ephem))]
                if bool(only_truth):
                    mask = _filter_ephem_to_truth(
                        ephem=ephem, target_idx=target_idx, truth_keys=truth_keys_tbl_strategy
                    )
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
                    pix = hp.ang2pix(int(healpix_nside), lon, lat, lonlat=True, nest=True).astype(
                        np.int64
                    )
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
                    cov_ll = cov6[:, 1:3, 1:3].astype(np.float64)  # (N,2,2) lon/lat in deg^2
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
                                elif fp in {"cov_polygon", "cov_polygon_moc"}:
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
                    if int(np.sum(lens)) == 0:
                        continue
                    sum_pred_pixels += int(np.sum(lens))
                    row_idx_rep = np.repeat(np.arange(int(len(ephem)), dtype=np.int64), lens)
                    hpix = np.concatenate(pix_list).astype(np.int64, copy=False)
                    pred = pa.table(
                        {
                            "orbit_id": pa.array(np.asarray(orbit_id_arr, dtype=object)[row_idx_rep], pa.large_string()),
                            "target_idx": pa.array(target_idx.astype(np.int64)[row_idx_rep], pa.int64()),
                            "healpixel": pa.array(hpix, pa.int64()),
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
                    strategy=name,
                    variant_kind=None,
                    footprint=str(footprint),
                    healpix_nside=int(healpix_nside),
                    n_rows_ephem=int(n_rows_used),
                    n_groups=int(n_rows_used),
                    n_rows_with_cov=int(n_rows_with_cov),
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
                # extra frames = selected_unique \ truth_unique
                n_in_truth = int(
                    sel_unique.join(
                        truth_unique, keys=["orbit_id", "target_idx", "healpixel"], join_type="inner"
                    ).num_rows
                )
                n_extra = int(n_selected - n_in_truth)
                # Optional persisted artifact for fast downstream inspection.
                keys_dir = run_dir / "selected_keys" / name / footprint
                _ensure_dir(keys_dir)
                TruthObservations.from_pyarrow(sel_unique).to_parquet(
                    str(keys_dir / "selected_keys_unique.parquet")
                )
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
                    strategy=name,
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

            orbit_ids_in_strategy: set[str] = set()
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
                        mask = _filter_ephem_to_truth(
                            ephem=ephem, target_idx=target_idx, truth_keys=truth_keys_tbl_strategy
                        )
                        hit = np.nonzero(mask)[0]
                        if hit.size == 0:
                            continue
                        ephem = ephem.take(hit.tolist())
                        target_idx = target_idx[hit]

                    if len(ephem) == 0:
                        continue
                    n_rows_used += int(len(ephem))

                    orbit_id = np.asarray(_ephem_key_table(ephem)["orbit_id"].to_pylist(), dtype=object)
                    lon = ephem.coordinates.lon.to_numpy(zero_copy_only=False).astype(np.float64)
                    lat = ephem.coordinates.lat.to_numpy(zero_copy_only=False).astype(np.float64)
                    order, starts = _group_slices_by_orbit_target(orbit_id, target_idx.astype(np.int64))
                    if starts.size == 0:
                        continue
                    # Add end sentinel.
                    ends = np.concatenate([starts[1:], np.array([len(order)], dtype=np.int64)])

                    # Accumulate predicted pixels per group into a key table.
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
                                collapsed = _collapse_variant_ephemeris_group(
                                    variants=vsub,
                                )
                                lon0 = float(collapsed.coordinates.lon[0].as_py())
                                lat0 = float(collapsed.coordinates.lat[0].as_py())
                                cov6 = collapsed.coordinates.covariance.to_matrix()[0].astype(np.float64)
                                cov_ll = cov6[1:3, 1:3]
                                if bool(persist_geometry):
                                    k = (str(oid), int(tidx))
                                    if k not in seen_geom:
                                        seen_geom.add(k)
                                        geom_id = f"{k[0]}|{k[1]}"
                                        if str(footprint) in {"cov_polygon_reconstructed", "cov_polygon_reconstructed_moc"}:
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
                                        if str(footprint) in {"sample_polygon", "sample_polygon_moc"}:
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
                                    polygon_mode=(
                                        "convex_hull"
                                        if polygon_mode == "convex_hull"
                                        else "angle_sort"
                                    ),
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
                            truth_obs.join(
                                selected, keys=["orbit_id", "target_idx", "healpixel"], join_type="inner"
                            ).num_rows
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
                    TruthObservations.from_pyarrow(sel_unique).to_parquet(
                        str(keys_dir / "selected_keys_unique.parquet")
                    )
                if bool(compute_extra_frames) and bool(persist_geometry):
                    write_geometry_artifacts(
                        run_dir=run_dir,
                        strategy=str(variant_root_name),
                        variant_kind=str(variant_kind),
                        footprint=str(out_fp),
                        geometry_rows=geom_rows,
                        points_rows=geom_points,
                    )
                if bool(compute_extra_frames) and bool(persist_geometry):
                    strat_key = str(variant_root_name) if variant_kind is None else f"{variant_root_name}:{variant_kind}"
                    write_geometry_artifacts(
                        run_dir=run_dir,
                        strategy=strat_key if ":" not in strat_key else str(variant_root_name),
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

    metrics = Stage3Metrics.from_pyarrow(pa.Table.from_pylist(metrics_rows))
    coverage = Stage3Coverage.from_pyarrow(pa.Table.from_pylist(coverage_rows))
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
        only_truth=bool(only_truth),
        compute_extra_frames=bool(compute_extra_frames),
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
    run_dir = run_stage3_healpixel_bench(
        subset_dir=Path(args.subset_dir),
        stage2_run_dir=Path(args.stage2_run_dir),
        out_dir=None if args.out_dir is None else Path(args.out_dir),
        healpix_nside=int(args.healpix_nside),
        n_sigma=float(args.n_sigma),
        polygon_vertices=int(args.polygon_vertices),
        cov_mc_num_samples=int(args.cov_mc_num_samples),
        cov_mc_seed=int(args.cov_mc_seed),
        corridor_radius_arcsec=float(args.corridor_radius_arcsec),
        corridor_step_arcsec=float(args.corridor_step_arcsec),
        strategies=strategies,
        only_truth=bool(args.only_truth),
        compute_extra_frames=bool(args.compute_extra_frames),
        persist_geometry=bool(args.persist_geometry),
    )
    print(f"run_dir={run_dir}")
    print(f"metrics_parquet={run_dir / 'metrics.parquet'}")
    print(f"coverage_parquet={run_dir / 'coverage.parquet'}")


if __name__ == "__main__":
    main()

