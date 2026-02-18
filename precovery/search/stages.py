from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import time
from adam_core.orbits.ephemeris import Ephemeris
from adam_core.photometry.bandpasses import map_to_canonical_filter_bands
from adam_core.photometry.magnitude import convert_magnitude
from adam_core.time import Timestamp
from adam_core.orbits import Orbits

from precovery.healpix_geom import radec_to_healpixel
from precovery.frame_db import HealpixFrame
from precovery.observation import ObservationsTable
from precovery.precovery_db import (
    FrameCandidates,
    PrecoveryCandidates,
    candidates_from_ephem,
)

from .detection_filter import innov_ellipse_keep_mask
from .metrics import SearchAgg
from .propagation import PropagationStrategy, predict_observations_in_frame
from .reconstruction import wrap_delta_lon_deg
from .types import FrameTimeTargets
from .protocols import SearchDB


def index_db_path(db) -> Path:
    uri = db.frames.idx.db_uri
    s = str(uri).replace("sqlite:///", "").split("?")[0]
    return Path(s)


def arrow_lookup(needles: pa.Array, keys: pa.Array, values: pa.Array) -> pa.Array:
    if len(keys) == 0:
        return pa.nulls(len(needles), type=pa.float64())
    idx = pc.fill_null(pc.index_in(needles, value_set=keys), -1)
    valid = pc.greater_equal(idx, 0)
    idx_safe = pc.cast(pc.if_else(valid, idx, 0), pa.int64())
    out = pc.take(values, idx_safe)
    return pc.if_else(valid, out, None)


def timestamp_key(days: pa.Array, nanos: pa.Array) -> pa.Array:
    nanos_in_day = pa.scalar(86_400_000_000_000, type=pa.int64())
    return pc.add_checked(pc.multiply_checked(days, nanos_in_day), nanos)


def refresh_runtime_state(db) -> None:
    """
    Best-effort refreshes of cache-like state. Kept separate for profiling.
    """
    try:
        db._refresh_limiting_magnitudes_cache_if_needed()  # noqa: SLF001
    except Exception:
        pass


def align_frames_to_targets(
    *, frames, targets: FrameTimeTargets
) -> tuple[np.ndarray, np.ndarray]:
    targets_key = pa.table(
        {
            "obscode": targets.obscode,
            "exposure_mjd_mid": targets.exposure_mjd_mid,
            "target_row": pa.array(np.arange(len(targets), dtype=np.int64), type=pa.int64()),
        }
    )
    frames_key = pa.table(
        {
            "frame_row": pa.array(np.arange(len(frames), dtype=np.int64), type=pa.int64()),
            "obscode": frames.obscode,
            "exposure_mjd_mid": frames.exposure_mjd_mid,
        }
    )
    aligned = frames_key.join(targets_key, keys=["obscode", "exposure_mjd_mid"], join_type="inner")
    if aligned.num_rows == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    frame_rows = aligned["frame_row"].to_numpy(zero_copy_only=False).astype(np.int64)
    target_rows = aligned["target_row"].to_numpy(zero_copy_only=False).astype(np.int64)
    return frame_rows, target_rows


def frame_candidates_from_frames_and_ephem(*, frames, ephem, orbit_id: str) -> FrameCandidates:
    if len(frames) == 0:
        return FrameCandidates.empty()

    healpix_id = pa.array(
        radec_to_healpixel(
            ephem.coordinates.lon.to_numpy(zero_copy_only=False),
            ephem.coordinates.lat.to_numpy(zero_copy_only=False),
            nside=2**15,
        ).astype(np.int64),
        type=pa.int64(),
    )

    pred_mag = pa.nulls(len(frames), type=pa.float64())
    if (
        hasattr(ephem, "predicted_magnitude_v")
        and not pc.all(pc.is_null(ephem.predicted_magnitude_v)).as_py()
    ):
        canonical = map_to_canonical_filter_bands(
            frames.obscode,
            frames.filter,
            allow_fallback_filters=True,
        )
        mag_v = pc.cast(ephem.predicted_magnitude_v, pa.float64())
        valid = pc.is_valid(mag_v)
        mags_v_np = pc.fill_null(mag_v, pa.scalar(np.nan, type=pa.float64())).to_numpy(
            zero_copy_only=False
        ).astype(np.float64)
        src = np.full(len(frames), "V", dtype=object)
        tgt = np.asarray(canonical, dtype=object)
        pred_mag_np = convert_magnitude(
            mags_v_np,
            source_filter_id=src,
            target_filter_id=tgt,
            composition="C",
        )
        pred_mag = pc.if_else(valid, pa.array(pred_mag_np, type=pa.float64()), None)

    return FrameCandidates.from_kwargs(
        exposure_time_start=Timestamp.from_mjd(frames.exposure_mjd_start, scale="utc"),
        exposure_time_mid=Timestamp.from_mjd(frames.exposure_mjd_mid, scale="utc"),
        filter=frames.filter,
        obscode=frames.obscode,
        exposure_id=frames.exposure_id,
        exposure_duration=frames.exposure_duration,
        dataset_id=frames.dataset_id,
        healpix_id=healpix_id,
        pred_ra_deg=ephem.coordinates.lon,
        pred_dec_deg=ephem.coordinates.lat,
        pred_vra_degpday=ephem.coordinates.vlon,
        pred_vdec_degpday=ephem.coordinates.vlat,
        pred_mag=pred_mag,
        rejected=pa.repeat(False, len(frames)),
        rejected_reason=pa.array([None] * len(frames), type=pa.large_string()),
        aberrated_coordinates=ephem.aberrated_coordinates,
        orbit_id=pa.repeat(str(orbit_id), len(frames)),
    )


def compute_faint_skip_mask(
    *,
    db: SearchDB,
    frame_cands: FrameCandidates,
    limit_keys: pa.Array,
    limit_vals: pa.Array,
    faint_margin: float,
) -> pa.Array:
    """
    Return a boolean Arrow array `too_faint` aligned to `frame_cands`.
    """
    too_faint = pa.repeat(False, len(frame_cands))
    if len(limit_keys) == 0 or len(frame_cands) == 0:
        return too_faint
    if pc.all(pc.is_null(frame_cands.pred_mag)).as_py():
        return too_faint

    canonical = map_to_canonical_filter_bands(
        frame_cands.obscode,
        frame_cands.filter,
        allow_fallback_filters=True,
    )
    canon_arr = pa.array(canonical, type=pa.large_string())
    sep = pa.scalar("|", type=pa.large_string())
    codefid = pc.binary_join_element_wise(frame_cands.obscode, canon_arr, sep)
    limit = arrow_lookup(codefid, limit_keys, limit_vals)
    limit_with_margin = pc.add(limit, float(faint_margin))
    too_faint = pc.fill_null(
        pc.and_(
            pc.is_valid(limit),
            pc.greater(frame_cands.pred_mag, limit_with_margin),
        ),
        False,
    )
    return too_faint


def load_observations_for_frames(
    *,
    frame_db,
    frames: HealpixFrame,
    active_idx: list[int],
) -> list[object | None]:
    """
    Batch-load observations (aligned to frames row order).
    """
    obs_by_i: list[object | None] = [None] * int(len(frames))
    if not active_idx:
        return obs_by_i

    obs_list = frame_db.get_observations_many(frames.take(active_idx))
    for k, i in enumerate(active_idx):
        obs_by_i[int(i)] = obs_list[int(k)]
    return obs_by_i


def gate_frame_observations(
    *,
    orbit: Orbits,
    frame: HealpixFrame,
    obs: ObservationsTable,
    eph_mid: Ephemeris,
    cov_ll_mid: np.ndarray,  # (1,2,2)
    propagation_strategy: PropagationStrategy,
    n_sigma: float,
    tolerance_deg: float | None,
    max_processes: int | None,
    metrics: SearchAgg | None,
) -> tuple[list[int], Ephemeris, np.ndarray]:
    """
    Return (hit_idx, eph_rep, cov_rep) for a frame.
    """
    if len(obs) == 0:
        return [], eph_mid, cov_ll_mid

    obs_time_utc = obs.time.rescale("utc")
    unique_times = obs_time_utc.unique().sort_by(["days", "nanos"])
    if len(unique_times) == 1:
        eph_rep = eph_mid.take(np.zeros(len(obs), dtype=np.int64).tolist())
        cov_rep = np.repeat(cov_ll_mid, len(obs), axis=0)
    else:
        t_pred = time.perf_counter() if metrics is not None else None
        pred_obs = predict_observations_in_frame(
            orbit=orbit,
            obscode=str(frame.obscode[0].as_py()),
            times_utc=unique_times,
            strategy=propagation_strategy,
            max_processes=max_processes,
        )
        if metrics is not None and t_pred is not None:
            metrics.propagate_sec += float(time.perf_counter() - t_pred)
        obs_key = timestamp_key(obs_time_utc.days, obs_time_utc.nanos)
        uniq_key = timestamp_key(unique_times.days, unique_times.nanos)
        idx = pc.fill_null(pc.index_in(obs_key, value_set=uniq_key), -1)
        idx64 = pc.cast(idx, pa.int64())
        eph_rep = pred_obs.ephem.take(idx64.to_numpy(zero_copy_only=False).tolist())
        cov_rep = pred_obs.cov_ll_deg2[idx64.to_numpy(zero_copy_only=False)]

    t_filter = time.perf_counter() if metrics is not None else None
    keep = innov_ellipse_keep_mask(
        obs_lon_deg=obs.ra.to_numpy(zero_copy_only=False),
        obs_lat_deg=obs.dec.to_numpy(zero_copy_only=False),
        obs_lon_sigma_deg=obs.ra_sigma.to_numpy(zero_copy_only=False),
        obs_lat_sigma_deg=obs.dec_sigma.to_numpy(zero_copy_only=False),
        pred_lon_deg=eph_rep.coordinates.lon.to_numpy(zero_copy_only=False),
        pred_lat_deg=eph_rep.coordinates.lat.to_numpy(zero_copy_only=False),
        pred_cov_ll_deg2=cov_rep,
        n_sigma=float(n_sigma),
        det_sigma_floor_arcsec=0.10,
    )
    if metrics is not None and t_filter is not None:
        metrics.filter_sec += float(time.perf_counter() - t_filter)

    if tolerance_deg is not None and float(tolerance_deg) > 0.0:
        cos_lat = np.cos(np.deg2rad(eph_rep.coordinates.lat.to_numpy(zero_copy_only=False)))
        cos_lat = np.where(np.isfinite(cos_lat) & (np.abs(cos_lat) > 1e-12), cos_lat, 1e-12)
        dx = wrap_delta_lon_deg(
            obs.ra.to_numpy(zero_copy_only=False),
            eph_rep.coordinates.lon.to_numpy(zero_copy_only=False),
        ) * cos_lat
        dy = obs.dec.to_numpy(zero_copy_only=False) - eph_rep.coordinates.lat.to_numpy(zero_copy_only=False)
        dist_deg = np.hypot(dx, dy)
        keep = np.asarray(keep, dtype=bool) & (dist_deg <= float(tolerance_deg))

    hit_idx = np.nonzero(keep)[0].tolist()
    return hit_idx, eph_rep, cov_rep


def build_candidates_for_hits(
    *,
    obs: ObservationsTable,
    eph_rep: Ephemeris,
    frame: HealpixFrame,
    orbit: Orbits,
    db: SearchDB,
    hit_idx: list[int],
    metrics: SearchAgg | None = None,
    require_magnitudes: bool = False,
) -> PrecoveryCandidates:
    if not hit_idx:
        return PrecoveryCandidates.empty()
    obs_hit = obs.take(hit_idx)
    eph_hit = eph_rep.take(hit_idx)

    pred_mag = pa.nulls(len(obs_hit), type=pa.float64())
    mag_residual = pa.nulls(len(obs_hit), type=pa.float64())
    if hasattr(eph_hit, "predicted_magnitude_v") and not pc.all(
        pc.is_null(eph_hit.predicted_magnitude_v)
    ).as_py():
        try:
            t_photo = time.perf_counter() if metrics is not None else None
            canonical = map_to_canonical_filter_bands(
                pa.repeat(frame.obscode[0].as_py(), len(obs_hit)),
                pa.repeat(frame.filter[0].as_py(), len(obs_hit)),
                allow_fallback_filters=True,
            )
            mag_v = pc.cast(eph_hit.predicted_magnitude_v, pa.float64())
            valid = pc.is_valid(mag_v)
            mags_v_np = pc.fill_null(
                mag_v, pa.scalar(np.nan, type=pa.float64())
            ).to_numpy(zero_copy_only=False).astype(np.float64)
            src = np.full(len(obs_hit), "V", dtype=object)
            tgt = np.asarray(canonical, dtype=object)
            pred_mag_np = convert_magnitude(
                mags_v_np,
                source_filter_id=src,
                target_filter_id=tgt,
                composition="C",
            )
            pred_mag = pc.if_else(
                valid, pa.array(pred_mag_np, type=pa.float64()), None
            )
            mag_residual = pc.subtract(pc.cast(obs_hit.mag, pa.float64()), pred_mag)
            if metrics is not None and t_photo is not None:
                metrics.photometry_sec += float(time.perf_counter() - t_photo)
        except Exception:
            if bool(require_magnitudes):
                raise

    return candidates_from_ephem(
        obs_hit,
        eph_hit,
        frame,
        pred_mag=pred_mag,
        mag_residual=mag_residual,
    )

