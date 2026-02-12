from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import quivr as qv
from adam_core.orbits import Orbits
from adam_core.time import Timestamp

from precovery.precovery_db import (
    REJECT_REASON_LIMITING_MAGNITUDE,
    REJECT_REASON_MAG_RESIDUAL,
    FrameCandidates,
    PrecoveryCandidates,
    candidates_from_ephem,
)

from .detection_filter import innov_ellipse_keep_mask
from .footprints import CovPolygonReconstructedMoc
from .frames_sqlite import fetch_frames_for_pixels_sqlite
from .propagation import (
    PropagationStrategy,
    TargetPrediction,
    predict_observations_in_frame,
    predict_targets,
)
from .targets import enumerate_distinct_frame_time_targets_sqlite
from .types import FrameTimeTargets, TargetPixels


def _index_db_path(db) -> Path:
    uri = db.frames.idx.db_uri
    s = str(uri).replace("sqlite:///", "").split("?")[0]
    return Path(s)


def _arrow_lookup(needles: pa.Array, keys: pa.Array, values: pa.Array) -> pa.Array:
    if len(keys) == 0:
        return pa.nulls(len(needles), type=pa.float64())
    idx = pc.fill_null(pc.index_in(needles, value_set=keys), -1)
    valid = pc.greater_equal(idx, 0)
    idx_safe = pc.cast(pc.if_else(valid, idx, 0), pa.int64())
    out = pc.take(values, idx_safe)
    return pc.if_else(valid, out, None)

def _timestamp_key(days: pa.Array, nanos: pa.Array) -> pa.Array:
    """
    Build an int64 composite key for Timestamp alignment:
      key = days*NANOS_IN_DAY + nanos

    We use this because (at least) pyarrow does not support `index_in()` over struct keys.
    """
    nanos_in_day = pa.scalar(86_400_000_000_000, type=pa.int64())
    return pc.add_checked(pc.multiply_checked(days, nanos_in_day), nanos)


def enumerate_targets(
    *, db, start_mjd: float, end_mjd: float, datasets: set[str] | None
) -> FrameTimeTargets:
    """
    Stage 1: enumerate distinct (obscode, exposure_mjd_mid) targets.

    Uses raw sqlite GROUP BY (experiment-proven hot path).
    """
    idx_db = _index_db_path(db)
    codes, mjd_mid, times_utc = enumerate_distinct_frame_time_targets_sqlite(
        index_db=idx_db,
        start_mjd=float(start_mjd),
        end_mjd=float(end_mjd),
        datasets=datasets,
    )
    return FrameTimeTargets.from_kwargs(obscode=codes, exposure_mjd_mid=mjd_mid, time=times_utc)


def targets_to_pixels(
    *,
    targets: FrameTimeTargets,
    pred: TargetPrediction,
    footprint: CovPolygonReconstructedMoc,
    nside: int,
) -> TargetPixels:
    """
    Stage 3: footprint → explode to (obscode, exposure_mjd_mid, healpixel) triples.

    This is intentionally a tight loop around mocpy (not vectorized), but it is fully
    encapsulated and returns an Arrow-friendly Quivr table for the DB join stage.
    """
    if len(targets) == 0:
        return TargetPixels.empty()

    codes_out: list[str] = []
    mjd_out: list[float] = []
    hpix_out: list[int] = []

    lon0 = pred.ephem.coordinates.lon.to_numpy(zero_copy_only=False).astype(np.float64)
    lat0 = pred.ephem.coordinates.lat.to_numpy(zero_copy_only=False).astype(np.float64)

    for i in range(int(len(pred.ephem))):
        px = footprint.pixels_for_prediction(
            lon0_deg=float(lon0[i]),
            lat0_deg=float(lat0[i]),
            cov_ll_deg2=pred.cov_ll_deg2[i],
            nside=int(nside),
        )
        if px.size == 0:
            continue
        code_i = str(targets.obscode[i].as_py())
        mjd_i = float(targets.exposure_mjd_mid[i].as_py())
        codes_out.extend([code_i] * int(px.size))
        mjd_out.extend([mjd_i] * int(px.size))
        hpix_out.extend([int(x) for x in px.tolist()])

    if not codes_out:
        return TargetPixels.empty()

    return TargetPixels.from_kwargs(
        obscode=codes_out,
        exposure_mjd_mid=mjd_out,
        healpixel=hpix_out,
    )


def _frame_candidates_from_frames_and_ephem(
    *,
    frames,
    ephem,
    orbit_id: str,
) -> FrameCandidates:
    from precovery.healpix_geom import radec_to_healpixel

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
        pred_mag=pa.nulls(len(frames), type=pa.float64()),
        rejected=pa.repeat(False, len(frames)),
        rejected_reason=pa.array([None] * len(frames), type=pa.large_string()),
        aberrated_coordinates=ephem.aberrated_coordinates,
        orbit_id=pa.repeat(str(orbit_id), len(frames)),
    )


def _apply_mag_residual_rejection(
    *,
    candidates: PrecoveryCandidates,
    max_mag_residual_fainter_mag: float | None,
    max_mag_residual_brighter_mag: float | None,
) -> PrecoveryCandidates:
    if len(candidates) == 0:
        return candidates
    if max_mag_residual_fainter_mag is None and max_mag_residual_brighter_mag is None:
        return candidates
    if pc.all(pc.is_null(candidates.mag_residual)).as_py():
        return candidates

    outlier = None
    if max_mag_residual_fainter_mag is not None:
        too_faint = pc.greater(
            candidates.mag_residual,
            pa.scalar(float(max_mag_residual_fainter_mag), type=pa.float64()),
        )
        outlier = too_faint if outlier is None else pc.or_(outlier, too_faint)
    if max_mag_residual_brighter_mag is not None:
        too_bright = pc.less(
            candidates.mag_residual,
            pa.scalar(-float(max_mag_residual_brighter_mag), type=pa.float64()),
        )
        outlier = too_bright if outlier is None else pc.or_(outlier, too_bright)

    if outlier is None:
        return candidates

    outlier = pc.fill_null(outlier, False)
    rejected = pc.cast(outlier, pa.bool_())
    reason = pc.if_else(
        outlier,
        pa.scalar(REJECT_REASON_MAG_RESIDUAL, type=pa.large_string()),
        pa.scalar(None, type=pa.large_string()),
    )
    return PrecoveryCandidates.from_pyarrow(
        candidates.set_column("rejected", rejected)
        .set_column("rejected_reason", reason)
        .table
    )


def precover_orbit(
    *,
    db,
    orbit: Orbits,
    start_mjd: float | None = None,
    end_mjd: float | None = None,
    datasets: set[str] | None = None,
    window_size_days: int = 7,
    propagation_strategy: PropagationStrategy = "assist_window_then_2body_variants:sigma_points",
    footprint: CovPolygonReconstructedMoc | None = None,
    n_sigma: float = 3.0,
    target_chunk_size: int = 10_000,
    max_processes: int | None = 1,
) -> tuple[PrecoveryCandidates, FrameCandidates]:
    """
    Performance-first precovery search for a single orbit.

    This is the new high-throughput pipeline. It intentionally does not preserve the old
    window-by-window state machine; instead it processes distinct (obscode,time) targets
    in chunks and streams candidates.
    """
    if len(orbit) != 1:
        raise ValueError("precover_orbit currently supports exactly one orbit (len==1).")

    # Refresh limiting magnitudes cache only if it changed on disk.
    try:
        db._refresh_limiting_magnitudes_cache_if_needed()  # noqa: SLF001
    except Exception:
        pass

    if start_mjd is None or end_mjd is None:
        first, last = db.frames.idx.mjd_bounds(datasets=datasets)
        start_mjd = first if start_mjd is None else float(start_mjd)
        end_mjd = last if end_mjd is None else float(end_mjd)

    footprint = footprint or CovPolygonReconstructedMoc(n_sigma=float(n_sigma), polygon_vertices=32)

    targets = enumerate_targets(
        db=db,
        start_mjd=float(start_mjd),
        # Use nextafter to ensure we include the max frame time even when `mjd_bounds()`
        # returns a float that is infinitesimally below the true stored max.
        end_mjd=float(np.nextafter(float(end_mjd), np.inf)),
        datasets=datasets,
    )
    if len(targets) == 0:
        return PrecoveryCandidates.empty(), FrameCandidates.empty()

    n_targets = int(len(targets))

    # Accumulate results as lists (concat once at end).
    all_candidates: list[PrecoveryCandidates] = []
    all_frame_candidates: list[FrameCandidates] = []

    orbit_id = orbit.orbit_id[0].as_py()

    # Limiting magnitude cache (loaded once at DB open).
    limit_keys = getattr(db, "_limit_codefid_keys", pa.array([], type=pa.large_string()))
    limit_vals = getattr(db, "_limit_codefid_vals", pa.array([], type=pa.float64()))
    faint_margin = float(getattr(db.config, "faint_frame_skip_margin_mag", 0.0) or 0.0)

    # Magnitude residual thresholds.
    max_faint = getattr(db.config, "max_mag_residual_fainter_mag", None)
    max_bright = getattr(db.config, "max_mag_residual_brighter_mag", None)

    idx_db = _index_db_path(db)
    for t0 in range(0, n_targets, int(target_chunk_size)):
        sl = slice(t0, min(n_targets, t0 + int(target_chunk_size)))
        targets_chunk = targets[sl]
        codes_chunk = targets_chunk.obscode
        times_chunk = targets_chunk.time

        pred: TargetPrediction = predict_targets(
            orbit=orbit,
            obscode=codes_chunk,
            times_utc=times_chunk,
            start_mjd=float(start_mjd),
            window_size_days=int(window_size_days),
            strategy=propagation_strategy,
            max_processes=max_processes,
        )
        if len(pred.ephem) == 0:
            continue

        pixels = targets_to_pixels(
            targets=targets_chunk,
            pred=pred,
            footprint=footprint,
            nside=int(db.frames.healpix_nside),
        )
        if len(pixels) == 0:
            continue

        frames = fetch_frames_for_pixels_sqlite(
            index_db=idx_db,
            pixels=pixels,
            datasets=datasets,
        )
        if len(frames) == 0:
            continue

        # Map (obscode,mjd_mid) -> target index within this chunk.
        key_to_idx: dict[tuple[str, float], int] = {
            (
                str(targets_chunk.obscode[i].as_py()),
                float(targets_chunk.exposure_mjd_mid[i].as_py()),
            ): int(i)
            for i in range(int(len(targets_chunk)))
        }

        # Precompute per-frame ephem rows (at exposure midpoints) via target mapping.
        frame_rows: list[int] = []
        target_idx_for_frame: list[int] = []
        for row in range(len(frames)):
            k = (str(frames.obscode[row].as_py()), float(frames.exposure_mjd_mid[row].as_py()))
            j = key_to_idx.get(k)
            if j is None:
                # Should not happen: join was on these fields.
                continue
            frame_rows.append(int(row))
            target_idx_for_frame.append(int(j))
        if not frame_rows:
            continue

        frames2 = frames.take(frame_rows)
        ephem_for_frames = pred.ephem.take(target_idx_for_frame)

        frame_cands = _frame_candidates_from_frames_and_ephem(
            frames=frames2, ephem=ephem_for_frames, orbit_id=str(orbit_id)
        )

        # Compute predicted magnitudes for frames (if possible) and decide faint skips.
        try:
            frame_cands_mag = db._attach_magnitudes(frame_cands, orbit)  # noqa: SLF001
        except Exception:
            frame_cands_mag = frame_cands

        # Faint-frame skip using limiting magnitudes cache (vectorized).
        too_faint = pa.repeat(False, len(frame_cands_mag))
        if len(limit_keys) > 0 and len(frame_cands_mag) > 0 and not pc.all(
            pc.is_null(frame_cands_mag.pred_mag)
        ).as_py():
            from adam_core.photometry.bandpasses import map_to_canonical_filter_bands

            canonical = map_to_canonical_filter_bands(
                frame_cands_mag.obscode,
                frame_cands_mag.filter,
                allow_fallback_filters=True,
            )
            canon_arr = pa.array(canonical, type=pa.large_string())
            sep = pa.scalar("|", type=pa.large_string())
            codefid = pc.binary_join_element_wise(frame_cands_mag.obscode, canon_arr, sep)
            limit = _arrow_lookup(codefid, limit_keys, limit_vals)
            limit_with_margin = pc.add(limit, faint_margin)
            too_faint = pc.fill_null(
                pc.and_(
                    pc.is_valid(limit),
                    pc.greater(frame_cands_mag.pred_mag, limit_with_margin),
                ),
                False,
            )

        # Process each frame: if too faint -> emit rejected FrameCandidate; else load obs and gate.
        for i, (frame, eph_mid) in enumerate(zip(frames2, ephem_for_frames)):
            if bool(too_faint[i].as_py()):
                fc = frame_cands_mag.take([i])
                fc = FrameCandidates.from_pyarrow(
                    fc.set_column("rejected", pa.array([True]))
                    .set_column(
                        "rejected_reason",
                        pa.array([REJECT_REASON_LIMITING_MAGNITUDE], type=pa.large_string()),
                    )
                    .table
                )
                all_frame_candidates.append(fc)
                continue

            obs = db.frames.get_observations(frame)
            if len(obs) == 0:
                all_frame_candidates.append(frame_cands_mag.take([i]))
                continue

            # Per-observation prediction; fast-path when all obs times are identical.
            obs_time_utc = obs.time.rescale("utc")
            unique_times = obs_time_utc.unique().sort_by(["days", "nanos"])
            if len(unique_times) == 1:
                eph = eph_mid
                cov_ll = pred.cov_ll_deg2[
                    target_idx_for_frame[i] : target_idx_for_frame[i] + 1
                ]
                # Repeat ephem row for each observation.
                eph_rep = eph.take(np.zeros(len(obs), dtype=np.int64).tolist())
                cov_rep = np.repeat(cov_ll, len(obs), axis=0)
            else:
                pred_obs = predict_observations_in_frame(
                    orbit=orbit,
                    obscode=str(frame.obscode[0].as_py()),
                    times_utc=unique_times,
                    strategy=propagation_strategy,
                    max_processes=max_processes,
                )
                # Broadcast per-unique-time predictions back to per-observation rows.
                obs_key = _timestamp_key(obs_time_utc.days, obs_time_utc.nanos)
                uniq_key = _timestamp_key(unique_times.days, unique_times.nanos)
                idx = pc.fill_null(pc.index_in(obs_key, value_set=uniq_key), -1)
                assert pc.all(pc.greater_equal(idx, 0)).as_py(), "Missing time mapping for observations"
                idx64 = pc.cast(idx, pa.int64())

                eph_rep = pred_obs.ephem.take(idx64.to_numpy(zero_copy_only=False).tolist())
                cov_rep = pred_obs.cov_ll_deg2[idx64.to_numpy(zero_copy_only=False)]

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
            if not bool(np.any(keep)):
                all_frame_candidates.append(frame_cands_mag.take([i]))
                continue

            hit_idx = np.nonzero(keep)[0].tolist()
            obs_hit = obs.take(hit_idx)
            eph_hit = eph_rep.take(hit_idx)

            cand = candidates_from_ephem(obs_hit, eph_hit, frame)
            try:
                cand = db._attach_magnitudes(cand, orbit)  # noqa: SLF001
            except Exception:
                pass
            cand = _apply_mag_residual_rejection(
                candidates=cand,
                max_mag_residual_fainter_mag=None if max_faint is None else float(max_faint),
                max_mag_residual_brighter_mag=None if max_bright is None else float(max_bright),
            )
            all_candidates.append(cand)

    candidates = qv.concatenate(all_candidates) if all_candidates else PrecoveryCandidates.empty()
    frames_out = qv.concatenate(all_frame_candidates) if all_frame_candidates else FrameCandidates.empty()

    # Match existing behavior: strip aberrated_coordinates before returning.
    if len(candidates) > 0:
        from adam_core.coordinates.cartesian import CartesianCoordinates
        from adam_core.coordinates.origin import Origin

        candidates = candidates.set_column(
            "aberrated_coordinates",
            CartesianCoordinates.from_kwargs(
                x=pa.nulls(len(candidates), type=pa.float64()),
                y=pa.nulls(len(candidates), type=pa.float64()),
                z=pa.nulls(len(candidates), type=pa.float64()),
                vx=pa.nulls(len(candidates), type=pa.float64()),
                vy=pa.nulls(len(candidates), type=pa.float64()),
                vz=pa.nulls(len(candidates), type=pa.float64()),
                time=candidates.time,
                origin=Origin.from_kwargs(code=pa.repeat("SUN", len(candidates))),
                frame="ecliptic",
            ),
        )
    if len(frames_out) > 0:
        from adam_core.coordinates.cartesian import CartesianCoordinates
        from adam_core.coordinates.origin import Origin

        frames_out = frames_out.set_column(
            "aberrated_coordinates",
            CartesianCoordinates.from_kwargs(
                x=pa.nulls(len(frames_out), type=pa.float64()),
                y=pa.nulls(len(frames_out), type=pa.float64()),
                z=pa.nulls(len(frames_out), type=pa.float64()),
                vx=pa.nulls(len(frames_out), type=pa.float64()),
                vy=pa.nulls(len(frames_out), type=pa.float64()),
                vz=pa.nulls(len(frames_out), type=pa.float64()),
                time=frames_out.exposure_time_mid,
                origin=Origin.from_kwargs(code=pa.repeat("SUN", len(frames_out))),
                frame="ecliptic",
            ),
        )

    return PrecoveryCandidates.from_pyarrow(candidates.table), FrameCandidates.from_pyarrow(frames_out.table)


def precover_orbits(
    *,
    db,
    orbits: Orbits,
    **kwargs,
) -> tuple[PrecoveryCandidates, FrameCandidates]:
    """
    Convenience wrapper to precover many orbits (concatenates results).
    """
    all_c: list[PrecoveryCandidates] = []
    all_f: list[FrameCandidates] = []
    for orbit in orbits:
        c, f = precover_orbit(db=db, orbit=orbit, **kwargs)
        all_c.append(c)
        all_f.append(f)
    return (
        qv.concatenate(all_c) if all_c else PrecoveryCandidates.empty(),
        qv.concatenate(all_f) if all_f else FrameCandidates.empty(),
    )

