from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import quivr as qv
import time
from adam_core.orbits import Orbits
from adam_core.coordinates.cartesian import CartesianCoordinates
from adam_core.coordinates.origin import Origin

from precovery.precovery_db import (
    REJECT_REASON_LIMITING_MAGNITUDE,
    REJECT_REASON_MAG_RESIDUAL,
    FrameCandidates,
    PrecoveryCandidates,
)

from .footprints import CovPolygonReconstructedMoc
from .frames_sqlite import fetch_frames_for_pixels_sqlite
from .propagation import (
    PropagationStrategy,
    TargetPrediction,
    predict_targets,
)
from .targets import enumerate_distinct_frame_time_targets_sqlite
from .types import FrameTimeTargets, TargetPixels
from .metrics import SearchAgg, SearchMetrics, metrics_row
from .protocols import SearchDB
from .stages import (
    build_candidates_for_hits,
    compute_faint_skip_mask,
    frame_candidates_from_frames_and_ephem,
    gate_frame_observations,
    index_db_path,
    load_observations_for_frames,
    refresh_runtime_state,
    align_frames_to_targets,
)

def _process_targets_chunk_ray_worker(
    *,
    db_dir: str,
    allow_version_mismatch: bool,
    orbit: Orbits,
    orbit_id: str,
    targets: FrameTimeTargets,
    t0: int,
    t1: int,
    tolerance: float | None,
    start_mjd: float,
    end_mjd: float,
    datasets: set[str] | None,
    window_size_days: int,
    n_sigma: float,
    propagation_max_processes: int | None,
    want_per_target_metrics: bool,
) -> tuple[PrecoveryCandidates, FrameCandidates, SearchAgg]:
    """
    Ray worker: open DB locally and process one target chunk end-to-end.
    """
    import quivr as qv

    from precovery.precovery_db import PrecoveryDatabase

    db = PrecoveryDatabase.from_dir(
        str(db_dir),
        create=False,
        mode="r",
        allow_version_mismatch=bool(allow_version_mismatch),
    )
    refresh_runtime_state(db)

    start = int(t0)
    stop = int(t1)
    targets_chunk = targets[slice(start, stop)]

    limit_keys, limit_vals, faint_margin, max_faint, max_bright = _load_search_constants(db)
    idx_db = index_db_path(db)

    agg = SearchAgg()
    if want_per_target_metrics:
        agg.enable_per_target()

    c_list, f_list = _process_targets_chunk(
        db=db,
        orbit=orbit,
        orbit_id=str(orbit_id),
        targets_chunk=targets_chunk,
        tolerance=tolerance,
        start_mjd=float(start_mjd),
        end_mjd=float(end_mjd),
        datasets=datasets,
        window_size_days=int(window_size_days),
        footprint=CovPolygonReconstructedMoc(
            n_sigma=float(n_sigma),
            polygon_vertices=int(DEFAULT_FOOTPRINT_POLYGON_VERTICES),
        ),
        n_sigma=float(n_sigma),
        max_processes=propagation_max_processes,
        limit_keys=limit_keys,
        limit_vals=limit_vals,
        faint_margin=float(faint_margin),
        max_faint=max_faint,
        max_bright=max_bright,
        idx_db=idx_db,
        metrics=agg,
    )

    c = qv.concatenate(c_list) if c_list else PrecoveryCandidates.empty()
    f = qv.concatenate(f_list) if f_list else FrameCandidates.empty()
    return PrecoveryCandidates.from_pyarrow(c.table), FrameCandidates.from_pyarrow(f.table), agg


DEFAULT_PROPAGATION_STRATEGY: PropagationStrategy = "assist_window_then_2body_variants:sigma_points"
DEFAULT_FOOTPRINT_POLYGON_VERTICES: int = 32

def enumerate_targets(
    *, db: SearchDB, start_mjd: float, end_mjd: float, datasets: set[str] | None
) -> FrameTimeTargets:
    """
    Stage 1: enumerate distinct (obscode, exposure_mjd_mid) targets.

    Uses raw sqlite GROUP BY (experiment-proven hot path).
    """
    idx_db = index_db_path(db)
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
    metrics: SearchAgg | None = None,
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
        t_fp_one = time.perf_counter() if metrics is not None and metrics.per_target_rows is not None else None
        px = footprint.pixels_for_prediction(
            lon0_deg=float(lon0[i]),
            lat0_deg=float(lat0[i]),
            cov_ll_deg2=pred.cov_ll_deg2[i],
            nside=int(nside),
        )
        if metrics is not None and metrics.per_target_rows is not None:
            metrics.per_target_rows.append(
                dict(
                    obscode=str(targets.obscode[i].as_py()),
                    exposure_mjd_mid=float(targets.exposure_mjd_mid[i].as_py()),
                    target_row=int(i),
                    n_predicted_pixels=int(px.size),
                    n_frames_joined=0,
                    n_frames_loaded=0,
                    n_observations_loaded=0,
                    n_accepted=0,
                    footprint_sec=None if t_fp_one is None else float(time.perf_counter() - t_fp_one),
                    io_sec=None,
                    filter_sec=None,
                )
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


def _resolve_bounds(
    *, db: SearchDB, start_mjd: float | None, end_mjd: float | None, datasets: set[str] | None
) -> tuple[float, float]:
    if start_mjd is None or end_mjd is None:
        first, last = db.frames.idx.mjd_bounds(datasets=datasets)
        start_mjd = first if start_mjd is None else float(start_mjd)
        end_mjd = last if end_mjd is None else float(end_mjd)
    return float(start_mjd), float(end_mjd)


def _load_search_constants(db: SearchDB) -> tuple[pa.Array, pa.Array, float, object, object]:
    """
    Pull config/cache-like values once per run.
    """
    limit_keys = getattr(db, "_limit_codefid_keys", pa.array([], type=pa.large_string()))
    limit_vals = getattr(db, "_limit_codefid_vals", pa.array([], type=pa.float64()))
    faint_margin = float(getattr(db.config, "faint_frame_skip_margin_mag", 0.0) or 0.0)
    max_faint = getattr(db.config, "max_mag_residual_fainter_mag", None)
    max_bright = getattr(db.config, "max_mag_residual_brighter_mag", None)
    return limit_keys, limit_vals, faint_margin, max_faint, max_bright


def _process_targets_chunk(
    *,
    db: SearchDB,
    orbit: Orbits,
    orbit_id: str,
    targets_chunk: FrameTimeTargets,
    tolerance: float | None,
    start_mjd: float,
    end_mjd: float,
    datasets: set[str] | None,
    window_size_days: int,
    footprint: CovPolygonReconstructedMoc,
    n_sigma: float,
    max_processes: int | None,
    limit_keys: pa.Array,
    limit_vals: pa.Array,
    faint_margin: float,
    max_faint: object,
    max_bright: object,
    idx_db,
    metrics: SearchAgg | None,
) -> tuple[list[PrecoveryCandidates], list[FrameCandidates]]:
    """
    Process one chunk of distinct (obscode, time) targets end-to-end.

    This is the primary "atomic" unit for profiling: it contains the per-chunk hot path
    (propagate -> footprint -> join frames -> IO -> filter -> photometry).
    """
    if len(targets_chunk) == 0:
        return [], []

    codes_chunk = targets_chunk.obscode
    times_chunk = targets_chunk.time

    t_prop = time.perf_counter() if metrics is not None else None
    pred: TargetPrediction = predict_targets(
        orbit=orbit,
        obscode=codes_chunk,
        times_utc=times_chunk,
        start_mjd=float(start_mjd),
        window_size_days=int(window_size_days),
        strategy=DEFAULT_PROPAGATION_STRATEGY,
        max_processes=max_processes,
    )
    if metrics is not None and t_prop is not None:
        metrics.propagate_sec += float(time.perf_counter() - t_prop)
    if len(pred.ephem) == 0:
        return [], []
    if metrics is not None:
        metrics.n_predicted_rows += int(len(pred.ephem))

    t_fp = time.perf_counter() if metrics is not None else None
    pixels = targets_to_pixels(
        targets=targets_chunk,
        pred=pred,
        footprint=footprint,
        nside=int(db.frames.healpix_nside),
        metrics=metrics,
    )
    if metrics is not None and t_fp is not None:
        metrics.footprint_sec += float(time.perf_counter() - t_fp)
    if len(pixels) == 0:
        return [], []
    if metrics is not None:
        metrics.n_predicted_pixels += int(len(pixels))

    t_join = time.perf_counter() if metrics is not None else None
    frames = fetch_frames_for_pixels_sqlite(index_db=idx_db, pixels=pixels, datasets=datasets)
    if metrics is not None and t_join is not None:
        metrics.join_frames_sec += float(time.perf_counter() - t_join)
    if len(frames) == 0:
        return [], []
    if metrics is not None:
        metrics.n_frames_joined += int(len(frames))

    frame_rows, target_rows = align_frames_to_targets(frames=frames, targets=targets_chunk)
    if frame_rows.size == 0:
        return [], []

    frames2 = frames.take(frame_rows.tolist())
    ephem_for_frames = pred.ephem.take(target_rows.tolist())
    frame_cands = frame_candidates_from_frames_and_ephem(
        frames=frames2, ephem=ephem_for_frames, orbit_id=str(orbit_id)
    )

    need_mag_residual = (max_faint is not None) or (max_bright is not None)
    need_limiting_mag = len(limit_keys) > 0
    need_any_mag = bool(need_mag_residual) or bool(need_limiting_mag)
    # Only require magnitudes when the orbit has an H parameter; otherwise magnitudes are
    # expected to remain null and downstream magnitude gates should no-op.
    H_v = None
    phys = getattr(orbit, "physical_parameters", None)
    if phys is not None and hasattr(phys, "H_v"):
        H_v = phys.H_v[0].as_py()
    require_any_mag = bool(need_any_mag) and (H_v is not None)
    require_cand_mag = bool(need_mag_residual) and bool(require_any_mag)

    too_faint = compute_faint_skip_mask(
        db=db,
        frame_cands=frame_cands,
        limit_keys=limit_keys,
        limit_vals=limit_vals,
        faint_margin=float(faint_margin),
    )

    # Batch-load observations for non-faint frames.
    active_idx = [i for i in range(len(frames2)) if not bool(too_faint[i].as_py())]
    t_io_batch = time.perf_counter() if metrics is not None else None
    obs_by_i = load_observations_for_frames(frame_db=db.frames, frames=frames2, active_idx=active_idx)
    if metrics is not None and t_io_batch is not None:
        metrics.io_sec += float(time.perf_counter() - t_io_batch)

    out_candidates: list[PrecoveryCandidates] = []
    out_frames: list[FrameCandidates] = []
    for i, (frame, eph_mid) in enumerate(zip(frames2, ephem_for_frames)):
        if bool(too_faint[i].as_py()):
            if metrics is not None:
                metrics.n_frames_faint_skipped += 1
            fc = frame_cands.take([i])
            fc = FrameCandidates.from_pyarrow(
                fc.set_column("rejected", pa.array([True]))
                .set_column(
                    "rejected_reason",
                    pa.array([REJECT_REASON_LIMITING_MAGNITUDE], type=pa.large_string()),
                )
                .table
            )
            out_frames.append(fc)
            continue

        if metrics is not None:
            metrics.n_frames_loaded += 1

        obs = obs_by_i[i]
        assert obs is not None
        if len(obs) == 0:
            out_frames.append(frame_cands.take([i]))
            continue
        if metrics is not None:
            metrics.n_observations_loaded += int(len(obs))

        cov_ll_mid = pred.cov_ll_deg2[target_rows[i] : target_rows[i] + 1]
        hit_idx, eph_rep, _cov_rep = gate_frame_observations(
            orbit=orbit,
            frame=frame,
            obs=obs,
            eph_mid=eph_mid,
            cov_ll_mid=cov_ll_mid,
            propagation_strategy=DEFAULT_PROPAGATION_STRATEGY,
            n_sigma=float(n_sigma),
            tolerance_deg=(None if tolerance is None else float(tolerance)),
            max_processes=max_processes,
            metrics=metrics,
        )
        if not hit_idx:
            out_frames.append(frame_cands.take([i]))
            continue
        if metrics is not None:
            metrics.n_accepted += int(len(hit_idx))

        cand = build_candidates_for_hits(
            obs=obs,
            eph_rep=eph_rep,
            frame=frame,
            orbit=orbit,
            db=db,
            hit_idx=hit_idx,
            metrics=metrics,
            require_magnitudes=require_cand_mag,
        )
        if metrics is not None:
            metrics.n_candidates += int(len(cand))
        cand = _apply_mag_residual_rejection(
            candidates=cand,
            max_mag_residual_fainter_mag=None if max_faint is None else float(max_faint),
            max_mag_residual_brighter_mag=None if max_bright is None else float(max_bright),
        )
        out_candidates.append(cand)

    return out_candidates, out_frames


def precover_orbit(
    *,
    db: SearchDB,
    orbit: Orbits,
    tolerance: float | None = None,
    start_mjd: float | None = None,
    end_mjd: float | None = None,
    datasets: set[str] | None = None,
    window_size_days: int = 7,
    n_sigma: float = 3.0,
    target_chunk_size: int = 10_000,
    max_processes: int | None = 1,
    metrics: SearchAgg | None = None,
) -> tuple[PrecoveryCandidates, FrameCandidates]:
    """
    Performance-first precovery search for a single orbit.

    This is the new high-throughput pipeline. It intentionally does not preserve the old
    window-by-window state machine; instead it processes distinct (obscode,time) targets
    in chunks and streams candidates.
    """
    if len(orbit) != 1:
        raise ValueError("precover_orbit currently supports exactly one orbit (len==1).")

    t_total = time.perf_counter() if metrics is not None else None

    refresh_runtime_state(db)

    start_mjd, end_mjd = _resolve_bounds(db=db, start_mjd=start_mjd, end_mjd=end_mjd, datasets=datasets)

    footprint = CovPolygonReconstructedMoc(
        n_sigma=float(n_sigma),
        polygon_vertices=int(DEFAULT_FOOTPRINT_POLYGON_VERTICES),
    )

    t_enum = time.perf_counter() if metrics is not None else None
    targets = enumerate_targets(
        db=db,
        start_mjd=float(start_mjd),
        # Use nextafter to ensure we include the max frame time even when `mjd_bounds()`
        # returns a float that is infinitesimally below the true stored max.
        end_mjd=float(np.nextafter(float(end_mjd), np.inf)),
        datasets=datasets,
    )
    if metrics is not None and t_enum is not None:
        metrics.enum_sec += float(time.perf_counter() - t_enum)
    if len(targets) == 0:
        if metrics is not None and t_total is not None:
            metrics.total_sec += float(time.perf_counter() - t_total)
        return PrecoveryCandidates.empty(), FrameCandidates.empty()

    n_targets = int(len(targets))
    if metrics is not None:
        metrics.n_targets += int(n_targets)

    # Accumulate results as lists (concat once at end).
    all_candidates: list[PrecoveryCandidates] = []
    all_frame_candidates: list[FrameCandidates] = []

    orbit_id = orbit.orbit_id[0].as_py()

    limit_keys, limit_vals, faint_margin, max_faint, max_bright = _load_search_constants(db)

    idx_db = index_db_path(db)

    # Optional Ray chunk parallelism (max_processes denotes worker budget).
    use_ray = False
    if max_processes is not None and int(max_processes) > 1:
        try:
            from adam_core.ray_cluster import initialize_use_ray

            use_ray = bool(initialize_use_ray(num_cpus=int(max_processes)))
        except Exception:
            use_ray = False

    if use_ray:
        import ray

        from precovery.precovery_db import __version__ as _running_version

        allow_version_mismatch = bool(getattr(db.config, "build_version", None) != _running_version)
        orbit_ref = ray.put(orbit)
        targets_ref = ray.put(targets)

        # If per-target metrics are enabled, each worker will populate its own list which we
        # merge in the driver.
        want_per_target = metrics is not None and metrics.per_target_rows is not None

        worker = ray.remote(_process_targets_chunk_ray_worker)

        futures: list[object] = []
        for t0 in range(0, n_targets, int(target_chunk_size)):
            t1 = min(n_targets, t0 + int(target_chunk_size))
            futures.append(
                worker.remote(
                    db_dir=str(db.directory),
                    allow_version_mismatch=allow_version_mismatch,
                    orbit=orbit_ref,
                    orbit_id=str(orbit_id),
                    targets=targets_ref,
                    t0=int(t0),
                    t1=int(t1),
                    tolerance=tolerance,
                    start_mjd=float(start_mjd),
                    end_mjd=float(end_mjd),
                    datasets=datasets,
                    window_size_days=int(window_size_days),
                    n_sigma=float(n_sigma),
                    propagation_max_processes=1,  # prevent CPU oversubscription
                    want_per_target_metrics=want_per_target,
                )
            )

            # Bound in-flight futures (experiment-proven scheduling pattern).
            if len(futures) >= int(max_processes) * 2:
                finished, futures = ray.wait(futures, num_returns=1)
                c_one, f_one, agg_one = ray.get(finished[0])
                all_candidates.append(c_one)
                all_frame_candidates.append(f_one)
                if metrics is not None:
                    metrics.n_targets += int(getattr(agg_one, "n_targets", 0))
                    metrics.n_predicted_rows += int(getattr(agg_one, "n_predicted_rows", 0))
                    metrics.n_predicted_pixels += int(getattr(agg_one, "n_predicted_pixels", 0))
                    metrics.n_frames_joined += int(getattr(agg_one, "n_frames_joined", 0))
                    metrics.n_frames_loaded += int(getattr(agg_one, "n_frames_loaded", 0))
                    metrics.n_frames_faint_skipped += int(getattr(agg_one, "n_frames_faint_skipped", 0))
                    metrics.n_observations_loaded += int(getattr(agg_one, "n_observations_loaded", 0))
                    metrics.n_accepted += int(getattr(agg_one, "n_accepted", 0))
                    metrics.n_candidates += int(getattr(agg_one, "n_candidates", 0))
                    metrics.enum_sec += float(getattr(agg_one, "enum_sec", 0.0))
                    metrics.propagate_sec += float(getattr(agg_one, "propagate_sec", 0.0))
                    metrics.footprint_sec += float(getattr(agg_one, "footprint_sec", 0.0))
                    metrics.join_frames_sec += float(getattr(agg_one, "join_frames_sec", 0.0))
                    metrics.io_sec += float(getattr(agg_one, "io_sec", 0.0))
                    metrics.filter_sec += float(getattr(agg_one, "filter_sec", 0.0))
                    metrics.photometry_sec += float(getattr(agg_one, "photometry_sec", 0.0))
                    if metrics.per_target_rows is not None and getattr(agg_one, "per_target_rows", None):
                        metrics.per_target_rows.extend(list(getattr(agg_one, "per_target_rows")))

        while futures:
            finished, futures = ray.wait(futures, num_returns=1)
            c_one, f_one, agg_one = ray.get(finished[0])
            all_candidates.append(c_one)
            all_frame_candidates.append(f_one)
            if metrics is not None:
                metrics.n_targets += int(getattr(agg_one, "n_targets", 0))
                metrics.n_predicted_rows += int(getattr(agg_one, "n_predicted_rows", 0))
                metrics.n_predicted_pixels += int(getattr(agg_one, "n_predicted_pixels", 0))
                metrics.n_frames_joined += int(getattr(agg_one, "n_frames_joined", 0))
                metrics.n_frames_loaded += int(getattr(agg_one, "n_frames_loaded", 0))
                metrics.n_frames_faint_skipped += int(getattr(agg_one, "n_frames_faint_skipped", 0))
                metrics.n_observations_loaded += int(getattr(agg_one, "n_observations_loaded", 0))
                metrics.n_accepted += int(getattr(agg_one, "n_accepted", 0))
                metrics.n_candidates += int(getattr(agg_one, "n_candidates", 0))
                metrics.enum_sec += float(getattr(agg_one, "enum_sec", 0.0))
                metrics.propagate_sec += float(getattr(agg_one, "propagate_sec", 0.0))
                metrics.footprint_sec += float(getattr(agg_one, "footprint_sec", 0.0))
                metrics.join_frames_sec += float(getattr(agg_one, "join_frames_sec", 0.0))
                metrics.io_sec += float(getattr(agg_one, "io_sec", 0.0))
                metrics.filter_sec += float(getattr(agg_one, "filter_sec", 0.0))
                metrics.photometry_sec += float(getattr(agg_one, "photometry_sec", 0.0))
                if metrics.per_target_rows is not None and getattr(agg_one, "per_target_rows", None):
                    metrics.per_target_rows.extend(list(getattr(agg_one, "per_target_rows")))
    else:
        for t0 in range(0, n_targets, int(target_chunk_size)):
            sl = slice(t0, min(n_targets, t0 + int(target_chunk_size)))
            targets_chunk = targets[sl]
            c_chunk, f_chunk = _process_targets_chunk(
                db=db,
                orbit=orbit,
                orbit_id=str(orbit_id),
                targets_chunk=targets_chunk,
                tolerance=tolerance,
                start_mjd=float(start_mjd),
                end_mjd=float(end_mjd),
                datasets=datasets,
                window_size_days=int(window_size_days),
                footprint=footprint,
                n_sigma=float(n_sigma),
                max_processes=max_processes,
                limit_keys=limit_keys,
                limit_vals=limit_vals,
                faint_margin=float(faint_margin),
                max_faint=max_faint,
                max_bright=max_bright,
                idx_db=idx_db,
                metrics=metrics,
            )
            all_candidates.extend(c_chunk)
            all_frame_candidates.extend(f_chunk)

    candidates = qv.concatenate(all_candidates) if all_candidates else PrecoveryCandidates.empty()
    frames_out = qv.concatenate(all_frame_candidates) if all_frame_candidates else FrameCandidates.empty()

    # Match existing behavior: strip aberrated_coordinates before returning.
    if len(candidates) > 0:
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

    if metrics is not None and t_total is not None:
        metrics.total_sec += float(time.perf_counter() - t_total)

    return PrecoveryCandidates.from_pyarrow(candidates.table), FrameCandidates.from_pyarrow(frames_out.table)


def precover_orbit_with_metrics(
    *,
    db: SearchDB,
    orbit: Orbits,
    tolerance: float | None = None,
    start_mjd: float | None = None,
    end_mjd: float | None = None,
    datasets: set[str] | None = None,
    window_size_days: int = 7,
    n_sigma: float = 3.0,
    target_chunk_size: int = 10_000,
    max_processes: int | None = 1,
) -> tuple[PrecoveryCandidates, FrameCandidates, SearchMetrics, SearchAgg]:
    """
    `precover_orbit`, but always returns aggregated metrics.

    This is intentionally small and "profiling-friendly": callers can `cProfile` or
    `pyinstrument` this wrapper and then inspect the returned `SearchAgg`/`SearchMetrics`.
    """
    agg = SearchAgg()
    cands, frames = precover_orbit(
        db=db,
        orbit=orbit,
        tolerance=tolerance,
        start_mjd=start_mjd,
        end_mjd=end_mjd,
        datasets=datasets,
        window_size_days=int(window_size_days),
        n_sigma=float(n_sigma),
        target_chunk_size=int(target_chunk_size),
        max_processes=max_processes,
        metrics=agg,
    )
    fp = CovPolygonReconstructedMoc(
        n_sigma=float(n_sigma),
        polygon_vertices=int(DEFAULT_FOOTPRINT_POLYGON_VERTICES),
    )
    run = metrics_row(
        orbit_id=orbit.orbit_id[0].as_py(),
        propagation_strategy=str(DEFAULT_PROPAGATION_STRATEGY),
        footprint=type(fp).__name__,
        healpix_nside=int(db.frames.healpix_nside),
        n_sigma=float(n_sigma),
        window_size_days=int(window_size_days),
        target_chunk_size=int(target_chunk_size),
        max_processes=None if max_processes is None else int(max_processes),
        agg=agg,
    )
    return cands, frames, run, agg


def precover_orbits(
    *,
    db: SearchDB,
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

