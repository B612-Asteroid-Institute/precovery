"""
Stage 2/3 pipeline build: orbit propagation, predicted positions, and healpix triples.

Produces PredictedTargets, PredictedTriples, and Stage3OrbitMetrics used by
Stage 4 (backend join + innovation gate) and by run/artifact writers.
"""
from __future__ import annotations

import time

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from adam_core.observers import Observers
from adam_core.photometry.bandpasses import map_to_canonical_filter_bands
from adam_core.photometry.magnitude import convert_magnitude
from adam_core.time import Timestamp
from adam_core.orbits import Orbits

from .pipeline_types import (
    BenchTargets,
    PredictedTargets,
    PredictedTriples,
    Stage3OrbitMetrics,
    pack_cov_ll,
)
from .target_link import map_times_to_target_idx_by_obscode
from .propagation import (
    predict_targets,
    predict_targets_batched_assist_variants_sigma_points,
    predict_targets_batched_assist_window_then_2body_variants_sigma_points,
)


def build_predictions_and_triples(
    *,
    orbits: Orbits,
    targets: BenchTargets,
    start_mjd_utc: float,
    window_size_days: int,
    stage2_strategy: str,
    healpix_nside: int,
    footprint,
    max_processes: int | None,
    limit_codefid_keys: pa.Array,
    limit_codefid_vals: pa.Array,
    faint_margin_mag: float,
    max_mag_residual_fainter_mag: float | None,
    max_mag_residual_brighter_mag: float | None,
    dataset_pixels_by_target: dict[tuple[str, int], np.ndarray] | None = None,
    truth_frames: pa.Table | None = None,
    detailed_timings: bool,
) -> tuple[
    PredictedTargets,
    PredictedTriples,
    Stage3OrbitMetrics,
    float,
    float,
    float,
    float,
    float,
    int,
    dict[str, float],
]:
    """
    Stage 2/3 build: propagate orbits to targets, build predicted positions and triples.

    Returns
    -------
    preds_table
        Table with schema compatible with `pipeline_types.PredictedTargets`.
    triples_table
        Table with schema compatible with `pipeline_types.PredictedTriples`.
    frame_metrics_table
        Per-orbit Stage-3 accounting (geometry-matched, lim-mag rejected, etc.).
    observer_creation_elapsed_s, propagation_elapsed_s, build_pred_mag_elapsed_s,
    build_footprint_elapsed_s, build_triples_elapsed_s
        Timing breakdown (seconds).
    n_frames_skipped_limiting_mag
        Count of (orbit_id, target_idx) pairs skipped due to limiting magnitude.
    micro_timings
        Dict of micro-timing keys for footprint and stage2 steps.
    """
    if len(orbits) == 0 or len(targets) == 0:
        return (
            PredictedTargets.empty(),
            PredictedTriples.empty(),
            Stage3OrbitMetrics.empty(),
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0,
            {},
        )

    targets_tbl = targets.table

    obsc_col = targets_tbl["obscode"]
    filt_col = targets_tbl["filter"]
    mid_col = targets_tbl["exposure_mjd_mid_utc"]
    key_col = targets_tbl["exposure_mjd_mid_key_us"]
    if isinstance(obsc_col, pa.ChunkedArray):
        obsc_col = obsc_col.combine_chunks()
    if isinstance(filt_col, pa.ChunkedArray):
        filt_col = filt_col.combine_chunks()
    if isinstance(mid_col, pa.ChunkedArray):
        mid_col = mid_col.combine_chunks()
    if isinstance(key_col, pa.ChunkedArray):
        key_col = key_col.combine_chunks()

    obsc = pc.cast(obsc_col, pa.large_string())
    filt = pc.cast(filt_col, pa.large_string())
    mids = np.asarray(
        pc.cast(mid_col, pa.float64()).to_numpy(zero_copy_only=False),
        dtype=np.float64,
    )
    key_us = pc.cast(key_col, pa.int64())

    times = Timestamp.from_mjd(mids, scale="utc")

    canonical = map_to_canonical_filter_bands(
        obsc,
        filt,
        allow_fallback_filters=True,
    )
    canon_arr = pa.array(canonical, type=pa.large_string())

    # Pre-build bandpass conversion IDs once (reused across orbits).
    src_filter_id = np.full(int(len(times)), "V", dtype=object)
    tgt_filter_id = np.asarray(canonical, dtype=object)

    # Precompute observers aligned to targets once; reused across orbits in stage2.
    t_obs0 = time.perf_counter()
    observers_all = Observers.from_codes(obsc, times)
    observer_creation_elapsed_s = float(time.perf_counter() - t_obs0)

    # Precompute limiting-magnitude lookup needles once (aligned to targets).
    sep = pa.scalar("|", type=pa.large_string())
    target_codefid = pc.binary_join_element_wise(obsc, canon_arr, sep)
    limit_for_target = pa.nulls(len(times), type=pa.float64())
    if len(limit_codefid_keys) > 0:
        idx = pc.fill_null(pc.index_in(target_codefid, value_set=limit_codefid_keys), -1)
        valid = pc.greater_equal(idx, 0)
        idx_safe = pc.cast(pc.if_else(valid, idx, 0), pa.int64())
        lim = pc.take(pc.cast(limit_codefid_vals, pa.float64()), idx_safe)
        limit_for_target = pc.if_else(valid, lim, None)
    limit_with_margin = pc.add(limit_for_target, float(faint_margin_mag))

    need_pred_mag = (
        len(limit_codefid_keys) > 0
        or max_mag_residual_fainter_mag is not None
        or max_mag_residual_brighter_mag is not None
    )

    truth_by_orbit_target: dict[str, dict[tuple[str, int], dict[int, int]]] = {}
    if truth_frames is not None and truth_frames.num_rows > 0:
        req_tf = {
            "orbit_id",
            "obscode",
            "exposure_mjd_mid_key_us",
            "healpixel",
            "n_truth_detections",
        }
        missing_tf = sorted(req_tf - set(truth_frames.column_names))
        if missing_tf:
            raise ValueError(f"truth_frames missing columns: {missing_tf}")
        oid_tf = pc.cast(truth_frames["orbit_id"], pa.large_string()).to_pylist()
        oc_tf = pc.cast(truth_frames["obscode"], pa.large_string()).to_pylist()
        key_tf = (
            pc.cast(truth_frames["exposure_mjd_mid_key_us"], pa.int64())
            .to_numpy(zero_copy_only=False)
            .astype(np.int64, copy=False)
        )
        pix_tf = (
            pc.cast(truth_frames["healpixel"], pa.int64())
            .to_numpy(zero_copy_only=False)
            .astype(np.int64, copy=False)
        )
        n_tf = (
            pc.cast(truth_frames["n_truth_detections"], pa.int64())
            .to_numpy(zero_copy_only=False)
            .astype(np.int64, copy=False)
        )
        for o, oc, k, p, n in zip(oid_tf, oc_tf, key_tf.tolist(), pix_tf.tolist(), n_tf.tolist()):
            if o is None or oc is None:
                continue
            o_s = str(o)
            oc_s = str(oc)
            k_i = int(k)
            p_i = int(p)
            n_i = int(n)
            by_t = truth_by_orbit_target.setdefault(o_s, {})
            by_pix = by_t.setdefault((oc_s, k_i), {})
            by_pix[p_i] = max(int(by_pix.get(p_i, 0)), n_i)

    # Column-oriented outputs.
    pred_orbit_id: list[str] = []
    pred_target_idx: list[int] = []
    pred_obscode: list[str] = []
    pred_exposure_mjd_mid_utc: list[float] = []
    pred_exposure_mjd_mid_key_us: list[int] = []
    pred_canonical_filter_id: list[str | None] = []
    pred_lon_deg: list[float] = []
    pred_lat_deg: list[float] = []
    pred_cov00: list[float] = []
    pred_cov01: list[float] = []
    pred_cov11: list[float] = []
    pred_mag_out: list[float | None] = []

    trip_orbit_id: list[str] = []
    trip_target_idx: list[int] = []
    trip_obscode: list[str] = []
    trip_exposure_mjd_mid_utc: list[float] = []
    trip_exposure_mjd_mid_key_us: list[int] = []
    trip_healpixel: list[int] = []

    met_orbit_id: list[str] = []
    met_frames_geom: list[int] = []
    met_frames_limmag_rej: list[int] = []
    met_frames_truth_geom: list[int] = []
    met_frames_truth_limmag_rej: list[int] = []
    met_truth_det_frame_candidates: list[int] = []

    propagation_elapsed_s = 0.0
    build_pred_mag_elapsed_s = 0.0
    build_footprint_elapsed_s = 0.0
    build_triples_elapsed_s = 0.0
    n_pairs_skipped_faint = 0

    footprint_timings: dict[str, float] = {}
    stage2_timings: dict[str, float] = {}

    # Reduce per-chunk overhead inside `predict_targets` by using larger time chunks.
    time_chunk_size = int(min(20_000, max(2_000, int(len(times)))))

    # Batched idiomatic path for sigma-point strategies: compute predictions for many orbits at once
    # (amortizes SPICE lookups) and then proceed with the usual Stage 3 build.
    if str(stage2_strategy) in (
        "assist_variants:sigma_points",
        "assist_window_then_2body_variants:sigma_points",
    ) and len(orbits) > 0:
        t_pred0 = time.perf_counter()
        if str(stage2_strategy) == "assist_variants:sigma_points":
            ephem_all, cov_ll_all = predict_targets_batched_assist_variants_sigma_points(
                orbits=orbits,
                observers_utc=observers_all,
                max_processes=max_processes,
                orbit_chunk_size=int(min(16, max(1, int(len(orbits))))),
                time_chunk_size=int(time_chunk_size),
                aberration_mode="none",
                default_mag_slope_G=(0.15 if need_pred_mag else None),
                timings=(stage2_timings if bool(detailed_timings) else None),
            )
        else:
            # Windowed ASSIST + 2-body variants, but batched across orbits and time chunks.
            ephem_all, cov_ll_all = predict_targets_batched_assist_window_then_2body_variants_sigma_points(
                orbits=orbits,
                obscode=pc.cast(obsc, pa.large_string()).to_numpy(zero_copy_only=False).astype(object),
                times_utc=times,
                start_mjd=float(start_mjd_utc),
                window_size_days=int(window_size_days),
                observers_all=observers_all,
                max_processes=max_processes,
                orbit_chunk_size=int(min(8, max(1, int(len(orbits))))),
                time_chunk_size=0,
                aberration_mode="none",
                default_mag_slope_G=(0.15 if need_pred_mag else None),
                timings=(stage2_timings if bool(detailed_timings) else None),
            )
        propagation_elapsed_s += float(time.perf_counter() - t_pred0)

        # Link predicted ephemeris rows back to target_idx robustly (exact join + NN fallback).
        cov00 = pa.array(cov_ll_all[:, 0, 0], type=pa.float64())
        cov01 = pa.array(cov_ll_all[:, 0, 1], type=pa.float64())
        cov11 = pa.array(cov_ll_all[:, 1, 1], type=pa.float64())

        t_ephem_utc = ephem_all.coordinates.time.rescale("utc")
        pred_codes = pc.cast(ephem_all.coordinates.origin.code, pa.large_string()).to_numpy(
            zero_copy_only=False
        ).astype(object)
        tidx = map_times_to_target_idx_by_obscode(
            obscode=pred_codes,
            time_utc=t_ephem_utc,
            targets=targets,
            dt_sec=1.0,
        )
        if np.any(tidx < 0):
            n_miss = int(np.count_nonzero(tidx < 0))
            raise RuntimeError(
                "Failed to re-link some batched predictions to targets (after NN fallback). "
                f"n_miss={n_miss} ephem_rows={int(len(ephem_all))}"
            )
        tidx_pa = pa.array(tidx.astype(np.int64, copy=False), type=pa.int64())

        obsc_for_pred = pc.take(obsc, tidx_pa)
        key_us_for_pred = pc.take(key_us, tidx_pa)
        canon_for_pred = pc.take(canon_arr, tidx_pa)
        mjd_for_pred = pc.take(pa.array(mids, type=pa.float64()), tidx_pa)

        # Convert predicted V magnitudes to canonical filter per row.
        pred_mag_arr: pa.Array = pa.nulls(int(len(ephem_all)), type=pa.float64())
        if need_pred_mag and int(len(ephem_all)) > 0:
            t_mag0 = time.perf_counter()
            mag_v = pc.cast(ephem_all.predicted_magnitude_v, pa.float64())
            valid = pc.is_valid(mag_v)
            mags_v_np = pc.fill_null(mag_v, pa.scalar(np.nan, type=pa.float64())).to_numpy(
                zero_copy_only=False
            ).astype(np.float64)
            tgt_f = pc.cast(canon_for_pred, pa.large_string()).to_numpy(zero_copy_only=False)
            pred_mag_np = convert_magnitude(
                mags_v_np,
                source_filter_id=np.full(len(mags_v_np), str(src_filter_id[0]), dtype=object),
                target_filter_id=tgt_f,
                composition="C",
            )
            pred_mag_arr = pc.if_else(valid, pa.array(pred_mag_np, type=pa.float64()), None)
            build_pred_mag_elapsed_s += float(time.perf_counter() - t_mag0)

        # Limiting-magnitude skip mask per (orbit, target) pair.
        too_faint = pa.array([False] * int(len(ephem_all)), type=pa.bool_())
        if len(limit_codefid_keys) > 0 and int(len(ephem_all)) > 0 and not pc.all(
            pc.is_null(pred_mag_arr)
        ).as_py():
            limit_row = pc.take(limit_for_target, tidx_pa)
            limit_margin_row = pc.take(limit_with_margin, tidx_pa)
            too_faint = pc.fill_null(
                pc.and_(
                    pc.is_valid(limit_row),
                    pc.greater(pred_mag_arr, limit_margin_row),
                ),
                False,
            )
            try:
                n_pairs_skipped_faint += int(
                    pc.sum(pc.cast(too_faint, pa.int64())).as_py()
                )
            except Exception:
                pass

        # Build PredictedTargets table (vectorized) and then generate triples via footprint.
        preds_tbl = pa.table(
            {
                "orbit_id": pc.cast(ephem_all.orbit_id, pa.large_string()),
                "target_idx": tidx_pa,
                "obscode": obsc_for_pred,
                "exposure_mjd_mid_utc": mjd_for_pred,
                "exposure_mjd_mid_key_us": pc.cast(key_us_for_pred, pa.int64()),
                "canonical_filter_id": canon_for_pred,
                "pred_lon_deg": pc.cast(ephem_all.coordinates.lon, pa.float64()),
                "pred_lat_deg": pc.cast(ephem_all.coordinates.lat, pa.float64()),
                "cov_ll_00": pc.cast(cov00, pa.float64()),
                "cov_ll_01": pc.cast(cov01, pa.float64()),
                "cov_ll_11": pc.cast(cov11, pa.float64()),
                "pred_mag": pred_mag_arr,
            }
        )

        # Sort so we can process by-orbit with simple contiguous slices.
        preds_tbl = preds_tbl.append_column("_too_faint", too_faint).sort_by(
            [("orbit_id", "ascending"), ("target_idx", "ascending")]
        )

        # Materialize columns needed for footprint computation (vectorized extraction).
        orbit_id_values = pc.cast(orbits.orbit_id, pa.large_string())
        orbit_code = pc.cast(pc.index_in(preds_tbl["orbit_id"], value_set=orbit_id_values), pa.int32()).to_numpy(
            zero_copy_only=False
        ).astype(np.int32, copy=False)
        if np.any(orbit_code < 0):
            raise RuntimeError("Internal error: encountered prediction orbit_id not present in input orbits.")

        target_idx = pc.cast(preds_tbl["target_idx"], pa.int64()).to_numpy(zero_copy_only=False).astype(np.int64)
        lon = pc.cast(preds_tbl["pred_lon_deg"], pa.float64()).to_numpy(zero_copy_only=False).astype(np.float64)
        lat = pc.cast(preds_tbl["pred_lat_deg"], pa.float64()).to_numpy(zero_copy_only=False).astype(np.float64)
        c00 = pc.cast(preds_tbl["cov_ll_00"], pa.float64()).to_numpy(zero_copy_only=False).astype(np.float64)
        c01 = pc.cast(preds_tbl["cov_ll_01"], pa.float64()).to_numpy(zero_copy_only=False).astype(np.float64)
        c11 = pc.cast(preds_tbl["cov_ll_11"], pa.float64()).to_numpy(zero_copy_only=False).astype(np.float64)
        faint = pc.cast(preds_tbl["_too_faint"], pa.bool_()).to_numpy(zero_copy_only=False)
        # Encode obscode as small integer codes (drastically reduces Ray object-store pressure).
        obscode_values_py = sorted({str(x) for x in pc.cast(obsc, pa.large_string()).to_pylist() if x is not None})
        obscode_values = pa.array(obscode_values_py, type=pa.large_string())
        obscode_code = pc.cast(pc.index_in(preds_tbl["obscode"], value_set=obscode_values), pa.int8()).to_numpy(
            zero_copy_only=False
        ).astype(np.int8, copy=False)
        if np.any(obscode_code < 0):
            raise RuntimeError("Internal error: encountered prediction obscode not present in targets.")
        mjd_list = pc.cast(preds_tbl["exposure_mjd_mid_utc"], pa.float64()).to_numpy(
            zero_copy_only=False
        ).astype(np.float64)
        key_us_list = pc.cast(preds_tbl["exposure_mjd_mid_key_us"], pa.int64()).to_numpy(
            zero_copy_only=False
        ).astype(np.int64)

        obscode_to_code = {str(o): int(i) for i, o in enumerate(obscode_values_py)}

        # Convert truth mapping (small) to use orbit_code + obscode_code keys and array payloads
        # (avoid inner per-pixel Python loops inside Ray workers).
        truth_by_orbit_target_code: dict[int, dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]] = {}
        if truth_by_orbit_target:
            orbit_ids_py = orbit_id_values.to_pylist()
            orbit_to_code = {str(oid): int(i) for i, oid in enumerate(orbit_ids_py)}
            for oid, by_t in truth_by_orbit_target.items():
                c = orbit_to_code.get(str(oid))
                if c is None:
                    continue
                out_t: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
                for (oc_s, k_i), by_pix in by_t.items():
                    oc_code = obscode_to_code.get(str(oc_s))
                    if oc_code is None:
                        continue
                    if not by_pix:
                        continue
                    pix = np.fromiter((int(p) for p in by_pix.keys()), dtype=np.int64)
                    ntruth = np.fromiter((int(n) for n in by_pix.values()), dtype=np.int64)
                    if pix.size == 0:
                        continue
                    out_t[(int(oc_code), int(k_i))] = (pix, ntruth)
                if out_t:
                    truth_by_orbit_target_code[int(c)] = out_t

        # Parallel footprint + triple generation using Ray when max_processes > 1.
        triples_tbl: pa.Table
        frame_metrics: Stage3OrbitMetrics

        use_ray = max_processes is None or int(max_processes) > 1
        if use_ray:
            try:
                import ray  # type: ignore
                from adam_core.ray_cluster import initialize_use_ray

                initialize_use_ray(num_cpus=None if max_processes is None else int(max_processes))

                data_ref = ray.put(
                    {
                        "orbit_code": orbit_code,
                        "target_idx": target_idx,
                        "lon": lon,
                        "lat": lat,
                        "c00": c00,
                        "c01": c01,
                        "c11": c11,
                        "faint": np.asarray(faint, dtype=bool),
                        "obscode_code": obscode_code,
                        "mjd": mjd_list,
                        "key_us": key_us_list,
                    }
                )
                ds_code: dict[tuple[int, int], np.ndarray] | None = None
                if dataset_pixels_by_target:
                    ds_code = {}
                    for (oc_s, k_i), pix in dataset_pixels_by_target.items():
                        oc_code = obscode_to_code.get(str(oc_s))
                        if oc_code is None:
                            continue
                        arr = np.asarray(pix, dtype=np.int64)
                        if arr.size == 0:
                            continue
                        ds_code[(int(oc_code), int(k_i))] = arr
                ds_ref = None if not ds_code else ray.put(ds_code)
                truth_ref = None if not truth_by_orbit_target_code else ray.put(truth_by_orbit_target_code)

                def _worker_chunk(start_i: int, stop_i: int, data, ds_map, truth_map):
                    import numpy as _np
                    import pyarrow as _pa
                    import time as _time

                    oc = data["orbit_code"]
                    ti = data["target_idx"]
                    lon0 = data["lon"]
                    lat0 = data["lat"]
                    c00_ = data["c00"]
                    c01_ = data["c01"]
                    c11_ = data["c11"]
                    faint_ = data["faint"]
                    obsc_ = data["obscode_code"]
                    mjd_ = data["mjd"]
                    key_ = data["key_us"]

                    triples_pix_list: list[_np.ndarray] = []
                    triples_lens: list[int] = []
                    triples_orbit_code: list[int] = []
                    triples_target_idx: list[int] = []
                    triples_obscode_code: list[int] = []
                    triples_mjd: list[float] = []
                    triples_key: list[int] = []

                    # Per-prediction metric rows; aggregate by orbit_code at end (no dict accumulators).
                    used_orbit_code: list[int] = []
                    used_n_pix: list[int] = []
                    used_is_faint: list[bool] = []
                    used_truth_hit_frames: list[int] = []
                    used_truth_hit_frames_faint: list[int] = []
                    used_truth_det_sum: list[int] = []

                    t_local: dict[str, float] = {}
                    for j in range(int(start_i), int(stop_i)):
                        ocj = int(oc[j])
                        tij = int(ti[j])
                        obs_code = int(obsc_[j])
                        mjdj = float(mjd_[j])
                        keyj = int(key_[j])
                        cov_ll = _np.array([[c00_[j], c01_[j]], [c01_[j], c11_[j]]], dtype=_np.float64)

                        t0 = _time.perf_counter()
                        pix = footprint.pixels_for_prediction(
                            lon0_deg=float(lon0[j]),
                            lat0_deg=float(lat0[j]),
                            cov_ll_deg2=cov_ll,
                            nside=int(healpix_nside),
                            timings=t_local,
                        )
                        t_local["pipeline_build.footprint_total_elapsed_s"] = t_local.get(
                            "pipeline_build.footprint_total_elapsed_s", 0.0
                        ) + (_time.perf_counter() - t0)

                        pix_arr = _np.unique(_np.asarray(pix, dtype=_np.int64))
                        if pix_arr.size <= 0:
                            continue

                        if ds_map is not None:
                            ds_pix = ds_map.get((obs_code, keyj))
                            if ds_pix is not None and int(getattr(ds_pix, "size", 0)) > 0:
                                m_ds = _np.isin(
                                    pix_arr, _np.asarray(ds_pix, dtype=_np.int64), assume_unique=False
                                )
                                pix_arr = pix_arr[m_ds]
                                if pix_arr.size <= 0:
                                    continue

                        n_pix = int(pix_arr.size)
                        is_faint = bool(faint_[j])

                        truth_hit_frames = 0
                        truth_det_sum = 0
                        if truth_map is not None:
                            truth_for_orbit = truth_map.get(ocj)
                            if truth_for_orbit is not None:
                                ent = truth_for_orbit.get((obs_code, keyj))
                                if ent is not None:
                                    pix_truth, ntruth = ent
                                    m = _np.isin(_np.asarray(pix_truth, dtype=_np.int64), pix_arr, assume_unique=False)
                                    truth_hit_frames = int(_np.count_nonzero(m))
                                    if truth_hit_frames > 0 and (not is_faint):
                                        truth_det_sum = int(_np.sum(_np.asarray(ntruth, dtype=_np.int64)[m]))

                        used_orbit_code.append(int(ocj))
                        used_n_pix.append(int(n_pix))
                        used_is_faint.append(bool(is_faint))
                        used_truth_hit_frames.append(int(truth_hit_frames))
                        used_truth_hit_frames_faint.append(int(truth_hit_frames if is_faint else 0))
                        used_truth_det_sum.append(int(truth_det_sum))

                        if is_faint:
                            continue

                        # Emit triples for Stage 4 join (vectorized concat later).
                        triples_pix_list.append(pix_arr.astype(_np.int64, copy=False))
                        triples_lens.append(int(pix_arr.size))
                        triples_orbit_code.append(int(ocj))
                        triples_target_idx.append(int(tij))
                        triples_obscode_code.append(int(obs_code))
                        triples_mjd.append(float(mjdj))
                        triples_key.append(int(keyj))

                    if triples_pix_list:
                        lens = _np.asarray(triples_lens, dtype=_np.int64)
                        pix_cat = _np.concatenate(triples_pix_list, axis=0)
                        rep = lambda x: _np.repeat(_np.asarray(x), lens)  # noqa: E731
                        triples = _pa.table(
                            {
                                "orbit_code": _pa.array(rep(triples_orbit_code), type=_pa.int32()),
                                "target_idx": _pa.array(rep(triples_target_idx), type=_pa.int64()),
                                "obscode_code": _pa.array(rep(triples_obscode_code), type=_pa.int8()),
                                "exposure_mjd_mid_utc": _pa.array(rep(triples_mjd), type=_pa.float64()),
                                "exposure_mjd_mid_key_us": _pa.array(rep(triples_key), type=_pa.int64()),
                                "healpixel": _pa.array(pix_cat, type=_pa.int64()),
                            }
                        )
                    else:
                        triples = _pa.table(
                            {
                                "orbit_code": _pa.array([], type=_pa.int32()),
                                "target_idx": _pa.array([], type=_pa.int64()),
                                "obscode_code": _pa.array([], type=_pa.int8()),
                                "exposure_mjd_mid_utc": _pa.array([], type=_pa.float64()),
                                "exposure_mjd_mid_key_us": _pa.array([], type=_pa.int64()),
                                "healpixel": _pa.array([], type=_pa.int64()),
                            }
                        )

                    if not used_orbit_code:
                        metrics = _pa.table(
                            {
                                "orbit_code": _pa.array([], type=_pa.int32()),
                                "n_frames_geometry_matched": _pa.array([], type=_pa.int64()),
                                "n_frames_lim_mag_rejected": _pa.array([], type=_pa.int64()),
                                "n_frames_truth_geometry_matched": _pa.array([], type=_pa.int64()),
                                "n_frames_lim_mag_truth_rejected": _pa.array([], type=_pa.int64()),
                                "n_detections_truth_frame_candidates": _pa.array([], type=_pa.int64()),
                            }
                        )
                        return triples, metrics, t_local

                    oc_u = _np.asarray(used_orbit_code, dtype=_np.int32)
                    n_pix_u = _np.asarray(used_n_pix, dtype=_np.int64)
                    faint_u = _np.asarray(used_is_faint, dtype=bool)
                    tf_u = _np.asarray(used_truth_hit_frames, dtype=_np.int64)
                    tf_f_u = _np.asarray(used_truth_hit_frames_faint, dtype=_np.int64)
                    td_u = _np.asarray(used_truth_det_sum, dtype=_np.int64)

                    n_orbits_local = int(_np.max(oc_u)) + 1 if oc_u.size else 0
                    geo = _np.zeros(n_orbits_local, dtype=_np.int64)
                    lim = _np.zeros(n_orbits_local, dtype=_np.int64)
                    tgeo = _np.zeros(n_orbits_local, dtype=_np.int64)
                    tlim = _np.zeros(n_orbits_local, dtype=_np.int64)
                    tdet = _np.zeros(n_orbits_local, dtype=_np.int64)

                    _np.add.at(geo, oc_u, n_pix_u)
                    if faint_u.any():
                        _np.add.at(lim, oc_u[faint_u], n_pix_u[faint_u])
                    _np.add.at(tgeo, oc_u, tf_u)
                    _np.add.at(tlim, oc_u, tf_f_u)
                    _np.add.at(tdet, oc_u, td_u)

                    have = (geo != 0) | (lim != 0) | (tgeo != 0) | (tlim != 0) | (tdet != 0)
                    ocs = _np.nonzero(have)[0].astype(_np.int32, copy=False)
                    metrics = _pa.table(
                        {
                            "orbit_code": _pa.array(ocs, type=_pa.int32()),
                            "n_frames_geometry_matched": _pa.array(geo[have], type=_pa.int64()),
                            "n_frames_lim_mag_rejected": _pa.array(lim[have], type=_pa.int64()),
                            "n_frames_truth_geometry_matched": _pa.array(tgeo[have], type=_pa.int64()),
                            "n_frames_lim_mag_truth_rejected": _pa.array(tlim[have], type=_pa.int64()),
                            "n_detections_truth_frame_candidates": _pa.array(tdet[have], type=_pa.int64()),
                        }
                    )
                    return triples, metrics, t_local

                remote_fn = ray.remote(_worker_chunk)

                t_fp_wall0 = time.perf_counter()
                rows_per_task = int(5_000)
                pending = []
                for s in range(0, int(len(orbit_code)), int(rows_per_task)):
                    e = int(min(int(len(orbit_code)), int(s) + int(rows_per_task)))
                    pending.append(remote_fn.remote(s, e, data_ref, ds_ref, truth_ref))

                triples_parts: list[pa.Table] = []
                metrics_parts: list[pa.Table] = []
                timing_parts: list[dict[str, float]] = []
                while pending:
                    finished, pending = ray.wait(pending, num_returns=1)
                    t_part, m_part, tim_part = ray.get(finished[0])
                    if t_part is not None and t_part.num_rows > 0:
                        triples_parts.append(t_part)
                    if m_part is not None and m_part.num_rows > 0:
                        metrics_parts.append(m_part)
                    if tim_part:
                        timing_parts.append(dict(tim_part))

                triples_tmp = pa.concat_tables(triples_parts, promote_options="default") if triples_parts else pa.table({})

                # Map orbit_code -> orbit_id string.
                if triples_tmp.num_rows > 0:
                    orbit_id_out = pc.take(orbit_id_values, pc.cast(triples_tmp["orbit_code"], pa.int64()))
                    obscode_out = pc.take(obscode_values, pc.cast(triples_tmp["obscode_code"], pa.int64()))
                    triples_tbl = pa.table(
                        {
                            "orbit_id": pc.cast(orbit_id_out, pa.large_string()),
                            "target_idx": triples_tmp["target_idx"],
                            "obscode": pc.cast(obscode_out, pa.large_string()),
                            "exposure_mjd_mid_utc": triples_tmp["exposure_mjd_mid_utc"],
                            "exposure_mjd_mid_key_us": triples_tmp["exposure_mjd_mid_key_us"],
                            "healpixel": triples_tmp["healpixel"],
                        }
                    )
                else:
                    triples_tbl = pa.table(
                        {
                            "orbit_id": pa.array([], type=pa.large_string()),
                            "target_idx": pa.array([], type=pa.int64()),
                            "obscode": pa.array([], type=pa.large_string()),
                            "exposure_mjd_mid_utc": pa.array([], type=pa.float64()),
                            "exposure_mjd_mid_key_us": pa.array([], type=pa.int64()),
                            "healpixel": pa.array([], type=pa.int64()),
                        }
                    )

                metrics_tmp = (
                    pa.concat_tables(metrics_parts, promote_options="default") if metrics_parts else pa.table({})
                )
                if metrics_tmp.num_rows > 0:
                    # Sum per orbit_code across worker chunks.
                    g = metrics_tmp.group_by(["orbit_code"]).aggregate(
                        [
                            ("n_frames_geometry_matched", "sum"),
                            ("n_frames_lim_mag_rejected", "sum"),
                            ("n_frames_truth_geometry_matched", "sum"),
                            ("n_frames_lim_mag_truth_rejected", "sum"),
                            ("n_detections_truth_frame_candidates", "sum"),
                        ]
                    )
                    g = g.rename_columns(
                        [
                            "orbit_code",
                            "n_frames_geometry_matched",
                            "n_frames_lim_mag_rejected",
                            "n_frames_truth_geometry_matched",
                            "n_frames_lim_mag_truth_rejected",
                            "n_detections_truth_frame_candidates",
                        ]
                    )
                    # Ensure all orbits are present (fill missing with 0).
                    all_codes = pa.array(np.arange(len(orbit_id_values), dtype=np.int32), type=pa.int32())
                    all_tbl = pa.table({"orbit_code": all_codes})
                    g = all_tbl.join(g, keys=["orbit_code"], join_type="left outer")
                    for c in g.column_names:
                        if c == "orbit_code":
                            continue
                        g = g.set_column(
                            g.schema.get_field_index(c),
                            c,
                            pc.cast(pc.fill_null(g[c], 0), pa.int64()),
                        )
                    oid_out = pc.take(orbit_id_values, pc.cast(g["orbit_code"], pa.int64()))
                    frame_metrics = Stage3OrbitMetrics.from_kwargs(
                        orbit_id=pc.cast(oid_out, pa.large_string()),
                        n_frames_geometry_matched=pc.cast(g["n_frames_geometry_matched"], pa.int64()),
                        n_frames_lim_mag_rejected=pc.cast(g["n_frames_lim_mag_rejected"], pa.int64()),
                        n_frames_truth_geometry_matched=pc.cast(g["n_frames_truth_geometry_matched"], pa.int64()),
                        n_frames_lim_mag_truth_rejected=pc.cast(g["n_frames_lim_mag_truth_rejected"], pa.int64()),
                        n_detections_truth_frame_candidates=pc.cast(
                            g["n_detections_truth_frame_candidates"], pa.int64()
                        ),
                    )
                else:
                    frame_metrics = Stage3OrbitMetrics.from_kwargs(
                        orbit_id=orbit_id_values,
                        n_frames_geometry_matched=pa.array([0] * len(orbit_id_values), type=pa.int64()),
                        n_frames_lim_mag_rejected=pa.array([0] * len(orbit_id_values), type=pa.int64()),
                        n_frames_truth_geometry_matched=pa.array([0] * len(orbit_id_values), type=pa.int64()),
                        n_frames_lim_mag_truth_rejected=pa.array([0] * len(orbit_id_values), type=pa.int64()),
                        n_detections_truth_frame_candidates=pa.array([0] * len(orbit_id_values), type=pa.int64()),
                    )

                # Merge footprint micro-timings across workers.
                for d in timing_parts:
                    for k, v in d.items():
                        footprint_timings[k] = footprint_timings.get(k, 0.0) + float(v)

                build_footprint_elapsed_s += float(time.perf_counter() - t_fp_wall0)

            except Exception:
                use_ray = False

        if not use_ray:
            # Fallback: keep the prior single-process implementation for environments without Ray.
            orbit_ids = preds_tbl["orbit_id"].to_pylist()
            obsc_list = pc.cast(preds_tbl["obscode"], pa.large_string()).to_pylist()
            met_orbit_id: list[str] = []
            met_n_frames_geometry_matched: list[int] = []
            met_n_frames_lim_mag_rejected: list[int] = []
            met_n_frames_truth_geometry_matched: list[int] = []
            met_n_frames_lim_mag_truth_rejected: list[int] = []
            met_n_detections_truth_frame_candidates: list[int] = []
            triples_orbit_id: list[str] = []
            triples_target_idx: list[int] = []
            triples_obscode: list[str] = []
            triples_exposure_mjd_mid_utc: list[float] = []
            triples_exposure_mjd_mid_key_us: list[int] = []
            triples_healpixel: list[int] = []

            start = 0
            while start < len(orbit_ids):
                oid = orbit_ids[start]
                end = start + 1
                while end < len(orbit_ids) and orbit_ids[end] == oid:
                    end += 1

                n_geo_frames = 0
                n_limrej_frames = 0
                n_truth_geo_frames = 0
                n_truth_limrej_frames = 0
                n_truth_frame_candidates = 0
                truth_for_orbit = truth_by_orbit_target.get(oid) if truth_by_orbit_target is not None else None

                for j in range(start, end):
                    ti = int(target_idx[j])
                    oc = str(obsc_list[j])
                    mjd_i = float(mjd_list[j])
                    key_i = int(key_us_list[j])
                    cov_ll = np.array([[c00[j], c01[j]], [c01[j], c11[j]]], dtype=np.float64)
                    pix = footprint.pixels_for_prediction(
                        lon0_deg=float(lon[j]),
                        lat0_deg=float(lat[j]),
                        cov_ll_deg2=cov_ll,
                        nside=int(healpix_nside),
                        timings=(footprint_timings if bool(detailed_timings) else None),
                    )
                    pix_arr = np.unique(np.asarray(pix, dtype=np.int64))
                    if pix_arr.size <= 0:
                        continue
                    ds_pix = (
                        None
                        if dataset_pixels_by_target is None
                        else dataset_pixels_by_target.get((oc, key_i))
                    )
                    if ds_pix is not None and int(getattr(ds_pix, "size", 0)) > 0:
                        m_ds = np.isin(pix_arr, np.asarray(ds_pix, dtype=np.int64), assume_unique=False)
                        pix_arr = pix_arr[m_ds]
                        if pix_arr.size <= 0:
                            continue
                    n_pix = int(pix_arr.size)
                    n_geo_frames += n_pix
                    is_faint = bool(faint[j])
                    if is_faint:
                        n_limrej_frames += n_pix
                    if truth_for_orbit is not None:
                        tf_pix = truth_for_orbit.get((oc, key_i))
                        if tf_pix:
                            pix_set = set(int(p) for p in pix_arr.tolist())
                            hit_frames = 0
                            hit_truth_det = 0
                            for hpix, ntruth in tf_pix.items():
                                if int(hpix) not in pix_set:
                                    continue
                                hit_frames += 1
                                if not is_faint:
                                    hit_truth_det += int(ntruth)
                            if hit_frames > 0:
                                n_truth_geo_frames += int(hit_frames)
                                if is_faint:
                                    n_truth_limrej_frames += int(hit_frames)
                                else:
                                    n_truth_frame_candidates += int(hit_truth_det)
                    if is_faint:
                        continue
                    triples_orbit_id.extend([oid] * n_pix)
                    triples_target_idx.extend([ti] * n_pix)
                    triples_obscode.extend([oc] * n_pix)
                    triples_exposure_mjd_mid_utc.extend([mjd_i] * n_pix)
                    triples_exposure_mjd_mid_key_us.extend([key_i] * n_pix)
                    triples_healpixel.extend([int(p) for p in pix_arr.tolist()])

                met_orbit_id.append(oid)
                met_n_frames_geometry_matched.append(int(n_geo_frames))
                met_n_frames_lim_mag_rejected.append(int(n_limrej_frames))
                met_n_frames_truth_geometry_matched.append(int(n_truth_geo_frames))
                met_n_frames_lim_mag_truth_rejected.append(int(n_truth_limrej_frames))
                met_n_detections_truth_frame_candidates.append(int(n_truth_frame_candidates))
                start = end

            triples_tbl = pa.table(
                {
                    "orbit_id": pa.array(triples_orbit_id, type=pa.large_string()),
                    "target_idx": pa.array(triples_target_idx, type=pa.int64()),
                    "obscode": pa.array(triples_obscode, type=pa.large_string()),
                    "exposure_mjd_mid_utc": pa.array(triples_exposure_mjd_mid_utc, type=pa.float64()),
                    "exposure_mjd_mid_key_us": pa.array(triples_exposure_mjd_mid_key_us, type=pa.int64()),
                    "healpixel": pa.array(triples_healpixel, type=pa.int64()),
                }
            )
            frame_metrics = Stage3OrbitMetrics.from_kwargs(
                orbit_id=pa.array(met_orbit_id, type=pa.large_string()),
                n_frames_geometry_matched=pa.array(met_n_frames_geometry_matched, type=pa.int64()),
                n_frames_lim_mag_rejected=pa.array(met_n_frames_lim_mag_rejected, type=pa.int64()),
                n_frames_truth_geometry_matched=pa.array(met_n_frames_truth_geometry_matched, type=pa.int64()),
                n_frames_lim_mag_truth_rejected=pa.array(met_n_frames_lim_mag_truth_rejected, type=pa.int64()),
                n_detections_truth_frame_candidates=pa.array(met_n_detections_truth_frame_candidates, type=pa.int64()),
            )

        preds_tbl_out = preds_tbl.drop(["_too_faint"])
        preds = PredictedTargets.from_pyarrow(preds_tbl_out)
        triples = PredictedTriples.from_pyarrow(triples_tbl)

        micro_timings = dict(stage2_timings)
        micro_timings.update(footprint_timings)
        return (
            preds,
            triples,
            frame_metrics,
            float(observer_creation_elapsed_s),
            float(propagation_elapsed_s),
            float(build_pred_mag_elapsed_s),
            float(build_footprint_elapsed_s),
            float(build_triples_elapsed_s),
            int(n_pairs_skipped_faint),
            micro_timings,
        )

    for i_orb in range(int(len(orbits))):
        orbit_single = orbits.take([int(i_orb)])
        oid = str(orbit_single.orbit_id[0].as_py())

        try:
            t_pred0 = time.perf_counter()
            pred = predict_targets(
                orbit=orbit_single,
                obscode=obsc,
                times_utc=times,
                start_mjd=float(start_mjd_utc),
                window_size_days=int(window_size_days),
                strategy=str(stage2_strategy),
                max_processes=max_processes,
                time_chunk_size=time_chunk_size,
                observers_all=observers_all,
                timings=(stage2_timings if bool(detailed_timings) else None),
                default_mag_slope_G=(0.15 if need_pred_mag else None),
            )
            propagation_elapsed_s += float(time.perf_counter() - t_pred0)
        except Exception:
            # Skip orbit on propagation failure (e.g. bad state or unsupported strategy).
            continue

        eph = pred.ephem
        if len(eph) != int(len(times)):
            raise RuntimeError("predict_targets returned unexpected length vs targets")

        lon = eph.coordinates.lon.to_numpy(zero_copy_only=False).astype(np.float64)
        lat = eph.coordinates.lat.to_numpy(zero_copy_only=False).astype(np.float64)

        pred_mag_arr: pa.Array = pa.nulls(len(times), type=pa.float64())
        if need_pred_mag:
            if hasattr(eph, "predicted_magnitude_v") and not pc.all(
                pc.is_null(eph.predicted_magnitude_v)
            ).as_py():
                t_mag0 = time.perf_counter()
                mag_v = pc.cast(eph.predicted_magnitude_v, pa.float64())
                valid = pc.is_valid(mag_v)
                mags_v_np = pc.fill_null(
                    mag_v, pa.scalar(np.nan, type=pa.float64())
                ).to_numpy(zero_copy_only=False).astype(np.float64)
                pred_mag_np = convert_magnitude(
                    mags_v_np,
                    source_filter_id=src_filter_id,
                    target_filter_id=tgt_filter_id,
                    composition="C",
                )
                pred_mag_arr = pc.if_else(
                    valid, pa.array(pred_mag_np, type=pa.float64()), None
                )
                build_pred_mag_elapsed_s += float(time.perf_counter() - t_mag0)

        pred_mag_list = pred_mag_arr.to_pylist()

        too_faint = None
        if len(limit_codefid_keys) > 0 and not pc.all(pc.is_null(pred_mag_arr)).as_py():
            too_faint = pc.fill_null(
                pc.and_(
                    pc.is_valid(limit_for_target),
                    pc.greater(pred_mag_arr, limit_with_margin),
                ),
                False,
            )

        for i in range(int(len(times))):
            cov_ll = pred.cov_ll_deg2[i]
            c00, c01, c11 = pack_cov_ll(cov_ll)

            pred_orbit_id.append(oid)
            pred_target_idx.append(int(i))
            pred_obscode.append(str(obsc[i].as_py()))
            pred_exposure_mjd_mid_utc.append(float(mids[i]))
            pred_exposure_mjd_mid_key_us.append(int(key_us[i].as_py()))
            pred_canonical_filter_id.append(
                str(canon_arr[i].as_py()) if canon_arr[i].as_py() is not None else None
            )
            pred_lon_deg.append(float(lon[i]))
            pred_lat_deg.append(float(lat[i]))
            pred_cov00.append(float(c00))
            pred_cov01.append(float(c01))
            pred_cov11.append(float(c11))
            pm = pred_mag_list[i]
            pred_mag_out.append(None if pm is None else float(pm))
            oc = str(obsc[i].as_py())
            mjd_i = float(mids[i])
            key_i = int(key_us[i].as_py())

            is_faint = False
            if too_faint is not None and bool(too_faint[i].as_py()):
                is_faint = True
                n_pairs_skipped_faint += 1

            t_fp0 = time.perf_counter()
            pix = footprint.pixels_for_prediction(
                lon0_deg=float(lon[i]),
                lat0_deg=float(lat[i]),
                cov_ll_deg2=cov_ll,
                nside=int(healpix_nside),
                timings=(footprint_timings if bool(detailed_timings) else None),
            )
            build_footprint_elapsed_s += float(time.perf_counter() - t_fp0)

            pix_arr = np.unique(np.asarray(pix, dtype=np.int64))
            if pix_arr.size <= 0:
                continue

            # Restrict to frame keys that actually exist in the dataset (when provided).
            ds_pix = (
                None if dataset_pixels_by_target is None else dataset_pixels_by_target.get((oc, key_i))
            )
            if ds_pix is not None and int(getattr(ds_pix, "size", 0)) > 0:
                m_ds = np.isin(pix_arr, np.asarray(ds_pix, dtype=np.int64), assume_unique=False)
                pix_arr = pix_arr[m_ds]
            n_pix = int(pix_arr.size)
            if n_pix <= 0:
                continue

            # Per-orbit frame accounting (geometry vs limiting-magnitude).
            if met_orbit_id and met_orbit_id[-1] == oid:
                met_frames_geom[-1] += n_pix
                if is_faint:
                    met_frames_limmag_rej[-1] += n_pix
            else:
                met_orbit_id.append(oid)
                met_frames_geom.append(n_pix)
                met_frames_limmag_rej.append(n_pix if is_faint else 0)
                met_frames_truth_geom.append(0)
                met_frames_truth_limmag_rej.append(0)
                met_truth_det_frame_candidates.append(0)

            # Optional truth overlap accounting during Stage 3.
            if truth_by_orbit_target:
                tf = truth_by_orbit_target.get(oid)
                if tf is not None:
                    tf_pix = tf.get((oc, key_i))
                    if tf_pix:
                        pix_set = set(pix_arr.tolist())
                        hit_frames = 0
                        hit_truth_det = 0
                        for hpix, ntruth in tf_pix.items():
                            if int(hpix) not in pix_set:
                                continue
                            hit_frames += 1
                            if not is_faint:
                                hit_truth_det += int(ntruth)
                        if hit_frames > 0:
                            met_frames_truth_geom[-1] += int(hit_frames)
                            if is_faint:
                                met_frames_truth_limmag_rej[-1] += int(hit_frames)
                            else:
                                met_truth_det_frame_candidates[-1] += int(hit_truth_det)

            # Only include frames that survive limiting-magnitude skipping in the join keys.
            if is_faint:
                continue

            t_tr0 = time.perf_counter()
            trip_orbit_id.extend([oid] * n_pix)
            trip_target_idx.extend([int(i)] * n_pix)
            trip_obscode.extend([oc] * n_pix)
            trip_exposure_mjd_mid_utc.extend([mjd_i] * n_pix)
            trip_exposure_mjd_mid_key_us.extend([key_i] * n_pix)
            trip_healpixel.extend(pix_arr.tolist())
            build_triples_elapsed_s += float(time.perf_counter() - t_tr0)

    preds_table = pa.table(
        {
            "orbit_id": pa.array(pred_orbit_id, type=pa.large_string()),
            "target_idx": pa.array(pred_target_idx, type=pa.int64()),
            "obscode": pa.array(pred_obscode, type=pa.large_string()),
            "exposure_mjd_mid_utc": pa.array(pred_exposure_mjd_mid_utc, type=pa.float64()),
            "exposure_mjd_mid_key_us": pa.array(pred_exposure_mjd_mid_key_us, type=pa.int64()),
            "canonical_filter_id": pa.array(
                pred_canonical_filter_id, type=pa.large_string()
            ),
            "pred_lon_deg": pa.array(pred_lon_deg, type=pa.float64()),
            "pred_lat_deg": pa.array(pred_lat_deg, type=pa.float64()),
            "cov_ll_00": pa.array(pred_cov00, type=pa.float64()),
            "cov_ll_01": pa.array(pred_cov01, type=pa.float64()),
            "cov_ll_11": pa.array(pred_cov11, type=pa.float64()),
            "pred_mag": pa.array(pred_mag_out, type=pa.float64()),
        }
    )
    triples_table = pa.table(
        {
            "orbit_id": pa.array(trip_orbit_id, type=pa.large_string()),
            "target_idx": pa.array(trip_target_idx, type=pa.int64()),
            "obscode": pa.array(trip_obscode, type=pa.large_string()),
            "exposure_mjd_mid_utc": pa.array(trip_exposure_mjd_mid_utc, type=pa.float64()),
            "exposure_mjd_mid_key_us": pa.array(trip_exposure_mjd_mid_key_us, type=pa.int64()),
            "healpixel": pa.array(trip_healpixel, type=pa.int64()),
        }
    )

    frame_metrics = Stage3OrbitMetrics.from_kwargs(
        orbit_id=met_orbit_id,
        n_frames_geometry_matched=np.asarray(met_frames_geom, dtype=np.int64),
        n_frames_lim_mag_rejected=np.asarray(met_frames_limmag_rej, dtype=np.int64),
        n_frames_truth_geometry_matched=np.asarray(met_frames_truth_geom, dtype=np.int64),
        n_frames_lim_mag_truth_rejected=np.asarray(met_frames_truth_limmag_rej, dtype=np.int64),
        n_detections_truth_frame_candidates=np.asarray(
            met_truth_det_frame_candidates, dtype=np.int64
        ),
    )

    micro_timings = dict(stage2_timings)
    micro_timings.update(footprint_timings)
    return (
        PredictedTargets.from_pyarrow(preds_table),
        PredictedTriples.from_pyarrow(triples_table),
        frame_metrics,
        float(observer_creation_elapsed_s),
        float(propagation_elapsed_s),
        float(build_pred_mag_elapsed_s),
        float(build_footprint_elapsed_s),
        float(build_triples_elapsed_s),
        int(n_pairs_skipped_faint),
        micro_timings,
    )

