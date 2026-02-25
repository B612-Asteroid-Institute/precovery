from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import time

from .detection_filter import innov_ellipse_keep_mask
from .pipeline_types import AcceptedCounts, CandidateDetections, PredictedTargets
from .backends.protocols import GateParams


def _det_sigma_floor_arcsec_by_candidate(
    *, candidates: CandidateDetections, gate: GateParams
) -> float | np.ndarray:
    """
    Return a scalar floor (common case) or a per-row floor array (arcsec).
    """
    m = gate.invalid_sigma_fill_floor_arcsec_by_obscode
    if not m:
        return float(gate.invalid_sigma_fill_floor_arcsec_global)

    # Vectorized Arrow mapping obscode -> floor, with default fallback.
    keys = list(m.keys())
    vals = [float(m[k]) for k in keys]
    if not keys:
        return float(gate.invalid_sigma_fill_floor_arcsec_global)

    obsc = candidates.table.column("obscode").combine_chunks()
    idx = pc.index_in(obsc, value_set=pa.array(keys, type=pa.large_string()))
    valid = pc.greater_equal(idx, 0)
    idx_safe = pc.cast(pc.if_else(valid, idx, 0), pa.int64())
    taken = pc.take(pa.array(vals, type=pa.float64()), idx_safe)
    out = pc.if_else(
        valid,
        taken,
        pa.scalar(float(gate.invalid_sigma_fill_floor_arcsec_global), type=pa.float64()),
    )
    return np.asarray(out.to_numpy(zero_copy_only=False), dtype=np.float64)


def _inflate_candidate_sigmas_for_gate(
    *,
    candidates: CandidateDetections,
    gate: GateParams,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (ra_sigma_deg_eff, dec_sigma_deg_eff) arrays aligned to candidates.

    Policy:
    - First, fill missing/invalid sigmas using the (possibly per-obscode) sigma floor.
      This matches the innov_ellipse behavior and ensures we have finite positive values.
    - Optionally apply a per-obscode systematic term in quadrature:
        sigma_eff = sqrt(sigma^2 + sigma_sys^2)
      but only when the (filled) reported sigma RMS is below a per-obscode threshold.
    """
    ra_sig = np.asarray(candidates.ra_sigma_deg.to_numpy(zero_copy_only=False), dtype=np.float64)
    dec_sig = np.asarray(candidates.dec_sigma_deg.to_numpy(zero_copy_only=False), dtype=np.float64)

    # Fill invalids with floor (in degrees), consistent with innov_ellipse policy.
    floor = _det_sigma_floor_arcsec_by_candidate(candidates=candidates, gate=gate)
    if np.ndim(floor) == 0:
        floor_deg = (float(floor) / 3600.0) if float(floor) > 0.0 else 0.0
    else:
        floor_arr = np.asarray(floor, dtype=np.float64)
        floor_deg = np.where(np.isfinite(floor_arr) & (floor_arr > 0.0), floor_arr / 3600.0, 0.0)

    ra_ok = np.isfinite(ra_sig) & (ra_sig > 0.0)
    dec_ok = np.isfinite(dec_sig) & (dec_sig > 0.0)
    ra_sig = np.where(ra_ok, ra_sig, floor_deg)
    dec_sig = np.where(dec_ok, dec_sig, floor_deg)

    sys_map = gate.sigma_systematic_arcsec_by_obscode
    trig_map = gate.apply_systematic_if_reported_rms_lt_arcsec_by_obscode
    if not sys_map or not trig_map:
        return ra_sig, dec_sig

    obsc = candidates.table.column("obscode").combine_chunks()
    obsc_np = np.asarray(obsc.to_numpy(zero_copy_only=False), dtype=object)

    # Intuition: some stations occasionally report sigmas that are *too optimistic* (or otherwise
    # systematically miscalibrated) relative to the actual orbit-prediction residuals. In that
    # regime the Mahalanobis gate becomes overconfident and rejects truth at multi-arcsecond
    # offsets. A small per-obscode systematic term added in quadrature is a targeted way to
    # restore calibration without globally loosening the gate for all detections.
    #
    # Vectorized mapping obscode -> sys/trig (default 0 / -inf so it never applies).
    sys_arc = np.zeros(len(candidates), dtype=np.float64)
    trig_arc = np.full(len(candidates), -np.inf, dtype=np.float64)
    for code, sys in sys_map.items():
        if code not in trig_map:
            continue
        m = obsc_np == code
        if not np.any(m):
            continue
        sys_arc[m] = float(sys)
        trig_arc[m] = float(trig_map[code])

    if not np.any(sys_arc > 0):
        return ra_sig, dec_sig

    rms_arc = np.sqrt(ra_sig * ra_sig + dec_sig * dec_sig) * 3600.0
    apply = (sys_arc > 0.0) & np.isfinite(trig_arc) & (rms_arc < trig_arc)
    if not np.any(apply):
        return ra_sig, dec_sig

    sys_deg = (sys_arc / 3600.0).astype(np.float64, copy=False)
    ra_sig = np.where(apply, np.sqrt(ra_sig * ra_sig + sys_deg * sys_deg), ra_sig)
    dec_sig = np.where(apply, np.sqrt(dec_sig * dec_sig + sys_deg * sys_deg), dec_sig)
    return ra_sig, dec_sig


@dataclass(frozen=True)
class GateTotals:
    n_candidates: int
    n_accepted: int
    n_rejected_innov_ellipse: int
    n_rejected_mag_residual: int


def _gate_keep_and_reject_masks(
    *,
    candidates: CandidateDetections,
    preds: PredictedTargets,
    gate: GateParams,
    timings: dict[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (keep_innov_ellipse, rejected_innov_ellipse, rejected_mag_residual, keep_final)
    boolean masks aligned to `candidates`.
    """
    if len(candidates) == 0:
        z = np.zeros(0, dtype=bool)
        return z, z, z, z
    if len(preds) == 0:
        raise ValueError("preds is empty but candidates is not")
    need_mag_residual_gate = (
        gate.max_mag_residual_fainter_mag is not None
        or gate.max_mag_residual_brighter_mag is not None
    )

    # Vectorized mapping from candidates -> prediction columns via Arrow join.
    cand_keys = candidates.table.select(["orbit_id", "target_idx"]).append_column(
        "_row_idx",
        pa.array(np.arange(len(candidates), dtype=np.int64), type=pa.int64()),
    )
    preds_cols = [
        "orbit_id",
        "target_idx",
        "pred_lon_deg",
        "pred_lat_deg",
        "cov_ll_00",
        "cov_ll_01",
        "cov_ll_11",
    ]
    if need_mag_residual_gate:
        preds_cols.append("pred_mag")
    preds_sub = preds.table.select([c for c in preds_cols if c in preds.table.column_names])
    joined = cand_keys.join(preds_sub, keys=["orbit_id", "target_idx"], join_type="left outer")
    if int(joined.num_rows) != int(len(candidates)):
        raise RuntimeError(
            "Failed to align candidates to preds via join. "
            f"cand_rows={int(len(candidates))} joined_rows={int(joined.num_rows)}"
        )
    joined = joined.sort_by([("_row_idx", "ascending")])
    if bool(pc.any(pc.is_null(joined["pred_lon_deg"])).as_py()):
        raise ValueError("Candidates contain (orbit_id,target_idx) not present in preds")

    lon0 = pc.cast(joined["pred_lon_deg"], pa.float64()).to_numpy(zero_copy_only=False).astype(
        np.float64, copy=False
    )
    lat0 = pc.cast(joined["pred_lat_deg"], pa.float64()).to_numpy(zero_copy_only=False).astype(
        np.float64, copy=False
    )

    c00 = pc.cast(joined["cov_ll_00"], pa.float64()).to_numpy(zero_copy_only=False).astype(
        np.float64, copy=False
    )
    c01 = pc.cast(joined["cov_ll_01"], pa.float64()).to_numpy(zero_copy_only=False).astype(
        np.float64, copy=False
    )
    c11 = pc.cast(joined["cov_ll_11"], pa.float64()).to_numpy(zero_copy_only=False).astype(
        np.float64, copy=False
    )
    cov = np.stack([c00, c01, c01, c11], axis=1).reshape(-1, 2, 2).astype(np.float64)

    # Innovation-ellipse gate first, then magnitude outlier rejection.
    t_innov0 = None if timings is None else time.perf_counter()
    ra_sig_eff, dec_sig_eff = _inflate_candidate_sigmas_for_gate(candidates=candidates, gate=gate)
    keep_innov = innov_ellipse_keep_mask(
        obs_lon_deg=candidates.ra_deg.to_numpy(zero_copy_only=False),
        obs_lat_deg=candidates.dec_deg.to_numpy(zero_copy_only=False),
        obs_lon_sigma_deg=ra_sig_eff,
        obs_lat_sigma_deg=dec_sig_eff,
        pred_lon_deg=lon0,
        pred_lat_deg=lat0,
        pred_cov_ll_deg2=cov,
        n_sigma=float(gate.innovation_gate_n_sigma),
    )
    keep_innov = np.asarray(keep_innov, dtype=bool)
    if timings is not None and t_innov0 is not None:
        dt = time.perf_counter() - t_innov0
        timings["gate.innov_ellipse_elapsed_s"] = timings.get("gate.innov_ellipse_elapsed_s", 0.0) + dt

    rejected_innov = ~keep_innov

    rejected_mag = np.zeros(len(candidates), dtype=bool)
    idx = np.nonzero(keep_innov)[0]
    if idx.size > 0:
        max_faint = gate.max_mag_residual_fainter_mag
        max_bright = gate.max_mag_residual_brighter_mag
        t_mag0 = None if timings is None else time.perf_counter()
        if need_mag_residual_gate:
            nan = pa.scalar(np.nan, type=pa.float64())
            cand_mag = pc.fill_null(pc.cast(candidates.mag, pa.float64()), nan).to_numpy(
                zero_copy_only=False
            )
            pred_mag = pc.fill_null(pc.cast(joined["pred_mag"], pa.float64()), nan).to_numpy(
                zero_copy_only=False
            )
            resid = cand_mag[idx] - pred_mag[idx]
            finite = np.isfinite(resid)
            rej_sub = np.zeros(int(idx.size), dtype=bool)
            if max_faint is not None:
                rej_sub |= finite & (resid > float(max_faint))
            if max_bright is not None:
                rej_sub |= finite & (resid < -float(max_bright))
            rejected_mag[idx] = np.asarray(rej_sub, dtype=bool)
        if timings is not None and t_mag0 is not None:
            dt = time.perf_counter() - t_mag0
            timings["gate.mag_residual_elapsed_s"] = timings.get("gate.mag_residual_elapsed_s", 0.0) + dt

    keep_final = keep_innov & (~rejected_mag)
    return keep_innov, rejected_innov, rejected_mag, keep_final


def gate_keep_and_reject_masks_python(
    *,
    candidates: CandidateDetections,
    preds: PredictedTargets,
    gate: GateParams,
    timings: dict[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return boolean masks aligned to `candidates`:

    - keep_innov_ellipse
    - rejected_innov_ellipse
    - rejected_mag_residual
    - keep_final
    """
    return _gate_keep_and_reject_masks(
        candidates=candidates, preds=preds, gate=gate, timings=timings
    )


def gate_totals_python(
    *,
    candidates: CandidateDetections,
    preds: PredictedTargets,
    gate: GateParams,
    timings: dict[str, float] | None = None,
) -> GateTotals:
    _keep_innov, rej_innov, rej_mag, keep_final = _gate_keep_and_reject_masks(
        candidates=candidates, preds=preds, gate=gate, timings=timings
    )
    return GateTotals(
        n_candidates=int(len(candidates)),
        n_accepted=int(np.count_nonzero(keep_final)),
        n_rejected_innov_ellipse=int(np.count_nonzero(rej_innov)),
        n_rejected_mag_residual=int(np.count_nonzero(rej_mag)),
    )


def accepted_observation_ids_and_totals_python(
    *,
    candidates: CandidateDetections,
    preds: PredictedTargets,
    gate: GateParams,
    timings: dict[str, float] | None = None,
) -> tuple[pa.Table, GateTotals]:
    """
    Return accepted rows and overall rejection totals.
    """
    _keep_innov, rej_innov, rej_mag, keep_final = _gate_keep_and_reject_masks(
        candidates=candidates, preds=preds, gate=gate, timings=timings
    )
    if len(candidates) == 0:
        accepted = pa.table(
            {
                "orbit_id": pa.array([], type=pa.large_string()),
                "target_idx": pa.array([], type=pa.int64()),
                "observation_id": pa.array([], type=pa.large_string()),
            }
        )
    else:
        t = candidates.table.select(["orbit_id", "target_idx", "observation_id"])
        t = t.append_column("_keep", pa.array(keep_final))
        accepted = t.filter(pc.field("_keep")).select(
            ["orbit_id", "target_idx", "observation_id"]
        )
    totals = GateTotals(
        n_candidates=int(len(candidates)),
        n_accepted=int(np.count_nonzero(keep_final)),
        n_rejected_innov_ellipse=int(np.count_nonzero(rej_innov)),
        n_rejected_mag_residual=int(np.count_nonzero(rej_mag)),
    )
    return accepted, totals


def accepted_counts_python(
    *,
    candidates: CandidateDetections,
    preds: PredictedTargets,
    gate: GateParams,
    timings: dict[str, float] | None = None,
) -> AcceptedCounts:
    """
    Compute accepted counts using the production `innov_ellipse_keep_mask` in Python.
    """
    if len(candidates) == 0:
        return AcceptedCounts.empty()
    _keep_innov, _rej_innov, _rej_mag, keep_final = _gate_keep_and_reject_masks(
        candidates=candidates, preds=preds, gate=gate, timings=timings
    )

    t = candidates.table
    t = t.append_column("_keep", pa.array(keep_final))
    gb = t.group_by(["orbit_id", "target_idx"]).aggregate(
        [
            ("observation_id", "count"),
            ("_keep", "sum"),
        ]
    )
    gb = gb.rename_columns(["orbit_id", "target_idx", "n_candidates", "n_accepted"])
    return AcceptedCounts.from_pyarrow(gb)


def accepted_counts_and_totals_python(
    *,
    candidates: CandidateDetections,
    preds: PredictedTargets,
    gate: GateParams,
    timings: dict[str, float] | None = None,
) -> tuple[AcceptedCounts, GateTotals]:
    if len(candidates) == 0:
        return AcceptedCounts.empty(), GateTotals(0, 0, 0, 0)

    _keep_innov, rej_innov, rej_mag, keep_final = _gate_keep_and_reject_masks(
        candidates=candidates, preds=preds, gate=gate, timings=timings
    )
    t = candidates.table
    t = t.append_column("_keep", pa.array(keep_final))
    gb = t.group_by(["orbit_id", "target_idx"]).aggregate(
        [
            ("observation_id", "count"),
            ("_keep", "sum"),
        ]
    )
    gb = gb.rename_columns(["orbit_id", "target_idx", "n_candidates", "n_accepted"])
    totals = GateTotals(
        n_candidates=int(len(candidates)),
        n_accepted=int(np.count_nonzero(keep_final)),
        n_rejected_innov_ellipse=int(np.count_nonzero(rej_innov)),
        n_rejected_mag_residual=int(np.count_nonzero(rej_mag)),
    )
    return AcceptedCounts.from_pyarrow(gb), totals


def accepted_observation_ids_python(
    *,
    candidates: CandidateDetections,
    preds: PredictedTargets,
    gate: GateParams,
    timings: dict[str, float] | None = None,
) -> pa.Table:
    """
    Return accepted (orbit_id, target_idx, observation_id) rows (for truth scoring).
    """
    accepted, _ = accepted_observation_ids_and_totals_python(
        candidates=candidates, preds=preds, gate=gate, timings=timings
    )
    return accepted


def gate_breakdown_masks_python(
    *,
    candidates: CandidateDetections,
    preds: PredictedTargets,
    gate: GateParams,
    timings: dict[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return per-candidate masks aligned to `candidates`:

    - keep_innov_ellipse
    - rejected_mag_residual (only evaluated on innovation-ellipse survivors)
    - keep_final (innov_ellipse and not rejected by mag residual)
    """
    keep_innov, _rej_innov, rej_mag, keep_final = _gate_keep_and_reject_masks(
        candidates=candidates, preds=preds, gate=gate, timings=timings
    )
    return keep_innov, rej_mag, keep_final
