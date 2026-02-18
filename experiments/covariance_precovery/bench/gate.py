from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from precovery.search.detection_filter import innov_ellipse_keep_mask

from .types import AcceptedCounts, CandidateDetections, PredictedTargets
from .backends.protocols import GateParams


def _key(orbit_id: str, target_idx: int) -> tuple[str, int]:
    return str(orbit_id), int(target_idx)


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (keep_final, rejected_innov_ellipse, rejected_mag_residual) boolean masks
    aligned to `candidates`.
    """
    if len(candidates) == 0:
        z = np.zeros(0, dtype=bool)
        return z, z, z

    if len(preds) == 0:
        raise ValueError("preds is empty but candidates is not")

    pred_orb = [str(x) for x in preds.orbit_id.to_pylist()]
    pred_t = preds.target_idx.to_numpy(zero_copy_only=False).astype(np.int64)
    idx_by_key: dict[tuple[str, int], int] = {}
    for i, (oid, tidx) in enumerate(zip(pred_orb, pred_t.tolist())):
        idx_by_key[_key(oid, int(tidx))] = int(i)

    cand_orb = np.asarray([str(x) for x in candidates.orbit_id.to_pylist()], dtype=object)
    cand_t = candidates.target_idx.to_numpy(zero_copy_only=False).astype(np.int64)

    pred_idx = np.full(len(candidates), -1, dtype=np.int64)
    for i in range(len(candidates)):
        pred_idx[i] = int(idx_by_key.get(_key(str(cand_orb[i]), int(cand_t[i])), -1))
    if np.any(pred_idx < 0):
        raise ValueError("Candidates contain (orbit_id,target_idx) not present in preds")

    lon0 = preds.pred_lon_deg.to_numpy(zero_copy_only=False).astype(np.float64)[pred_idx]
    lat0 = preds.pred_lat_deg.to_numpy(zero_copy_only=False).astype(np.float64)[pred_idx]

    c00 = preds.cov_ll_00.to_numpy(zero_copy_only=False).astype(np.float64)[pred_idx]
    c01 = preds.cov_ll_01.to_numpy(zero_copy_only=False).astype(np.float64)[pred_idx]
    c11 = preds.cov_ll_11.to_numpy(zero_copy_only=False).astype(np.float64)[pred_idx]
    cov = np.stack([c00, c01, c01, c11], axis=1).reshape(-1, 2, 2).astype(np.float64)

    # Apply magnitude residual rejection first (cheap scalar compare), then only run
    # the more expensive innovation-ellipse gate on survivors.
    rejected_mag = np.zeros(len(candidates), dtype=bool)
    max_faint = gate.max_mag_residual_fainter_mag
    max_bright = gate.max_mag_residual_brighter_mag
    if (max_faint is not None) or (max_bright is not None):
        nan = pa.scalar(np.nan, type=pa.float64())
        cand_mag = pc.fill_null(pc.cast(candidates.mag, pa.float64()), nan).to_numpy(
            zero_copy_only=False
        )
        pred_mag_all = pc.fill_null(pc.cast(preds.pred_mag, pa.float64()), nan).to_numpy(
            zero_copy_only=False
        )
        pred_mag = pred_mag_all[pred_idx]
        resid = cand_mag - pred_mag
        finite = np.isfinite(resid)
        if max_faint is not None:
            rejected_mag |= finite & (resid > float(max_faint))
        if max_bright is not None:
            rejected_mag |= finite & (resid < -float(max_bright))

    keep_after_mag = ~rejected_mag
    idx = np.nonzero(keep_after_mag)[0]
    keep_geom = np.zeros(len(candidates), dtype=bool)
    if idx.size > 0:
        keep_geom_sub = innov_ellipse_keep_mask(
            obs_lon_deg=candidates.ra_deg.to_numpy(zero_copy_only=False)[idx],
            obs_lat_deg=candidates.dec_deg.to_numpy(zero_copy_only=False)[idx],
            obs_lon_sigma_deg=candidates.ra_sigma_deg.to_numpy(zero_copy_only=False)[idx],
            obs_lat_sigma_deg=candidates.dec_sigma_deg.to_numpy(zero_copy_only=False)[idx],
            pred_lon_deg=lon0[idx],
            pred_lat_deg=lat0[idx],
            pred_cov_ll_deg2=cov[idx, :, :],
            n_sigma=float(gate.n_sigma),
            det_sigma_floor_arcsec=float(gate.det_sigma_floor_arcsec),
        )
        keep_geom[idx] = np.asarray(keep_geom_sub, dtype=bool)

    keep_final = keep_after_mag & keep_geom
    # Non-overlapping reject reasons:
    rejected_innov = keep_after_mag & (~keep_geom)
    return keep_final, rejected_innov, rejected_mag


def gate_totals_python(
    *,
    candidates: CandidateDetections,
    preds: PredictedTargets,
    gate: GateParams,
) -> GateTotals:
    keep, rej_innov, rej_mag = _gate_keep_and_reject_masks(candidates=candidates, preds=preds, gate=gate)
    return GateTotals(
        n_candidates=int(len(candidates)),
        n_accepted=int(np.count_nonzero(keep)),
        n_rejected_innov_ellipse=int(np.count_nonzero(rej_innov)),
        n_rejected_mag_residual=int(np.count_nonzero(rej_mag)),
    )


def accepted_observation_ids_and_totals_python(
    *,
    candidates: CandidateDetections,
    preds: PredictedTargets,
    gate: GateParams,
) -> tuple[pa.Table, GateTotals]:
    """
    Return accepted rows and overall rejection totals.
    """
    keep, rej_innov, rej_mag = _gate_keep_and_reject_masks(candidates=candidates, preds=preds, gate=gate)
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
        t = t.append_column("_keep", pa.array(keep))
        accepted = t.filter(pc.field("_keep")).select(["orbit_id", "target_idx", "observation_id"])
    totals = GateTotals(
        n_candidates=int(len(candidates)),
        n_accepted=int(np.count_nonzero(keep)),
        n_rejected_innov_ellipse=int(np.count_nonzero(rej_innov)),
        n_rejected_mag_residual=int(np.count_nonzero(rej_mag)),
    )
    return accepted, totals


def accepted_counts_python(
    *,
    candidates: CandidateDetections,
    preds: PredictedTargets,
    gate: GateParams,
) -> AcceptedCounts:
    """
    Compute accepted counts using the production `innov_ellipse_keep_mask` in Python.

    This is the reference implementation used when a backend cannot (or should not)
    push down gating to SQL.
    """
    if len(candidates) == 0:
        return AcceptedCounts.empty()
    keep, _, _ = _gate_keep_and_reject_masks(candidates=candidates, preds=preds, gate=gate)

    # Aggregate per key (vectorized-ish with Arrow).
    t = candidates.table
    t = t.append_column("_keep", pa.array(keep))
    # group by orbit_id,target_idx
    gb = t.group_by(["orbit_id", "target_idx"]).aggregate(
        [
            ("observation_id", "count"),
            ("_keep", "sum"),
        ]
    )
    # Rename columns to canonical output schema
    gb = gb.rename_columns(
        [
            "orbit_id",
            "target_idx",
            "n_candidates",
            "n_accepted",
        ]
    )
    return AcceptedCounts.from_pyarrow(gb)


def accepted_observation_ids_python(
    *,
    candidates: CandidateDetections,
    preds: PredictedTargets,
    gate: GateParams,
) -> pa.Table:
    """
    Return accepted (orbit_id, target_idx, observation_id) rows (for truth scoring).
    """
    accepted, _ = accepted_observation_ids_and_totals_python(candidates=candidates, preds=preds, gate=gate)
    return accepted

