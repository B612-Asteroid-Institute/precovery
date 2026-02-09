from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyarrow as pa
import quivr as qv
from adam_core.orbits import Orbits


class CovarianceSeverity(qv.Table):
    """
    Simple covariance magnitude/shape summaries computed from SBDB 6×6 covariance.

    These are intended for *stratification*, not for exact on-sky uncertainty prediction.
    """

    orbit_id = qv.LargeStringColumn()

    cov_ok = qv.BooleanColumn()
    cov_invalid_reason = qv.LargeStringColumn(nullable=True)

    # Position block severity.
    sigma_pos_rms = qv.Float64Column(nullable=True)
    sigma_pos_max = qv.Float64Column(nullable=True)
    anisotropy_pos = qv.Float64Column(nullable=True)

    # Velocity block severity.
    sigma_vel_rms = qv.Float64Column(nullable=True)
    sigma_vel_max = qv.Float64Column(nullable=True)
    anisotropy_vel = qv.Float64Column(nullable=True)


def _eig_stats_psd(m: np.ndarray) -> tuple[float, float, float] | None:
    """
    Return (sigma_rms, sigma_max, anisotropy) for a PSD-ish 3×3 covariance block.
    """
    mm = np.asarray(m, dtype=np.float64)
    if mm.shape != (3, 3):
        return None
    mm = 0.5 * (mm + mm.T)
    try:
        w = np.linalg.eigvalsh(mm)
    except Exception:
        return None
    if not np.isfinite(w).all():
        return None
    # Clamp negative eigenvalues (numerical noise).
    w = np.maximum(w, 0.0)
    sig_rms = float(np.sqrt(float(np.sum(w))))
    sig_max = float(np.sqrt(float(np.max(w))))
    w_min = float(np.min(w))
    w_max = float(np.max(w))
    denom = max(w_min, 1e-30)
    anis = float(np.sqrt(w_max / denom)) if w_max > 0 else 1.0
    return sig_rms, sig_max, anis


def compute_covariance_severity(*, orbits: Orbits) -> CovarianceSeverity:
    """
    Compute covariance severity metrics for each orbit in `orbits`.
    """
    orbit_id = [str(x) for x in orbits.orbit_id.to_pylist()]

    cov = getattr(orbits.coordinates, "covariance", None)
    if cov is None:
        return CovarianceSeverity.from_kwargs(
            orbit_id=orbit_id,
            cov_ok=[False] * len(orbit_id),
            cov_invalid_reason=["missing_covariance"] * len(orbit_id),
            sigma_pos_rms=[None] * len(orbit_id),
            sigma_pos_max=[None] * len(orbit_id),
            anisotropy_pos=[None] * len(orbit_id),
            sigma_vel_rms=[None] * len(orbit_id),
            sigma_vel_max=[None] * len(orbit_id),
            anisotropy_vel=[None] * len(orbit_id),
        )

    try:
        m = cov.to_matrix().astype(np.float64)
    except Exception:
        return CovarianceSeverity.from_kwargs(
            orbit_id=orbit_id,
            cov_ok=[False] * len(orbit_id),
            cov_invalid_reason=["covariance_to_matrix_failed"] * len(orbit_id),
            sigma_pos_rms=[None] * len(orbit_id),
            sigma_pos_max=[None] * len(orbit_id),
            anisotropy_pos=[None] * len(orbit_id),
            sigma_vel_rms=[None] * len(orbit_id),
            sigma_vel_max=[None] * len(orbit_id),
            anisotropy_vel=[None] * len(orbit_id),
        )

    if m.ndim != 3 or m.shape[1:] != (6, 6):
        raise ValueError(f"Unexpected covariance matrix shape: {m.shape}")

    out_cov_ok: list[bool] = []
    out_reason: list[str | None] = []
    sp_rms: list[float | None] = []
    sp_max: list[float | None] = []
    ap: list[float | None] = []
    sv_rms: list[float | None] = []
    sv_max: list[float | None] = []
    av: list[float | None] = []

    for i in range(m.shape[0]):
        mi = np.asarray(m[i], dtype=np.float64)
        if not np.isfinite(mi).all():
            out_cov_ok.append(False)
            out_reason.append("non_finite")
            sp_rms.append(None)
            sp_max.append(None)
            ap.append(None)
            sv_rms.append(None)
            sv_max.append(None)
            av.append(None)
            continue

        pos = mi[0:3, 0:3]
        vel = mi[3:6, 3:6]
        ps = _eig_stats_psd(pos)
        vs = _eig_stats_psd(vel)
        if ps is None or vs is None:
            out_cov_ok.append(False)
            out_reason.append("eig_failed")
            sp_rms.append(None)
            sp_max.append(None)
            ap.append(None)
            sv_rms.append(None)
            sv_max.append(None)
            av.append(None)
            continue

        out_cov_ok.append(True)
        out_reason.append(None)
        sp_rms.append(float(ps[0]))
        sp_max.append(float(ps[1]))
        ap.append(float(ps[2]))
        sv_rms.append(float(vs[0]))
        sv_max.append(float(vs[1]))
        av.append(float(vs[2]))

    return CovarianceSeverity.from_kwargs(
        orbit_id=orbit_id,
        cov_ok=out_cov_ok,
        cov_invalid_reason=out_reason,
        sigma_pos_rms=sp_rms,
        sigma_pos_max=sp_max,
        anisotropy_pos=ap,
        sigma_vel_rms=sv_rms,
        sigma_vel_max=sv_max,
        anisotropy_vel=av,
    )

