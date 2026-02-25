from __future__ import annotations

from typing import Final

import numpy as np

from adam_core.coordinates.covariances import CoordinateCovariances
from adam_core.orbits import Orbits


def repair_covariance_matrix_psd(
    cov: np.ndarray,
    *,
    abs_tol: float = 1e-15,
    rel_tol: float = 1e-10,
    sym_tol: float = 1e-12,
) -> tuple[np.ndarray | None, bool]:
    """
    Repair a 6x6 covariance matrix to be PSD-enough for sampling.

    Returns
    -------
    (cov_out, modified)
      - cov_out is None when the matrix is too non-PSD to safely repair.
      - modified indicates whether the returned matrix differs from the input.

    Policy
    ------
    - Do not change matrices that are already PSD and symmetric-enough.
    - If the most-negative eigenvalue is only slightly negative (within tolerance),
      clip negative eigenvalues to 0.
    - Enforce symmetry after reconstruction.
    - If strict eigenvalue checks still see a tiny negative, add the minimum diagonal
      jitter needed to push the smallest eigenvalue above zero (plus a small epsilon).
    """
    c_raw = np.asarray(cov, dtype=np.float64)
    if c_raw.shape != (6, 6):
        raise ValueError(f"cov must be shape (6,6), got {c_raw.shape}")
    if not np.isfinite(c_raw).all():
        return None, False

    sym_err = float(np.max(np.abs(c_raw - c_raw.T)))
    if sym_err <= float(sym_tol):
        c_sym = c_raw
        modified = False
    else:
        c_sym = 0.5 * (c_raw + c_raw.T)
        modified = True

    try:
        w, v = np.linalg.eigh(c_sym)
    except Exception:  # noqa: BLE001
        return None, False
    if not np.isfinite(w).all():
        return None, False

    w_min = float(w.min())
    w_max = float(w.max())
    tol = float(max(float(abs_tol), float(rel_tol) * float(w_max)))
    if w_min < -tol:
        return None, False

    c_out = c_sym
    if w_min < 0.0:
        w = np.where(w < 0.0, 0.0, w)
        c_out = (v * w) @ v.T
        modified = True

    if modified:
        c_out = 0.5 * (c_out + c_out.T)

    min_ev = float(np.min(np.real(np.linalg.eigvals(c_out))))
    if min_ev < 0.0:
        eps: Final[float] = float(abs_tol) * 10.0
        c_out = c_out + np.eye(6, dtype=np.float64) * (-min_ev + eps)
        modified = True

    return c_out.astype(np.float64, copy=False), modified


def repair_covariance_matrix_psd_with_reason(
    cov: np.ndarray,
    *,
    abs_tol: float = 1e-15,
    rel_tol: float = 1e-10,
    sym_tol: float = 1e-12,
) -> tuple[np.ndarray | None, bool, str | None]:
    """
    Like `repair_covariance_matrix_psd`, but also returns a reason when repair fails.

    reason values are stable strings intended for logging/metrics:
      - "non_finite"
      - "eigh_failed"
      - "non_finite_eigenvalues"
      - "too_non_psd"
    """
    c_raw = np.asarray(cov, dtype=np.float64)
    if c_raw.shape != (6, 6):
        raise ValueError(f"cov must be shape (6,6), got {c_raw.shape}")
    if not np.isfinite(c_raw).all():
        return None, False, "non_finite"

    sym_err = float(np.max(np.abs(c_raw - c_raw.T)))
    if sym_err <= float(sym_tol):
        c_sym = c_raw
        modified = False
    else:
        c_sym = 0.5 * (c_raw + c_raw.T)
        modified = True

    try:
        w, v = np.linalg.eigh(c_sym)
    except Exception:  # noqa: BLE001
        return None, False, "eigh_failed"
    if not np.isfinite(w).all():
        return None, False, "non_finite_eigenvalues"

    w_min = float(w.min())
    w_max = float(w.max())
    tol = float(max(float(abs_tol), float(rel_tol) * float(w_max)))
    if w_min < -tol:
        return None, False, "too_non_psd"

    c_out = c_sym
    if w_min < 0.0:
        w = np.where(w < 0.0, 0.0, w)
        c_out = (v * w) @ v.T
        modified = True

    if modified:
        c_out = 0.5 * (c_out + c_out.T)

    min_ev = float(np.min(np.real(np.linalg.eigvals(c_out))))
    if min_ev < 0.0:
        eps: Final[float] = float(abs_tol) * 10.0
        c_out = c_out + np.eye(6, dtype=np.float64) * (-min_ev + eps)
        modified = True

    return c_out.astype(np.float64, copy=False), modified, None


def repair_orbits_covariance_psd_for_sampling(
    orbits: Orbits,
    *,
    abs_tol: float = 1e-15,
    rel_tol: float = 1e-10,
    sym_tol: float = 1e-12,
) -> tuple[Orbits, int, int]:
    """
    Repair/reject orbit covariances so sampling (e.g. sigma points) is safe.

    Returns
    -------
    (orbits_out, n_repaired, n_rejected)
      - orbits_out: subset of input orbits with repaired covariances (if needed).
      - n_repaired: number of orbits whose covariance matrix was modified.
      - n_rejected: number of orbits dropped due to invalid/non-PSD covariance.
    """
    if len(orbits) == 0:
        return orbits, 0, 0

    cov = getattr(orbits.coordinates, "covariance", None)
    if cov is None or cov.is_all_nan():
        return Orbits.empty(), 0, int(len(orbits))

    m_raw = cov.to_matrix().astype(np.float64, copy=False)

    keep: list[int] = []
    cov_out: list[np.ndarray] = []
    n_repaired = 0
    n_rejected = 0

    for i in range(int(m_raw.shape[0])):
        c = m_raw[i]
        if not np.isfinite(c).all():
            n_rejected += 1
            continue
        c_out, modified, _reason = repair_covariance_matrix_psd_with_reason(
            c,
            abs_tol=float(abs_tol),
            rel_tol=float(rel_tol),
            sym_tol=float(sym_tol),
        )
        if c_out is None:
            n_rejected += 1
            continue
        if bool(modified):
            n_repaired += 1
        keep.append(int(i))
        cov_out.append(c_out.astype(np.float64, copy=False))

    if not keep:
        return Orbits.empty(), int(n_repaired), int(n_rejected)

    out = orbits.take(keep)
    cov_fixed = np.stack(cov_out, axis=0)
    out = out.set_column(
        "coordinates.covariance",
        CoordinateCovariances.from_matrix(cov_fixed),
    )
    return out, int(n_repaired), int(n_rejected)

