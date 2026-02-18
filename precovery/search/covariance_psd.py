from __future__ import annotations

from typing import Final

import numpy as np


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

