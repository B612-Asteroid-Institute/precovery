from __future__ import annotations

import math

import numpy as np

from precovery.observation import ObservationsTable


def cov_xy_deg2_from_cov_ll_deg2(*, cov_ll_deg2: np.ndarray, lat0_deg: float) -> np.ndarray:
    """
    Convert lon/lat covariance (deg^2) into local tangent-plane covariance (deg^2) using:

      x = Δlon * cos(lat0)
      y = Δlat
    """
    cov_ll = np.asarray(cov_ll_deg2, dtype=np.float64)
    if cov_ll.shape != (2, 2):
        raise ValueError(f"cov_ll_deg2 must be shape (2,2), got {cov_ll.shape}")
    cos_lat = float(np.cos(np.deg2rad(float(lat0_deg))))
    cos_lat = cos_lat if np.isfinite(cos_lat) and abs(cos_lat) > 1e-12 else 1e-12
    A = np.array([[cos_lat, 0.0], [0.0, 1.0]], dtype=np.float64)
    cov_xy = A @ cov_ll @ A.T
    return 0.5 * (cov_xy + cov_xy.T)


def sigma_major_arcsec_from_cov_ll_deg2(*, cov_ll_deg2: np.ndarray, lat0_deg: float) -> float:
    """
    1-sigma major-axis uncertainty (arcsec) in the local tangent plane.
    """
    cov_xy = cov_xy_deg2_from_cov_ll_deg2(cov_ll_deg2=cov_ll_deg2, lat0_deg=float(lat0_deg))
    w = np.linalg.eigvalsh(cov_xy)
    w = np.maximum(w, 0.0)
    sig_deg = float(np.sqrt(float(np.max(w))))
    return float(sig_deg * 3600.0)


def ellipse_area_deg2_from_cov_ll_deg2(
    *, cov_ll_deg2: np.ndarray, lat0_deg: float, n_sigma: float
) -> float:
    """
    Area (deg^2) of the N-sigma ellipse implied by a 2x2 sky-plane covariance.

    For x ~ N(0, C), the set {x | x^T C^{-1} x <= n_sigma^2} has area:
        area = π * n_sigma^2 * sqrt(det(C))
    """
    sig = float(n_sigma)
    if not np.isfinite(sig) or sig <= 0:
        raise ValueError(f"n_sigma must be finite and > 0, got {n_sigma!r}")
    cov_xy = cov_xy_deg2_from_cov_ll_deg2(cov_ll_deg2=cov_ll_deg2, lat0_deg=float(lat0_deg))
    det = float(np.linalg.det(cov_xy))
    det = max(det, 0.0)
    return float(math.pi * (sig * sig) * math.sqrt(det))


def bytes_per_observation(*, mean_id_len_bytes: float = 16.0) -> float:
    """
    Approximate bytes-per-observation in `.data` frame blobs.

    Each record is:
      - fixed-size numeric payload of `ObservationsTable.datagram_size`, plus
      - variable-length `id` bytes.
    """
    m = float(mean_id_len_bytes)
    if not np.isfinite(m) or m < 0:
        raise ValueError(f"mean_id_len_bytes must be finite and >= 0, got {mean_id_len_bytes!r}")
    return float(ObservationsTable.datagram_size + m)


def expected_observations_from_bytes(*, data_length_bytes: float, bytes_per_obs: float) -> float:
    """
    Convert `frames.data_length` bytes into an expected observation count.

    This is approximate (ID length varies), but can be useful for human-friendly thresholds
    like “~1 candidate per exposure”.
    """
    b = float(bytes_per_obs)
    if not np.isfinite(b) or b <= 0:
        raise ValueError(f"bytes_per_obs must be finite and > 0, got {bytes_per_obs!r}")
    x = float(data_length_bytes)
    if not np.isfinite(x) or x < 0:
        return float("nan")
    return float(x / b)

