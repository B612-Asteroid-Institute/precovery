from __future__ import annotations

import numpy as np


def wrap_delta_lon_deg(lon_deg: np.ndarray, lon0_deg: np.ndarray) -> np.ndarray:
    """
    Wrap longitude residuals into [-180, 180) degrees.

    Parameters
    ----------
    lon_deg
        Longitudes in degrees.
    lon0_deg
        Reference longitudes in degrees (broadcastable to lon_deg).
    """
    return (lon_deg - lon0_deg + 180.0) % 360.0 - 180.0


def reconstruct_cov_ll_from_sigma_point_cloud(
    *,
    lon0_deg: np.ndarray,  # (N,)
    lat0_deg: np.ndarray,  # (N,)
    lon_samples_deg: np.ndarray,  # (K,N)
    lat_samples_deg: np.ndarray,  # (K,N)
    weights_cov: np.ndarray,  # (K,)
) -> np.ndarray:
    """
    Reconstruct a (lon,lat) covariance from a sigma-point ephemeris cloud.

    Why this exists
    ---------------
    - We want the mean prediction to remain the nominal/mean ephemeris (lon0,lat0).
    - We want the covariance to reflect nonlinear effects captured by the sigma-point cloud.
    - We keep memory bounded by avoiding large VariantEphemeris objects.

    Notes
    -----
    - The covariance is computed in degrees^2 in the (lon,lat) basis, with wrap-safe
      longitude deltas relative to the nominal longitude.
    - The returned covariance is symmetric and shaped (N,2,2).
    """
    lon0 = np.asarray(lon0_deg, dtype=np.float64)
    lat0 = np.asarray(lat0_deg, dtype=np.float64)
    lon_s = np.asarray(lon_samples_deg, dtype=np.float64)
    lat_s = np.asarray(lat_samples_deg, dtype=np.float64)
    w = np.asarray(weights_cov, dtype=np.float64).reshape(-1)  # (K,)

    if lon_s.ndim != 2 or lat_s.ndim != 2:
        raise ValueError("lon_samples_deg and lat_samples_deg must be 2D arrays (K,N).")
    if lon_s.shape != lat_s.shape:
        raise ValueError("lon_samples_deg and lat_samples_deg must have matching shape (K,N).")
    if lon_s.shape[1] != lon0.shape[0] or lon0.shape != lat0.shape:
        raise ValueError("Nominal lon0/lat0 must be shape (N,) matching sample second dimension.")
    if w.shape[0] != lon_s.shape[0]:
        raise ValueError("weights_cov must be shape (K,) matching sample first dimension.")

    if not np.isfinite(w).all():
        raise ValueError("weights_cov contains non-finite values.")
    wsum = float(np.sum(w))
    if not np.isfinite(wsum) or wsum == 0.0:
        raise ValueError("weights_cov sum must be finite and non-zero.")
    # Sigma-point covariance weights should already sum to ~1; normalize defensively.
    w = w / wsum
    w2 = w.reshape(-1, 1)  # (K,1)

    dlon = wrap_delta_lon_deg(lon_s, lon0[None, :])
    dlat = lat_s - lat0[None, :]

    cov00 = np.sum(w2 * dlon * dlon, axis=0)  # (N,)
    cov11 = np.sum(w2 * dlat * dlat, axis=0)
    cov01 = np.sum(w2 * dlon * dlat, axis=0)

    cov = np.empty((lon0.shape[0], 2, 2), dtype=np.float64)
    cov[:, 0, 0] = cov00
    cov[:, 1, 1] = cov11
    cov[:, 0, 1] = cov01
    cov[:, 1, 0] = cov01
    # Symmetrize and clamp tiny negative diagonals from numeric noise.
    cov[:, 0, 0] = np.maximum(cov[:, 0, 0], 0.0)
    cov[:, 1, 1] = np.maximum(cov[:, 1, 1], 0.0)
    return cov

