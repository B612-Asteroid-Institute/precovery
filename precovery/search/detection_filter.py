from __future__ import annotations

import numpy as np

from .reconstruction import wrap_delta_lon_deg


def innov_ellipse_keep_mask(
    *,
    obs_lon_deg: np.ndarray,
    obs_lat_deg: np.ndarray,
    obs_lon_sigma_deg: np.ndarray,
    obs_lat_sigma_deg: np.ndarray,
    pred_lon_deg: np.ndarray,
    pred_lat_deg: np.ndarray,
    pred_cov_ll_deg2: np.ndarray,  # (N,2,2)
    n_sigma: float = 3.0,
    det_sigma_floor_arcsec: float = 0.10,
) -> np.ndarray:
    """
    Innovation-ellipse keep mask (Mahalanobis gate) in a local tangent plane.

    This is the detection-level filter `innov_ellipse@N` from the experiment harness.
    We combine predicted covariance (from reconstructed sigma-point covariances) with
    observational uncertainties and test:

      chi2 = Δᵀ (Σ_pred + Σ_obs)^(-1) Δ  <=  n_sigma^2

    with tangent-plane residuals:
      x = Δlon * cos(lat0)
      y = Δlat

    Parameters
    ----------
    obs_lon_deg, obs_lat_deg
        Observed sky positions in degrees.
    obs_lon_sigma_deg, obs_lat_sigma_deg
        1-sigma observational uncertainties in degrees (diagonal-only model).
    pred_lon_deg, pred_lat_deg
        Predicted sky positions in degrees.
    pred_cov_ll_deg2
        Predicted covariance in (lon,lat) degrees^2, shape (N,2,2).
    det_sigma_floor_arcsec
        Optional floor applied to observational sigmas to avoid pathological zeros.

    Returns
    -------
    keep : ndarray[bool] (N,)
        Boolean mask selecting accepted observations.
    """
    lon_o = np.asarray(obs_lon_deg, dtype=np.float64)
    lat_o = np.asarray(obs_lat_deg, dtype=np.float64)
    sig_lon = np.asarray(obs_lon_sigma_deg, dtype=np.float64)
    sig_lat = np.asarray(obs_lat_sigma_deg, dtype=np.float64)
    lon_p = np.asarray(pred_lon_deg, dtype=np.float64)
    lat_p = np.asarray(pred_lat_deg, dtype=np.float64)
    cov_ll = np.asarray(pred_cov_ll_deg2, dtype=np.float64)

    if lon_o.shape != lat_o.shape or lon_o.shape != lon_p.shape or lon_o.shape != lat_p.shape:
        raise ValueError("Observed and predicted lon/lat must have matching shapes.")
    if sig_lon.shape != lon_o.shape or sig_lat.shape != lon_o.shape:
        raise ValueError("Observational sigma arrays must match lon/lat shape.")
    if cov_ll.shape != (lon_o.shape[0], 2, 2):
        raise ValueError("pred_cov_ll_deg2 must be shape (N,2,2).")

    # Tangent-plane basis uses cos(lat0) at the predicted latitude.
    cos_lat = np.cos(np.deg2rad(lat_p))
    cos_lat = np.where(np.isfinite(cos_lat) & (np.abs(cos_lat) > 1e-12), cos_lat, 1e-12)

    # Residuals in degrees in tangent plane.
    dlon = wrap_delta_lon_deg(lon_o, lon_p)
    x = dlon * cos_lat
    y = lat_o - lat_p

    # Predicted covariance in tangent plane: cov_xy = A * cov_ll * A^T, where A=diag(cos_lat,1).
    a_p = (cos_lat * cos_lat) * cov_ll[:, 0, 0]
    d_p = cov_ll[:, 1, 1]
    b_p = cos_lat * cov_ll[:, 0, 1]

    # Observational diagonal covariance in tangent plane.
    floor_deg = float(det_sigma_floor_arcsec) / 3600.0
    sig_lon = np.where(np.isfinite(sig_lon) & (sig_lon > 0.0), sig_lon, 0.0)
    sig_lat = np.where(np.isfinite(sig_lat) & (sig_lat > 0.0), sig_lat, 0.0)
    if floor_deg > 0.0:
        sig_lon = np.maximum(sig_lon, floor_deg)
        sig_lat = np.maximum(sig_lat, floor_deg)
    var_x = (sig_lon * cos_lat) ** 2
    var_y = sig_lat**2

    a = a_p + var_x
    d = d_p + var_y
    b = b_p

    # Fast 2x2 inversion via determinant.
    det = a * d - b * b
    det = np.where(np.isfinite(det) & (det > 0.0), det, np.inf)
    chi2 = (x * x * d + y * y * a - 2.0 * x * y * b) / det
    chi2 = np.where(np.isfinite(chi2), chi2, np.inf)

    return chi2 <= float(n_sigma) ** 2

