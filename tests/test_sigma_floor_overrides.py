from __future__ import annotations

import numpy as np

from precovery.search.detection_filter import innov_ellipse_keep_mask


def test_innov_ellipse_supports_per_row_sigma_floor() -> None:
    """
    Per-row floors should be accepted and should influence keep/reject decisions.
    """
    # One candidate right at the predicted location with zero reported sigmas.
    obs_lon = np.array([10.0], dtype=np.float64)
    obs_lat = np.array([0.0], dtype=np.float64)
    obs_sig_lon = np.array([0.0], dtype=np.float64)
    obs_sig_lat = np.array([0.0], dtype=np.float64)

    pred_lon = np.array([10.0], dtype=np.float64)
    pred_lat = np.array([0.0], dtype=np.float64)
    pred_cov = np.zeros((1, 2, 2), dtype=np.float64)

    # With a large floor, the innovation ellipse is large => keep.
    keep_big = innov_ellipse_keep_mask(
        obs_lon_deg=obs_lon,
        obs_lat_deg=obs_lat,
        obs_lon_sigma_deg=obs_sig_lon,
        obs_lat_sigma_deg=obs_sig_lat,
        pred_lon_deg=pred_lon,
        pred_lat_deg=pred_lat,
        pred_cov_ll_deg2=pred_cov,
        n_sigma=3.0,
        invalid_sigma_fill_floor_arcsec=np.array([10.0], dtype=np.float64),
    )
    assert bool(keep_big[0])

    # With a zero floor and zero cov, this is still keep (exact match).
    keep_zero = innov_ellipse_keep_mask(
        obs_lon_deg=obs_lon,
        obs_lat_deg=obs_lat,
        obs_lon_sigma_deg=obs_sig_lon,
        obs_lat_sigma_deg=obs_sig_lat,
        pred_lon_deg=pred_lon,
        pred_lat_deg=pred_lat,
        pred_cov_ll_deg2=pred_cov,
        n_sigma=3.0,
        invalid_sigma_fill_floor_arcsec=np.array([0.0], dtype=np.float64),
    )
    assert bool(keep_zero[0])
