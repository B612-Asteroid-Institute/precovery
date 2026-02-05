import math

import numpy as np

from experiments.covariance_precovery.methods.covariance_metrics import (
    bytes_per_observation,
    ellipse_area_deg2_from_cov_ll_deg2,
    expected_observations_from_bytes,
    sigma_major_arcsec_from_cov_ll_deg2,
)


def test_sigma_major_arcsec_diagonal_cov() -> None:
    cov_ll = np.array([[4e-6, 0.0], [0.0, 1e-6]], dtype=np.float64)  # deg^2
    # At lat0=0, xy==ll. Major sigma is sqrt(4e-6)=0.002 deg = 7.2 arcsec.
    sig = sigma_major_arcsec_from_cov_ll_deg2(cov_ll_deg2=cov_ll, lat0_deg=0.0)
    assert abs(sig - 7.2) < 1e-9


def test_ellipse_area_matches_det_formula() -> None:
    cov_ll = np.array([[9e-6, 0.0], [0.0, 4e-6]], dtype=np.float64)  # deg^2
    n_sigma = 3.0
    # At lat0=0, cov_xy=cov_ll and det=36e-12 -> sqrt(det)=6e-6.
    expected = math.pi * (n_sigma**2) * 6e-6
    area = ellipse_area_deg2_from_cov_ll_deg2(cov_ll_deg2=cov_ll, lat0_deg=0.0, n_sigma=n_sigma)
    assert abs(area - expected) < 1e-15


def test_bytes_per_observation_is_positive() -> None:
    b = bytes_per_observation(mean_id_len_bytes=10.0)
    assert b > 0


def test_expected_observations_from_bytes() -> None:
    bpo = 50.0
    assert expected_observations_from_bytes(data_length_bytes=0.0, bytes_per_obs=bpo) == 0.0
    assert expected_observations_from_bytes(data_length_bytes=100.0, bytes_per_obs=bpo) == 2.0

