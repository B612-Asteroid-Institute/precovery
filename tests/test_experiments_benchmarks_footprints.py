import numpy as np
import pytest

from experiments.covariance_precovery.methods.footprints import (
    corridor_pixels_from_samples,
    disc_pixels_from_cov,
    ellipse_polygon_pixels_from_cov,
    mc_pixels_from_cov,
)


@pytest.mark.benchmark(group="exp_footprints")
def test_benchmark_disc_pixels(benchmark):
    lon0, lat0 = 10.0, 20.0
    cov_ll = np.array([[1e-6, 0.0], [0.0, 1e-6]], dtype=float)

    def case():
        disc_pixels_from_cov(lon0_deg=lon0, lat0_deg=lat0, cov_ll_deg2=cov_ll, nside=256, n_sigma=3.0)

    benchmark(case)


@pytest.mark.benchmark(group="exp_footprints")
def test_benchmark_polygon_pixels(benchmark):
    lon0, lat0 = 10.0, 20.0
    cov_ll = np.array([[1e-6, 0.0], [0.0, 1e-6]], dtype=float)

    def case():
        ellipse_polygon_pixels_from_cov(
            lon0_deg=lon0, lat0_deg=lat0, cov_ll_deg2=cov_ll, nside=256, n_sigma=3.0, num_vertices=32
        )

    benchmark(case)


@pytest.mark.benchmark(group="exp_footprints")
def test_benchmark_mc_pixels(benchmark):
    lon0, lat0 = 10.0, 20.0
    cov_ll = np.array([[1e-6, 0.0], [0.0, 1e-6]], dtype=float)

    def case():
        mc_pixels_from_cov(
            lon0_deg=lon0, lat0_deg=lat0, cov_ll_deg2=cov_ll, nside=256, n_sigma=3.0, num_samples=128, seed=0
        )

    benchmark(case)


@pytest.mark.benchmark(group="exp_footprints")
def test_benchmark_corridor_pixels(benchmark):
    lon0, lat0 = 10.0, 20.0
    # A short path with mild curvature
    lon = lon0 + np.linspace(0.0, 0.2, 50)
    lat = lat0 + 0.01 * np.sin(np.linspace(0.0, 2.0 * np.pi, 50))

    def case():
        corridor_pixels_from_samples(
            lon0_deg=lon0,
            lat0_deg=lat0,
            lon_deg=lon,
            lat_deg=lat,
            nside=256,
            radius_arcsec=30.0,
            step_arcsec=30.0,
        )

    benchmark(case)

