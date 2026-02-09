import numpy as np

from experiments.covariance_precovery.methods.footprints import (
    CorridorFootprint,
    EllipseFootprint,
    disc_pixels_from_cov,
    ellipse_polygon_pixels_from_cov_moc,
    mc_pixels_from_cov,
)


def test_covariance_pixel_methods_return_nonempty() -> None:
    lon0, lat0 = 10.0, 20.0
    cov_ll = np.array([[1e-6, 0.0], [0.0, 1e-6]], dtype=float)  # ~3.6 arcsec 1-sigma
    nside = 64
    disc = disc_pixels_from_cov(lon0_deg=lon0, lat0_deg=lat0, cov_ll_deg2=cov_ll, nside=nside)
    poly = ellipse_polygon_pixels_from_cov_moc(
        lon0_deg=lon0, lat0_deg=lat0, cov_ll_deg2=cov_ll, nside=nside, num_vertices=16
    )
    mc = mc_pixels_from_cov(
        lon0_deg=lon0, lat0_deg=lat0, cov_ll_deg2=cov_ll, nside=nside, num_samples=64, seed=0
    )
    assert disc.size > 0
    assert poly.size > 0
    assert mc.size > 0


def test_ellipse_contains_center() -> None:
    lon0, lat0 = 10.0, 20.0
    cov_ll = np.array([[1e-6, 0.0], [0.0, 1e-6]], dtype=float)
    fp = EllipseFootprint(lon0_deg=lon0, lat0_deg=lat0, cov_ll_deg2=cov_ll, n_sigma=3.0)
    mask = fp.contains(np.array([lon0]), np.array([lat0]))
    assert mask.tolist() == [True]


def test_corridor_contains_path_points() -> None:
    lon0, lat0 = 10.0, 20.0
    lon_path = np.array([lon0, lon0 + 0.02], dtype=float)
    lat_path = np.array([lat0, lat0], dtype=float)
    fp = CorridorFootprint(
        lon0_deg=lon0, lat0_deg=lat0, path_lon_deg=lon_path, path_lat_deg=lat_path, radius_arcsec=60.0
    )
    mask = fp.contains(lon_path, lat_path)
    assert mask.all()

