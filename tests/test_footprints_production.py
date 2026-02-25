from __future__ import annotations

import healpy as hp
import numpy as np

from precovery.search.footprints import CovPolygonReconstructedMoc


def test_cov_polygon_reconstructed_moc_contains_center_pixel() -> None:
    fp = CovPolygonReconstructedMoc(n_sigma=3.0, polygon_vertices=32, pad_neighbors=False)
    lon0, lat0 = 10.0, 20.0
    cov_ll = np.array([[1e-6, 0.0], [0.0, 1e-6]], dtype=np.float64)  # ~3.6 arcsec 1-sigma
    nside = 64

    pix = fp.pixels_for_prediction(lon0_deg=lon0, lat0_deg=lat0, cov_ll_deg2=cov_ll, nside=nside)
    assert pix.size > 0

    center = int(hp.ang2pix(int(nside), float(lon0), float(lat0), nest=True, lonlat=True))
    assert center in set(int(x) for x in np.asarray(pix, dtype=np.int64).tolist())


def test_pad_neighbors_increases_or_keeps_pixel_set() -> None:
    lon0, lat0 = 10.0, 20.0
    cov_ll = np.array([[1e-5, 0.0], [0.0, 1e-5]], dtype=np.float64)
    nside = 64

    fp0 = CovPolygonReconstructedMoc(n_sigma=3.0, polygon_vertices=32, pad_neighbors=False)
    fp1 = CovPolygonReconstructedMoc(n_sigma=3.0, polygon_vertices=32, pad_neighbors=True)

    p0 = fp0.pixels_for_prediction(lon0_deg=lon0, lat0_deg=lat0, cov_ll_deg2=cov_ll, nside=nside)
    p1 = fp1.pixels_for_prediction(lon0_deg=lon0, lat0_deg=lat0, cov_ll_deg2=cov_ll, nside=nside)

    assert p0.size > 0
    assert p1.size >= p0.size
