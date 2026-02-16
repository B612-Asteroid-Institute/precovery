from __future__ import annotations

from dataclasses import dataclass

import healpy as hp
import numpy as np

from .geometry import (
    ellipse_boundary_vertices_lonlat_deg_from_cov,
    healpix_order_from_nside,
)

def moc_pixels_from_polygon(
    *,
    lon_deg: np.ndarray,
    lat_deg: np.ndarray,
    nside: int,
    lon0_deg: float,
    lat0_deg: float,
) -> np.ndarray:
    """
    Rasterize a polygon to HEALPix pixels using MOC.

    Why MOC
    -------
    `healpy.query_polygon` can be brittle (convexity/validity assumptions and potential hard aborts).
    `mocpy` provides a robust rasterizer for arbitrary polygons and is the chosen production strategy.
    """
    from mocpy import MOC
    import astropy.units as u

    lon = np.asarray(lon_deg, dtype=np.float64) % 360.0
    lat = np.asarray(lat_deg, dtype=np.float64)
    lat = np.clip(lat, -89.999999, 89.999999)
    if lon.size < 3:
        return np.array([], dtype=np.int64)

    order = healpix_order_from_nside(int(nside))
    try:
        # Avoid SkyCoord construction overhead; mocpy supports lon/lat quantities directly.
        moc = MOC.from_polygon(lon * u.deg, lat * u.deg, max_depth=int(order))
    except BaseException as e:  # noqa: BLE001
        raise ValueError(f"MOC polygon rasterization failed: {type(e).__name__}: {e}") from e

    pix = np.asarray(moc.to_order(int(order)).flatten(), dtype=np.int64)
    pix_set: set[int] = set(pix.tolist())

    # Safety margin WITHOUT discs:
    # - include neighbors of polygon pixels (protects discretization boundaries)
    # - always include the nominal point pixel and its neighbors (protects thin polygons)
    if pix_set:
        neigh = hp.get_all_neighbours(
            int(nside), np.asarray(list(pix_set), dtype=np.int64), nest=True
        )
        if neigh is not None:
            for p in np.asarray(neigh).ravel().tolist():
                if int(p) >= 0:
                    pix_set.add(int(p))

    center_pix = int(
        hp.ang2pix(int(nside), float(lon0_deg), float(lat0_deg), lonlat=True, nest=True)
    )
    pix_set.add(center_pix)
    center_neigh = hp.get_all_neighbours(int(nside), np.asarray([center_pix], dtype=np.int64), nest=True)
    if center_neigh is not None:
        for p in np.asarray(center_neigh).ravel().tolist():
            if int(p) >= 0:
                pix_set.add(int(p))
    return np.unique(np.fromiter(pix_set, dtype=np.int64))


@dataclass(frozen=True)
class CovPolygonReconstructedMoc:
    """
    `cov_polygon_reconstructed_moc` footprint strategy.
    """

    n_sigma: float = 3.0
    polygon_vertices: int = 32

    def pixels_for_prediction(
        self, *, lon0_deg: float, lat0_deg: float, cov_ll_deg2: np.ndarray, nside: int
    ) -> np.ndarray:
        lonv, latv = ellipse_boundary_vertices_lonlat_deg_from_cov(
            lon0_deg=float(lon0_deg),
            lat0_deg=float(lat0_deg),
            cov_ll_deg2=np.asarray(cov_ll_deg2, dtype=np.float64),
            n_sigma=float(self.n_sigma),
            num_vertices=int(self.polygon_vertices),
        )
        return moc_pixels_from_polygon(
            lon_deg=lonv, lat_deg=latv, nside=int(nside), lon0_deg=float(lon0_deg), lat0_deg=float(lat0_deg)
        )

