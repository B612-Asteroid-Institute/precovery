from __future__ import annotations

from dataclasses import dataclass

import healpy as hp
import numpy as np


def _normalize_unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    n = np.where(n > 0, n, 1.0)
    return v / n


def _local_tangent_basis(lon0_deg: float, lat0_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (center, east, north) unit vectors at (lon0,lat0) on the unit sphere.
    """
    c = np.asarray(hp.ang2vec(float(lon0_deg), float(lat0_deg), lonlat=True), dtype=np.float64)
    c = _normalize_unit(c)
    z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    e = np.cross(z, c)
    if float(np.linalg.norm(e)) < 1e-12:
        x = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        e = np.cross(x, c)
    e = _normalize_unit(e)
    n = _normalize_unit(np.cross(c, e))
    return c, e, n


def _safe_sqrtm_2x2(c: np.ndarray) -> np.ndarray:
    c = 0.5 * (c + c.T)
    w, v = np.linalg.eigh(c)
    w = np.maximum(w, 0.0)
    return v @ np.diag(np.sqrt(w)) @ v.T


def ellipse_boundary_vertices_lonlat_deg_from_cov(
    *,
    lon0_deg: float,
    lat0_deg: float,
    cov_ll_deg2: np.ndarray,
    n_sigma: float,
    num_vertices: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate an N-sigma ellipse boundary polygon on the sphere.

    Why this exists
    ---------------
    We want a robust boundary polygon that is safe under longitude wrapping and
    near-pole behavior. We build the ellipse in a local tangent plane and map
    it to the sphere using an exponential map (tangent-plane vector -> great-circle).
    """
    V = max(int(num_vertices), 8)
    angles = np.linspace(0.0, 2.0 * np.pi, V, endpoint=False)
    unit = np.stack([np.cos(angles), np.sin(angles)], axis=1)  # (V,2)

    cov_ll = np.asarray(cov_ll_deg2, dtype=np.float64)
    if cov_ll.shape != (2, 2):
        raise ValueError("cov_ll_deg2 must be shape (2,2).")

    cos_lat = float(np.cos(np.deg2rad(float(lat0_deg))))
    cos_lat = cos_lat if np.isfinite(cos_lat) and abs(cos_lat) > 1e-12 else 1e-12
    A = np.array([[cos_lat, 0.0], [0.0, 1.0]], dtype=np.float64)
    cov_xy = A @ cov_ll @ A.T
    S = _safe_sqrtm_2x2(cov_xy)
    pts_xy_deg = (unit @ S.T) * float(n_sigma)  # (V,2) in tangent-plane degrees

    c, e, n = _local_tangent_basis(float(lon0_deg), float(lat0_deg))
    d = (np.deg2rad(pts_xy_deg[:, 0])[:, None] * e[None, :]) + (
        np.deg2rad(pts_xy_deg[:, 1])[:, None] * n[None, :]
    )  # (V,3)
    r = np.linalg.norm(d, axis=1)
    r_safe = np.where(r > 0, r, 1.0)
    dir_u = d / r_safe[:, None]
    v = (np.cos(r)[:, None] * c[None, :]) + (np.sin(r)[:, None] * dir_u)
    v = _normalize_unit(v)

    lon_poly, lat_poly = hp.vec2ang(v, lonlat=True)
    lon_poly = np.asarray(lon_poly, dtype=np.float64) % 360.0
    lat_poly = np.asarray(lat_poly, dtype=np.float64)
    return lon_poly, lat_poly


def _healpix_order_from_nside(nside: int) -> int:
    n = int(nside)
    if n <= 0:
        raise ValueError("nside must be > 0")
    if (n & (n - 1)) != 0:
        raise ValueError(f"nside must be a power of two, got {nside}")
    return int(np.log2(n))


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
    from astropy.coordinates import SkyCoord
    from mocpy import MOC

    lon = np.asarray(lon_deg, dtype=np.float64) % 360.0
    lat = np.asarray(lat_deg, dtype=np.float64)
    lat = np.clip(lat, -89.999999, 89.999999)
    if lon.size < 3:
        return np.array([], dtype=np.int64)

    order = _healpix_order_from_nside(int(nside))
    sc = SkyCoord(lon, lat, unit="deg", frame="icrs")
    try:
        moc = MOC.from_polygon_skycoord(sc, max_depth=int(order))
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

