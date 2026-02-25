from __future__ import annotations

from functools import lru_cache

import healpy as hp
import numpy as np


def normalize_unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    n = np.where(n > 0, n, 1.0)
    return v / n


def local_tangent_basis(lon0_deg: float, lat0_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (center, east, north) unit vectors at (lon0,lat0) on the unit sphere.
    """
    c = np.asarray(hp.ang2vec(float(lon0_deg), float(lat0_deg), lonlat=True), dtype=np.float64)
    c = normalize_unit(c)
    z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    e = np.cross(z, c)
    if float(np.linalg.norm(e)) < 1e-12:
        x = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        e = np.cross(x, c)
    e = normalize_unit(e)
    n = normalize_unit(np.cross(c, e))
    return c, e, n


def safe_sqrtm_2x2(c: np.ndarray) -> np.ndarray:
    c2 = 0.5 * (c + c.T)
    w, v = np.linalg.eigh(c2)
    w = np.maximum(w, 0.0)
    return v @ np.diag(np.sqrt(w)) @ v.T


def healpix_order_from_nside(nside: int) -> int:
    n = int(nside)
    if n <= 0:
        raise ValueError("nside must be > 0")
    if (n & (n - 1)) != 0:
        raise ValueError(f"nside must be a power of two, got {nside}")
    return int(np.log2(n))


@lru_cache(maxsize=64)
def _unit_circle_vertices(num_vertices: int) -> np.ndarray:
    v = max(int(num_vertices), 8)
    angles = np.linspace(0.0, 2.0 * np.pi, v, endpoint=False, dtype=np.float64)
    unit = np.stack([np.cos(angles), np.sin(angles)], axis=1).astype(np.float64, copy=False)  # (V,2)
    unit.setflags(write=False)
    return unit


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

    We build the ellipse in a local tangent plane and map it to the sphere using an
    exponential map (tangent-plane vector -> great-circle). This is wrap-safe and robust
    near poles.
    """
    V = max(int(num_vertices), 8)
    unit = _unit_circle_vertices(V)

    cov_ll = np.asarray(cov_ll_deg2, dtype=np.float64)
    if cov_ll.shape != (2, 2):
        raise ValueError("cov_ll_deg2 must be shape (2,2).")

    cos_lat = float(np.cos(np.deg2rad(float(lat0_deg))))
    cos_lat = cos_lat if np.isfinite(cos_lat) and abs(cos_lat) > 1e-12 else 1e-12
    A = np.array([[cos_lat, 0.0], [0.0, 1.0]], dtype=np.float64)
    cov_xy = A @ cov_ll @ A.T
    S = safe_sqrtm_2x2(cov_xy)
    pts_xy_deg = (unit @ S.T) * float(n_sigma)  # (V,2) in tangent-plane degrees

    c, e, n = local_tangent_basis(float(lon0_deg), float(lat0_deg))
    d = (np.deg2rad(pts_xy_deg[:, 0])[:, None] * e[None, :]) + (
        np.deg2rad(pts_xy_deg[:, 1])[:, None] * n[None, :]
    )  # (V,3)
    r = np.linalg.norm(d, axis=1)
    r_safe = np.where(r > 0, r, 1.0)
    dir_u = d / r_safe[:, None]
    v = (np.cos(r)[:, None] * c[None, :]) + (np.sin(r)[:, None] * dir_u)
    v = normalize_unit(v)

    lon_poly, lat_poly = hp.vec2ang(v, lonlat=True)
    lon_poly = np.asarray(lon_poly, dtype=np.float64) % 360.0
    lat_poly = np.asarray(lat_poly, dtype=np.float64)
    return lon_poly, lat_poly

