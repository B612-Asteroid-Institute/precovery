from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import healpy as hp
import numpy as np


def _wrap_delta_lon_deg(lon_deg: np.ndarray, lon0_deg: float) -> np.ndarray:
    """
    Wrap Δlon into [-180, 180) degrees.
    """
    return (lon_deg - lon0_deg + 180.0) % 360.0 - 180.0


def _tangent_xy_deg(lon_deg: np.ndarray, lat_deg: np.ndarray, lon0_deg: float, lat0_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Local small-angle tangent-plane coordinates:
      x = Δlon * cos(lat0)
      y = Δlat
    """
    dlon = _wrap_delta_lon_deg(lon_deg, lon0_deg)
    x = dlon * np.cos(np.deg2rad(lat0_deg))
    y = lat_deg - lat0_deg
    return x.astype(np.float64), y.astype(np.float64)


def _safe_sqrtm_2x2(c: np.ndarray) -> np.ndarray:
    c = 0.5 * (c + c.T)
    w, v = np.linalg.eigh(c)
    w = np.maximum(w, 0.0)
    return v @ np.diag(np.sqrt(w)) @ v.T


def _normalize_unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    n = np.where(n > 0, n, 1.0)
    return v / n


def _local_tangent_basis(lon0_deg: float, lat0_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (center, east, north) unit vectors at (lon0, lat0) on the unit sphere.
    """
    c = np.asarray(hp.ang2vec(float(lon0_deg), float(lat0_deg), lonlat=True), dtype=np.float64)
    c = _normalize_unit(c)
    z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    e = np.cross(z, c)
    if float(np.linalg.norm(e)) < 1e-12:
        # Near the poles, pick a different reference axis.
        x = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        e = np.cross(x, c)
    e = _normalize_unit(e)
    n = _normalize_unit(np.cross(c, e))
    return c, e, n


def _ellipse_boundary_lonlat_deg_from_cov(
    *,
    lon0_deg: float,
    lat0_deg: float,
    cov_ll_deg2: np.ndarray,
    n_sigma: float,
    num_vertices: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate ellipse boundary vertices on the sphere (lon/lat in deg).

    We compute the N-sigma ellipse in the local tangent plane, then map those offsets
    to the unit sphere using the exponential map. This avoids longitude wrap issues and
    guarantees lat stays within [-90, 90].
    """
    V = max(int(num_vertices), 8)
    angles = np.linspace(0.0, 2.0 * np.pi, V, endpoint=False)
    unit = np.stack([np.cos(angles), np.sin(angles)], axis=1)  # (V, 2)

    cos_lat = float(np.cos(np.deg2rad(lat0_deg)))
    cos_lat = cos_lat if np.isfinite(cos_lat) and abs(cos_lat) > 1e-12 else 1e-12
    A = np.array([[cos_lat, 0.0], [0.0, 1.0]], dtype=np.float64)
    cov_xy = A @ cov_ll_deg2 @ A.T
    S = _safe_sqrtm_2x2(cov_xy)
    pts_xy_deg = (unit @ S.T) * float(n_sigma)  # (V,2) in tangent-plane degrees

    # Exponential map from tangent plane to sphere.
    c, e, n = _local_tangent_basis(float(lon0_deg), float(lat0_deg))
    d = (np.deg2rad(pts_xy_deg[:, 0])[:, None] * e[None, :]) + (
        np.deg2rad(pts_xy_deg[:, 1])[:, None] * n[None, :]
    )  # (V,3) in radians along tangent basis
    r = np.linalg.norm(d, axis=1)
    r_safe = np.where(r > 0, r, 1.0)
    dir_u = d / r_safe[:, None]
    v = (np.cos(r)[:, None] * c[None, :]) + (np.sin(r)[:, None] * dir_u)
    v = _normalize_unit(v)

    lon_poly, lat_poly = hp.vec2ang(v, lonlat=True)
    lon_poly = np.asarray(lon_poly, dtype=np.float64) % 360.0
    lat_poly = np.asarray(lat_poly, dtype=np.float64)
    return lon_poly, lat_poly


def ellipse_boundary_vertices_lonlat_deg_from_cov(
    *,
    lon0_deg: float,
    lat0_deg: float,
    cov_ll_deg2: np.ndarray,
    n_sigma: float,
    num_vertices: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Public wrapper: N-sigma ellipse boundary vertices as (lon,lat) deg arrays.

    This is a geometry primitive that can be persisted and reused across Stage 3/4
    to avoid reconstructing footprints twice.
    """
    return _ellipse_boundary_lonlat_deg_from_cov(
        lon0_deg=float(lon0_deg),
        lat0_deg=float(lat0_deg),
        cov_ll_deg2=np.asarray(cov_ll_deg2, dtype=np.float64),
        n_sigma=float(n_sigma),
        num_vertices=int(num_vertices),
    )


def _major_axis_sigma_deg(cov_xy_deg2: np.ndarray) -> float:
    cov_xy_deg2 = 0.5 * (cov_xy_deg2 + cov_xy_deg2.T)
    w = np.linalg.eigvalsh(cov_xy_deg2)
    lam_max = float(np.max(w))
    return float(np.sqrt(max(lam_max, 0.0)))


def _healpix_order_from_nside(nside: int) -> int:
    """
    MOC/HEALPix "order" is log2(nside) and requires power-of-two nside.
    """
    n = int(nside)
    if n <= 0:
        raise ValueError("nside must be > 0")
    if (n & (n - 1)) != 0:
        raise ValueError(f"nside must be a power of two, got {nside}")
    return int(np.log2(n))


def _moc_pixels_from_polygon(
    *,
    lon_deg: np.ndarray,
    lat_deg: np.ndarray,
    nside: int,
    include_center_disc: bool = True,
    lon0_deg: float | None = None,
    lat0_deg: float | None = None,
) -> np.ndarray:
    """
    Rasterize a spherical polygon to HEALPix pixels using MOC.

    This is a robust alternative to `healpy.query_polygon`:
    - concave and even self-intersecting polygons are accepted by mocpy
    - avoids healpy's hard abort behavior on invalid polygons
    """
    from astropy.coordinates import SkyCoord
    from mocpy import MOC

    if lon_deg.size < 3:
        return np.array([], dtype=np.int64)

    lon_deg = np.asarray(lon_deg, dtype=np.float64)
    lat_deg = np.asarray(lat_deg, dtype=np.float64)
    if lon_deg.shape != lat_deg.shape:
        raise ValueError("lon_deg and lat_deg must have the same shape")
    if not np.isfinite(lon_deg).all() or not np.isfinite(lat_deg).all():
        raise ValueError("polygon vertices contain non-finite lon/lat")

    lon_deg = lon_deg % 360.0
    # Defensive: ensure lat is in [-90, 90]. Some upstream values can be NaN or slightly
    # outside range due to numeric transforms; mocpy/cdshealpix will panic on invalid lat.
    lat_deg = np.clip(lat_deg, -89.999999, 89.999999)

    order = _healpix_order_from_nside(int(nside))

    # mocpy expects RA/Dec in degrees. Our lon/lat are already in degrees in equatorial frame
    # (Stage 3 uses ephemeris.coordinates lon/lat).
    sc = SkyCoord(lon_deg, lat_deg, unit="deg", frame="icrs")
    try:
        moc = MOC.from_polygon_skycoord(sc, max_depth=int(order))
    except BaseException as e:  # noqa: BLE001
        # mocpy can raise a pyo3 PanicException if cdshealpix asserts; normalize to ValueError.
        raise ValueError(f"moc polygon rasterization failed: {type(e).__name__}: {e}") from e

    # Convert to the exact order we need; uniq_hpx are HEALPix cell indices at nside=2**order.
    pix = moc.to_order(int(order)).uniq_hpx.astype(np.int64, copy=False)
    pix_set: set[int] = set(pix.tolist())

    # Match the existing healpy polygon behavior: always include a minimal disc at the center
    # to protect against discretization edge cases.
    if include_center_disc and lon0_deg is not None and lat0_deg is not None:
        center_vec = hp.ang2vec(float(lon0_deg), float(lat0_deg), lonlat=True)
        pix_set.update(
            hp.query_disc(
                int(nside),
                center_vec,
                float(hp.max_pixrad(int(nside))),
                inclusive=True,
                nest=True,
            ).tolist()
        )

    return np.unique(np.fromiter(pix_set, dtype=np.int64))


def disc_pixels_from_cov(
    *,
    lon0_deg: float,
    lat0_deg: float,
    cov_ll_deg2: np.ndarray,
    nside: int,
    n_sigma: float = 3.0,
) -> np.ndarray:
    """
    Conservative disc bound using the major-axis sigma in the tangent plane.
    """
    cos_lat = float(np.cos(np.deg2rad(lat0_deg)))
    cos_lat = cos_lat if np.isfinite(cos_lat) and abs(cos_lat) > 1e-12 else 1e-12
    A = np.array([[cos_lat, 0.0], [0.0, 1.0]], dtype=np.float64)
    cov_xy = A @ cov_ll_deg2 @ A.T
    sigma_major_deg = _major_axis_sigma_deg(cov_xy)
    r_rad = float(np.deg2rad(float(n_sigma) * sigma_major_deg)) + float(hp.max_pixrad(nside))
    vec = hp.ang2vec(float(lon0_deg), float(lat0_deg), lonlat=True)
    pix = hp.query_disc(nside, vec, r_rad, inclusive=True, nest=True)
    return np.unique(np.asarray(pix, dtype=np.int64))


def ellipse_polygon_pixels_from_cov_moc(
    *,
    lon0_deg: float,
    lat0_deg: float,
    cov_ll_deg2: np.ndarray,
    nside: int,
    n_sigma: float = 3.0,
    num_vertices: int = 32,
) -> np.ndarray:
    """
    Approximate N-sigma ellipse boundary in tangent plane and rasterize with mocpy (MOC).
    """
    lon_poly, lat_poly = _ellipse_boundary_lonlat_deg_from_cov(
        lon0_deg=float(lon0_deg),
        lat0_deg=float(lat0_deg),
        cov_ll_deg2=cov_ll_deg2,
        n_sigma=float(n_sigma),
        num_vertices=int(num_vertices),
    )

    return _moc_pixels_from_polygon(
        lon_deg=lon_poly.astype(np.float64, copy=False),
        lat_deg=lat_poly.astype(np.float64, copy=False),
        nside=int(nside),
        include_center_disc=True,
        lon0_deg=float(lon0_deg),
        lat0_deg=float(lat0_deg),
    )


def mc_pixels_from_cov(
    *,
    lon0_deg: float,
    lat0_deg: float,
    cov_ll_deg2: np.ndarray,
    nside: int,
    n_sigma: float = 3.0,
    num_samples: int = 256,
    seed: int = 0,
) -> np.ndarray:
    """
    Sample from the 2×2 sky covariance (Gaussian), map samples to pixels, include neighbors.
    """
    if int(num_samples) <= 0:
        return disc_pixels_from_cov(lon0_deg=lon0_deg, lat0_deg=lat0_deg, cov_ll_deg2=cov_ll_deg2, nside=nside, n_sigma=n_sigma)

    cos_lat = float(np.cos(np.deg2rad(lat0_deg)))
    cos_lat = cos_lat if np.isfinite(cos_lat) and abs(cos_lat) > 1e-12 else 1e-12
    A = np.array([[cos_lat, 0.0], [0.0, 1.0]], dtype=np.float64)
    cov_xy = A @ cov_ll_deg2 @ A.T

    S = _safe_sqrtm_2x2(cov_xy)
    rng = np.random.default_rng(int(seed))
    z = rng.standard_normal((int(num_samples), 2))
    dxy = (z @ S.T) * float(n_sigma)
    dlon = dxy[:, 0] / cos_lat
    dlat = dxy[:, 1]

    lon_s = (float(lon0_deg) + dlon) % 360.0
    lat_s = np.clip(float(lat0_deg) + dlat, -89.999999, 89.999999)
    lon_all = np.concatenate([lon_s, np.array([float(lon0_deg)])])
    lat_all = np.concatenate([lat_s, np.array([float(lat0_deg)])])

    pix = hp.ang2pix(nside, lon_all, lat_all, lonlat=True, nest=True)
    pix_set = set(np.asarray(pix, dtype=np.int64).tolist())

    neigh = hp.get_all_neighbours(nside, np.asarray(list(pix_set), dtype=np.int64), nest=True)
    if neigh is not None:
        for p in np.asarray(neigh).ravel().tolist():
            if int(p) >= 0:
                pix_set.add(int(p))

    center_vec = hp.ang2vec(float(lon0_deg), float(lat0_deg), lonlat=True)
    pix_set.update(
        hp.query_disc(nside, center_vec, float(hp.max_pixrad(nside)), inclusive=True, nest=True).tolist()
    )
    return np.unique(np.fromiter(pix_set, dtype=np.int64))


def sample_pixels_direct(
    *,
    lon_deg: np.ndarray,
    lat_deg: np.ndarray,
    nside: int,
    include_neighbors: bool = True,
    include_center_disc: bool = True,
) -> np.ndarray:
    pix = hp.ang2pix(nside, lon_deg, lat_deg, lonlat=True, nest=True)
    pix_set = set(np.asarray(pix, dtype=np.int64).ravel().tolist())
    if include_neighbors and pix_set:
        neigh = hp.get_all_neighbours(nside, np.asarray(list(pix_set), dtype=np.int64), nest=True)
        if neigh is not None:
            for p in np.asarray(neigh).ravel().tolist():
                if int(p) >= 0:
                    pix_set.add(int(p))
    if include_center_disc and len(lon_deg) > 0:
        lon0 = float(lon_deg[0])
        lat0 = float(lat_deg[0])
        center_vec = hp.ang2vec(lon0, lat0, lonlat=True)
        pix_set.update(
            hp.query_disc(nside, center_vec, float(hp.max_pixrad(nside)), inclusive=True, nest=True).tolist()
        )
    return np.unique(np.fromiter(pix_set, dtype=np.int64))


def _convex_hull_indices(points: np.ndarray) -> np.ndarray:
    """
    2D monotone chain convex hull. Returns indices into points.
    """
    pts = points
    order = np.lexsort((pts[:, 1], pts[:, 0]))
    pts_sorted = pts[order]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[int] = []
    for idx in range(len(pts_sorted)):
        while len(lower) >= 2 and cross(pts_sorted[lower[-2]], pts_sorted[lower[-1]], pts_sorted[idx]) <= 0:
            lower.pop()
        lower.append(idx)

    upper: list[int] = []
    for idx in range(len(pts_sorted) - 1, -1, -1):
        while len(upper) >= 2 and cross(pts_sorted[upper[-2]], pts_sorted[upper[-1]], pts_sorted[idx]) <= 0:
            upper.pop()
        upper.append(idx)

    hull = lower[:-1] + upper[:-1]
    hull_idx_sorted = order[np.array(hull, dtype=np.int64)]
    return hull_idx_sorted


def perimeter_polygon_from_samples(
    *,
    lon0_deg: float,
    lat0_deg: float,
    lon_deg: np.ndarray,
    lat_deg: np.ndarray,
    mode: Literal["angle_sort", "convex_hull"] = "convex_hull",
) -> np.ndarray:
    """
    Return an ordered polygon (lon,lat) vertices that wrap the samples (naive).
    """
    x, y = _tangent_xy_deg(lon_deg, lat_deg, lon0_deg, lat0_deg)
    pts = np.stack([x, y], axis=1)
    if len(pts) < 3:
        return np.stack([lon_deg, lat_deg], axis=1)

    if mode == "angle_sort":
        cx = float(np.mean(x))
        cy = float(np.mean(y))
        ang = np.arctan2(y - cy, x - cx)
        order = np.argsort(ang)
    else:
        order = _convex_hull_indices(pts)

    lon_poly = lon_deg[order]
    lat_poly = lat_deg[order]
    return np.stack([lon_poly, lat_poly], axis=1)


def corridor_path_lonlat_deg_from_samples(
    *,
    lon0_deg: float,
    lat0_deg: float,
    lon_deg: np.ndarray,
    lat_deg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Construct an ordered polyline (lon_path, lat_path) from unordered sample points.

    Current heuristic (same as corridor rasterization):
      - project to tangent plane
      - order by first principal component (PCA via SVD)
    """
    lon_deg = np.asarray(lon_deg, dtype=np.float64)
    lat_deg = np.asarray(lat_deg, dtype=np.float64)
    if lon_deg.size == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    if lon_deg.size == 1:
        return lon_deg.astype(np.float64, copy=False), lat_deg.astype(np.float64, copy=False)

    x, y = _tangent_xy_deg(lon_deg, lat_deg, float(lon0_deg), float(lat0_deg))
    pts = np.stack([x, y], axis=1)
    pts0 = pts - np.mean(pts, axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(pts0, full_matrices=False)
    u = pts0 @ vt[0].T
    order = np.argsort(u)
    return lon_deg[order], lat_deg[order]


def sample_perimeter_polygon_pixels_moc(
    *,
    lon0_deg: float,
    lat0_deg: float,
    lon_deg: np.ndarray,
    lat_deg: np.ndarray,
    nside: int,
    mode: Literal["angle_sort", "convex_hull"] = "convex_hull",
    include_center_disc: bool = True,
) -> np.ndarray:
    """
    Build a perimeter polygon from samples (tangent-plane heuristic) and rasterize with mocpy.

    Unlike `healpy.query_polygon`, mocpy accepts concave and self-intersecting polygons, so
    this is more robust for "boundary polygon" experiments.
    """
    poly = perimeter_polygon_from_samples(
        lon0_deg=float(lon0_deg),
        lat0_deg=float(lat0_deg),
        lon_deg=lon_deg,
        lat_deg=lat_deg,
        mode=mode,
    )
    if poly.shape[0] < 3:
        return sample_pixels_direct(lon_deg=lon_deg, lat_deg=lat_deg, nside=int(nside))

    lon_poly = float(lon0_deg) + _wrap_delta_lon_deg(poly[:, 0].astype(np.float64), float(lon0_deg))
    lat_poly = poly[:, 1].astype(np.float64, copy=False)

    return _moc_pixels_from_polygon(
        lon_deg=lon_poly.astype(np.float64),
        lat_deg=lat_poly.astype(np.float64),
        nside=int(nside),
        include_center_disc=bool(include_center_disc),
        lon0_deg=float(lon0_deg),
        lat0_deg=float(lat0_deg),
    )


def corridor_pixels_from_samples(
    *,
    lon0_deg: float,
    lat0_deg: float,
    lon_deg: np.ndarray,
    lat_deg: np.ndarray,
    nside: int,
    radius_arcsec: float,
    step_arcsec: float = 30.0,
) -> np.ndarray:
    """
    Rasterize a buffered corridor defined by the sample path:
      - project to tangent plane, order by first principal component
      - use that ordered polyline as the centerline
      - union query_disc along the polyline with radius
    """
    lon_s, lat_s = corridor_path_lonlat_deg_from_samples(
        lon0_deg=float(lon0_deg),
        lat0_deg=float(lat0_deg),
        lon_deg=np.asarray(lon_deg, dtype=np.float64),
        lat_deg=np.asarray(lat_deg, dtype=np.float64),
    )
    if lon_s.size == 0:
        return np.array([], dtype=np.int64)

    r_rad = np.deg2rad(float(radius_arcsec) / 3600.0)
    max_step_rad = np.deg2rad(float(step_arcsec) / 3600.0)

    pix_set: set[int] = set()
    for i in range(len(lon_s) - 1):
        lon_a, lat_a = float(lon_s[i]), float(lat_s[i])
        lon_b, lat_b = float(lon_s[i + 1]), float(lat_s[i + 1])
        va = hp.ang2vec(lon_a, lat_a, lonlat=True)
        vb = hp.ang2vec(lon_b, lat_b, lonlat=True)
        ang = float(np.arccos(np.clip(np.dot(va, vb), -1.0, 1.0)))
        nseg = max(1, int(np.ceil(ang / max_step_rad)))
        for j in range(nseg + 1):
            f = j / nseg
            # Slerp-ish by linear interpolation of unit vectors (good enough for small steps).
            v = (1.0 - f) * va + f * vb
            v = v / np.linalg.norm(v)
            pix = hp.query_disc(nside, v, r_rad, inclusive=True, nest=True)
            pix_set.update(int(p) for p in np.asarray(pix).tolist())

    # Always include sample points as well.
    pix_set.update(sample_pixels_direct(lon_deg=lon_deg, lat_deg=lat_deg, nside=nside).tolist())
    return np.unique(np.fromiter(pix_set, dtype=np.int64))


def corridor_contains_points(
    *,
    lon0_deg: float,
    lat0_deg: float,
    lon_path_deg: np.ndarray,
    lat_path_deg: np.ndarray,
    radius_arcsec: float,
    test_lon_deg: np.ndarray,
    test_lat_deg: np.ndarray,
) -> np.ndarray:
    """
    Point-in-corridor test in tangent plane using distance-to-polyline <= radius.
    """
    if len(lon_path_deg) == 0:
        return np.zeros(len(test_lon_deg), dtype=bool)

    x_path, y_path = _tangent_xy_deg(lon_path_deg, lat_path_deg, lon0_deg, lat0_deg)
    x, y = _tangent_xy_deg(test_lon_deg, test_lat_deg, lon0_deg, lat0_deg)
    r2 = (float(radius_arcsec) / 3600.0) ** 2

    # Compute min squared distance from each point to any segment.
    min_d2 = np.full(len(x), np.inf, dtype=np.float64)
    for i in range(len(x_path) - 1):
        ax, ay = x_path[i], y_path[i]
        bx, by = x_path[i + 1], y_path[i + 1]
        vx, vy = bx - ax, by - ay
        denom = vx * vx + vy * vy
        if denom <= 0:
            continue
        t = ((x - ax) * vx + (y - ay) * vy) / denom
        t = np.clip(t, 0.0, 1.0)
        px = ax + t * vx
        py = ay + t * vy
        d2 = (x - px) ** 2 + (y - py) ** 2
        min_d2 = np.minimum(min_d2, d2)
    return min_d2 <= r2


class Footprint(Protocol):
    def pixels(self, nside: int) -> np.ndarray: ...

    def contains(self, lon_deg: np.ndarray, lat_deg: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class EllipseFootprint:
    lon0_deg: float
    lat0_deg: float
    cov_ll_deg2: np.ndarray  # 2x2 in (lon,lat) deg^2
    n_sigma: float = 3.0
    polygon_vertices: int = 64
    raster_mode: Literal["disc", "polygon"] = "polygon"

    def pixels(self, nside: int) -> np.ndarray:
        if self.raster_mode == "disc":
            return disc_pixels_from_cov(
                lon0_deg=self.lon0_deg,
                lat0_deg=self.lat0_deg,
                cov_ll_deg2=self.cov_ll_deg2,
                nside=nside,
                n_sigma=self.n_sigma,
            )
        # Use the MOC rasterizer for robust polygon pixels.
        return ellipse_polygon_pixels_from_cov_moc(
            lon0_deg=self.lon0_deg,
            lat0_deg=self.lat0_deg,
            cov_ll_deg2=self.cov_ll_deg2,
            nside=nside,
            n_sigma=self.n_sigma,
            num_vertices=self.polygon_vertices,
        )

    def contains(self, lon_deg: np.ndarray, lat_deg: np.ndarray) -> np.ndarray:
        # Mahalanobis gate in tangent plane using the ellipse covariance.
        cos_lat = float(np.cos(np.deg2rad(self.lat0_deg)))
        cos_lat = cos_lat if np.isfinite(cos_lat) and abs(cos_lat) > 1e-12 else 1e-12
        A = np.array([[cos_lat, 0.0], [0.0, 1.0]], dtype=np.float64)
        cov_xy = A @ self.cov_ll_deg2 @ A.T
        cov_xy = 0.5 * (cov_xy + cov_xy.T)
        inv = np.linalg.pinv(cov_xy)
        x, y = _tangent_xy_deg(lon_deg, lat_deg, self.lon0_deg, self.lat0_deg)
        d = np.stack([x, y], axis=1)
        chi2 = np.einsum("ni,ij,nj->n", d, inv, d)
        return chi2 <= float(self.n_sigma) ** 2


@dataclass(frozen=True)
class FixedPolygonFootprint:
    """
    Footprint defined by explicit polygon vertices (lon/lat degrees).

    This is used when we persist polygon vertices in Stage 3 and want to reuse them in Stage 4
    without re-running the sample->polygon construction.
    """

    lon0_deg: float
    lat0_deg: float
    vertex_lon_deg: np.ndarray
    vertex_lat_deg: np.ndarray
    buffer_arcsec: float = 0.0

    def pixels(self, nside: int) -> np.ndarray:
        # Prefer robust MOC rasterization for arbitrary polygons.
        return _moc_pixels_from_polygon(
            lon_deg=np.asarray(self.vertex_lon_deg, dtype=np.float64),
            lat_deg=np.asarray(self.vertex_lat_deg, dtype=np.float64),
            nside=int(nside),
            include_center_disc=True,
            lon0_deg=float(self.lon0_deg),
            lat0_deg=float(self.lat0_deg),
        )

    def contains(self, lon_deg: np.ndarray, lat_deg: np.ndarray) -> np.ndarray:
        # Point-in-polygon in tangent plane; optional buffer by distance-to-edges.
        poly = np.stack(
            [np.asarray(self.vertex_lon_deg, dtype=np.float64), np.asarray(self.vertex_lat_deg, dtype=np.float64)],
            axis=1,
        )
        if poly.shape[0] < 3:
            return np.zeros(len(lon_deg), dtype=bool)

        px, py = _tangent_xy_deg(poly[:, 0], poly[:, 1], float(self.lon0_deg), float(self.lat0_deg))
        x, y = _tangent_xy_deg(np.asarray(lon_deg, dtype=np.float64), np.asarray(lat_deg, dtype=np.float64), float(self.lon0_deg), float(self.lat0_deg))

        inside = np.zeros(len(x), dtype=bool)
        j = len(px) - 1
        for i in range(len(px)):
            xi, yi = px[i], py[i]
            xj, yj = px[j], py[j]
            cond = ((yi > y) != (yj > y)) & (
                x < (xj - xi) * (y - yi) / (yj - yi + 1e-30) + xi
            )
            inside ^= cond
            j = i

        # Boundary tolerance.
        eps2 = 1e-24
        min_d2 = np.full(len(x), np.inf, dtype=np.float64)
        for i in range(len(px)):
            ax, ay = px[i], py[i]
            bx, by = px[(i + 1) % len(px)], py[(i + 1) % len(py)]
            vx, vy = bx - ax, by - ay
            denom = vx * vx + vy * vy
            if denom <= 0:
                continue
            t = ((x - ax) * vx + (y - ay) * vy) / denom
            t = np.clip(t, 0.0, 1.0)
            qx = ax + t * vx
            qy = ay + t * vy
            d2 = (x - qx) ** 2 + (y - qy) ** 2
            min_d2 = np.minimum(min_d2, d2)
        inside = inside | (min_d2 <= eps2)

        if float(self.buffer_arcsec) <= 0:
            return inside
        r2 = (float(self.buffer_arcsec) / 3600.0) ** 2
        return inside | (min_d2 <= r2)


@dataclass(frozen=True)
class CorridorFootprint:
    lon0_deg: float
    lat0_deg: float
    path_lon_deg: np.ndarray
    path_lat_deg: np.ndarray
    radius_arcsec: float
    raster_step_arcsec: float = 30.0

    def pixels(self, nside: int) -> np.ndarray:
        return corridor_pixels_from_samples(
            lon0_deg=self.lon0_deg,
            lat0_deg=self.lat0_deg,
            lon_deg=self.path_lon_deg,
            lat_deg=self.path_lat_deg,
            nside=nside,
            radius_arcsec=self.radius_arcsec,
            step_arcsec=self.raster_step_arcsec,
        )

    def contains(self, lon_deg: np.ndarray, lat_deg: np.ndarray) -> np.ndarray:
        return corridor_contains_points(
            lon0_deg=self.lon0_deg,
            lat0_deg=self.lat0_deg,
            lon_path_deg=self.path_lon_deg,
            lat_path_deg=self.path_lat_deg,
            radius_arcsec=self.radius_arcsec,
            test_lon_deg=lon_deg,
            test_lat_deg=lat_deg,
        )

