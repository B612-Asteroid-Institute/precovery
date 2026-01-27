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


def _major_axis_sigma_deg(cov_xy_deg2: np.ndarray) -> float:
    cov_xy_deg2 = 0.5 * (cov_xy_deg2 + cov_xy_deg2.T)
    w = np.linalg.eigvalsh(cov_xy_deg2)
    lam_max = float(np.max(w))
    return float(np.sqrt(max(lam_max, 0.0)))


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


def ellipse_polygon_pixels_from_cov(
    *,
    lon0_deg: float,
    lat0_deg: float,
    cov_ll_deg2: np.ndarray,
    nside: int,
    n_sigma: float = 3.0,
    num_vertices: int = 32,
) -> np.ndarray:
    """
    Approximate N-sigma ellipse boundary in tangent plane and rasterize via `query_polygon`.
    """
    V = max(int(num_vertices), 8)
    angles = np.linspace(0.0, 2.0 * np.pi, V, endpoint=False)
    unit = np.stack([np.cos(angles), np.sin(angles)], axis=1)  # (V, 2)

    cos_lat = float(np.cos(np.deg2rad(lat0_deg)))
    cos_lat = cos_lat if np.isfinite(cos_lat) and abs(cos_lat) > 1e-12 else 1e-12
    A = np.array([[cos_lat, 0.0], [0.0, 1.0]], dtype=np.float64)
    cov_xy = A @ cov_ll_deg2 @ A.T
    S = _safe_sqrtm_2x2(cov_xy)
    pts = (unit @ S.T) * float(n_sigma)  # (V, 2) in degrees

    dlon = pts[:, 0] / cos_lat
    dlat = pts[:, 1]
    lon_poly = (float(lon0_deg) + dlon) % 360.0
    lat_poly = np.clip(float(lat0_deg) + dlat, -89.999999, 89.999999)

    verts = hp.ang2vec(lon_poly, lat_poly, lonlat=True)
    pix = hp.query_polygon(nside, verts, inclusive=True, nest=True)
    pix_set = set(np.asarray(pix, dtype=np.int64).tolist())

    # Safety margin for discretization.
    center_vec = hp.ang2vec(float(lon0_deg), float(lat0_deg), lonlat=True)
    pix_set.update(
        hp.query_disc(nside, center_vec, float(hp.max_pixrad(nside)), inclusive=True, nest=True).tolist()
    )
    return np.unique(np.fromiter(pix_set, dtype=np.int64))


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
    x, y = _tangent_xy_deg(lon_deg, lat_deg, lon0_deg, lat0_deg)
    pts = np.stack([x, y], axis=1)
    if len(pts) == 0:
        return np.array([], dtype=np.int64)

    # PCA via SVD.
    pts0 = pts - np.mean(pts, axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(pts0, full_matrices=False)
    u = pts0 @ vt[0].T
    order = np.argsort(u)
    lon_s = lon_deg[order]
    lat_s = lat_deg[order]

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
        return ellipse_polygon_pixels_from_cov(
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
class SamplePerimeterPolygonFootprint:
    lon0_deg: float
    lat0_deg: float
    sample_lon_deg: np.ndarray
    sample_lat_deg: np.ndarray
    polygon_mode: Literal["angle_sort", "convex_hull"] = "convex_hull"
    buffer_arcsec: float = 0.0

    def _poly_vertices(self) -> np.ndarray:
        return perimeter_polygon_from_samples(
            lon0_deg=self.lon0_deg,
            lat0_deg=self.lat0_deg,
            lon_deg=self.sample_lon_deg,
            lat_deg=self.sample_lat_deg,
            mode=self.polygon_mode,
        )

    def pixels(self, nside: int) -> np.ndarray:
        poly = self._poly_vertices()
        if poly.shape[0] < 3:
            return sample_pixels_direct(
                lon_deg=self.sample_lon_deg, lat_deg=self.sample_lat_deg, nside=nside
            )
        verts = hp.ang2vec(poly[:, 0], poly[:, 1], lonlat=True)
        pix = hp.query_polygon(nside, verts, inclusive=True, nest=True)
        pix_set = set(np.asarray(pix, dtype=np.int64).tolist())
        # Add safety margin for buffer/pixelization.
        if float(self.buffer_arcsec) > 0:
            r_rad = np.deg2rad(float(self.buffer_arcsec) / 3600.0) + float(hp.max_pixrad(nside))
            for lon, lat in poly:
                v = hp.ang2vec(float(lon), float(lat), lonlat=True)
                pix_set.update(
                    hp.query_disc(nside, v, r_rad, inclusive=True, nest=True).tolist()
                )
        return np.unique(np.fromiter(pix_set, dtype=np.int64))

    def contains(self, lon_deg: np.ndarray, lat_deg: np.ndarray) -> np.ndarray:
        # Point-in-polygon in tangent plane; optional buffer by distance-to-edges.
        poly = self._poly_vertices()
        if poly.shape[0] < 3:
            return np.zeros(len(lon_deg), dtype=bool)

        px, py = _tangent_xy_deg(poly[:, 0], poly[:, 1], self.lon0_deg, self.lat0_deg)
        x, y = _tangent_xy_deg(lon_deg, lat_deg, self.lon0_deg, self.lat0_deg)

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

        # Always treat boundary points as inside (robustness for discrete catalogs).
        eps2 = 1e-24  # (deg^2) ~ 1e-12 deg tolerance
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

        on_boundary = min_d2 <= eps2
        inside = inside | on_boundary

        if float(self.buffer_arcsec) <= 0:
            return inside

        # Buffer: accept points within buffer distance of any edge.
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

