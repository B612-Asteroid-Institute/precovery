from __future__ import annotations

from dataclasses import dataclass

import healpy as hp
import numpy as np
import time

from .geometry import (
    ellipse_boundary_vertices_lonlat_deg_from_cov,
    healpix_order_from_nside,
)

def _pad_neighbors(*, pixels: np.ndarray, nside: int) -> np.ndarray:
    """
    Optional neighbor padding around a rasterized pixel set.

    This is intentionally **off by default** at the footprint strategy level, since it
    increases join volume. Keep it only as a truth-regression knob.
    """
    pix = np.unique(np.asarray(pixels, dtype=np.int64))
    if pix.size == 0:
        return pix
    neigh = hp.get_all_neighbours(int(nside), pix, nest=True)
    if neigh is None:
        return pix
    nn = np.asarray(neigh, dtype=np.int64).ravel()
    nn = nn[nn >= 0]
    if nn.size == 0:
        return pix
    return np.unique(np.concatenate([pix, nn], axis=0))


def _moc_pixels_from_polygon_raw(
    *,
    lon_deg: np.ndarray,
    lat_deg: np.ndarray,
    nside: int,
) -> np.ndarray:
    """
    Rasterize a polygon to HEALPix pixels using MOC (no padding).

    Padding/guards are applied by callers.
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
        moc = MOC.from_polygon(lon * u.deg, lat * u.deg, max_depth=int(order))
    except BaseException as e:  # noqa: BLE001
        raise ValueError(f"MOC polygon rasterization failed: {type(e).__name__}: {e}") from e
    return np.asarray(moc.to_order(int(order)).flatten(), dtype=np.int64)


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
    `mocpy` provides a robust rasterizer for arbitrary polygons and is the chosen production strategy.
    """
    _ = (lon0_deg, lat0_deg)
    return np.unique(_moc_pixels_from_polygon_raw(lon_deg=lon_deg, lat_deg=lat_deg, nside=int(nside)))


@dataclass(frozen=True)
class CovPolygonReconstructedMoc:
    """
    `cov_polygon_reconstructed_moc` footprint strategy.
    """

    n_sigma: float = 3.0
    polygon_vertices: int = 32
    pad_neighbors: bool = False
    point_radius_frac_of_pixel: float | None = None

    def pixels_for_prediction(
        self,
        *,
        lon0_deg: float,
        lat0_deg: float,
        cov_ll_deg2: np.ndarray,
        nside: int,
        timings: dict[str, float] | None = None,
    ) -> np.ndarray:
        # Optional fast path: when the N-sigma region is guaranteed to fit within a single
        # HEALPix pixel, avoid polygon rasterization entirely.
        if self.point_radius_frac_of_pixel is not None:
            frac = float(self.point_radius_frac_of_pixel)
            if np.isfinite(frac) and frac > 0.0:
                cov = np.asarray(cov_ll_deg2, dtype=np.float64)
                if cov.shape == (2, 2):
                    cos_lat = float(np.cos(np.deg2rad(float(lat0_deg))))
                    cos_lat = cos_lat if np.isfinite(cos_lat) and abs(cos_lat) > 1e-12 else 1e-12
                    # Upper-bound the max eigenvalue of the locally-scaled covariance via row-sum.
                    a00 = float((cos_lat * cos_lat) * cov[0, 0])
                    a01 = float(cos_lat * cov[0, 1])
                    a11 = float(cov[1, 1])
                    max_eig_ub = max(abs(a00) + abs(a01), abs(a11) + abs(a01), 0.0)
                    r_deg_ub = float(self.n_sigma) * float(np.sqrt(max_eig_ub))
                    pix_deg = float(np.degrees(hp.nside2resol(int(nside))))
                    if np.isfinite(r_deg_ub) and np.isfinite(pix_deg) and r_deg_ub <= frac * pix_deg:
                        pix0 = np.array(
                            [hp.ang2pix(int(nside), float(lon0_deg), float(lat0_deg), lonlat=True, nest=True)],
                            dtype=np.int64,
                        )
                        if not bool(self.pad_neighbors):
                            return pix0
                        return _pad_neighbors(pixels=pix0, nside=int(nside))

        t0 = time.perf_counter() if timings is not None else None
        lonv, latv = ellipse_boundary_vertices_lonlat_deg_from_cov(
            lon0_deg=float(lon0_deg),
            lat0_deg=float(lat0_deg),
            cov_ll_deg2=np.asarray(cov_ll_deg2, dtype=np.float64),
            n_sigma=float(self.n_sigma),
            num_vertices=int(self.polygon_vertices),
        )
        if timings is not None and t0 is not None:
            timings["footprint.vertices_elapsed_s"] = timings.get("footprint.vertices_elapsed_s", 0.0) + (
                time.perf_counter() - t0
            )

        t1 = time.perf_counter() if timings is not None else None
        pix = _moc_pixels_from_polygon_raw(lon_deg=lonv, lat_deg=latv, nside=int(nside))
        _ = (lon0_deg, lat0_deg)
        pix = np.unique(np.asarray(pix, dtype=np.int64))
        if timings is not None and t1 is not None:
            timings["footprint.rasterize_elapsed_s"] = timings.get("footprint.rasterize_elapsed_s", 0.0) + (
                time.perf_counter() - t1
            )
        if not bool(self.pad_neighbors):
            return pix
        t2 = time.perf_counter() if timings is not None else None
        out = _pad_neighbors(pixels=pix, nside=int(nside))
        if timings is not None and t2 is not None:
            timings["footprint.pad_neighbors_elapsed_s"] = timings.get(
                "footprint.pad_neighbors_elapsed_s", 0.0
            ) + (time.perf_counter() - t2)
        return out

