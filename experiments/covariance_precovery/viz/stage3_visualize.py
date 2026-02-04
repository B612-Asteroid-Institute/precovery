from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import healpy as hp
import numpy as np
import pyarrow.parquet as pq

from adam_core.time import Timestamp

from ..harness.stage3_healpixel_bench import (
    _predicted_pixels_from_mean_row,
    _predicted_pixels_from_samples,
    _timestamp_to_utc,
    FrameTimeTargets,
)
from ..methods.footprints import ellipse_boundary_vertices_lonlat_deg_from_cov


@dataclass(frozen=True)
class VizCase:
    strategy: str
    variant_kind: str | None
    orbit_id: str
    obscode: str
    time_utc: Timestamp  # length-1


def _tangent_xy_deg(
    lon_deg: np.ndarray, lat_deg: np.ndarray, lon0_deg: float, lat0_deg: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Gnomonic tangent-plane projection (in degrees).

    The previous small-angle approximation (Δlon*cos(lat0), Δlat) behaves badly for
    pixels far from the tangent point and can yield enormous polygons that “paint”
    the whole plot. The gnomonic projection keeps geometry well-behaved for typical
    small fields-of-view.
    """
    lon = np.deg2rad(lon_deg.astype(np.float64, copy=False))
    lat = np.deg2rad(lat_deg.astype(np.float64, copy=False))
    lon0 = float(np.deg2rad(float(lon0_deg)))
    lat0 = float(np.deg2rad(float(lat0_deg)))
    dlon = (lon - lon0 + np.pi) % (2 * np.pi) - np.pi

    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_lat0 = np.sin(lat0)
    cos_lat0 = np.cos(lat0)
    cos_dlon = np.cos(dlon)

    cosc = sin_lat0 * sin_lat + cos_lat0 * cos_lat * cos_dlon
    # Points >= 90deg from tangent point are not representable (cosc <= 0).
    with np.errstate(divide="ignore", invalid="ignore"):
        x = (cos_lat * np.sin(dlon)) / cosc
        y = (cos_lat0 * sin_lat - sin_lat0 * cos_lat * cos_dlon) / cosc
    x = np.rad2deg(x)
    y = np.rad2deg(y)
    x = np.where(cosc > 0, x, np.nan)
    y = np.where(cosc > 0, y, np.nan)
    return x.astype(np.float64), y.astype(np.float64)


def _pix_centers_extent_arcsec(*, nside: int, pix: np.ndarray, lon0_deg: float, lat0_deg: float) -> float:
    if pix.size == 0:
        return 0.0
    lon_c, lat_c = hp.pix2ang(int(nside), pix, nest=True, lonlat=True)
    x_c, y_c = _tangent_xy_deg(lon_c, lat_c, float(lon0_deg), float(lat0_deg))
    x = np.abs(x_c * 3600.0)
    y = np.abs(y_c * 3600.0)
    return float(max(float(np.max(x)) if x.size else 0.0, float(np.max(y)) if y.size else 0.0))


def _inverse_gnomonic_lonlat_deg(
    *, x_arcsec: np.ndarray, y_arcsec: np.ndarray, lon0_deg: float, lat0_deg: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Inverse gnomonic projection.

    Inputs are tangent-plane coordinates in arcsec; outputs are (lon,lat) in degrees.
    """
    x = np.deg2rad(x_arcsec.astype(np.float64, copy=False) / 3600.0)
    y = np.deg2rad(y_arcsec.astype(np.float64, copy=False) / 3600.0)
    lon0 = float(np.deg2rad(float(lon0_deg)))
    lat0 = float(np.deg2rad(float(lat0_deg)))

    rho = np.sqrt(x * x + y * y)
    c = np.arctan(rho)
    sin_c = np.sin(c)
    cos_c = np.cos(c)
    sin_lat0 = float(np.sin(lat0))
    cos_lat0 = float(np.cos(lat0))

    # Avoid division by zero at the origin.
    rho_safe = np.where(rho == 0.0, 1.0, rho)
    lat = np.arcsin(cos_c * sin_lat0 + (y * sin_c * cos_lat0) / rho_safe)
    lon = lon0 + np.arctan2(
        x * sin_c,
        rho * cos_lat0 * cos_c - y * sin_lat0 * sin_c,
    )
    lon = (lon + 2 * np.pi) % (2 * np.pi)
    return np.rad2deg(lon), np.rad2deg(lat)


def _imshow_healpix_set(
    *,
    ax: object,
    nside: int,
    pix: np.ndarray,
    lon0_deg: float,
    lat0_deg: float,
    lim_arcsec: float,
    color: str,
    alpha: float,
    grid: int = 700,
    zorder: int = 1,
) -> None:
    """
    Rasterize a HEALPix pixel-set onto the current tangent-plane axes.

    This avoids relying on `healpy.boundaries`, which can be unreliable for
    plotting pixel polygons depending on healpy version/build.
    """
    if pix.size == 0:
        return
    try:
        import matplotlib.colors as mcolors
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("matplotlib is required for visualization") from e

    xs = np.linspace(-float(lim_arcsec), float(lim_arcsec), int(grid), dtype=np.float64)
    ys = np.linspace(-float(lim_arcsec), float(lim_arcsec), int(grid), dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys)
    lon, lat = _inverse_gnomonic_lonlat_deg(x_arcsec=xx, y_arcsec=yy, lon0_deg=lon0_deg, lat0_deg=lat0_deg)
    pix_grid = hp.ang2pix(int(nside), lon, lat, lonlat=True, nest=True).astype(np.int64, copy=False)

    s = set([int(x) for x in np.asarray(pix, dtype=np.int64).tolist()])
    m = np.isin(pix_grid, list(s))
    if not np.any(m):
        return
    rgba = mcolors.to_rgba(color, alpha=float(alpha))
    img = np.zeros((m.shape[0], m.shape[1], 4), dtype=np.float64)
    img[m, :] = rgba  # type: ignore[index]
    ax.imshow(
        img,
        extent=[-float(lim_arcsec), float(lim_arcsec), -float(lim_arcsec), float(lim_arcsec)],
        origin="lower",
        interpolation="nearest",
        zorder=int(zorder),
        rasterized=True,
    )


def _plot_healpix_outlines_arcsec(
    *,
    ax: object,
    nside: int,
    pix: np.ndarray,
    lon0_deg: float,
    lat0_deg: float,
    color: str = "0.85",
    lw: float = 0.8,
) -> None:
    """
    Plot HEALPix pixel boundaries (not just centers) in the tangent plane.
    """
    if pix.size == 0:
        return
    try:
        import matplotlib.collections as mc
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("matplotlib is required for visualization") from e

    segs: list[np.ndarray] = []
    for p in pix.tolist():
        b = hp.boundaries(int(nside), int(p), step=1, nest=True)  # (3, Nv)
        lon_b, lat_b = hp.vec2ang(b, lonlat=True)
        x_b, y_b = _tangent_xy_deg(lon_b, lat_b, float(lon0_deg), float(lat0_deg))
        if not np.all(np.isfinite(x_b)) or not np.all(np.isfinite(y_b)):
            continue
        seg = np.column_stack([x_b * 3600.0, y_b * 3600.0])
        if seg.shape[0] >= 2:
            segs.append(np.vstack([seg, seg[0:1, :]]))

    if not segs:
        return
    ax.add_collection(mc.LineCollection(segs, colors=color, linewidths=lw))


def _plot_healpix_filled_pixels_arcsec(
    *,
    ax: object,
    nside: int,
    pix: np.ndarray,
    lon0_deg: float,
    lat0_deg: float,
    facecolor: str,
    edgecolor: str | None = None,
    alpha: float = 0.25,
    lw: float = 0.8,
) -> None:
    """
    Plot HEALPix pixels as filled polygons in the tangent plane.
    """
    if pix.size == 0:
        return
    try:
        import matplotlib.collections as mc
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("matplotlib is required for visualization") from e

    polys: list[np.ndarray] = []
    for p in pix.tolist():
        b = hp.boundaries(int(nside), int(p), step=1, nest=True)  # (3, Nv)
        lon_b, lat_b = hp.vec2ang(b, lonlat=True)
        x_b, y_b = _tangent_xy_deg(lon_b, lat_b, float(lon0_deg), float(lat0_deg))
        if not np.all(np.isfinite(x_b)) or not np.all(np.isfinite(y_b)):
            # Skip pixels that are too far from tangent point for a stable projection.
            continue
        poly = np.column_stack([x_b * 3600.0, y_b * 3600.0])
        if poly.shape[0] >= 3:
            polys.append(poly)
    if not polys:
        return
    pcoll = mc.PolyCollection(
        polys,
        facecolors=facecolor,
        edgecolors=(facecolor if edgecolor is None else edgecolor),
        linewidths=lw,
        alpha=float(alpha),
        closed=True,
    )
    ax.add_collection(pcoll)


def _load_frame_healpixels_for_target(
    *,
    subset_dir: Path,
    obscode: str,
    mjd_mid_utc: float,
    eps_days: float = 1e-8,
) -> np.ndarray:
    """
    Return all frame healpixels available for a single (obscode, exposure_mjd_mid).
    """
    conn = sqlite3.connect(str(subset_dir / "index.db"))
    try:
        rows = conn.execute(
            """
            SELECT healpixel
            FROM frames
            WHERE obscode = ?
              AND exposure_mjd_mid BETWEEN ? AND ?
            """,
            (str(obscode), float(mjd_mid_utc) - float(eps_days), float(mjd_mid_utc) + float(eps_days)),
        ).fetchall()
    finally:
        conn.close()
    if not rows:
        return np.array([], dtype=np.int64)
    a = np.unique(np.asarray([int(r[0]) for r in rows], dtype=np.int64))
    a.sort()
    return a


def _stage2_target_index(*, stage2_run_dir: Path, obscode: str, t_utc: Timestamp) -> tuple[int, float]:
    """
    Return (target_idx, mjd_mid_utc) for a given (obscode, time) from Stage 2 inputs.
    """
    targets = FrameTimeTargets.from_parquet(str(stage2_run_dir / "inputs" / "frame_time_targets.parquet"))
    t_utc = _timestamp_to_utc(t_utc)
    codes = np.asarray(targets.obscode.to_pylist(), dtype=object)
    days = targets.time.days.to_numpy(zero_copy_only=False).astype(np.int64)
    nanos = targets.time.nanos.to_numpy(zero_copy_only=False).astype(np.int64)
    mjd = targets.time.mjd().to_numpy(zero_copy_only=False).astype(np.float64)
    # Avoid exact (days,nanos) equality: some strategies produce tiny ns-level offsets.
    # We round to microseconds and allow a nearest match within 60s.
    step_ns = 1_000  # microsecond
    day_ns = 86_400 * 1_000_000_000
    d0 = int(t_utc.days[0].as_py())
    n0 = int(t_utc.nanos[0].as_py())
    n0 = int(((n0 + (step_ns // 2)) // step_ns) * step_ns)
    if n0 >= day_ns:
        d0 += (n0 // day_ns)
        n0 = n0 % day_ns
    key0 = d0 * day_ns + n0

    m = np.asarray([str(x) for x in codes], dtype=object) == str(obscode)
    if not m.any():
        raise KeyError(f"obscode not present in Stage 2 inputs: {obscode}")
    d = days[m].astype(np.int64, copy=False)
    n = nanos[m].astype(np.int64, copy=False)
    n = ((n + (step_ns // 2)) // step_ns) * step_ns
    carry = n // day_ns
    if np.any(carry != 0):
        d = d + carry
        n = n - carry * day_ns
    k = d * day_ns + n

    # Exact key match after rounding.
    hit = np.nonzero(k == key0)[0]
    if hit.size:
        i = int(np.nonzero(m)[0][int(hit[0])])
        return i, float(mjd[i])

    # Nearest within 60 seconds.
    order = np.argsort(k)
    ks = k[order]
    js = np.searchsorted(ks, key0)
    j0 = int(np.clip(js - 1, 0, len(ks) - 1))
    j1 = int(np.clip(js, 0, len(ks) - 1))
    cand = [(j0, abs(int(ks[j0]) - key0)), (j1, abs(int(ks[j1]) - key0))]
    jbest, dbest = min(cand, key=lambda x: x[1])
    if dbest <= 60 * 1_000_000_000:
        i_local = int(order[jbest])
        i = int(np.nonzero(m)[0][i_local])
        return i, float(mjd[i])

    key = (str(obscode), int(t_utc.days[0].as_py()), int(t_utc.nanos[0].as_py()))
    raise KeyError(f"Target (obscode,time) not found in Stage 2 inputs (rounded us, tol 60s): {key}")


def plot_stage3_case(
    *,
    subset_dir: Path,
    stage2_run_dir: Path,
    case: VizCase,
    healpix_nside: int,
    footprints: list[str],
    n_sigma: float = 3.0,
    polygon_vertices: int = 32,
    cov_mc_num_samples: int = 64,
    cov_mc_seed: int = 0,
    corridor_radius_arcsec: float = 30.0,
    corridor_step_arcsec: float = 30.0,
    frame_draw_mode: str = "outline",  # "outline" or "centers"
    min_view_arcsec: float = 30.0,
) -> None:
    """
    Quick-look visualization for a single (orbit, time) case:
      - frame healpixels at that exposure time (as pixel centers)
      - predicted footprint pixels (as pixel centers)
      - optional sample cloud (for variant strategies)

    This is meant to be called from an interactive session or notebook.
    """
    try:
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("matplotlib is required for visualization") from e

    t_utc = _timestamp_to_utc(case.time_utc)
    target_idx, mjd_mid_utc = _stage2_target_index(stage2_run_dir=stage2_run_dir, obscode=case.obscode, t_utc=t_utc)
    frame_pix = _load_frame_healpixels_for_target(subset_dir=subset_dir, obscode=case.obscode, mjd_mid_utc=mjd_mid_utc)

    # Pixel centers for frames.
    lon_f, lat_f = hp.pix2ang(int(healpix_nside), frame_pix, nest=True, lonlat=True)

    # Load ephemeris data for the case.
    if case.variant_kind is None:
        strat_meta = json.loads((stage2_run_dir / "strategies" / case.strategy / "meta.json").read_text())
        chunk = int(strat_meta.get("time_chunk_size", 1024))
        part_idx = int(target_idx // chunk)
        local = int(target_idx - part_idx * chunk)
        part_path = stage2_run_dir / "strategies" / case.strategy / "mean_ephemeris" / f"part-{part_idx:06d}.parquet"
        tbl = pq.read_table(str(part_path))

        orbit_ids = np.asarray(tbl.column("orbit_id").to_pylist(), dtype=object)
        # orbit-major order: orbit_idx*chunk + local
        try:
            orbit_idx = int(case.orbit_id)
        except Exception:
            orbit_idx = None
        row_idx = None if orbit_idx is None else int(orbit_idx * chunk + local)
        if row_idx is None or row_idx >= len(tbl) or str(orbit_ids[row_idx]) != str(case.orbit_id):
            # Fallback: filter by orbit_id and time key.
            c = tbl.column("coordinates")
            days = np.asarray(c.field("time").field("days").to_numpy(zero_copy_only=False), dtype=np.int64)
            nanos = np.asarray(c.field("time").field("nanos").to_numpy(zero_copy_only=False), dtype=np.int64)
            code = np.asarray(c.field("origin").field("code").to_pylist(), dtype=object)
            mask = (
                (orbit_ids == str(case.orbit_id))
                & (code == str(case.obscode))
                & (days == int(t_utc.days[0].as_py()))
                & (nanos == int(t_utc.nanos[0].as_py()))
            )
            idx = np.nonzero(mask)[0]
            if idx.size == 0:
                raise KeyError("Could not locate mean ephemeris row for case")
            row_idx = int(idx[0])

        coords = tbl.column("coordinates").combine_chunks()
        lon0 = float(coords.field("lon")[row_idx].as_py())
        lat0 = float(coords.field("lat")[row_idx].as_py())

        cov_ll = None
        try:
            cov_vals = coords.field("covariance").field("values")[row_idx].as_py()
            if cov_vals is not None:
                cov6 = np.asarray(cov_vals, dtype=np.float64).reshape(6, 6)
                cov_ll = cov6[1:3, 1:3].astype(np.float64)
        except Exception:
            cov_ll = None

        pix_by_fp: dict[str, np.ndarray] = {}
        for fp in footprints:
            pix_by_fp[fp] = _predicted_pixels_from_mean_row(
                lon_deg=lon0,
                lat_deg=lat0,
                cov_ll_deg2=cov_ll,
                nside=int(healpix_nside),
                footprint=str(fp),
                n_sigma=float(n_sigma),
                polygon_vertices=int(polygon_vertices),
                mc_num_samples=int(cov_mc_num_samples),
                mc_seed=int(cov_mc_seed),
            )

        # Plot in tangent plane around nominal.
        x_f, y_f = _tangent_xy_deg(lon_f, lat_f, lon0, lat0)
        fig, ax = plt.subplots(figsize=(7, 7))
        if str(frame_draw_mode).strip().lower() == "centers":
            ax.scatter(x_f * 3600, y_f * 3600, s=10, c="0.8", label="frame pixels", rasterized=True)
        else:
            _plot_healpix_filled_pixels_arcsec(
                ax=ax,
                nside=int(healpix_nside),
                pix=frame_pix,
                lon0_deg=lon0,
                lat0_deg=lat0,
                facecolor="0.9",
                edgecolor="none",
                alpha=0.15,
                lw=0.0,
            )
            _plot_healpix_outlines_arcsec(
                ax=ax,
                nside=int(healpix_nside),
                pix=frame_pix,
                lon0_deg=lon0,
                lat0_deg=lat0,
                color="0.70",
                lw=1.0,
            )
        ax.scatter([0.0], [0.0], s=40, c="k", marker="x", label="nominal")
        for i, (fp, pix) in enumerate(pix_by_fp.items()):
            c = f"C{int(i % 10)}"
            _plot_healpix_filled_pixels_arcsec(
                ax=ax,
                nside=int(healpix_nside),
                pix=pix,
                lon0_deg=lon0,
                lat0_deg=lat0,
                facecolor=c,
                edgecolor="none",
                alpha=0.06,
                lw=0.0,
            )
            _plot_healpix_outlines_arcsec(
                ax=ax,
                nside=int(healpix_nside),
                pix=pix,
                lon0_deg=lon0,
                lat0_deg=lat0,
                color="k",
                lw=1.2,
            )
            # Add a dummy handle so the footprint shows up in the legend.
            ax.plot([], [], color=c, lw=6, alpha=0.25, label=str(fp))
        lim = float(min_view_arcsec)
        # Ensure the view window is large enough to show full HEALPix pixel shapes.
        # At low nside (e.g. 32), a single pixel can be >1 degree across.
        pix_scale_arcsec = float(hp.nside2resol(int(healpix_nside), arcmin=False) * (180.0 / np.pi) * 3600.0)
        lim = max(lim, pix_scale_arcsec * 3.0)
        # Derive a reasonable zoom from predicted + frame pixel centers.
        ext = _pix_centers_extent_arcsec(nside=int(healpix_nside), pix=frame_pix, lon0_deg=lon0, lat0_deg=lat0)
        for pix in pix_by_fp.values():
            ext = max(ext, _pix_centers_extent_arcsec(nside=int(healpix_nside), pix=pix, lon0_deg=lon0, lat0_deg=lat0))
        if ext > 0:
            lim = max(lim, float(ext) * 1.2)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel("Δlon*cos(lat) [arcsec]")
        ax.set_ylabel("Δlat [arcsec]")
        ax.set_title(f"{case.strategy} orbit={case.orbit_id} {case.obscode} mjd={mjd_mid_utc:.6f}")
        ax.legend(markerscale=2)
        ax.set_aspect("equal", adjustable="box")
        fig.tight_layout()
        return

    # Variant case: load all samples for (orbit, time) and plot sample-derived footprints.
    #
    # IMPORTANT: Do not assume orbit-major chunking here (windowed strategies are not
    # chunked by `target_idx`). Instead, scan parts and pick the closest time key.
    strat_dir = stage2_run_dir / "strategies" / case.strategy / str(case.variant_kind)
    ephem_dir = strat_dir / "variants_ephemeris"
    part_files = sorted(ephem_dir.glob("part-*.parquet"))
    if not part_files:
        raise FileNotFoundError(f"Missing variants_ephemeris parts: {ephem_dir}")

    step_ns = 1_000  # microsecond
    day_ns = 86_400 * 1_000_000_000
    d0 = int(t_utc.days[0].as_py())
    n0 = int(t_utc.nanos[0].as_py())
    n0 = int(((n0 + (step_ns // 2)) // step_ns) * step_ns)
    if n0 >= day_ns:
        d0 += (n0 // day_ns)
        n0 = n0 % day_ns
    key0 = d0 * day_ns + n0

    best_lonlat: tuple[np.ndarray, np.ndarray] | None = None
    best_dist: int | None = None

    for pf in part_files:
        tbl = pq.read_table(str(pf), columns=["orbit_id", "coordinates"])
        if tbl.num_rows == 0:
            continue
        orbit_ids = np.asarray(tbl.column("orbit_id").to_pylist(), dtype=object)
        coords = tbl.column("coordinates").combine_chunks()
        code = np.asarray(coords.field("origin").field("code").to_pylist(), dtype=object)
        base = (orbit_ids == str(case.orbit_id)) & (code == str(case.obscode))
        base_idx = np.nonzero(base)[0]
        if base_idx.size == 0:
            continue

        days = np.asarray(coords.field("time").field("days").to_numpy(zero_copy_only=False), dtype=np.int64)
        nanos = np.asarray(coords.field("time").field("nanos").to_numpy(zero_copy_only=False), dtype=np.int64)
        lon = np.asarray(coords.field("lon").to_numpy(zero_copy_only=False), dtype=np.float64)
        lat = np.asarray(coords.field("lat").to_numpy(zero_copy_only=False), dtype=np.float64)

        d = days[base_idx].astype(np.int64, copy=False)
        n = nanos[base_idx].astype(np.int64, copy=False)
        n = ((n + (step_ns // 2)) // step_ns) * step_ns
        carry = n // day_ns
        if np.any(carry != 0):
            d = d + carry
            n = n - carry * day_ns
        k = d * day_ns + n

        # Choose the closest time in this part; then take all rows at that time (all variants).
        jbest = int(np.argmin(np.abs(k - key0)))
        key_best = int(k[jbest])
        dist = int(abs(key_best - int(key0)))

        if best_dist is None or dist < best_dist:
            m = k == key_best
            if not np.any(m):
                continue
            idx = base_idx[np.nonzero(m)[0]]
            best_lonlat = (lon[idx], lat[idx])
            best_dist = dist

    if best_lonlat is None:
        raise KeyError("Could not locate variant sample rows for case (orbit_id/obscode across parts)")

    lon_s, lat_s = best_lonlat
    lon0 = float(lon_s[0])
    lat0 = float(lat_s[0])

    pix_by_fp: dict[str, np.ndarray] = {}
    for fp in footprints:
        fp_key = str(fp)
        polygon_mode = None
        footprint = fp_key
        if fp_key.startswith("sample_polygon_moc:"):
            footprint = "sample_polygon_moc"
            polygon_mode = fp_key.split(":", 1)[1].strip() or None
        pix_by_fp[fp_key] = _predicted_pixels_from_samples(
            lon_deg=lon_s,
            lat_deg=lat_s,
            nside=int(healpix_nside),
            footprint=str(footprint),
            polygon_mode=polygon_mode,
            corridor_radius_arcsec=float(corridor_radius_arcsec),
            corridor_step_arcsec=float(corridor_step_arcsec),
        )

    x_f, y_f = _tangent_xy_deg(lon_f, lat_f, lon0, lat0)
    x_s, y_s = _tangent_xy_deg(lon_s, lat_s, lon0, lat0)
    fig, ax = plt.subplots(figsize=(7, 7))
    if str(frame_draw_mode).strip().lower() == "centers":
        ax.scatter(x_f * 3600, y_f * 3600, s=10, c="0.85", label="frame pixels", rasterized=True)
    else:
        _plot_healpix_filled_pixels_arcsec(
            ax=ax,
            nside=int(healpix_nside),
            pix=frame_pix,
            lon0_deg=lon0,
            lat0_deg=lat0,
            facecolor="0.9",
            edgecolor="none",
            alpha=0.15,
            lw=0.0,
        )
        _plot_healpix_outlines_arcsec(
            ax=ax,
            nside=int(healpix_nside),
            pix=frame_pix,
            lon0_deg=lon0,
            lat0_deg=lat0,
            color="0.70",
            lw=1.0,
        )
    ax.scatter(x_s * 3600, y_s * 3600, s=14, c="k", alpha=0.6, label="samples", rasterized=True)
    for i, (fp, pix) in enumerate(pix_by_fp.items()):
        c = f"C{int(i % 10)}"
        _plot_healpix_filled_pixels_arcsec(
            ax=ax,
            nside=int(healpix_nside),
            pix=pix,
            lon0_deg=lon0,
            lat0_deg=lat0,
            facecolor=c,
            edgecolor="none",
            alpha=0.06,
            lw=0.0,
        )
        _plot_healpix_outlines_arcsec(
            ax=ax,
            nside=int(healpix_nside),
            pix=pix,
            lon0_deg=lon0,
            lat0_deg=lat0,
            color="k",
            lw=1.2,
        )
        ax.plot([], [], color=c, lw=6, alpha=0.25, label=str(fp))
    lim = float(min_view_arcsec)
    pix_scale_arcsec = float(hp.nside2resol(int(healpix_nside), arcmin=False) * (180.0 / np.pi) * 3600.0)
    lim = max(lim, pix_scale_arcsec * 3.0)
    if x_s.size:
        lim = max(lim, float(np.max(np.abs(x_s * 3600))) * 1.2, float(np.max(np.abs(y_s * 3600))) * 1.2)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("Δlon*cos(lat) [arcsec]")
    ax.set_ylabel("Δlat [arcsec]")
    ax.set_title(f"{case.strategy}:{case.variant_kind} orbit={case.orbit_id} {case.obscode} mjd={mjd_mid_utc:.6f}")
    ax.legend(markerscale=2)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()


def plot_covariance_polygon_vs_truth(
    *,
    subset_dir: Path,
    stage2_run_dir: Path,
    stage3_run_dir: Path,
    strategy_key: str,
    orbit_id: str,
    target_idx: int,
    footprint: str = "cov_polygon_reconstructed_moc",
    facecolor: str = "C0",
    alpha: float = 0.25,
    dpi: int = 200,
    min_polygon_pixels: int = 50,
    n_sigma_override: float | None = None,
    draw_detection_uncertainty: bool = True,
    detection_sigma_levels: tuple[float, ...] = (1.0, 4.0),
    include_pred_mean: bool = True,
    zoom_include_pred_mean: bool = True,
    add_inset_when_scales_separate: bool = True,
    inset_scale_ratio_threshold: float = 50.0,
    layout: str = "single",
) -> Path:
    """
    Single-case debug plot:
    - draw N-sigma covariance polygon (filled, semi-transparent)
    - draw truth detection point
    - draw predicted mean point

    Uses Stage 3 geometry products (lon0/lat0 + cov_ll) and the existing covariance→polygon
    primitive (`ellipse_boundary_vertices_lonlat_deg_from_cov`).
    """
    try:
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("matplotlib is required for visualization") from e

    stage2_run_dir = Path(stage2_run_dir)
    stage3_run_dir = Path(stage3_run_dir)

    # Load Stage 2 target (obscode, time) for this target_idx.
    targets = FrameTimeTargets.from_parquet(str(stage2_run_dir / "inputs" / "frame_time_targets.parquet"))
    if int(target_idx) < 0 or int(target_idx) >= len(targets):
        raise IndexError(f"target_idx out of range: {target_idx}")
    obscode = str(targets.obscode[int(target_idx)].as_py())
    t_utc = _timestamp_to_utc(targets.time.take([int(target_idx)]).rescale("utc"))
    mjd_mid_utc = float(t_utc.mjd()[0].as_py())

    # Truth detection point for this (designation, obscode, time).
    truth_path = Path(subset_dir) / "artifacts" / "truth_precovery_crossmatch.parquet"
    truth = pq.read_table(
        str(truth_path),
        columns=[
            "matched",
            "designation",
            "obscode",
            "truth_ra_deg",
            "truth_dec_deg",
            "match_time_mjd_utc",
            "match_ra_deg",
            "match_dec_deg",
            "match_dataset_id",
            "match_exposure_id",
            "match_observation_id",
        ],
    ).combine_chunks()
    import pyarrow.compute as pc

    truth = truth.filter(pc.equal(truth["matched"], True))
    truth = truth.filter(pc.equal(truth["designation"], str(orbit_id)))
    truth = truth.filter(pc.equal(truth["obscode"], str(obscode)))
    if truth.num_rows == 0:
        raise KeyError(f"No truth row for designation={orbit_id} obscode={obscode}")
    mt = truth["match_time_mjd_utc"].to_numpy(zero_copy_only=False)
    j = int(np.argmin(np.abs(mt.astype(float) - float(mjd_mid_utc))))
    truth_ra = float(truth["truth_ra_deg"][j].as_py())
    truth_dec = float(truth["truth_dec_deg"][j].as_py())
    match_ra_py = truth["match_ra_deg"][j].as_py()
    match_dec_py = truth["match_dec_deg"][j].as_py()
    match_ra = float(match_ra_py) if match_ra_py is not None else float("nan")
    match_dec = float(match_dec_py) if match_dec_py is not None else float("nan")
    match_dataset_id = truth["match_dataset_id"][j].as_py()
    match_exposure_id = truth["match_exposure_id"][j].as_py()
    match_observation_id = truth["match_observation_id"][j].as_py()

    # Load Stage 3 geometry row for this (orbit_id, target_idx).
    geom_path = (
        Path(stage3_run_dir)
        / "footprint_geometry"
        / str(strategy_key)
        / str(footprint)
        / "geometry.parquet"
    )
    geom = pq.read_table(
        str(geom_path),
        columns=[
            "orbit_id",
            "target_idx",
            "lon0_deg",
            "lat0_deg",
            "cov_ll_00",
            "cov_ll_01",
            "cov_ll_10",
            "cov_ll_11",
            "n_sigma",
            "polygon_vertices",
        ],
    ).combine_chunks()
    geom = geom.filter(pc.equal(geom["orbit_id"], str(orbit_id)))
    geom = geom.filter(pc.equal(geom["target_idx"], int(target_idx)))
    if geom.num_rows == 0:
        raise KeyError(f"No geometry row for orbit_id={orbit_id} target_idx={target_idx} in {geom_path}")
    lon0 = float(geom["lon0_deg"][0].as_py())
    lat0 = float(geom["lat0_deg"][0].as_py())
    cov_ll = np.array(
        [
            [float(geom["cov_ll_00"][0].as_py()), float(geom["cov_ll_01"][0].as_py())],
            [float(geom["cov_ll_10"][0].as_py()), float(geom["cov_ll_11"][0].as_py())],
        ],
        dtype=np.float64,
    )
    n_sigma = float(geom["n_sigma"][0].as_py())
    if n_sigma_override is not None:
        n_sigma = float(n_sigma_override)
    n_vert = int(geom["polygon_vertices"][0].as_py())

    # Build covariance boundary polygon on the sphere.
    lon_poly, lat_poly = ellipse_boundary_vertices_lonlat_deg_from_cov(
        lon0_deg=lon0,
        lat0_deg=lat0,
        cov_ll_deg2=cov_ll,
        n_sigma=n_sigma,
        num_vertices=n_vert,
    )

    # Tangent plane centered on truth detection.
    x0, y0 = _tangent_xy_deg(np.array([lon0]), np.array([lat0]), truth_ra, truth_dec)
    xp, yp = _tangent_xy_deg(lon_poly, lat_poly, truth_ra, truth_dec)
    # convert to arcsec
    x0a, y0a = float(x0[0] * 3600.0), float(y0[0] * 3600.0)
    xp, yp = xp * 3600.0, yp * 3600.0

    # Matched (measured) detection point and its astrometric uncertainty ellipse (optional).
    xma = yma = float("nan")
    if np.isfinite(match_ra) and np.isfinite(match_dec):
        xm, ym = _tangent_xy_deg(np.array([match_ra]), np.array([match_dec]), truth_ra, truth_dec)
        xma, yma = float(xm[0] * 3600.0), float(ym[0] * 3600.0)

    det_sig_x_arcsec: float | None = None
    det_sig_y_arcsec: float | None = None
    if bool(draw_detection_uncertainty) and np.isfinite(xma) and np.isfinite(yma) and (match_observation_id is not None):
        try:
            from precovery.healpix_geom import radec_to_healpixel
            from precovery.precovery_db import PrecoveryDatabase
        except Exception:  # noqa: BLE001
            radec_to_healpixel = None  # type: ignore[assignment]
            PrecoveryDatabase = None  # type: ignore[assignment]

        if radec_to_healpixel is not None and PrecoveryDatabase is not None:
            try:
                db = PrecoveryDatabase.from_dir(str(subset_dir), mode="r", allow_version_mismatch=True)
                try:
                    nside_db = int(getattr(db.config, "nside", 32))
                except Exception:  # noqa: BLE001
                    nside_db = 32

                hpix_db = int(radec_to_healpixel(float(match_ra), float(match_dec), int(nside_db)))
                frames = db.frames.idx.frames_for_healpixel(int(hpix_db), str(obscode))
                if match_dataset_id is not None:
                    frames = frames.apply_mask(pc.equal(frames.dataset_id, str(match_dataset_id)))
                if match_exposure_id is not None:
                    frames = frames.apply_mask(pc.equal(frames.exposure_id, str(match_exposure_id)))
                if len(frames) > 1:
                    mjd = frames.exposure_mjd_mid.to_numpy(zero_copy_only=False).astype(np.float64)
                    k = int(np.argmin(np.abs(mjd - float(mjd_mid_utc))))
                    frames = frames.take([k])

                if len(frames) == 1:
                    obs = db.frames.get_observations(frames)
                    want = str(match_observation_id).encode()
                    ids = np.asarray(obs.id.to_pylist(), dtype=object)
                    hit = np.nonzero(ids == want)[0]
                    if hit.size > 0:
                        ii = int(hit[0])
                        ra_sig_deg = float(obs.ra_sigma[ii].as_py())
                        dec_sig_deg = float(obs.dec_sigma[ii].as_py())
                        ra_sig_arcsec = ra_sig_deg * 3600.0
                        dec_sig_arcsec = dec_sig_deg * 3600.0
                        det_sig_x_arcsec = float(ra_sig_arcsec * np.cos(np.deg2rad(float(match_dec))))
                        det_sig_y_arcsec = float(dec_sig_arcsec)

                try:
                    db.frames.close()
                except Exception:  # noqa: BLE001
                    pass
            except Exception:  # noqa: BLE001
                det_sig_x_arcsec = None
                det_sig_y_arcsec = None

    # Choose zoom: make polygon span at least `min_polygon_pixels` on the saved image.
    # Use a 6-inch square canvas.
    fig_w_in = 6.0
    axes_px = fig_w_in * float(dpi)
    poly_w = float(np.nanmax(xp) - np.nanmin(xp))
    poly_h = float(np.nanmax(yp) - np.nanmin(yp))
    poly_span = max(poly_w, poly_h, 1e-6)
    # Want poly_span * axes_px / (2*lim) >= min_polygon_pixels  => lim <= poly_span*axes_px/(2*min_polygon_pixels)
    lim = float(poly_span * axes_px / (2.0 * float(min_polygon_pixels)))
    # Also ensure we include detection uncertainty ellipse (if available) + some margin.
    if (det_sig_x_arcsec is not None) and (det_sig_y_arcsec is not None) and len(detection_sigma_levels) > 0:
        kmax = float(np.max(np.asarray(detection_sigma_levels, dtype=np.float64)))
        det_span = max(abs(float(det_sig_x_arcsec)), abs(float(det_sig_y_arcsec)), 0.0) * kmax * 2.0
        lim = max(lim, det_span * 2.0)
    # Keep a small margin around the polygon even if it is extremely tiny.
    lim = max(lim, poly_span * 0.6)
    # Optionally include the predicted mean point in view.
    if bool(zoom_include_pred_mean):
        lim = max(lim, float(max(abs(x0a), abs(y0a))) * 1.2)

    if str(layout) not in {"single", "two_panel"}:
        raise ValueError("layout must be 'single' or 'two_panel'")

    if str(layout) == "two_panel":
        fig, (ax_det, ax_cov) = plt.subplots(ncols=2, figsize=(10.5, 5.25), dpi=int(dpi))

        # Left: detection uncertainty scale.
        ax_det.scatter([0.0], [0.0], c="k", s=40, marker="x", label="truth detection")
        ax_det.scatter([x0a], [y0a], c="k", s=28, marker="o", label="cov center")
        if np.isfinite(xma) and np.isfinite(yma):
            ax_det.scatter([xma], [yma], c="k", s=28, marker="+", label="matched detection")
        # Draw covariance outline (will usually be tiny on this scale).
        ax_det.plot(xp, yp, color=str(facecolor), linewidth=2.0, alpha=0.9, label="cov 4σ footprint")

        if (det_sig_x_arcsec is not None) and (det_sig_y_arcsec is not None) and np.isfinite(xma) and np.isfinite(yma):
            try:
                from matplotlib.patches import Ellipse
            except Exception:  # noqa: BLE001
                Ellipse = None  # type: ignore[assignment]
            if Ellipse is not None and len(detection_sigma_levels) > 0:
                first = True
                for ksig in detection_sigma_levels:
                    ls = "--" if float(ksig) == 1.0 else "-"
                    ax_det.add_patch(
                        Ellipse(
                            (float(xma), float(yma)),
                            width=2.0 * float(ksig) * float(det_sig_x_arcsec),
                            height=2.0 * float(ksig) * float(det_sig_y_arcsec),
                            angle=0.0,
                            fill=False,
                            edgecolor="k",
                            linewidth=1.25,
                            linestyle=ls,
                            label=("det uncertainty" if first else None),
                        )
                    )
                    first = False

        # Set left limits based on detection uncertainty (4σ) when available.
        lim_det = 5.0
        if (det_sig_x_arcsec is not None) and (det_sig_y_arcsec is not None) and len(detection_sigma_levels) > 0:
            kmax = float(np.max(np.asarray(detection_sigma_levels, dtype=np.float64)))
            det_span = max(abs(float(det_sig_x_arcsec)), abs(float(det_sig_y_arcsec))) * kmax
            lim_det = max(lim_det, det_span * 1.6)
        # also include truth↔cov center distance
        lim_det = max(lim_det, float(max(abs(x0a), abs(y0a))) * 1.3)
        ax_det.set_xlim(-lim_det, lim_det)
        ax_det.set_ylim(-lim_det, lim_det)
        ax_det.set_aspect("equal", adjustable="box")
        ax_det.set_xlabel("Δlon*cos(lat) [arcsec]")
        ax_det.set_ylabel("Δlat [arcsec]")
        ax_det.set_title("Detection uncertainty scale")
        ax_det.legend(loc="upper right")

        # Right: covariance scale (show truth + matched + cov center + polygon).
        ax_cov.fill(xp, yp, facecolor=str(facecolor), alpha=float(alpha), edgecolor="none", zorder=1)
        ax_cov.plot(xp, yp, color=str(facecolor), linewidth=3.0, zorder=2, label="cov 4σ footprint")
        ax_cov.scatter([0.0], [0.0], c="k", s=40, marker="x", zorder=3, label="truth detection")
        ax_cov.scatter([x0a], [y0a], c="k", s=28, marker="o", zorder=3, label="cov center")
        if np.isfinite(xma) and np.isfinite(yma):
            ax_cov.scatter([xma], [yma], c="k", s=28, marker="+", zorder=3, label="matched detection")

        # Covariance-focused view: include truth/cov/matched but do NOT try to include full det-4σ ellipse.
        lim_cov = max(float(max(abs(x0a), abs(y0a))), float(max(abs(xma), abs(yma))) if np.isfinite(xma) else 0.0) * 1.5
        lim_cov = max(lim_cov, poly_span * 8.0, 0.05)
        ax_cov.set_xlim(-lim_cov, lim_cov)
        ax_cov.set_ylim(-lim_cov, lim_cov)
        ax_cov.set_aspect("equal", adjustable="box")
        ax_cov.set_xlabel("Δlon*cos(lat) [arcsec]")
        ax_cov.set_ylabel("Δlat [arcsec]")
        ax_cov.set_title("Covariance scale")
        ax_cov.legend(loc="upper right")

        fig.suptitle(f"{strategy_key} {orbit_id} {obscode} target={int(target_idx)} mjd={mjd_mid_utc:.6f}")
        fig.tight_layout()
    else:
        fig, ax = plt.subplots(figsize=(fig_w_in, fig_w_in), dpi=int(dpi))
        ax.fill(xp, yp, facecolor=str(facecolor), alpha=float(alpha), edgecolor=str(facecolor), linewidth=1.5)
        ax.scatter([0.0], [0.0], c="k", s=45, marker="x", label="truth detection")
        if np.isfinite(xma) and np.isfinite(yma):
            ax.scatter([xma], [yma], c="k", s=35, marker="+", label="matched detection")
            if (det_sig_x_arcsec is not None) and (det_sig_y_arcsec is not None):
                try:
                    from matplotlib.patches import Ellipse
                except Exception:  # noqa: BLE001
                    Ellipse = None  # type: ignore[assignment]
                if Ellipse is not None:
                    for ksig in detection_sigma_levels:
                        ls = "--" if float(ksig) == 1.0 else "-"
                        ax.add_patch(
                            Ellipse(
                                (float(xma), float(yma)),
                                width=2.0 * float(ksig) * float(det_sig_x_arcsec),
                                height=2.0 * float(ksig) * float(det_sig_y_arcsec),
                                angle=0.0,
                                fill=False,
                                edgecolor="k",
                                linewidth=1.25,
                                linestyle=ls,
                                label=f"det {float(ksig):g}σ",
                            )
                        )
        if bool(include_pred_mean):
            ax.scatter([x0a], [y0a], c="k", s=35, marker="o", label="pred mean")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Δlon*cos(lat) [arcsec]")
        ax.set_ylabel("Δlat [arcsec]")
        ax.set_title(
            f"{strategy_key} {orbit_id} {obscode} target={int(target_idx)} mjd={mjd_mid_utc:.6f}"
        )
        ax.legend()
        fig.tight_layout()

    out_dir = Path("experiments/covariance_precovery/tmp")
    out_dir.mkdir(parents=True, exist_ok=True)
    ns = f"{float(n_sigma):g}".replace(".", "p")
    out = out_dir / (
        f"cov_vs_truth_{strategy_key.replace('/', '_')}_{orbit_id}_t{int(target_idx)}_{ns}sigma.png"
    )
    fig.savefig(out)
    plt.close(fig)
    return out
