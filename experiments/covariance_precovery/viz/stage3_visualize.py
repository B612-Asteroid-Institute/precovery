from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import healpy as hp
import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

from adam_core.time import Timestamp

from ..harness.stage3_healpixel_bench import (
    _predicted_pixels_from_mean_row,
    _predicted_pixels_from_samples,
    _timestamp_to_utc,
    FrameTimeTargets,
)


@dataclass(frozen=True)
class VizCase:
    strategy: str
    variant_kind: str | None
    orbit_id: str
    obscode: str
    time_utc: Timestamp  # length-1


def _tangent_xy_deg(lon_deg: np.ndarray, lat_deg: np.ndarray, lon0_deg: float, lat0_deg: float) -> tuple[np.ndarray, np.ndarray]:
    dlon = (lon_deg - lon0_deg + 180.0) % 360.0 - 180.0
    x = dlon * np.cos(np.deg2rad(lat0_deg))
    y = lat_deg - lat0_deg
    return x.astype(np.float64), y.astype(np.float64)


def _load_frame_healpixels_for_target(*, subset_dir: Path, obscode: str, mjd_mid_utc: float) -> np.ndarray:
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
              AND exposure_mjd_mid = ?
            """,
            (str(obscode), float(mjd_mid_utc)),
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

    key = (str(obscode), int(t_utc.days[0].as_py()), int(t_utc.nanos[0].as_py()))
    for i in range(len(codes)):
        if str(codes[i]) == key[0] and int(days[i]) == key[1] and int(nanos[i]) == key[2]:
            return int(i), float(mjd[i])
    raise KeyError(f"Target (obscode,time) not found in Stage 2 inputs: {key}")


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

        coords = tbl.column("coordinates")
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
        ax.scatter(x_f * 3600, y_f * 3600, s=10, c="0.8", label="frame pixels", rasterized=True)
        ax.scatter([0.0], [0.0], s=40, c="k", marker="x", label="nominal")
        for fp, pix in pix_by_fp.items():
            lon_p, lat_p = hp.pix2ang(int(healpix_nside), pix, nest=True, lonlat=True)
            x_p, y_p = _tangent_xy_deg(lon_p, lat_p, lon0, lat0)
            ax.scatter(x_p * 3600, y_p * 3600, s=8, label=fp, rasterized=True)
        ax.set_xlabel("Δlon*cos(lat) [arcsec]")
        ax.set_ylabel("Δlat [arcsec]")
        ax.set_title(f"{case.strategy} orbit={case.orbit_id} {case.obscode} mjd={mjd_mid_utc:.6f}")
        ax.legend(markerscale=2)
        ax.set_aspect("equal", adjustable="box")
        fig.tight_layout()
        return

    # Variant case: load all samples for (orbit, time) and plot sample-derived footprints.
    strat_dir = stage2_run_dir / "strategies" / case.strategy / str(case.variant_kind)
    strat_meta = json.loads((strat_dir / "meta.json").read_text())
    chunk = int(strat_meta.get("time_chunk_size", 1024))
    part_idx = int(target_idx // chunk)
    part_path = strat_dir / "variants_ephemeris" / f"part-{part_idx:06d}.parquet"
    tbl = pq.read_table(str(part_path))
    orbit_ids = np.asarray(tbl.column("orbit_id").to_pylist(), dtype=object)
    coords = tbl.column("coordinates")
    code = np.asarray(coords.field("origin").field("code").to_pylist(), dtype=object)
    days = np.asarray(coords.field("time").field("days").to_numpy(zero_copy_only=False), dtype=np.int64)
    nanos = np.asarray(coords.field("time").field("nanos").to_numpy(zero_copy_only=False), dtype=np.int64)
    lon = np.asarray(coords.field("lon").to_numpy(zero_copy_only=False), dtype=np.float64)
    lat = np.asarray(coords.field("lat").to_numpy(zero_copy_only=False), dtype=np.float64)
    mask = (
        (orbit_ids == str(case.orbit_id))
        & (code == str(case.obscode))
        & (days == int(t_utc.days[0].as_py()))
        & (nanos == int(t_utc.nanos[0].as_py()))
    )
    idx = np.nonzero(mask)[0]
    if idx.size == 0:
        raise KeyError("Could not locate variant sample rows for case")
    lon_s = lon[idx]
    lat_s = lat[idx]
    lon0 = float(lon_s[0])
    lat0 = float(lat_s[0])

    pix_by_fp: dict[str, np.ndarray] = {}
    for fp in footprints:
        pix_by_fp[fp] = _predicted_pixels_from_samples(
            lon_deg=lon_s,
            lat_deg=lat_s,
            nside=int(healpix_nside),
            footprint=str(fp),
            corridor_radius_arcsec=float(corridor_radius_arcsec),
            corridor_step_arcsec=float(corridor_step_arcsec),
        )

    x_f, y_f = _tangent_xy_deg(lon_f, lat_f, lon0, lat0)
    x_s, y_s = _tangent_xy_deg(lon_s, lat_s, lon0, lat0)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(x_f * 3600, y_f * 3600, s=10, c="0.85", label="frame pixels", rasterized=True)
    ax.scatter(x_s * 3600, y_s * 3600, s=10, c="k", alpha=0.6, label="samples", rasterized=True)
    for fp, pix in pix_by_fp.items():
        lon_p, lat_p = hp.pix2ang(int(healpix_nside), pix, nest=True, lonlat=True)
        x_p, y_p = _tangent_xy_deg(lon_p, lat_p, lon0, lat0)
        ax.scatter(x_p * 3600, y_p * 3600, s=8, label=fp, rasterized=True)
    ax.set_xlabel("Δlon*cos(lat) [arcsec]")
    ax.set_ylabel("Δlat [arcsec]")
    ax.set_title(f"{case.strategy}:{case.variant_kind} orbit={case.orbit_id} {case.obscode} mjd={mjd_mid_utc:.6f}")
    ax.legend(markerscale=2)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()

