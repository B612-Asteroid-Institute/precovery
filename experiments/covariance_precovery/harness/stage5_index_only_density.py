from __future__ import annotations

import json
import math
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import quivr as qv
from adam_core.orbits import Orbits
from adam_core.orbits.ephemeris import Ephemeris

AU_KM = 149_597_870.700

from ..methods.covariance_metrics import (
    ellipse_area_deg2_from_cov_ll_deg2,
    expected_observations_from_bytes,
    sigma_major_arcsec_from_cov_ll_deg2,
)
from .stage3_healpixel_bench import FrameTimeTargets, _predicted_pixels_from_mean_row


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, obj: dict[str, object]) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _fmt_sec(x: float) -> str:
    x = float(x)
    if not np.isfinite(x) or x < 0:
        return "?"
    if x < 60:
        return f"{x:.1f}s"
    if x < 3600:
        return f"{x/60.0:.1f}m"
    return f"{x/3600.0:.2f}h"


def _log(msg: str) -> None:
    # Keep harness output readable in long runs.
    print(f"[stage5 { _now_utc() }] {msg}", flush=True)


def _designation_from_object_id(object_id: str) -> str:
    s = str(object_id).strip()
    if s.startswith("(") and s.endswith(")") and len(s) >= 3:
        return s[1:-1].strip()
    return s.split()[0].strip()


def _orbit_epoch_mjd_utc_by_key(orbits: Orbits) -> dict[str, float]:
    """
    Map stable orbit keys (designation-normalized) to orbit epoch MJD (UTC).
    """
    # Be robust: prefer object_id when present.
    obj_col = getattr(orbits, "object_id", None)
    if obj_col is None:
        keys = [str(x) for x in orbits.orbit_id.to_pylist()]
    else:
        keys = []
        for i, x in enumerate(obj_col.to_pylist()):
            s = "" if x is None else str(x).strip()
            keys.append(
                _designation_from_object_id(s) if s else str(orbits.orbit_id[i].as_py())
            )

    t = orbits.coordinates.time.rescale("utc")
    mjd = np.asarray(t.mjd().to_numpy(zero_copy_only=False), dtype=np.float64)
    out: dict[str, float] = {}
    for k, m in zip(keys, mjd.tolist()):
        kk = str(k)
        if kk and (kk not in out):
            out[kk] = float(m)
    return out


def _orbit_keys_from_ephem(ephem: Ephemeris) -> np.ndarray:
    obj_col = getattr(ephem, "object_id", None)
    if obj_col is None:
        return np.asarray([str(x) for x in ephem.orbit_id.to_pylist()], dtype=object)
    out: list[str] = []
    orbit_fallback = ephem.orbit_id.to_pylist()
    for i, x in enumerate(obj_col.to_pylist()):
        s = "" if x is None else str(x).strip()
        if s:
            out.append(_designation_from_object_id(s))
        else:
            out.append(str(orbit_fallback[i]) if i < len(orbit_fallback) else "")
    return np.asarray(out, dtype=object)


def _strategy_part_idx(p: Path) -> int:
    # part-000123.parquet -> 123
    stem = p.stem
    if "-" not in stem:
        raise ValueError(f"Unexpected part filename: {p.name}")
    return int(stem.split("-")[-1])


def _target_idx_for_part(
    *,
    part_idx: int,
    time_chunk_size: int,
    n_time_targets: int,
    n_orbits: int,
    n_rows: int,
) -> np.ndarray:
    start = int(part_idx) * int(time_chunk_size)
    chunk_len = int(min(int(time_chunk_size), int(n_time_targets) - int(start)))
    if chunk_len <= 0:
        return np.full(int(n_rows), -1, dtype=np.int64)
    tidx = np.tile((np.arange(chunk_len, dtype=np.int64) + int(start)), int(n_orbits))
    return tidx[: int(n_rows)].astype(np.int64, copy=False)


def _batch_id(abs_dt_days: float, batch_days: float) -> int:
    if not np.isfinite(abs_dt_days):
        return -1
    return int(math.floor(float(abs_dt_days) / float(batch_days)))


def _sample_target_indices(*, n: int, max_sampled: int | None) -> np.ndarray | None:
    """
    Return a sorted array of target_idx values to keep, sampled roughly uniformly over [0, n).
    """
    if max_sampled is None:
        return None
    m = int(max_sampled)
    if m <= 0:
        raise ValueError("max_sampled_targets must be > 0 when provided")
    n = int(n)
    if n <= 0:
        return np.array([], dtype=np.int64)
    if m >= n:
        return None
    idx = np.unique(np.rint(np.linspace(0, n - 1, m)).astype(np.int64))
    return np.sort(idx)


def _effective_obs_sigma_arcsec(astro_sigma_arcsec: float | None) -> float:
    """
    Observation 1-sigma astrometric error (arcsec) used to form an innovation covariance.

    We treat observation sigma as always > 0. If missing/invalid/<=0, we use a notional
    default of 0.1 arcsec.
    """
    default = 0.1
    if astro_sigma_arcsec is None:
        return float(default)
    s = float(astro_sigma_arcsec)
    if not np.isfinite(s) or s <= 0.0:
        return float(default)
    return float(s)


def _innov_cov_ll_deg2(
    *, cov_ll_deg2: np.ndarray, lat0_deg: float, obs_sigma_arcsec: float
) -> np.ndarray:
    """
    Innovation-style covariance inflation in the local tangent plane:

        C_xy_eff = C_xy_pred + sigma_obs^2 * I

    where x = Δlon * cos(lat0), y = Δlat (both in degrees), and sigma_obs is in degrees.

    Returns the equivalent lon/lat covariance (deg^2) so downstream footprint code can
    remain unchanged.
    """
    cov_ll = np.asarray(cov_ll_deg2, dtype=np.float64)
    cos_lat = float(np.cos(np.deg2rad(float(lat0_deg))))
    cos_lat = cos_lat if np.isfinite(cos_lat) and abs(cos_lat) > 1e-12 else 1e-12

    # Convert lon/lat -> tangent plane: C_xy = A C_ll A^T, A=diag(cos(lat0), 1).
    A = np.array([[cos_lat, 0.0], [0.0, 1.0]], dtype=np.float64)
    cov_xy = A @ cov_ll @ A.T
    cov_xy = 0.5 * (cov_xy + cov_xy.T)

    sig_deg = float(obs_sigma_arcsec) / 3600.0
    cov_xy_eff = cov_xy + (sig_deg * sig_deg) * np.eye(2, dtype=np.float64)

    # Convert back: C_ll = A^{-1} C_xy A^{-T}, A^{-1}=diag(1/cos(lat0), 1).
    Ainv = np.array([[1.0 / cos_lat, 0.0], [0.0, 1.0]], dtype=np.float64)
    cov_ll_eff = Ainv @ cov_xy_eff @ Ainv.T
    cov_ll_eff = 0.5 * (cov_ll_eff + cov_ll_eff.T)
    return cov_ll_eff.astype(np.float64, copy=False)


def _fractional_parent_weights_nested(
    *,
    pix_parent: np.ndarray,
    pix_child: np.ndarray,
    nside_parent: int,
    nside_child: int,
) -> dict[int, float]:
    """
    Compute fractional parent-pixel coverage from a child-pixel rasterization.

    Assumes NESTED indexing. For a ratio r = nside_child / nside_parent = 2**k, each parent
    pixel has r**2 child pixels, and the parent index is child >> (2*k).

    Returns a mapping {parent_pixel -> fraction_in_[0,1]} for all pixels in `pix_parent`.
    Pixels not hit by any child pixel get fraction=0.0.
    """
    pix_parent = np.unique(np.asarray(pix_parent, dtype=np.int64))
    if pix_parent.size == 0:
        return {}

    nside_parent = int(nside_parent)
    nside_child = int(nside_child)
    if nside_parent <= 0 or nside_child <= 0:
        raise ValueError("nsides must be positive")
    if nside_child < nside_parent:
        raise ValueError("nside_child must be >= nside_parent")
    ratio = int(nside_child // nside_parent)
    if ratio * nside_parent != nside_child:
        raise ValueError("nside_child must be an integer multiple of nside_parent")
    if ratio & (ratio - 1) != 0:
        raise ValueError(
            "nside_child / nside_parent must be a power of two for NESTED mapping"
        )
    k = int(np.log2(ratio))
    shift = int(2 * k)
    children_per_parent = int(ratio * ratio)

    pix_child = np.unique(np.asarray(pix_child, dtype=np.int64))
    if pix_child.size == 0:
        return {int(p): 0.0 for p in pix_parent.tolist()}

    parent_from_child = (pix_child >> shift).astype(np.int64, copy=False)
    parents, counts = np.unique(parent_from_child, return_counts=True)
    frac_by_parent = {
        int(p): float(c) / float(children_per_parent)
        for p, c in zip(parents.tolist(), counts.tolist())
    }
    return {int(p): float(frac_by_parent.get(int(p), 0.0)) for p in pix_parent.tolist()}


def _parent_pixels_and_weights_from_child_nested(
    *,
    pix_child: np.ndarray,
    nside_parent: int,
    nside_child: int,
) -> tuple[np.ndarray, dict[int, float]]:
    """
    Derive parent pixels and fractional weights from a child rasterization (NESTED).

    Returns:
      - parent_pixels (unique, sorted int64)
      - weights dict {parent_pixel -> fraction_in_[0,1]}
    """
    nside_parent = int(nside_parent)
    nside_child = int(nside_child)
    if nside_parent <= 0 or nside_child <= 0:
        raise ValueError("nsides must be positive")
    if nside_child < nside_parent:
        raise ValueError("nside_child must be >= nside_parent")
    ratio = int(nside_child // nside_parent)
    if ratio * nside_parent != nside_child:
        raise ValueError("nside_child must be an integer multiple of nside_parent")
    if ratio & (ratio - 1) != 0:
        raise ValueError(
            "nside_child / nside_parent must be a power of two for NESTED mapping"
        )
    k = int(np.log2(ratio))
    shift = int(2 * k)
    children_per_parent = int(ratio * ratio)

    pix_child = np.unique(np.asarray(pix_child, dtype=np.int64))
    if pix_child.size == 0:
        return np.array([], dtype=np.int64), {}

    parent_from_child = (pix_child >> shift).astype(np.int64, copy=False)
    parents, counts = np.unique(parent_from_child, return_counts=True)
    weights = {
        int(p): float(c) / float(children_per_parent)
        for p, c in zip(parents.tolist(), counts.tolist())
    }
    return parents.astype(np.int64, copy=False), weights


@dataclass
class _GroupAgg:
    n_targets: int = 0
    n_targets_hit: int = 0
    sum_pred_pixels: int = 0
    max_pred_pixels: int = 0

    sum_frames_touched: int = 0
    sum_data_length_bytes: int = 0
    sum_weighted_data_length_bytes: float = 0.0
    n_upper_bound_all_frames: int = 0

    # Observer->object range (from Stage 2 ephemeris `coordinates.rho`).
    n_rho: int = 0
    sum_rho_au: float = 0.0
    min_rho_au: float = float("inf")
    max_rho_au: float = float("-inf")

    sum_sigma_major_arcsec: float = 0.0
    max_sigma_major_arcsec: float = 0.0
    sum_ellipse_area_deg2: float = 0.0
    max_ellipse_area_deg2: float = 0.0

    # Physical-space covariance size (major axis 1σ of 3D position covariance), in km.
    n_sigma_major_pos_km: int = 0
    sum_sigma_major_pos_km: float = 0.0
    max_sigma_major_pos_km: float = 0.0

    # Predicted-covariance-only metrics (no observational variance added).
    sum_sigma_major_pred_arcsec: float = 0.0
    max_sigma_major_pred_arcsec: float = 0.0
    sum_ellipse_area_pred_deg2: float = 0.0
    max_ellipse_area_pred_deg2: float = 0.0

    abs_dt_min_days: float | None = None
    abs_dt_max_days: float | None = None

    def update_dt(self, abs_dt_days: float) -> None:
        x = float(abs_dt_days)
        if not np.isfinite(x):
            return
        if self.abs_dt_min_days is None or x < self.abs_dt_min_days:
            self.abs_dt_min_days = x
        if self.abs_dt_max_days is None or x > self.abs_dt_max_days:
            self.abs_dt_max_days = x


class Stage5IndexOnlyTimeSeries(qv.Table):
    subset_dir = qv.LargeStringColumn()
    stage2_run_dir = qv.LargeStringColumn()
    strategy = qv.LargeStringColumn()
    footprint = qv.LargeStringColumn()
    healpix_nside = qv.Int64Column()
    n_sigma = qv.Float64Column()
    batch_days = qv.Float64Column()
    direction = qv.LargeStringColumn()  # "abs" | "backward" | "forward"

    orbit_id = qv.LargeStringColumn()
    batch_id = qv.Int64Column()
    abs_dt_min_days = qv.Float64Column(nullable=True)
    abs_dt_max_days = qv.Float64Column(nullable=True)

    n_targets = qv.Int64Column()
    n_targets_hit = qv.Int64Column()
    hit_rate = qv.Float64Column(nullable=True)
    sum_pred_pixels = qv.Int64Column()
    max_pred_pixels = qv.Int64Column()

    sum_frames_touched = qv.Int64Column()
    sum_data_length_bytes = qv.Int64Column()
    bytes_per_exposure = qv.Float64Column()
    bytes_per_hit_exposure = qv.Float64Column(nullable=True)
    sum_weighted_data_length_bytes = qv.Float64Column(nullable=True)
    weighted_bytes_per_exposure = qv.Float64Column(nullable=True)
    expected_obs_per_exposure = qv.Float64Column(nullable=True)
    weighted_bytes_per_hit_exposure = qv.Float64Column(nullable=True)
    expected_obs_per_hit_exposure = qv.Float64Column(nullable=True)
    frames_per_exposure = qv.Float64Column()
    frames_per_hit_exposure = qv.Float64Column(nullable=True)

    n_upper_bound_all_frames = qv.Int64Column()

    rho_au_mean = qv.Float64Column(nullable=True)
    rho_au_min = qv.Float64Column(nullable=True)
    rho_au_max = qv.Float64Column(nullable=True)

    sigma_major_arcsec_mean = qv.Float64Column(nullable=True)
    sigma_major_arcsec_max = qv.Float64Column(nullable=True)
    ellipse_area_deg2_mean = qv.Float64Column(nullable=True)
    ellipse_area_deg2_max = qv.Float64Column(nullable=True)

    sigma_major_pos_km_mean = qv.Float64Column(nullable=True)
    sigma_major_pos_km_max = qv.Float64Column(nullable=True)

    sigma_major_pred_arcsec_mean = qv.Float64Column(nullable=True)
    sigma_major_pred_arcsec_max = qv.Float64Column(nullable=True)
    ellipse_area_pred_deg2_mean = qv.Float64Column(nullable=True)
    ellipse_area_pred_deg2_max = qv.Float64Column(nullable=True)


def run_stage5_index_only_density(
    *,
    subset_dir: Path,
    stage2_run_dir: Path,
    strategy: str,
    footprint: str,
    healpix_nside: int,
    n_sigma: float,
    polygon_vertices: int,
    batch_days: float = 7.0,
    out_dir: Path | None = None,
    orbits_parquet: Path | None = None,
    max_orbits: int | None = None,
    max_targets: int | None = None,
    max_sampled_targets: int | None = None,
    max_parts: int | None = None,
    max_pixels_exact: int = 20000,
    report_signed: bool = True,
    bytes_per_obs: float | None = None,
    astro_sigma_arcsec: float | None = None,
    bytes_per_exposure_max: float | None = None,
    consecutive_batches: int = 3,
    fractional_nside: int | None = None,
    orbit_ids: list[str] | None = None,
) -> Path:
    """
    Index-only Stage 5:
      - builds predicted footprint pixels from Stage 2 mean ephemerides (with covariance),
      - intersects with `index.db` frames,
      - aggregates `data_length` bytes as a proxy for candidate density vs |Δt|.
    """
    subset_dir = Path(subset_dir)
    stage2_run_dir = Path(stage2_run_dir)
    if out_dir is None:
        out_dir = subset_dir / "artifacts" / "stage5"
    _ensure_dir(out_dir)
    run_dir = out_dir / stage2_run_dir.name
    _ensure_dir(run_dir)

    index_db = subset_dir / "index.db"
    config_json = subset_dir / "config.json"
    if not index_db.exists():
        raise FileNotFoundError(f"Missing index.db: {index_db}")
    if not config_json.exists():
        raise FileNotFoundError(f"Missing config.json: {config_json}")

    # Default orbits parquet.
    if orbits_parquet is None:
        orbits_parquet = subset_dir / "artifacts" / "orbits_selected_sbdb.parquet"
    orbits_parquet = Path(orbits_parquet)
    if not orbits_parquet.exists():
        raise FileNotFoundError(
            f"Missing orbits parquet (needed for orbit epochs): {orbits_parquet}"
        )

    # Targets.
    targets = FrameTimeTargets.from_parquet(
        str(stage2_run_dir / "inputs" / "frame_time_targets.parquet")
    )
    targ_obscode = np.asarray(targets.obscode.to_pylist(), dtype=object)
    targ_mjd = np.asarray(
        targets.time.mjd().to_numpy(zero_copy_only=False), dtype=np.float64
    )
    n_time_targets = int(len(targets))
    if max_targets is not None:
        n_time_targets = int(min(n_time_targets, int(max_targets)))
        targ_obscode = targ_obscode[:n_time_targets]
        targ_mjd = targ_mjd[:n_time_targets]

    sampled_target_idx = _sample_target_indices(
        n=int(n_time_targets), max_sampled=max_sampled_targets
    )
    sampled_keep: np.ndarray | None = None
    if sampled_target_idx is not None:
        sampled_keep = np.zeros(int(n_time_targets), dtype=bool)
        sampled_keep[sampled_target_idx] = True

    # Orbit epochs.
    orbits = Orbits.from_parquet(str(orbits_parquet))
    if max_orbits is not None:
        orbits = orbits[: int(max_orbits)]
    epoch_by_key = _orbit_epoch_mjd_utc_by_key(orbits)
    if not epoch_by_key:
        raise ValueError("No orbit epochs found in orbits parquet.")

    # Strategy ephemeris parts.
    strat_dir = stage2_run_dir / "strategies" / str(strategy) / "mean_ephemeris"
    meta_path = stage2_run_dir / "strategies" / str(strategy) / "meta.json"
    if not strat_dir.exists():
        raise FileNotFoundError(f"Missing Stage2 mean ephemeris dir: {strat_dir}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing Stage2 strategy meta.json: {meta_path}")

    meta = json.loads(meta_path.read_text())
    time_chunk_size = int(meta.get("time_chunk_size", n_time_targets))
    n_orbits_meta = int(meta.get("n_orbits", len(orbits)))
    # `n_time_targets` in meta is the original target count; use our possibly capped value.
    n_time_targets_eff = int(n_time_targets)

    part_files = sorted(strat_dir.glob("part-*.parquet"))
    if not part_files:
        raise FileNotFoundError(f"No ephemeris parts found in {strat_dir}")
    if max_parts is not None:
        part_files = part_files[: int(max_parts)]

    # Basic run header.
    obs_sigma_arcsec_used = _effective_obs_sigma_arcsec(astro_sigma_arcsec)
    _log(
        "START"
        f" subset={subset_dir}"
        f" stage2={stage2_run_dir.name}"
        f" strategy={strategy}"
        f" footprint={footprint}"
        f" nside={int(healpix_nside)}"
        f" frac_nside={(None if fractional_nside is None else int(fractional_nside))}"
        f" n_sigma={float(n_sigma):g}"
        f" poly_v={int(polygon_vertices)}"
        f" astro_sigma_arcsec={(None if astro_sigma_arcsec is None else float(astro_sigma_arcsec))}"
        f" astro_sigma_arcsec_used={float(obs_sigma_arcsec_used)}"
        f" targets={int(n_time_targets_eff)}"
        f" sampled_targets={(None if max_sampled_targets is None else int(max_sampled_targets))}"
        f" parts={len(part_files)}"
        f" time_chunk={int(time_chunk_size)}"
        f" max_pixels_exact={int(max_pixels_exact)}"
        f" orbit_ids={(None if not orbit_ids else ','.join([str(x) for x in orbit_ids]))}"
    )

    # Cache total-by-exposure for upper-bound fallback.
    total_cache: dict[tuple[str, float], tuple[int, int]] = {}

    def _total_for_exposure(
        *, conn: sqlite3.Connection, obscode: str, exposure_mjd_mid: float
    ) -> tuple[int, int]:
        k = (str(obscode), float(exposure_mjd_mid))
        if k in total_cache:
            return total_cache[k]
        row = conn.execute(
            """
            SELECT COUNT(*), SUM(data_length)
            FROM frames
            WHERE obscode = ?
              AND exposure_mjd_mid = ?
            """,
            (str(obscode), float(exposure_mjd_mid)),
        ).fetchone()
        n = int(row[0]) if row and row[0] is not None else 0
        b = int(row[1]) if row and row[1] is not None else 0
        total_cache[k] = (n, b)
        return n, b

    # Aggregation: (orbit_id, batch_id, direction) -> _GroupAgg
    agg: dict[tuple[str, int, str], _GroupAgg] = {}

    # We store per-target info only within a part, then aggregate immediately.
    conn = sqlite3.connect(str(index_db))
    try:
        conn.execute("PRAGMA temp_store=MEMORY;")
        conn.execute("PRAGMA journal_mode=OFF;")
        conn.execute("PRAGMA synchronous=OFF;")
        conn.execute("DROP TABLE IF EXISTS pred;")
        conn.execute(
            "CREATE TEMP TABLE pred (orbit_id TEXT, target_idx INTEGER, obscode TEXT, exposure_mjd_mid REAL, healpixel INTEGER, weight REAL)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS pred_idx ON pred (obscode, exposure_mjd_mid, healpixel)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS pred_key_idx ON pred (orbit_id, target_idx)"
        )
        conn.commit()

        t_run0 = time.perf_counter()
        part_times: list[float] = []
        for p_i, pf in enumerate(part_files):
            t_part0 = time.perf_counter()
            ephem = Ephemeris.from_parquet(str(pf))
            if len(ephem) == 0:
                continue
            n_rows_raw = int(len(ephem))

            # Compute target_idx for this part.
            part_idx = _strategy_part_idx(pf)
            target_idx = _target_idx_for_part(
                part_idx=part_idx,
                time_chunk_size=int(time_chunk_size),
                n_time_targets=int(n_time_targets_eff),
                n_orbits=int(n_orbits_meta),
                n_rows=int(len(ephem)),
            )
            # Mask invalid or capped targets.
            m_ok = (target_idx >= 0) & (target_idx < int(n_time_targets_eff))
            if sampled_keep is not None:
                m_ok = m_ok & sampled_keep[target_idx]
            if not bool(np.any(m_ok)):
                continue
            if not bool(np.all(m_ok)):
                hit = np.nonzero(m_ok)[0]
                ephem = ephem.take(hit.tolist())
                target_idx = target_idx[hit]
            n_rows_used = int(len(ephem))

            orbit_key = _orbit_keys_from_ephem(ephem)
            if orbit_ids:
                wanted = {str(x).strip() for x in orbit_ids if str(x).strip()}
                if wanted:
                    m_orb = np.isin(orbit_key, np.asarray(sorted(wanted), dtype=object))
                    if not bool(np.any(m_orb)):
                        continue
                    if not bool(np.all(m_orb)):
                        hit = np.nonzero(m_orb)[0]
                        ephem = ephem.take(hit.tolist())
                        target_idx = target_idx[hit]
                        orbit_key = orbit_key[hit]
                        n_rows_used = int(len(ephem))

            lon = ephem.coordinates.lon.to_numpy(zero_copy_only=False).astype(
                np.float64
            )
            lat = ephem.coordinates.lat.to_numpy(zero_copy_only=False).astype(
                np.float64
            )
            rho = ephem.coordinates.rho.to_numpy(zero_copy_only=False).astype(
                np.float64
            )

            cov6 = None
            if ephem.coordinates.covariance is not None and (
                not ephem.coordinates.covariance.is_all_nan()
            ):
                cov6 = ephem.coordinates.covariance.to_matrix().astype(np.float64)
            else:
                raise ValueError(
                    f"Strategy {strategy!r} ephemeris has no covariance; Stage 5 index-only requires covariance."
                )

            # Physical-space position covariance (km): transform spherical (rho, lon, lat)
            # covariance into Cartesian (x,y,z) covariance via Jacobian, then take the
            # 1σ major axis length of the 3D position covariance.
            #
            # Inputs:
            #   - rho in AU
            #   - lon/lat in degrees (and covariance in deg^2 / AU*deg)
            #
            # We convert angular units to radians inside the covariance before applying
            # the Jacobian in radian form.
            sigma_major_pos_km = np.full(int(len(ephem)), np.nan, dtype=np.float64)
            try:
                n_ep = int(len(ephem))
                if n_ep > 0:
                    lon_rad = np.deg2rad(lon)
                    lat_rad = np.deg2rad(lat)
                    r_au = rho.astype(np.float64, copy=False)

                    # Position covariance in (rho, lon, lat) with lon/lat in *radians*.
                    pos_cov = cov6[:, 0:3, 0:3].astype(np.float64, copy=True)
                    d2r = float(np.pi / 180.0)
                    # Scale rows/cols for lon/lat from degrees -> radians.
                    pos_cov[:, 1, :] *= d2r
                    pos_cov[:, 2, :] *= d2r
                    pos_cov[:, :, 1] *= d2r
                    pos_cov[:, :, 2] *= d2r

                    clat = np.cos(lat_rad)
                    slat = np.sin(lat_rad)
                    clon = np.cos(lon_rad)
                    slon = np.sin(lon_rad)

                    J = np.zeros((n_ep, 3, 3), dtype=np.float64)
                    # x = r cos(lat) cos(lon)
                    J[:, 0, 0] = clat * clon
                    J[:, 0, 1] = -r_au * clat * slon
                    J[:, 0, 2] = -r_au * slat * clon
                    # y = r cos(lat) sin(lon)
                    J[:, 1, 0] = clat * slon
                    J[:, 1, 1] = r_au * clat * clon
                    J[:, 1, 2] = -r_au * slat * slon
                    # z = r sin(lat)
                    J[:, 2, 0] = slat
                    J[:, 2, 1] = 0.0
                    J[:, 2, 2] = r_au * clat

                    tmp = np.einsum("nij,njk->nik", J, pos_cov)
                    pos_cov_xyz = np.einsum("nij,nkj->nik", tmp, J)
                    pos_cov_xyz = 0.5 * (
                        pos_cov_xyz + np.swapaxes(pos_cov_xyz, 1, 2)
                    )
                    ok = np.isfinite(pos_cov_xyz).all(axis=(1, 2))
                    if bool(np.any(ok)):
                        eig = np.linalg.eigvalsh(pos_cov_xyz[ok])
                        lam_max = eig[:, -1]
                        lam_max = np.maximum(lam_max, 0.0)
                        sigma_major_pos_km[ok] = np.sqrt(lam_max) * float(AU_KM)
            except Exception:  # noqa: BLE001
                sigma_major_pos_km = np.full(int(len(ephem)), np.nan, dtype=np.float64)

            # Per-target info.
            key_info: dict[tuple[str, int], dict[str, object]] = {}
            pred_rows: list[tuple[str, int, str, float, int, float]] = []

            t_row0 = time.perf_counter()
            last_row_log_t = float(t_row0)
            for i in range(len(ephem)):
                if (time.perf_counter() - float(last_row_log_t)) > 30.0 and len(
                    ephem
                ) > 0:
                    now = time.perf_counter()
                    frac = float(i) / float(len(ephem))
                    rate = float(i) / max(1e-9, float(now - float(t_row0)))
                    remain = float(len(ephem) - i) / max(1e-9, float(rate))
                    _log(
                        f"part {p_i+1}/{len(part_files)} {pf.name} rows {i}/{len(ephem)} "
                        f"({100.0*frac:.1f}%) rate={rate:.1f}/s eta_part={_fmt_sec(remain)}"
                    )
                    last_row_log_t = float(now)

                oid = str(orbit_key[i])
                tidx = int(target_idx[i])
                if not oid:
                    continue
                epoch_mjd = epoch_by_key.get(oid)
                if epoch_mjd is None:
                    # Orbit not present in the orbits parquet (e.g., Stage2 ephem produced from a different file).
                    continue

                obscode = str(targ_obscode[tidx])
                mjd_mid = float(targ_mjd[tidx])
                dt_days = float(mjd_mid - float(epoch_mjd))
                abs_dt = float(abs(dt_days))
                rho_au = float(rho[i])
                sigma_pos_km = float(sigma_major_pos_km[i])

                cov_ll = cov6[i, 1:3, 1:3].astype(np.float64, copy=False)
                sigma_major_pred_arcsec = sigma_major_arcsec_from_cov_ll_deg2(
                    cov_ll_deg2=cov_ll, lat0_deg=float(lat[i])
                )
                ellipse_area_pred_deg2 = ellipse_area_deg2_from_cov_ll_deg2(
                    cov_ll_deg2=cov_ll,
                    lat0_deg=float(lat[i]),
                    n_sigma=float(n_sigma),
                )
                cov_ll_eff = _innov_cov_ll_deg2(
                    cov_ll_deg2=cov_ll,
                    lat0_deg=float(lat[i]),
                    obs_sigma_arcsec=float(obs_sigma_arcsec_used),
                )

                sigma_major_arcsec = sigma_major_arcsec_from_cov_ll_deg2(
                    cov_ll_deg2=cov_ll_eff, lat0_deg=float(lat[i])
                )
                ellipse_area_deg2 = ellipse_area_deg2_from_cov_ll_deg2(
                    cov_ll_deg2=cov_ll_eff,
                    lat0_deg=float(lat[i]),
                    n_sigma=float(n_sigma),
                )

                weights: dict[int, float] | None = None
                if fractional_nside is None:
                    pix = _predicted_pixels_from_mean_row(
                        lon_deg=float(lon[i]),
                        lat_deg=float(lat[i]),
                        cov_ll_deg2=cov_ll_eff,
                        nside=int(healpix_nside),
                        footprint=str(footprint),
                        n_sigma=float(n_sigma),
                        polygon_vertices=int(polygon_vertices),
                        mc_num_samples=64,
                        mc_seed=0,
                    )
                    pix = np.unique(np.asarray(pix, dtype=np.int64))
                else:
                    # Fast precheck: if a conservative disc at parent nside is already huge,
                    # skip expensive high-order rasterization and use an upper bound.
                    if str(footprint) == "cov_polygon_moc":
                        pix_pre = _predicted_pixels_from_mean_row(
                            lon_deg=float(lon[i]),
                            lat_deg=float(lat[i]),
                            cov_ll_deg2=cov_ll_eff,
                            nside=int(healpix_nside),
                            footprint="cov_disc",
                            n_sigma=float(n_sigma),
                            polygon_vertices=int(polygon_vertices),
                            mc_num_samples=64,
                            mc_seed=0,
                        )
                        pix_pre = np.unique(np.asarray(pix_pre, dtype=np.int64))
                        if int(pix_pre.size) > int(max_pixels_exact):
                            pix = pix_pre
                            weights = {int(p): 1.0 for p in pix.tolist()}
                        else:
                            pix_child = _predicted_pixels_from_mean_row(
                                lon_deg=float(lon[i]),
                                lat_deg=float(lat[i]),
                                cov_ll_deg2=cov_ll_eff,
                                nside=int(fractional_nside),
                                footprint=str(footprint),
                                n_sigma=float(n_sigma),
                                polygon_vertices=int(polygon_vertices),
                                mc_num_samples=64,
                                mc_seed=0,
                            )
                            pix, weights = _parent_pixels_and_weights_from_child_nested(
                                pix_child=np.asarray(pix_child, dtype=np.int64),
                                nside_parent=int(healpix_nside),
                                nside_child=int(fractional_nside),
                            )
                    else:
                        pix_child = _predicted_pixels_from_mean_row(
                            lon_deg=float(lon[i]),
                            lat_deg=float(lat[i]),
                            cov_ll_deg2=cov_ll_eff,
                            nside=int(fractional_nside),
                            footprint=str(footprint),
                            n_sigma=float(n_sigma),
                            polygon_vertices=int(polygon_vertices),
                            mc_num_samples=64,
                            mc_seed=0,
                        )
                        pix, weights = _parent_pixels_and_weights_from_child_nested(
                            pix_child=np.asarray(pix_child, dtype=np.int64),
                            nside_parent=int(healpix_nside),
                            nside_child=int(fractional_nside),
                        )

                    pix = np.unique(np.asarray(pix, dtype=np.int64))
                    # Ensure we always have a weight for each parent pixel.
                    if weights is None:
                        weights = {int(p): 1.0 for p in pix.tolist()}
                    else:
                        weights = {
                            int(p): float(weights.get(int(p), 0.0))
                            for p in pix.tolist()
                        }

                n_pix = int(pix.size)

                k = (oid, tidx)
                key_info[k] = dict(
                    orbit_id=oid,
                    target_idx=tidx,
                    obscode=obscode,
                    exposure_mjd_mid=mjd_mid,
                    dt_days=dt_days,
                    abs_dt_days=abs_dt,
                    rho_au=rho_au,
                    sigma_major_arcsec=float(sigma_major_arcsec),
                    sigma_major_pred_arcsec=float(sigma_major_pred_arcsec),
                    sigma_major_pos_km=float(sigma_pos_km),
                    ellipse_area_deg2=float(ellipse_area_deg2),
                    ellipse_area_pred_deg2=float(ellipse_area_pred_deg2),
                    n_pred_pixels=int(n_pix),
                    upper_bound_all_frames=False,
                    upper_n_frames=0,
                    upper_sum_bytes=0,
                    upper_sum_weighted_bytes=float("nan"),
                )

                if n_pix == 0:
                    continue
                if int(n_pix) > int(max_pixels_exact):
                    # Upper bound: the footprint is so large that we'd likely stop anyway.
                    n_all, b_all = _total_for_exposure(
                        conn=conn, obscode=obscode, exposure_mjd_mid=mjd_mid
                    )
                    key_info[k]["upper_bound_all_frames"] = True
                    key_info[k]["upper_n_frames"] = int(n_all)
                    key_info[k]["upper_sum_bytes"] = int(b_all)
                    key_info[k]["upper_sum_weighted_bytes"] = float(b_all)
                    continue

                if weights is None:
                    pred_rows.extend(
                        (oid, tidx, obscode, mjd_mid, int(h), 1.0) for h in pix.tolist()
                    )
                else:
                    pred_rows.extend(
                        (
                            oid,
                            tidx,
                            obscode,
                            mjd_mid,
                            int(h),
                            float(weights.get(int(h), 0.0)),
                        )
                        for h in pix.tolist()
                    )

            # Insert predicted rows and aggregate via join.
            conn.execute("DELETE FROM pred;")
            if pred_rows:
                conn.executemany(
                    "INSERT INTO pred (orbit_id, target_idx, obscode, exposure_mjd_mid, healpixel, weight) VALUES (?, ?, ?, ?, ?, ?)",
                    pred_rows,
                )
                conn.commit()

            # Aggregate exact intersections for the keys we inserted.
            exact_map: dict[tuple[str, int], tuple[int, int, float]] = {}
            if pred_rows:
                rows = conn.execute(
                    """
                    SELECT
                        p.orbit_id,
                        p.target_idx,
                        COUNT(*) AS n_frames,
                        SUM(f.data_length) AS sum_bytes,
                        SUM(p.weight * f.data_length) AS sum_weighted_bytes
                    FROM pred p
                    INNER JOIN frames f
                      ON f.obscode = p.obscode
                     AND f.exposure_mjd_mid = p.exposure_mjd_mid
                     AND f.healpixel = p.healpixel
                    GROUP BY p.orbit_id, p.target_idx
                    """
                ).fetchall()
                for r in rows:
                    ok = str(r[0])
                    tk = int(r[1])
                    nf = int(r[2]) if r[2] is not None else 0
                    sb = int(r[3]) if r[3] is not None else 0
                    sw = float(r[4]) if r[4] is not None else 0.0
                    exact_map[(ok, tk)] = (nf, sb, sw)

            # Update group aggregations.
            for k, info in key_info.items():
                oid = str(info["orbit_id"])
                tidx = int(info["target_idx"])
                dt_days = float(info["dt_days"])
                abs_dt = float(info["abs_dt_days"])
                n_pix = int(info["n_pred_pixels"])
                sig_arcsec = float(info["sigma_major_arcsec"])
                sig_pred_arcsec = float(info["sigma_major_pred_arcsec"])
                area_deg2 = float(info["ellipse_area_deg2"])
                area_pred_deg2 = float(info["ellipse_area_pred_deg2"])

                # Frames/bytes.
                if bool(info["upper_bound_all_frames"]):
                    n_frames = int(info["upper_n_frames"])
                    sum_bytes = int(info["upper_sum_bytes"])
                    sum_w_bytes = float(info["upper_sum_weighted_bytes"])
                    upper = True
                else:
                    n_frames, sum_bytes, sum_w_bytes = exact_map.get(
                        (oid, tidx), (0, 0, 0.0)
                    )
                    upper = False

                # Binning by |dt|.
                bid = _batch_id(abs_dt, float(batch_days))
                if bid < 0:
                    continue

                def _update(direction: str) -> None:
                    gk = (oid, int(bid), str(direction))
                    a = agg.get(gk)
                    if a is None:
                        a = _GroupAgg()
                        agg[gk] = a
                    a.n_targets += 1
                    if int(n_frames) > 0 or bool(upper):
                        a.n_targets_hit += 1
                    a.sum_pred_pixels += int(n_pix)
                    a.max_pred_pixels = max(int(a.max_pred_pixels), int(n_pix))
                    a.sum_frames_touched += int(n_frames)
                    a.sum_data_length_bytes += int(sum_bytes)
                    a.sum_weighted_data_length_bytes += float(sum_w_bytes)
                    if upper:
                        a.n_upper_bound_all_frames += 1
                    rho_au = float(info.get("rho_au", float("nan")))
                    if np.isfinite(rho_au):
                        a.n_rho += 1
                        a.sum_rho_au += float(rho_au)
                        a.min_rho_au = min(float(a.min_rho_au), float(rho_au))
                        a.max_rho_au = max(float(a.max_rho_au), float(rho_au))
                    if np.isfinite(sig_arcsec):
                        a.sum_sigma_major_arcsec += float(sig_arcsec)
                        a.max_sigma_major_arcsec = max(
                            float(a.max_sigma_major_arcsec), float(sig_arcsec)
                        )
                    sig_pos_km = float(info.get("sigma_major_pos_km", float("nan")))
                    if np.isfinite(sig_pos_km) and sig_pos_km >= 0.0:
                        a.n_sigma_major_pos_km += 1
                        a.sum_sigma_major_pos_km += float(sig_pos_km)
                        a.max_sigma_major_pos_km = max(
                            float(a.max_sigma_major_pos_km), float(sig_pos_km)
                        )
                    if np.isfinite(area_deg2):
                        a.sum_ellipse_area_deg2 += float(area_deg2)
                        a.max_ellipse_area_deg2 = max(
                            float(a.max_ellipse_area_deg2), float(area_deg2)
                        )
                    if np.isfinite(sig_pred_arcsec):
                        a.sum_sigma_major_pred_arcsec += float(sig_pred_arcsec)
                        a.max_sigma_major_pred_arcsec = max(
                            float(a.max_sigma_major_pred_arcsec), float(sig_pred_arcsec)
                        )
                    if np.isfinite(area_pred_deg2):
                        a.sum_ellipse_area_pred_deg2 += float(area_pred_deg2)
                        a.max_ellipse_area_pred_deg2 = max(
                            float(a.max_ellipse_area_pred_deg2), float(area_pred_deg2)
                        )
                    a.update_dt(abs_dt)

                _update("abs")
                if bool(report_signed) and dt_days != 0.0:
                    _update("backward" if dt_days < 0 else "forward")

            dt_part = float(time.perf_counter() - float(t_part0))
            part_times.append(dt_part)
            avg = float(np.mean(part_times)) if part_times else float("nan")
            rem = (
                float(avg) * float(max(0, len(part_files) - (p_i + 1)))
                if np.isfinite(avg)
                else float("nan")
            )
            elapsed = float(time.perf_counter() - float(t_run0))
            _log(
                f"part {p_i+1}/{len(part_files)} DONE {pf.name}"
                f" rows_raw={n_rows_raw} rows_used={n_rows_used}"
                f" keys={len(key_info)} pred_rows={len(pred_rows)} groups={len(agg)}"
                f" dt={_fmt_sec(dt_part)} elapsed={_fmt_sec(elapsed)} eta={_fmt_sec(rem)}"
            )

    finally:
        conn.close()

    # Emit timeseries rows.
    out_rows: list[dict[str, object]] = []
    for (oid, bid, direction), a in agg.items():
        n_t = int(a.n_targets)
        n_hit = int(a.n_targets_hit)
        sum_bytes = int(a.sum_data_length_bytes)
        sum_w_bytes = float(a.sum_weighted_data_length_bytes)
        sum_frames = int(a.sum_frames_touched)
        bytes_per_exp = (float(sum_bytes) / float(n_t)) if n_t > 0 else float("nan")
        w_bytes_per_exp = (float(sum_w_bytes) / float(n_t)) if n_t > 0 else float("nan")
        frames_per_exp = (float(sum_frames) / float(n_t)) if n_t > 0 else float("nan")
        bytes_per_hit_exp = (
            (float(sum_bytes) / float(n_hit)) if n_hit > 0 else float("nan")
        )
        w_bytes_per_hit_exp = (
            (float(sum_w_bytes) / float(n_hit)) if n_hit > 0 else float("nan")
        )
        frames_per_hit_exp = (
            (float(sum_frames) / float(n_hit)) if n_hit > 0 else float("nan")
        )

        sig_mean = (float(a.sum_sigma_major_arcsec) / float(n_t)) if (n_t > 0) else None
        area_mean = (float(a.sum_ellipse_area_deg2) / float(n_t)) if (n_t > 0) else None
        sig_pred_mean = (
            (float(a.sum_sigma_major_pred_arcsec) / float(n_t)) if (n_t > 0) else None
        )
        area_pred_mean = (
            (float(a.sum_ellipse_area_pred_deg2) / float(n_t)) if (n_t > 0) else None
        )
        sig_pos_km_mean = (
            (float(a.sum_sigma_major_pos_km) / float(a.n_sigma_major_pos_km))
            if int(a.n_sigma_major_pos_km) > 0
            else None
        )

        rho_mean: float | None = None
        rho_min: float | None = None
        rho_max: float | None = None
        if int(a.n_rho) > 0:
            rho_mean = float(a.sum_rho_au) / float(a.n_rho)
            if np.isfinite(float(a.min_rho_au)):
                rho_min = float(a.min_rho_au)
            if np.isfinite(float(a.max_rho_au)):
                rho_max = float(a.max_rho_au)

        exp_obs_per_exp: float | None = None
        if (
            bytes_per_obs is not None
            and np.isfinite(w_bytes_per_exp)
            and float(bytes_per_obs) > 0.0
        ):
            exp_obs_per_exp = float(w_bytes_per_exp) / float(bytes_per_obs)

        exp_obs_per_hit_exp: float | None = None
        if (
            bytes_per_obs is not None
            and np.isfinite(w_bytes_per_hit_exp)
            and float(bytes_per_obs) > 0.0
        ):
            exp_obs_per_hit_exp = float(w_bytes_per_hit_exp) / float(bytes_per_obs)

        out_rows.append(
            dict(
                subset_dir=str(subset_dir),
                stage2_run_dir=str(stage2_run_dir),
                strategy=str(strategy),
                footprint=str(footprint),
                healpix_nside=int(healpix_nside),
                n_sigma=float(n_sigma),
                batch_days=float(batch_days),
                direction=str(direction),
                orbit_id=str(oid),
                batch_id=int(bid),
                abs_dt_min_days=(
                    None if a.abs_dt_min_days is None else float(a.abs_dt_min_days)
                ),
                abs_dt_max_days=(
                    None if a.abs_dt_max_days is None else float(a.abs_dt_max_days)
                ),
                n_targets=int(n_t),
                n_targets_hit=int(n_hit),
                hit_rate=(None if n_t <= 0 else float(n_hit) / float(n_t)),
                sum_pred_pixels=int(a.sum_pred_pixels),
                max_pred_pixels=int(a.max_pred_pixels),
                sum_frames_touched=int(sum_frames),
                sum_data_length_bytes=int(sum_bytes),
                bytes_per_exposure=float(bytes_per_exp),
                bytes_per_hit_exposure=(
                    None
                    if not np.isfinite(bytes_per_hit_exp)
                    else float(bytes_per_hit_exp)
                ),
                sum_weighted_data_length_bytes=(
                    None if not np.isfinite(sum_w_bytes) else float(sum_w_bytes)
                ),
                weighted_bytes_per_exposure=(
                    None if not np.isfinite(w_bytes_per_exp) else float(w_bytes_per_exp)
                ),
                expected_obs_per_exposure=(
                    None if exp_obs_per_exp is None else float(exp_obs_per_exp)
                ),
                weighted_bytes_per_hit_exposure=(
                    None
                    if not np.isfinite(w_bytes_per_hit_exp)
                    else float(w_bytes_per_hit_exp)
                ),
                expected_obs_per_hit_exposure=(
                    None if exp_obs_per_hit_exp is None else float(exp_obs_per_hit_exp)
                ),
                frames_per_exposure=float(frames_per_exp),
                frames_per_hit_exposure=(
                    None
                    if not np.isfinite(frames_per_hit_exp)
                    else float(frames_per_hit_exp)
                ),
                n_upper_bound_all_frames=int(a.n_upper_bound_all_frames),

                rho_au_mean=(None if rho_mean is None else float(rho_mean)),
                rho_au_min=(None if rho_min is None else float(rho_min)),
                rho_au_max=(None if rho_max is None else float(rho_max)),

                sigma_major_arcsec_mean=(None if sig_mean is None else float(sig_mean)),
                sigma_major_arcsec_max=(
                    None if n_t <= 0 else float(a.max_sigma_major_arcsec)
                ),
                ellipse_area_deg2_mean=(
                    None if area_mean is None else float(area_mean)
                ),
                ellipse_area_deg2_max=(
                    None if n_t <= 0 else float(a.max_ellipse_area_deg2)
                ),

                sigma_major_pos_km_mean=(
                    None if sig_pos_km_mean is None else float(sig_pos_km_mean)
                ),
                sigma_major_pos_km_max=(
                    None
                    if int(a.n_sigma_major_pos_km) <= 0
                    else float(a.max_sigma_major_pos_km)
                ),

                sigma_major_pred_arcsec_mean=(
                    None if sig_pred_mean is None else float(sig_pred_mean)
                ),
                sigma_major_pred_arcsec_max=(
                    None if n_t <= 0 else float(a.max_sigma_major_pred_arcsec)
                ),
                ellipse_area_pred_deg2_mean=(
                    None if area_pred_mean is None else float(area_pred_mean)
                ),
                ellipse_area_pred_deg2_max=(
                    None if n_t <= 0 else float(a.max_ellipse_area_pred_deg2)
                ),
            )
        )

    if out_rows:
        ts = Stage5IndexOnlyTimeSeries.from_pyarrow(pa.Table.from_pylist(out_rows))
    else:
        ts = Stage5IndexOnlyTimeSeries.empty()
    ts.to_parquet(str(run_dir / "timeseries.parquet"))

    # Meta + heuristic artifact.
    meta_out: dict[str, object] = dict(
        subset_dir=str(subset_dir),
        index_db=str(index_db),
        stage2_run_dir=str(stage2_run_dir),
        strategy=str(strategy),
        footprint=str(footprint),
        healpix_nside=int(healpix_nside),
        n_sigma=float(n_sigma),
        polygon_vertices=int(polygon_vertices),
        batch_days=float(batch_days),
        max_pixels_exact=int(max_pixels_exact),
        max_sampled_targets=(
            None if max_sampled_targets is None else int(max_sampled_targets)
        ),
        report_signed=bool(report_signed),
        bytes_per_obs=(None if bytes_per_obs is None else float(bytes_per_obs)),
        astro_sigma_arcsec=float(obs_sigma_arcsec_used),
        astro_sigma_arcsec_arg=(
            None if astro_sigma_arcsec is None else float(astro_sigma_arcsec)
        ),
        bytes_per_exposure_max=(
            None if bytes_per_exposure_max is None else float(bytes_per_exposure_max)
        ),
        consecutive_batches=int(consecutive_batches),
        fractional_nside=(None if fractional_nside is None else int(fractional_nside)),
        generated_at_utc=_now_utc(),
    )
    _write_json(run_dir / "meta.json", meta_out)

    heur: dict[str, object] = dict(
        bytes_per_exposure_max=(
            None if bytes_per_exposure_max is None else float(bytes_per_exposure_max)
        ),
        consecutive_batches=int(consecutive_batches),
        bytes_per_obs=(None if bytes_per_obs is None else float(bytes_per_obs)),
    )

    # Optional: add an "expected_obs_per_exposure_max" human-friendly equivalent.
    if bytes_per_obs is not None and bytes_per_exposure_max is not None:
        heur["expected_obs_per_exposure_max"] = float(
            expected_observations_from_bytes(
                data_length_bytes=float(bytes_per_exposure_max),
                bytes_per_obs=float(bytes_per_obs),
            )
        )
    _write_json(run_dir / "heuristic.json", heur)
    return run_dir


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(
        description="Stage 5 (index-only): estimate candidate density vs |Δt| using only index.db + Stage2 ephemerides-with-covariance."
    )
    p.add_argument("--subset-dir", type=str, required=True)
    p.add_argument("--stage2-run-dir", type=str, required=True)
    p.add_argument("--strategy", type=str, default="2body_with_covariance")
    p.add_argument(
        "--footprint",
        type=str,
        default="cov_disc",
        choices=["point", "cov_disc", "cov_polygon_moc", "cov_mc"],
    )
    p.add_argument("--healpix-nside", type=int, required=True)
    p.add_argument("--n-sigma", type=float, default=3.0)
    p.add_argument("--polygon-vertices", type=int, default=32)
    p.add_argument("--batch-days", type=float, default=7.0)
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--orbits-parquet", type=str, default=None)
    p.add_argument("--max-orbits", type=int, default=None)
    p.add_argument("--max-targets", type=int, default=None)
    p.add_argument(
        "--max-sampled-targets",
        type=int,
        default=None,
        help="Optional: sample this many target_idx values uniformly across the full target list (reduces Stage5 runtime).",
    )
    p.add_argument("--max-parts", type=int, default=None)
    p.add_argument("--max-pixels-exact", type=int, default=20000)
    p.add_argument("--report-signed", action="store_true")
    p.add_argument(
        "--bytes-per-obs",
        type=float,
        default=None,
        help="Optional constant used to convert bytes_per_exposure -> expected_obs_per_exposure in heuristic.json.",
    )
    p.add_argument(
        "--astro-sigma-arcsec",
        type=float,
        default=0.1,
        help=(
            "Observation 1-sigma astrometric error (arcsec) used to form an innovation covariance "
            "(predicted + observational variance) in the tangent plane. "
            "If <= 0, a notional default of 0.1 arcsec is used."
        ),
    )
    p.add_argument(
        "--fractional-nside",
        type=int,
        default=None,
        help=(
            "Optional: compute fractional per-healpixel coverage by rasterizing the footprint at this nside "
            "(must be a power-of-two multiple of --healpix-nside; assumes NESTED ordering)."
        ),
    )
    p.add_argument(
        "--orbit-ids",
        type=str,
        default=None,
        help="Optional comma-separated list of orbit_id/object_id keys to include (use for 1-orbit exploration).",
    )
    p.add_argument("--bytes-per-exposure-max", type=float, default=None)
    p.add_argument("--consecutive-batches", type=int, default=3)
    args = p.parse_args()

    run_dir = run_stage5_index_only_density(
        subset_dir=Path(args.subset_dir),
        stage2_run_dir=Path(args.stage2_run_dir),
        strategy=str(args.strategy),
        footprint=str(args.footprint),
        healpix_nside=int(args.healpix_nside),
        n_sigma=float(args.n_sigma),
        polygon_vertices=int(args.polygon_vertices),
        batch_days=float(args.batch_days),
        out_dir=None if args.out_dir is None else Path(args.out_dir),
        orbits_parquet=None
        if args.orbits_parquet is None
        else Path(args.orbits_parquet),
        max_orbits=args.max_orbits,
        max_targets=args.max_targets,
        max_sampled_targets=args.max_sampled_targets,
        max_parts=args.max_parts,
        max_pixels_exact=int(args.max_pixels_exact),
        report_signed=bool(args.report_signed),
        bytes_per_obs=None if args.bytes_per_obs is None else float(args.bytes_per_obs),
        astro_sigma_arcsec=None
        if args.astro_sigma_arcsec is None
        else float(args.astro_sigma_arcsec),
        bytes_per_exposure_max=None
        if args.bytes_per_exposure_max is None
        else float(args.bytes_per_exposure_max),
        consecutive_batches=int(args.consecutive_batches),
        fractional_nside=None
        if args.fractional_nside is None
        else int(args.fractional_nside),
        orbit_ids=(
            None
            if args.orbit_ids is None
            else [s.strip() for s in str(args.orbit_ids).split(",") if s.strip()]
        ),
    )
    print(f"run_dir={run_dir}")
    print(f"timeseries_parquet={run_dir / 'timeseries.parquet'}")
    print(f"meta_json={run_dir / 'meta.json'}")
    print(f"heuristic_json={run_dir / 'heuristic.json'}")


if __name__ == "__main__":
    main()
