from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

from precovery.search.pipeline_types import BenchTargets, MonthWindow, SubsetPaths
from precovery.search.time_key import mjd_to_time_key_us


def _mjd_utc_from_ymd(year: int, month: int, day: int) -> float:
    """
    Convert a UTC Gregorian calendar date at 00:00:00 to MJD.

    Copied conceptually from the historical subset-DB tooling so the benchmarking harness
    can define month windows without importing CLI helpers.
    """
    # Standard JD algorithm for proleptic Gregorian calendar.
    y = int(year)
    m = int(month)
    d = float(day)

    if m <= 2:
        y -= 1
        m += 12

    A = y // 100
    B = 2 - A + (A // 4)
    jd = int(365.25 * (y + 4716)) + int(30.6001 * (m + 1)) + d + B - 1524.5
    return float(jd - 2400000.5)


def month_bounds_mjd_utc(year_month: str) -> tuple[float, float]:
    year_s, month_s = str(year_month).split("-", 1)
    year = int(year_s)
    month = int(month_s)
    if not (1 <= month <= 12):
        raise ValueError(f"Invalid month in {year_month!r}")
    if month == 12:
        end_year, end_month = year + 1, 1
    else:
        end_year, end_month = year, month + 1
    return _mjd_utc_from_ymd(year, month, 1), _mjd_utc_from_ymd(end_year, end_month, 1)


def window_bounds_mjd_utc(months: Iterable[str]) -> tuple[float, float]:
    mm = [str(x) for x in months]
    if not mm:
        raise ValueError("months cannot be empty")
    bounds = [month_bounds_mjd_utc(m) for m in mm]
    start = float(min(a for a, _b in bounds))
    end = float(max(b for _a, b in bounds))
    return start, end


@dataclass(frozen=True)
class WorkloadSpec:
    """
    Canonical benchmark workload.

    This is the *configuration* layer; it intentionally does not depend on any backend.
    """

    subset_dir: Path
    window: MonthWindow

    # Algorithm knobs (kept constant for backend benchmarking)
    # Stage-2 windowing: how far 2-body propagation can drift from ASSIST window centers.
    # Canonical default: 7 day window, ASSIST window + 2-body (assist_window_then_2body_variants:sigma_points).
    window_size_days: int = 7
    n_sigma: float = 3.0
    healpix_nside: int = 32
    footprint: str = "cov_polygon_reconstructed_moc"
    stage2_strategy: str = "assist_window_then_2body_variants:sigma_points"
    gate: str = "innov_ellipse@3"

    def subset_paths(self) -> SubsetPaths:
        return SubsetPaths(subset_dir=Path(self.subset_dir))

    def bounds_mjd_utc(self) -> tuple[float, float]:
        return window_bounds_mjd_utc(self.window.year_months)


def discover_stage_run_dirs(*, subset_dir: Path) -> dict[str, Path]:
    """
    Best-effort discovery of existing stage run dirs under `<subset_dir>/artifacts/`.

    Returns a mapping like:
      {"stage2": Path(...), "stage3": Path(...), "stage4": Path(...)}
    for the most recent matching runs, if present.
    """
    art = Path(subset_dir) / "artifacts"
    if not art.exists():
        return {}

    def _latest(glob_pat: str) -> Path | None:
        hits = sorted([p for p in art.glob(glob_pat) if p.is_dir()])
        return hits[-1] if hits else None

    out: dict[str, Path] = {}
    for k in ("stage2", "stage3", "stage4"):
        p = _latest(f"{k}/*")
        if p is not None:
            out[k] = p
    return out


def read_stage_meta_json(stage_run_dir: Path) -> dict[str, object]:
    p = Path(stage_run_dir) / "meta.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def benchmark_targets_from_sqlite_enumeration(
    *, obscodes: tuple[str, ...], exposure_mjd_mid: np.ndarray, filters: list[str] | np.ndarray | None = None
) -> BenchTargets:
    """
    Convert raw arrays into the canonical `BenchTargets` table.
    """
    tkey = mjd_to_time_key_us(exposure_mjd_mid)
    filt_out: list[str]
    if filters is None:
        filt_out = ["V"] * int(len(obscodes))
    else:
        filt_out = [str(x) for x in list(filters)]
    return BenchTargets.from_kwargs(
        obscode=list(map(str, obscodes)),
        exposure_mjd_mid_utc=exposure_mjd_mid.astype(np.float64, copy=False),
        filter=filt_out,
        exposure_mjd_mid_key_us=tkey,
    )


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

