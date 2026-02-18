from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import quivr as qv


class BenchTargets(qv.Table):
    """
    One row per distinct (obscode, exposure_mjd_mid) propagation target.
    """

    obscode = qv.LargeStringColumn()
    exposure_mjd_mid_utc = qv.Float64Column()
    # Frame/exposure band reported by the source dataset (used for canonicalization + photometry).
    filter = qv.LargeStringColumn()
    # Stable equality join key derived from exposure_mjd_mid_utc:
    #   key_us = round(exposure_mjd_mid_utc * 86400 * 1e6)
    exposure_mjd_mid_key_us = qv.Int64Column()


class PredictedTargets(qv.Table):
    """
    One row per (orbit_id, target_idx) prediction at exposure midpoint.
    """

    orbit_id = qv.LargeStringColumn()
    target_idx = qv.Int64Column()
    obscode = qv.LargeStringColumn()
    exposure_mjd_mid_utc = qv.Float64Column()
    exposure_mjd_mid_key_us = qv.Int64Column()

    # Canonical filter id (used by limiting-magnitude + mag residual rejection).
    canonical_filter_id = qv.LargeStringColumn(nullable=True)

    pred_lon_deg = qv.Float64Column()
    pred_lat_deg = qv.Float64Column()

    # 2x2 lon/lat covariance in degrees^2, serialized as 3 unique values
    # for ease of passing through SQL engines.
    cov_ll_00 = qv.Float64Column()
    cov_ll_01 = qv.Float64Column()
    cov_ll_11 = qv.Float64Column()

    # Predicted apparent magnitude in the canonical band (nullable when H is missing).
    pred_mag = qv.Float64Column(nullable=True)


class PredictedTriples(qv.Table):
    """
    Exploded (orbit_id, target_idx, obscode, exposure_time_key_us, healpixel) join keys.
    """

    orbit_id = qv.LargeStringColumn()
    target_idx = qv.Int64Column()
    obscode = qv.LargeStringColumn()
    exposure_mjd_mid_utc = qv.Float64Column()
    exposure_mjd_mid_key_us = qv.Int64Column()
    healpixel = qv.Int64Column()


class CandidateDetections(qv.Table):
    """
    Candidate detections returned by a backend join.

    This is deliberately minimal: just what innovation gating + truth scoring needs.
    """

    orbit_id = qv.LargeStringColumn()
    target_idx = qv.Int64Column()

    observation_id = qv.LargeStringColumn()
    obstime_mjd_utc = qv.Float64Column()
    ra_deg = qv.Float64Column()
    dec_deg = qv.Float64Column()
    ra_sigma_deg = qv.Float64Column()
    dec_sigma_deg = qv.Float64Column()

    mag = qv.Float64Column(nullable=True)
    mag_sigma = qv.Float64Column(nullable=True)


class AcceptedCounts(qv.Table):
    """
    Aggregated accepted count results per (orbit_id, target_idx).
    """

    orbit_id = qv.LargeStringColumn()
    target_idx = qv.Int64Column()
    n_candidates = qv.Int64Column()
    n_accepted = qv.Int64Column()


@dataclass(frozen=True)
class MonthWindow:
    """
    A benchmark window defined by one or more calendar months (UTC).
    """

    year_months: tuple[str, ...]  # YYYY-MM
    obscodes: tuple[str, ...]

    def label(self) -> str:
        ym = "_".join(self.year_months)
        codes = "_".join(self.obscodes)
        return f"months_{ym}__{codes}"


@dataclass(frozen=True)
class SubsetPaths:
    subset_dir: Path

    @property
    def index_db(self) -> Path:
        return self.subset_dir / "index.db"

    @property
    def config_json(self) -> Path:
        return self.subset_dir / "config.json"

    @property
    def artifacts_dir(self) -> Path:
        return self.subset_dir / "artifacts"


def pack_cov_ll(cov_ll_deg2: np.ndarray) -> tuple[float, float, float]:
    c = np.asarray(cov_ll_deg2, dtype=np.float64)
    if c.shape != (2, 2):
        raise ValueError("cov_ll_deg2 must be shape (2,2)")
    return float(c[0, 0]), float(c[0, 1]), float(c[1, 1])


def unpack_cov_ll(*, c00: float, c01: float, c11: float) -> np.ndarray:
    return np.array([[float(c00), float(c01)], [float(c01), float(c11)]], dtype=np.float64)

