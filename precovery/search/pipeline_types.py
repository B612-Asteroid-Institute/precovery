"""
Shared data types for the precovery search pipeline.

Defines tables and structs for targets, predictions, triples, per-stage metrics,
and paths used across Stage 2–4 and by backend adapters.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import quivr as qv


@dataclass(frozen=True)
class MonthWindow:
    """
    A search window defined by one or more calendar months (UTC).
    """

    year_months: tuple[str, ...]  # YYYY-MM
    obscodes: tuple[str, ...]

    def label(self) -> str:
        ym = "_".join(self.year_months)
        codes = "_".join(self.obscodes)
        return f"months_{ym}__{codes}"


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


class Stage3OrbitMetrics(qv.Table):
    """
    Per-orbit Stage-3 accounting emitted by the build step.
    """

    orbit_id = qv.LargeStringColumn()
    n_frames_geometry_matched = qv.Int64Column()
    n_frames_stage3_rejected_any = qv.Int64Column()
    n_frames_lim_mag_rejected = qv.Int64Column()
    n_frames_uncertainty_rejected = qv.Int64Column()
    n_frames_truth_geometry_matched = qv.Int64Column()
    n_frames_truth_stage3_rejected_any = qv.Int64Column()
    n_frames_lim_mag_truth_rejected = qv.Int64Column()
    n_frames_truth_uncertainty_rejected = qv.Int64Column()
    n_detections_truth_frame_candidates = qv.Int64Column()
    n_targets_preprop_viability_rejected = qv.Int64Column()
    n_targets_preprop_time_limited = qv.Int64Column()
    n_targets_failfast_dynamics_error = qv.Int64Column()
    n_targets_eval_total = qv.Int64Column()
    n_targets_eval_after_policy = qv.Int64Column()
    completed_full_time_period_check = qv.BooleanColumn()
    preprop_decision = qv.LargeStringColumn(nullable=True)
    preprop_reason = qv.LargeStringColumn(nullable=True)
    preprop_trigger_metric = qv.LargeStringColumn(nullable=True)
    preprop_trigger_value = qv.Float64Column(nullable=True)
    preprop_trigger_threshold = qv.Float64Column(nullable=True)
    preprop_time_limit_days_applied = qv.Float64Column(nullable=True)
    preprop_first_excluded_target_mjd_utc = qv.Float64Column(nullable=True)
    failfast_stage = qv.LargeStringColumn(nullable=True)
    failfast_reason = qv.LargeStringColumn(nullable=True)
    failfast_time_mjd_tdb = qv.Float64Column(nullable=True)
    failfast_t0_mjd_tdb = qv.Float64Column(nullable=True)
    failfast_t1_mjd_tdb = qv.Float64Column(nullable=True)
    failfast_dt_days = qv.Float64Column(nullable=True)


class Stage4OrbitMetrics(qv.Table):
    """
    Per-orbit Stage-4 detection accounting emitted by the gate step.

    These are backend-dependent because candidate joins depend on the backend adapter.
    """

    orbit_id = qv.LargeStringColumn()

    n_detections_candidates = qv.Int64Column()
    # Innovation-ellipse survivors (before magnitude outlier rejection).
    n_detections_gate_matched = qv.Int64Column()
    # Rejected by innovation-ellipse gate.
    n_detections_innov_ellipse_rejected = qv.Int64Column()
    # Magnitude outlier rejections (evaluated only on innovation-ellipse survivors).
    n_detections_magnitude_rejected = qv.Int64Column()

    # Truth-scored detection counts (0 when no truth is provided).
    n_detections_truth_gate_matched = qv.Int64Column()
    n_detections_truth_magnitude_rejected = qv.Int64Column()
    n_detections_truth_final = qv.Int64Column()

    # Final accepted-detection accounting (0 when no truth is provided for unknown_final).
    n_detections_false_positive_final = qv.Int64Column()
    n_detections_unknown_final = qv.Int64Column()


class CandidateDetections(qv.Table):
    """
    Candidate detections returned by a backend join.

    This is deliberately minimal: just what innovation gating + truth scoring needs.
    """

    orbit_id = qv.LargeStringColumn()
    target_idx = qv.Int64Column()

    # Frame-level join keys (useful for grouping + additional metrics).
    obscode = qv.LargeStringColumn()
    exposure_mjd_mid_key_us = qv.Int64Column()
    healpixel = qv.Int64Column()
    filter = qv.LargeStringColumn(nullable=True)

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
class SubsetPaths:
    """
    Paths for a subset (config, index, artifacts) used by the pipeline and backends.
    """

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
