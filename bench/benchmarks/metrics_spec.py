from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MetricDefinition:
    key: str
    description: str
    level: str  # "run", "backend", "orbit", "bin"
    notes: str | None = None


BENCHMARK_GOAL = (
    "Optimize for fastest possible recovery of complete truth detections while minimizing false positives."
)

# Authoritative benchmark requirements/specs live in:
#   bench/BENCHMARK_SPECS.md

# Terminology used across benchmark outputs.
TERMS = {
    "frame_key": "(obscode, exposure_mjd_mid_key_us, healpixel) at nside=32",
    "truth_detection": "a truth row with a matched precovery observation_id (match_observation_id) within the window",
    "truth_frame": "a frame_key that contains at least one truth_detection",
    "false_positive_final": (
        "within each (orbit_id, obscode, exposure_mjd_mid_key_us) exposure key, "
        "every accepted detection beyond the first"
    ),
}


METRICS: tuple[MetricDefinition, ...] = (
    # Run-level build timings (Stage 2/3).
    MetricDefinition(
        key="build_elapsed_s",
        description="Total build time (predict + photometry + footprint + triples).",
        level="run",
    ),
    MetricDefinition(
        key="build_observers_elapsed_s",
        description="Time to construct unique observers aligned to targets (Stage 2 input).",
        level="run",
    ),
    MetricDefinition(
        key="build_predict_elapsed_s",
        description="Stage 2 propagation + ephemeris generation time.",
        level="run",
    ),
    MetricDefinition(
        key="build_pred_mag_elapsed_s",
        description="Time to convert predicted V magnitudes to canonical band (for gating).",
        level="run",
    ),
    MetricDefinition(
        key="build_footprint_elapsed_s",
        description="Time to rasterize covariance footprints into healpixels (Stage 3).",
        level="run",
    ),
    MetricDefinition(
        key="build_triples_elapsed_s",
        description="Time to build exploded triples join keys from selected healpixels (Stage 3).",
        level="run",
    ),
    # Run-level sizes.
    MetricDefinition(
        key="n_frames_candidates",
        description="Dataset-wide unique frame keys in the window.",
        level="run",
        notes="Frame key = (obscode, exposure_mjd_mid_key_us, healpixel).",
    ),
    MetricDefinition(
        key="n_frames_final",
        description="Total frames surviving geometry and limiting-magnitude filtering (sum across orbits).",
        level="run",
    ),
    MetricDefinition(
        key="n_frames_truth_final",
        description="Truth frames surviving Stage 3 (sum across orbits).",
        level="run",
    ),
    # Backend-level join + gating sizes.
    MetricDefinition(
        key="n_detections_candidates",
        description="Total candidate detections retrieved by backend join (pre Stage 4 gating).",
        level="backend",
    ),
    MetricDefinition(
        key="n_detections_truth_final",
        description="Truth detections recovered (survive all filtering).",
        level="backend",
    ),
    MetricDefinition(
        key="n_detections_false_positive_final",
        description="Known false positives among final accepted detections.",
        level="backend",
    ),
    MetricDefinition(
        key="n_detections_unknown_final",
        description="Final accepted detections in exposures with no truth detections.",
        level="backend",
        notes="Only defined when truth is provided.",
    ),
)

