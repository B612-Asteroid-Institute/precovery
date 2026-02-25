"""
High-throughput precovery search pipeline.

This package contains a performance-first refactor of the precovery search algorithm.
It is intentionally organized as small, typed functions (no large classes) and leans
on `adam-core` for optimized propagation/ephemeris kernels and Ray parallelism.

Production defaults (implementation contract)
---------------------------------------------
- Propagation: ASSIST window-center propagation + 2-body within window, using sigma-point sampling
  (`assist_window_then_2body_variants:sigma_points`).
- Footprints: `cov_polygon_reconstructed_moc` for healpixel intersection.
- Detection gating: innovation ellipse (Mahalanobis) at the production standard of `innov_ellipse@3`
  (tunable via `n_sigma`, default 3.0).
"""

from .metrics import SearchAgg, SearchMetrics, metrics_row  # noqa: F401

# Backend-oriented (DuckDB/ClickHouse/BigQuery) pipeline components.
from .pipeline_types import (  # noqa: F401
    AcceptedCounts,
    BenchTargets,
    CandidateDetections,
    MonthWindow,
    PredictedTargets,
    PredictedTriples,
    SubsetPaths,
    Stage4OrbitMetrics,
)
from .backend_pipeline import (  # noqa: F401
    build_stage23,
    run_stage1_to_stage4_python,
    run_stage1_to_stage4_rows_python,
    stage4_fetch_and_gate_python,
    stage4_fetch_and_gate_rows_python,
)
from .backends.protocols import BackendCapabilities, GateParams, SearchBackend  # noqa: F401
from .results import AcceptedDetections  # noqa: F401
from .run import BackendSearchRun, precover_orbits_backend  # noqa: F401

