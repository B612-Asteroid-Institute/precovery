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
from .search import precover_orbit, precover_orbit_with_metrics, precover_orbits  # noqa: F401

