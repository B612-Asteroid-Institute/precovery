"""Backend benchmarking harness for covariance precovery experiments.

This package is intentionally *not* wired into production `precovery/` search paths.
It provides a cost-controlled, backend-agnostic way to benchmark:
- keyed joins by (obscode, exposure_time, healpix)
- innovation-ellipse gating
- truth recovery metrics (when truth artifacts are available)
"""

