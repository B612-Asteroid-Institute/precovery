"""
High-throughput precovery search pipeline.

This package contains a performance-first refactor of the precovery search algorithm.
It is intentionally organized as small, typed functions (no large classes) and leans
on `adam-core` for optimized propagation/ephemeris kernels and Ray parallelism.
"""

from .search import precover_orbit, precover_orbits  # noqa: F401

