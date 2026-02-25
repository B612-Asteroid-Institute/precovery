import logging
from typing import Optional, Tuple

from adam_core.orbits import Orbits

from .search.pipeline_types import AcceptedCounts
from .search.results import AcceptedDetections
from .search.run import precover_orbits_backend

logger = logging.getLogger("precovery")
logging.basicConfig()
logger.setLevel(logging.INFO)


def precover(
    orbits: Orbits,
    database_directory: str,
    tolerance: float = 1 / 3600,
    start_mjd: Optional[float] = None,
    end_mjd: Optional[float] = None,
    window_size: int = 7,
    allow_version_mismatch: bool = False,
    datasets: Optional[set[str]] = None,
    max_processes: Optional[int] = None,
    n_sigma: float = 3.0,
) -> Tuple[AcceptedDetections, AcceptedCounts]:
    """
    Connect to a subset directory and run precovery for the input orbit(s).

    This entrypoint now uses the backend-adapter pipeline (DuckDB/ClickHouse/BigQuery).
    The legacy SQLite blob-store search path is deprecated.
    """
    _ = tolerance
    _ = allow_version_mismatch
    _ = datasets

    if start_mjd is None or end_mjd is None:
        raise ValueError("start_mjd and end_mjd are required for backend search")

    run = precover_orbits_backend(
        orbits=orbits,
        subset_dir=database_directory,
        start_mjd=float(start_mjd),
        end_mjd=float(end_mjd),
        obscodes=(),
        window_size_days=int(window_size),
        stage2_strategy="assist_window_then_2body_variants:sigma_points",
        max_processes=max_processes,
        n_sigma=float(n_sigma),
        polygon_vertices=32,
        detailed_timings=False,
    )
    return run.accepted, run.accepted_counts
