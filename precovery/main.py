import logging
from typing import Optional, Tuple, Type

import quivr as qv
from adam_core.orbits import Orbits
from adam_core.propagator import Propagator

from .precovery_db import FrameCandidates, PrecoveryCandidates, PrecoveryDatabase

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
    propagator_class: Optional[Type[Propagator]] = None,
    max_processes: Optional[int] = None,
    match_method: str = "circle",
    n_sigma: float = 3.0,
    covariance_polygon_vertices: int = 32,
    covariance_mc_num_samples: int = 64,
    covariance_mc_seed: int = 0,
) -> Tuple[PrecoveryCandidates, FrameCandidates]:
    """
    Connect to database directory and run precovery for the input orbit.
    """
    precovery_db = PrecoveryDatabase.from_dir(
        database_directory,
        create=False,
        mode="r",
        allow_version_mismatch=allow_version_mismatch,
    )

    precovery_candidates = PrecoveryCandidates.empty()
    frame_candidates = FrameCandidates.empty()

    for orbit in orbits:
        candidates, frames = precovery_db.precover(
            orbit,
            tolerance=tolerance,
            start_mjd=start_mjd,
            end_mjd=end_mjd,
            window_size=window_size,
            datasets=datasets,
            propagator_class=propagator_class,
            max_processes=max_processes,
            match_method=match_method,
            n_sigma=n_sigma,
            covariance_polygon_vertices=covariance_polygon_vertices,
            covariance_mc_num_samples=covariance_mc_num_samples,
            covariance_mc_seed=covariance_mc_seed,
        )
        precovery_candidates = qv.concatenate([precovery_candidates, candidates])
        frame_candidates = qv.concatenate([frame_candidates, frames])

    return precovery_candidates, frame_candidates
