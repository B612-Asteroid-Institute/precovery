import logging
import multiprocessing
from typing import Optional, Tuple

import quivr as qv
import ray
from adam_core.orbits import Orbits as AdamOrbits
from adam_core.ray_cluster import initialize_use_ray

from .precovery_db import FrameCandidatesQv, PrecoveryCandidatesQv, PrecoveryDatabase

logger = logging.getLogger("precovery")
logging.basicConfig()
logger.setLevel(logging.INFO)


def _collect_precovery_results(futures, precovery_candidates, frame_candidates):
    finished, futures = ray.wait(futures, num_returns=1)
    precovery_candidates_chunk, frame_candidates_chunk = ray.get(finished[0])
    precovery_candidates = qv.concatenate(
        [precovery_candidates, precovery_candidates_chunk]
    )
    frame_candidates = qv.concatenate([frame_candidates, frame_candidates_chunk])
    return futures, precovery_candidates, frame_candidates


def precover_many(
    orbits: AdamOrbits,
    database_directory: str,
    tolerance: float = 1 / 3600,
    start_mjd: Optional[float] = None,
    end_mjd: Optional[float] = None,
    window_size: int = 7,
    allow_version_mismatch: bool = False,
    datasets: Optional[set[str]] = None,
    n_workers: Optional[int] = None,
) -> Tuple[PrecoveryCandidatesQv, FrameCandidatesQv]:
    """
    Run a precovery search algorithm against many orbits at once.
    """
    if n_workers is None:
        n_workers = multiprocessing.cpu_count()

    logger.info(f"Running precovery with {n_workers} workers")

    initialize_use_ray(num_cpus=n_workers)

    inputs = [
        (
            o,
            database_directory,
            tolerance,
            start_mjd,
            end_mjd,
            window_size,
            allow_version_mismatch,
            datasets,
        )
        for o in orbits
    ]

    precovery_candidates = PrecoveryCandidatesQv.empty()
    frame_candidates = FrameCandidatesQv.empty()

    futures = []
    completed = 0
    for orbit in orbits:
        futures.append(
            precover_remote.remote(
                orbit,
                database_directory,
                tolerance,
                start_mjd,
                end_mjd,
                window_size,
                allow_version_mismatch,
                datasets,
            )
        )
        if len(futures) >= n_workers * 1.5:
            futures, precovery_candidates, frame_candidates = (
                _collect_precovery_results(
                    futures, precovery_candidates, frame_candidates
                )
            )
            completed += 1
            logger.info(f"Completed {completed}/{len(orbits)} precovery jobs")
    logger.info("Finished submitting precovery jobs")

    while len(futures) > 0:
        futures, precovery_candidates, frame_candidates = _collect_precovery_results(
            futures, precovery_candidates, frame_candidates
        )
        completed += 1
        logger.info(f"Completed {completed}/{len(orbits)} precovery jobs")

    return precovery_candidates, frame_candidates


def precover(
    orbit: AdamOrbits,
    database_directory: str,
    tolerance: float = 1 / 3600,
    start_mjd: Optional[float] = None,
    end_mjd: Optional[float] = None,
    window_size: int = 7,
    allow_version_mismatch: bool = False,
    datasets: Optional[set[str]] = None,
) -> Tuple[PrecoveryCandidatesQv, FrameCandidatesQv]:
    """
    Connect to database directory and run precovery for the input orbit.

    Parameters
    ----------
    orbit : `precovery.Orbit`
        Orbit to propagate through indexed observations.
    database_directory : str
        Path to database directory. Assumes the index database has already been created
        and the observations indexed. Access through this function is read-only by
        design.
    tolerance : float, optional
        The on-sky angular tolerance in degrees to which any PrecoveryCandidates should be
        returned.
    start_mjd : float, optional
        Limit precovery search to all MJD UTC times beyond this time.
    end_mjd : float, optional
        Limit precovery search to all MJD UTC times before this time.
    window_size : int, optional
        To decrease computational cost, the index observations are searched in windows of this size.
        The orbit is propagated with N-body dynamics to the midpoint of each window. From the midpoint,
        the orbit is then propagated using 2-body dynamics to find which HealpixFrames intersect the
        trajectory. Once the list of HealpixFrames has been made, the orbit is then propagated via
        n-body dynamics to each frame and the angular distance to each observation in that
        frame is checked.
    allow_version_mismatch : bool, optional
        Allows using a precovery db version that does not match the library version.
    datasets : set[str], optional
        Filter down searches to only scan selected datasets

    Returns
    -------
    candidates : Tuple[List[PrecoveryCandidate], List[FrameCandidate]]
        PrecoveryCandidate observations that may belong to this orbit. FrameCandidates of any frames
        that intersected the orbit's trajectory but did not have any observations (PrecoveryCandidates)
        found within the angular tolerance.
    """
    precovery_db = PrecoveryDatabase.from_dir(
        database_directory,
        create=False,
        mode="r",
        allow_version_mismatch=allow_version_mismatch,
    )

    precovery_candidates, frame_candidates = precovery_db.precover(
        orbit,
        tolerance=tolerance,
        start_mjd=start_mjd,
        end_mjd=end_mjd,
        window_size=window_size,
        datasets=datasets,
    )

    return precovery_candidates, frame_candidates


precover_remote = ray.remote(precover)
precover_remote.options(
    num_returns=1,
    num_cpus=1,
)
