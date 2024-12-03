import logging
import multiprocessing
from typing import Optional, Tuple, Type

import quivr as qv
import ray
from adam_core.orbits import Orbits
from adam_core.propagator import Propagator
from adam_core.ray_cluster import initialize_use_ray

from .precovery_db import FrameCandidates, PrecoveryCandidates, PrecoveryDatabase

logger = logging.getLogger("precovery")
logging.basicConfig()
logger.setLevel(logging.INFO)



def precover_worker(
    orbit: Orbits,
    database_directory: str,
    tolerance: float = 1 / 3600,
    start_mjd: Optional[float] = None,
    end_mjd: Optional[float] = None,
    window_size: int = 7,
    allow_version_mismatch: bool = False,
    datasets: Optional[set[str]] = None,
    propagator_class: Optional[Type[Propagator]] = None,
    max_processes: Optional[int] = None,
) -> Tuple[PrecoveryCandidates, FrameCandidates]:
    """
    Wraps the precover function to return the orbit_id for mapping.
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
        propagator_class=propagator_class,
        max_processes=max_processes,
    )

    return (precovery_candidates, frame_candidates)


precover_worker_remote = ray.remote(precover_worker)


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
) -> Tuple[PrecoveryCandidates, FrameCandidates]:
    """
    Connect to database directory and run precovery for the input orbit.

    Parameters
    ----------
    orbit : `adam_core.orbits.Orbits`
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
    propagator_class : Type[Propagator], optional
        An adam_core.propagator.Propagator subclass to use for propagating the orbit.

    Returns
    -------
    candidates : Tuple[PrecoveryCandidates, FrameCandidates]
        PrecoveryCandidate observations that may belong to this orbit. FrameCandidates of any frames
        that intersected the orbit's trajectory but did not have any observations (PrecoveryCandidates)
        found within the angular tolerance.
    """
    if max_processes is not None and max_processes > 1:
        initialize_use_ray(max_processes)

        futures = []
        precovery_candidates = PrecoveryCandidates.empty()
        frame_candidates = FrameCandidates.empty()
        for o in orbits:
            futures.append(
                precover_worker_remote.remote(
                    o,
                    database_directory,
                    tolerance,
                    start_mjd,
                    end_mjd,
                    window_size,
                    allow_version_mismatch,
                    datasets,
                    propagator_class,
                )
            )

            if len(futures) == max_processes * 2:
                finished, futures = ray.wait(futures, num_returns=1)
                job_candidates, job_frame_candidates = ray.get(finished[0])
                precovery_candidates = qv.concatenate(
                    [precovery_candidates, job_candidates]
                )
                frame_candidates = qv.concatenate(
                    [frame_candidates, job_frame_candidates]
                )

        while futures:
            finished, futures = ray.wait(futures, num_returns=1)
            job_candidates, job_frame_candidates = ray.get(finished[0])
            precovery_candidates = qv.concatenate(
                [precovery_candidates, job_candidates]
            )
            frame_candidates = qv.concatenate(
                [frame_candidates, job_frame_candidates]
            )
    
    else:
        precovery_candidates, frame_candidates = precover_worker(
            orbits,
            database_directory,
            tolerance,
            start_mjd,
            end_mjd,
            window_size,
            allow_version_mismatch,
            datasets,
            propagator_class,
        )
    return precovery_candidates, frame_candidates
