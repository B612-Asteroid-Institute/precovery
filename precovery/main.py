import logging
import multiprocessing
from typing import Optional, Tuple

import quivr as qv
from adam_core.orbits import Orbits
from adam_core.propagator import Propagator

from .precovery_db import FrameCandidates, PrecoveryCandidates, PrecoveryDatabase

logger = logging.getLogger("precovery")
logging.basicConfig()
logger.setLevel(logging.INFO)


def precover_many(
    orbits: Orbits,
    database_directory: str,
    tolerance: float = 1 / 3600,
    start_mjd: Optional[float] = None,
    end_mjd: Optional[float] = None,
    window_size: int = 7,
    allow_version_mismatch: bool = False,
    datasets: Optional[set[str]] = None,
    n_workers: int = multiprocessing.cpu_count(),
    propagator: Optional[
        Propagator
    ] = None,  # should we initialize an assist propagator?
) -> Tuple[PrecoveryCandidates, FrameCandidates]:
    """
    Run a precovery search algorithm against many orbits at once.
    """
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
            propagator,
        )
        for o in orbits
    ]

    pool = multiprocessing.Pool(processes=n_workers)
    results = pool.starmap(
        precover_worker,
        inputs,
    )
    pool.close()
    pool.join()

    precovery_candidates = PrecoveryCandidates.empty()
    frame_candidates = FrameCandidates.empty()

    for orb_precovery_candidate, orb_frame_candidate in results:
        precovery_candidates = qv.concatenate(
            [precovery_candidates, orb_precovery_candidate]
        )
        frame_candidates = qv.concatenate([frame_candidates, orb_frame_candidate])

    return precovery_candidates, frame_candidates


def precover_worker(
    orbit: Orbits,
    database_directory: str,
    tolerance: float = 1 / 3600,
    start_mjd: Optional[float] = None,
    end_mjd: Optional[float] = None,
    window_size: int = 7,
    allow_version_mismatch: bool = False,
    datasets: Optional[set[str]] = None,
) -> Tuple[PrecoveryCandidates, FrameCandidates]:
    """
    Wraps the precover function to return the orbit_id for mapping.
    """
    precovery_candidates, frame_candidates = precover(
        orbit,
        database_directory,
        tolerance,
        start_mjd,
        end_mjd,
        window_size,
        allow_version_mismatch,
        datasets,
    )

    return (
        precovery_candidates,
        frame_candidates,
    )


def precover(
    orbit: Orbits,
    database_directory: str,
    tolerance: float = 1 / 3600,
    start_mjd: Optional[float] = None,
    end_mjd: Optional[float] = None,
    window_size: int = 7,
    allow_version_mismatch: bool = False,
    datasets: Optional[set[str]] = None,
    propagator: Optional[
        Propagator
    ] = None,  # should we initialize an assist propagator?
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

    Returns
    -------
    candidates : Tuple[PrecoveryCandidates, FrameCandidates]
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
        propagator=propagator,
    )

    return precovery_candidates, frame_candidates
