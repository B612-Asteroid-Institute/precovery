import logging
import multiprocessing
from typing import List, Optional, Tuple

from .orbit import Orbit
from .precovery_db import FrameCandidate, PrecoveryCandidate, PrecoveryDatabase

logger = logging.getLogger("precovery")
logging.basicConfig()
logger.setLevel(logging.INFO)


def precover_many(
    orbits: List[Orbit],
    database_directory: str,
    tolerance: float = 1 / 3600,
    start_mjd: Optional[float] = None,
    end_mjd: Optional[float] = None,
    window_size: int = 7,
    allow_version_mismatch: bool = False,
    datasets: Optional[set[str]] = None,
    n_workers: int = multiprocessing.cpu_count(),
) -> dict[int, Tuple[List[PrecoveryCandidate], List[FrameCandidate]]]:
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

    result_dict = {}
    if len(results) == 0:
        return {}

    for orbit_id, precovery_candidates, frame_candidates in results:
        result_dict[orbit_id] = (precovery_candidates, frame_candidates)

    return result_dict


def precover_worker(
    orbit: Orbit,
    database_directory: str,
    tolerance: float = 1 / 3600,
    start_mjd: Optional[float] = None,
    end_mjd: Optional[float] = None,
    window_size: int = 7,
    allow_version_mismatch: bool = False,
    datasets: Optional[set[str]] = None,
) -> Tuple[int, List[PrecoveryCandidate], List[FrameCandidate]]:
    """
    Wraps the precover function to return the orbit_id for mapping.
    """
    orbit_id = orbit.orbit_id
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
    return orbit_id, precovery_candidates, frame_candidates


def precover(
    orbit: Orbit,
    database_directory: str,
    tolerance: float = 1 / 3600,
    start_mjd: Optional[float] = None,
    end_mjd: Optional[float] = None,
    window_size: int = 7,
    allow_version_mismatch: bool = False,
    datasets: Optional[set[str]] = None,
) -> Tuple[List[PrecoveryCandidate], List[FrameCandidate]]:
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
