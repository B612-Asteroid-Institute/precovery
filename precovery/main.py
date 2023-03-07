import logging
import multiprocessing
from typing import List, Optional

import numpy as np
import pandas as pd
from adam_core.orbits.query import sbdb

from .orbit import EpochTimescale, Orbit
from .precovery_db import PrecoveryDatabase

logger = logging.getLogger("precovery")
logging.basicConfig()
logger.setLevel(logging.INFO)


def _candidates_to_dict(candidates):
    data = {
        "mjd": [],
        "ra_deg": [],
        "dec_deg": [],
        "ra_sigma_arcsec": [],
        "dec_sigma_arcsec": [],
        "mag": [],
        "mag_sigma": [],
        "filter": [],
        "obscode": [],
        "exposure_id": [],
        "exposure_mjd_start": [],
        "exposure_mjd_mid": [],
        "exposure_duration": [],
        "observation_id": [],
        "healpix_id": [],
        "pred_ra_deg": [],
        "pred_dec_deg": [],
        "pred_vra_degpday": [],
        "pred_vdec_degpday": [],
        "delta_ra_arcsec": [],
        "delta_dec_arcsec": [],
        "distance_arcsec": [],
        "dataset_id": [],
    }
    for c in candidates:
        for k in data.keys():
            if k in c.__dict__.keys():
                data[k].append(c.__dict__[k])
            else:
                data[k].append(np.NaN)

    return data


def precover_many(
    orbits: List[Orbit],
    database_directory: str,
    tolerance: float = 1 / 3600,
    max_matches: Optional[int] = None,
    start_mjd: Optional[float] = None,
    end_mjd: Optional[float] = None,
    window_size: int = 7,
    include_frame_candidates: bool = False,
    allow_version_mismatch: bool = False,
    n_workers: int = multiprocessing.cpu_count(),
) -> dict[int, pd.DataFrame]:
    """
    Run a precovery search algorithm against many orbits at once.
    """

    inputs = [
        (
            o,
            database_directory,
            tolerance,
            max_matches,
            start_mjd,
            end_mjd,
            window_size,
            include_frame_candidates,
            allow_version_mismatch,
        )
        for o in orbits
    ]

    pool = multiprocessing.Pool(processes=n_workers)
    results = pool.starmap(
        precover,
        inputs,
    )
    pool.close()
    pool.join()

    result_dict = {}
    for r in results:
        orbit_id = r["orbit_id"][0]
        result_dict[orbit_id] = r

    return result_dict


def precover(
    orbit: Orbit,
    database_directory: str,
    tolerance: float = 1 / 3600,
    max_matches: Optional[int] = None,
    start_mjd: Optional[float] = None,
    end_mjd: Optional[float] = None,
    window_size: int = 7,
    include_frame_candidates: bool = False,
    allow_version_mismatch: bool = False,
) -> pd.DataFrame:
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
    max_matches : int, optional
        Don't return more than this many potential PrecoveryCandidates.
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
    include_frame_candidates : bool, optional
        If no observations are found within the given angular tolerance, return the HealpixFrame
        where the trajectory intersected the Healpixel but no observations were found. This is useful
        for negative observation campaigns. Note that camera footprints are not modeled, all datasets
        are mapped onto a Healpixel space and this simply returns the Healpixel equivalent exposure
        information.
    allow_version_mismatch : bool, optional
        Allows using a precovery db version that does not match the library version.

    Returns
    -------
    candidates : List[PrecoveryCandidate, FrameCandidate]
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

    candidates = [
        c
        for c in precovery_db.precover(
            orbit,
            tolerance=tolerance,
            max_matches=max_matches,
            start_mjd=start_mjd,
            end_mjd=end_mjd,
            window_size=window_size,
            include_frame_candidates=include_frame_candidates,
        )
    ]

    df = pd.DataFrame(_candidates_to_dict(candidates))
    df.loc[:, "observation_id"] = df.loc[:, "observation_id"].astype(str)
    df["orbit_id"] = orbit.orbit_id
    df.sort_values(by=["mjd", "obscode"], inplace=True, ignore_index=True)
    return df


def precover_objects(
    object_ids: List[str],
    database_directory: str,
    tolerance: float = 1 / 3600,
    max_matches: Optional[int] = None,
    start_mjd: Optional[float] = None,
    end_mjd: Optional[float] = None,
    window_size: int = 7,
    include_frame_candidates: bool = False,
    allow_version_mismatch: bool = False,
) -> dict[str, pd.DataFrame]:

    orbid_to_objid = {}
    logging.info("looking up orbital elements for objects")
    orbits = []
    for i, obj_id in enumerate(object_ids):
        logging.info(f"resolving {obj_id} (orbit_id={i})")
        sbdb_orbit = sbdb.query_sbdb([obj_id])
        precovery_orbit = Orbit.keplerian(
            orbit_id=i,
            semimajor_axis_au=sbdb_orbit.keplerian.a[0],
            eccentricity=sbdb_orbit.keplerian.e[0],
            inclination_deg=sbdb_orbit.keplerian.i[0],
            ascending_node_longitude_deg=sbdb_orbit.keplerian.raan[0],
            periapsis_argument_deg=sbdb_orbit.keplerian.ap[0],
            mean_anomaly_deg=sbdb_orbit.keplerian.M[0],
            osculating_element_epoch_mjd=sbdb_orbit.keplerian.times[0].tt.mjd,
            epoch_timescale=EpochTimescale.TT,
            abs_magnitude=20,
            photometric_slope_parameter=0.15,
        )
        orbid_to_objid[i] = obj_id
        orbits.append(precovery_orbit)
    logging.info("executing precovery")
    results = precover_many(
        orbits,
        database_directory,
        tolerance,
        max_matches,
        start_mjd,
        end_mjd,
        window_size,
        include_frame_candidates,
        allow_version_mismatch,
    )
    logging.info("got all results")

    final_dataframe = pd.DataFrame()
    for orbid, dataframe in results.items():
        obj_id = orbid_to_objid[orbid]
        dataframe["object_id"] = obj_id
        final_dataframe = pd.concat([final_dataframe, dataframe])
    return final_dataframe
