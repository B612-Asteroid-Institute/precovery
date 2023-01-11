from typing import List, Optional, Union

import numpy as np
import pandas as pd

from .orbit import Orbit
from .precovery_db import FrameCandidate, PrecoveryCandidate, PrecoveryDatabase


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


def precover(
    orbit: Orbit,
    database_directory: str,
    tolerance: float = 1 / 3600,
    max_matches: Optional[int] = None,
    start_mjd: Optional[float] = None,
    end_mjd: Optional[float] = None,
    window_size: int = 7,
    include_frame_candidates: bool = False,
) -> List[Union[PrecoveryCandidate, FrameCandidate]]:
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

    Returns
    -------
    candidates : List[PrecoveryCandidate, FrameCandidate]
        PrecoveryCandidate observations that may belong to this orbit. FrameCandidates of any frames
        that intersected the orbit's trajectory but did not have any observations (PrecoveryCandidates)
        found within the angular tolerance.
    """
    precovery_db = PrecoveryDatabase.from_dir(
        database_directory, create=False, mode="r"
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
    df.sort_values(by=["mjd", "obscode"], inplace=True, ignore_index=True)
    return df
