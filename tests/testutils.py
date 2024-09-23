import os
import random
import string
from typing import Optional, Tuple

import healpy
import pytest
from adam_core.observers import Observers
from adam_core.orbits import Orbits
from adam_core.propagator.adam_assist import ASSISTPropagator
from adam_core.time import Timestamp

from precovery.sourcecatalog import SourceFrame, SourceObservation


def make_sourceobs(
    exposure_id: str = "exposure",
    id: Optional[bytes] = None,
    obscode: str = "obs",
    healpixel: Optional[int] = None,
    nside: int = 32,
    ra: float = 1.0,
    dec: float = 2.0,
    mjd: float = 50000.0,
    exposure_mjd_start: float = 50000.0,
    exposure_mjd_mid: float = 50000.0,
) -> SourceObservation:
    """Constructor for SourceObservations which provides default
    values for anything unspecified, which makes test setup less
    verbose.

    """
    if id is None:
        id = random_string(16).encode("utf8")

    if healpixel is not None:
        ra, dec = radec_for_healpixel(healpixel, nside)

    return SourceObservation(
        exposure_id=exposure_id,
        obscode=obscode,
        id=id,
        mjd=mjd,
        ra=ra,
        dec=dec,
        ra_sigma=3.0,
        dec_sigma=4.0,
        mag=5.0,
        mag_sigma=6.0,
        filter="filter",
        exposure_mjd_start=exposure_mjd_start,
        exposure_mjd_mid=exposure_mjd_mid,
        exposure_duration=30,
    )


def make_sourceobs_of_orbit(
    orbit: Orbits,
    obscode: str,
    mjd: float = 50000.0,
):
    propagator = ASSISTPropagator()

    times = Timestamp.from_mjd([mjd], scale="utc")
    # create observers
    observers = Observers.from_code(obscode, times)
    ephem = propagator.generate_ephemeris(orbit, observers)
    obs = make_sourceobs(
        mjd=mjd,
        exposure_mjd_mid=mjd,
        obscode=obscode,
        ra=ephem.coordinates.lon[0].as_py(),
        dec=ephem.coordinates.lat[0].as_py(),
    )
    return obs


def make_sourceframe_with_observations(
    n_observations: int,
    exposure_id: str = "exposure",
    obscode: str = "obs",
    exposure_mjd_start: float = 50000.0,
    exposure_mjd_mid: float = 50000.0,
    healpixel: int = 1,
) -> SourceFrame:
    """Constructor for SourceFrames which provides default
    values for anything unspecified, which makes test setup less
    verbose. SourceObservations are generated and included.

    """
    observations = [
        make_sourceobs(
            exposure_id=exposure_id,
            obscode=obscode,
            healpixel=healpixel,
            mjd=exposure_mjd_mid,
            exposure_mjd_start=exposure_mjd_start,
            exposure_mjd_mid=exposure_mjd_mid,
        )
        for _ in range(n_observations)
    ]

    return SourceFrame(
        exposure_id=exposure_id,
        obscode=obscode,
        filter="filter",
        exposure_mjd_start=exposure_mjd_start,
        exposure_mjd_mid=exposure_mjd_mid,
        exposure_duration=30.0,
        healpixel=1,
        observations=observations,
    )


def radec_for_healpixel(healpixel: int, nside: int) -> Tuple[float, float]:
    """
    Compute the ra and dec associated with a healpixel
    """
    return healpy.pix2ang(nside=nside, ipix=healpixel, nest=True, lonlat=True)


def random_string(length: int):
    return "".join(random.choice(string.ascii_lowercase) for i in range(length))


# maybe replace this with a check that we have downloaded jpl ephemeris files?
requires_jpl_ephem_data = pytest.mark.skipif(
    "ASSIST_DATA_DIR" not in os.environ,
    reason="test requires propagation, so ASSIST_DATA_DIR environment variable must be set",
)
