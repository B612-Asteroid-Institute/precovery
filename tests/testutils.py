import random
from typing import Optional, Tuple

import healpy

from precovery.sourcecatalog import SourceObservation


def make_sourceobs(
    exposure_id: str = "exposure",
    id: Optional[bytes] = None,
    healpixel: Optional[int] = None,
    nside: int = 32,
    ra: float = 1.0,
    dec: float = 2.0,
    obscode: str = "obs",
) -> SourceObservation:

    if id is None:
        id = random.randbytes(16)

    if healpixel is not None:
        ra, dec = radec_for_healpixel(healpixel, nside)

    return SourceObservation(
        exposure_id=exposure_id,
        obscode=obscode,
        id=id,
        mjd=50000,
        ra=ra,
        dec=dec,
        ra_sigma=3.0,
        dec_sigma=4.0,
        mag=5.0,
        mag_sigma=6.0,
        filter="filter",
        exposure_mjd_start=50000,
        exposure_mjd_mid=50000,
        exposure_duration=30,
    )


def radec_for_healpixel(healpixel: int, nside: int) -> Tuple[float, float]:
    """
    Compute the ra and dec associated with a healpixel
    """
    return healpy.pix2ang(nside=nside, ipix=healpixel, nest=True, lonlat=True)
