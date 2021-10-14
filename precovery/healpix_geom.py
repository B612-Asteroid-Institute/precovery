from typing import List, Tuple

import healpy
import numpy as np
import numpy.typing as npt


def radec_to_healpixels(
    ra_decs: List[Tuple[float, float]], nside: int
) -> npt.NDArray[np.float64]:
    healpixels = np.empty(shape=len(ra_decs), dtype=np.int64)
    for i, (ra, dec) in enumerate(ra_decs):
        healpixels[i] = radec_to_healpixel(ra, dec, nside)
    return healpixels


def radec_to_thetaphi(ra: float, dec: float) -> Tuple[float, float]:
    theta = 0.5 * np.pi - np.deg2rad(dec)
    phi = np.deg2rad(ra)
    return theta, phi


def radec_to_healpixel(ra: float, dec: float, nside: int) -> int:
    theta, phi = radec_to_thetaphi(ra, dec)
    return healpy.ang2pix(nside, theta, phi, nest=True)
