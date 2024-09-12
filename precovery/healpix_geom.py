from typing import overload

import healpy as hp
import numpy as np
import numpy.typing as npt


@overload
def radec_to_healpixel(ra: float, dec: float, nside: int) -> int: ...


@overload
def radec_to_healpixel(
    ra: npt.NDArray[np.float64], dec: npt.NDArray[np.float64], nside: int
) -> npt.NDArray[np.int64]: ...


def radec_to_healpixel(ra, dec, nside):
    return hp.ang2pix(nside, ra, dec, nest=True, lonlat=True).astype(np.int64)
