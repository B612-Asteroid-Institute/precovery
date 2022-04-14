import healpy as hp
import numpy as np
import numpy.typing as npt

def radec_to_healpixel(ra: float, dec: float, nside: int) -> int:
    return hp.ang2pix(nside, ra, dec, nest=True, lonlat=True)

def radec_to_healpixel(ra: npt.NDArray[np.float64], dec: npt.NDArray[np.float64], nside: int) -> npt.NDArray[np.intc]:
    return hp.ang2pix(nside, ra, dec, nest=True, lonlat=True).astype(np.intc)