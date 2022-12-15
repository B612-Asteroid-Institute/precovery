import healpy as hp
import numpy as np
import numpy.typing as npt

def radec_to_healpixel(ra: npt.NDArray[np.float64], dec: npt.NDArray[np.float64], nside: int) -> npt.NDArray[np.int64]:
    return hp.ang2pix(nside, ra, dec, nest=True, lonlat=True).astype(np.int64)

def radec_to_healpixel_with_neighbors(ra: npt.NDArray[np.float64], dec: npt.NDArray[np.float64], nside: int) -> npt.NDArray[np.int64]:
    healpix =  hp.ang2pix(nside, ra, dec, nest=True, lonlat=True).astype(np.int64)
    neighbors = hp.pixelfunc.get_all_neighbours(nside, ra, dec, nest=True, lonlat=True).astype(np.int64)
    return np.append(neighbors, [healpix], axis=0)