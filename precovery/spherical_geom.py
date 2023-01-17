import numba
import numpy as np


@numba.jit(nopython=True, cache=True)
def haversine_distance(ra1: float, ra2: float, dec1: float, dec2: float) -> float:
    """
    Computes the great-circle distance between two points on a sphere, using
    the RA and Dec coordinate system (in radians).
    """
    s1 = np.sin((dec2 - dec1) / 2)
    s2 = np.sin((ra2 - ra1) / 2)
    val = 2 * np.arcsin(np.sqrt(s1 * s1 + np.cos(dec1) * np.cos(dec2) * s2 * s2))
    return val


@numba.jit(nopython=True, cache=True)
def haversine_distance_deg(ra1: float, ra2: float, dec1: float, dec2: float) -> float:
    return np.rad2deg(
        haversine_distance(
            np.deg2rad(ra1), np.deg2rad(ra2), np.deg2rad(dec1), np.deg2rad(dec2)
        )
    )
