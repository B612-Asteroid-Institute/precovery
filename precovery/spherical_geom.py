from typing import Tuple

import numba
import numpy as np


@numba.jit
def propagate_linearly(
    ra0: float, dec0: float, vra: float, vdec: float, dt: float
) -> Tuple[float, float]:
    """
    Linearly propagate a position across the sky, given its spherical
    coordinates and the velocities in spherical coordinate terms.

    All inputs and outputs are in radians.

    ra0 and dec0 are the initial right ascension and declination - that is, the
    initial position.

    vra and vdec are the angular velocity of RA and Dec.

    dt is the time interval.

    Returns the resulting (ra, dec).

    Note that this is really only accurate for a few days. This is about 5x
    faster than doing a 2-body propagation.
    """
    # First, convert to Cartesian coordinates.
    cos_ra = np.cos(ra0)
    cos_dec = np.cos(dec0)
    sin_ra = np.sin(ra0)
    sin_dec = np.sin(dec0)

    x0 = cos_dec * cos_ra
    y0 = cos_dec * sin_ra
    z0 = sin_dec

    vx = -y0 * vra - sin_dec * cos_ra * vdec
    vy = x0 * vra - sin_dec * sin_ra * vdec
    vz = cos_dec * vdec

    # Now, propagate in cartesian space.
    x1 = x0 + vx * dt
    y1 = y0 + vy * dt
    z1 = z0 + vz * dt

    # Transform back to RA and Dec.
    r1 = x1 * x1 + y1 * y1 + z1 * z1
    ra1 = np.arctan2(y1, x1)
    dec1 = np.arcsin(z1 / r1)

    # Normalize values, since RA is always in [0, 2pi) and dec is in [-pi/2, pi/2].
    if ra1 < 0:
        ra1 += 2 * np.pi
    elif ra1 >= 2 * np.pi:
        ra1 -= 2 * np.pi

    if dec1 < -(np.pi / 2):
        dec1 += np.pi
    elif dec1 > (np.pi / 2):
        dec1 -= np.pi

    return ra1, dec1


@numba.jit
def haversine_distance(ra1: float, ra2: float, dec1: float, dec2: float) -> float:
    """
    Computes the great-circle distance between two points on a sphere, using
    the RA and Dec coordinate system (in radians).
    """
    s1 = np.sin((dec2 - dec1) / 2)
    s2 = np.sin((ra2 - ra1) / 2)
    val = 2 * np.arcsin(np.sqrt(s1 * s1 + np.cos(dec1) * np.cos(dec2) * s2 * s2))
    return val


def haversine_distance_deg(ra1: float, ra2: float, dec1: float, dec2: float) -> float:
    return np.rad2deg(
        haversine_distance(
            np.deg2rad(ra1), np.deg2rad(ra2), np.deg2rad(dec1), np.deg2rad(dec2)
        )
    )
