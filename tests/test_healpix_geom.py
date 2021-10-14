import numpy as np
import pytest

from precovery.healpix_geom import radec_to_healpixels, radec_to_thetaphi


@pytest.mark.parametrize(
    "ra,dec,theta,phi",
    [
        (0, 90, 0, 0),
        (90, 90, 0, np.pi / 2),
        (0, 0, np.pi / 2, 0),
        (0, -90, np.pi, 0),
        (0, -45, np.pi * 3 / 4, 0),
        (180, 45, np.pi / 4, np.pi),
    ],
)
def test_radec_to_thetaphi(ra, dec, theta, phi):
    have_theta, have_phi = radec_to_thetaphi(ra, dec)
    assert theta == have_theta
    assert phi == have_phi


@pytest.mark.parametrize(
    "radecs,nside,healpixels",
    [
        ([], 1 << 8, []),
        (
            [
                (45, 60),
                (135, 60),
                (225, 60),
                (315, 60),
                (0, 0),
                (90, 0),
                (180, 0),
                (270, 0),
                (45, -60),
                (135, -60),
                (225, -60),
                (315, -60),
            ],
            1,
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        ),
        (
            [
                (45, 30),
                (67.5, 50),
                (22.5, 50),
                (45, 75),
            ],
            2,
            [0, 1, 2, 3],
        ),
    ],
)
def test_radec_to_healpixels(radecs, nside, healpixels):
    have = radec_to_healpixels(radecs, nside)
    assert list(have) == list(healpixels)
