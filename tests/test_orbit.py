import numpy as np

from precovery import orbit
from precovery.observation import Observation, ObservationArray

from .testutils import requires_openorb_data


def test_orbit_initialization_from_state_vector():
    state_vector = np.array(
        [
            [
                0,
                1.46905,
                0.33435,
                np.deg2rad(14.3024),
                np.deg2rad(224.513),
                np.deg2rad(27.5419),
                np.deg2rad(324.697),
                3,
                51544.5,
                1,
                12.5,
                0.15,
            ]
        ],
        dtype=np.double,
        order="F",
    )
    o = orbit.Orbit(0, state_vector)
    assert o._orbit_type == orbit.OrbitElementType.KEPLERIAN


@requires_openorb_data
def test_orbit_propagation():
    state_vector = np.array(
        [
            [
                0,
                1.46905,
                0.33435,
                np.deg2rad(14.3024),
                np.deg2rad(224.513),
                np.deg2rad(27.5419),
                np.deg2rad(324.697),
                3,
                51544.5,
                1,
                12.5,
                0.15,
            ]
        ],
        dtype=np.double,
        order="F",
    )
    o = orbit.Orbit(0, state_vector)

    propagated = o.propagate([51232.23])[0]
    assert propagated._epoch == 51232.23

    # Original should be unchanged
    assert o._epoch == 51544.5


@requires_openorb_data
def test_orbit_ephemeris_computation():
    state_vector = np.array(
        [
            [
                0,
                1.46905,
                0.33435,
                np.deg2rad(14.3024),
                np.deg2rad(224.513),
                np.deg2rad(27.5419),
                np.deg2rad(324.697),
                3,
                51544.5,
                1,
                12.5,
                0.15,
            ]
        ],
        dtype=np.double,
        order="F",
    )
    o = orbit.Orbit(0, state_vector)

    obscode = "534"
    epoch = 51544.5

    ephem = o.compute_ephemeris(obscode, [epoch])
    assert ephem[0].mjd == epoch


def test_ephemeris_distance():
    ephemeris = orbit.Ephemeris(mjd=1, ra=2, dec=3, ra_velocity=4, dec_velocity=5)

    observations = ObservationArray(
        [
            Observation(
                mjd=1,
                ra=2,
                dec=3,
                id=b"exact",
                ra_sigma=0,
                dec_sigma=0,
                mag=0,
                mag_sigma=0,
            ),
            Observation(
                mjd=1,
                ra=2,
                dec=4,
                id=b"one_degree_dec",
                ra_sigma=0,
                dec_sigma=0,
                mag=0,
                mag_sigma=0,
            ),
            Observation(
                mjd=1,
                ra=362,
                dec=3,
                id=b"wraparound",
                ra_sigma=0,
                dec_sigma=0,
                mag=0,
                mag_sigma=0,
            ),
        ]
    )

    distance = ephemeris.distance(observations)
    np.testing.assert_almost_equal(distance, [0, 1, 0])
