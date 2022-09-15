import numpy as np

from precovery import orbit, spherical_geom


def test_propagate_linearly_at_poles():
    one_degree = np.deg2rad(1)
    # Should work at the poles, acting basically linearly over small velocity
    ra0 = 0
    dec0 = np.pi / 4

    vra = one_degree
    vdec = one_degree
    dt = 1.0
    ra1, dec1 = spherical_geom.propagate_linearly(ra0, dec0, vra, vdec, dt)
    assert np.isclose(ra0 + one_degree, ra1, atol=0.001)

    # Go again, but this time loop backwards over the pole
    ra0 = 0
    dec0 = np.pi / 4

    vra = -one_degree
    vdec = one_degree
    dt = 1.0
    ra1, dec1 = spherical_geom.propagate_linearly(ra0, dec0, vra, vdec, dt)
    assert np.isclose(2 * np.pi + vra, ra1, atol=0.001)

    # Now do the north pole
    ra0 = np.pi / 4
    dec0 = np.pi / 2

    vra = 0
    vdec = (np.pi / 180) / 10
    dt = 1.0
    ra1, dec1 = spherical_geom.propagate_linearly(ra0, dec0, vra, vdec, dt)

    # We should have flipped over the top, so RA is offset by 180 degrees
    assert np.isclose(ra0 + np.pi, ra1, atol=0.001)

    # And declination should have been decreased by going over the top
    expected = dec0 - vdec
    assert np.isclose(expected, dec1, atol=0.001)


def test_propagate_linearly_vs_integration():
    # compare propagate_linearly to n-body integration
    t0 = 51544.5
    obscode = "534"

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
                t0,
                1,
                12.5,
                0.15,
            ]
        ],
        dtype=np.double,
        order="F",
    )
    o = orbit.Orbit(0, state_vector)
    initial_position = o.compute_ephemeris(obscode, [t0])[0]

    # Zero dt should result in zero motion
    ra, dec = np.rad2deg(
        spherical_geom.propagate_linearly(
            np.deg2rad(initial_position.ra),
            np.deg2rad(initial_position.dec),
            np.deg2rad(initial_position.ra_velocity),
            np.deg2rad(initial_position.dec_velocity),
            0,
        )
    )
    assert np.isclose(ra, initial_position.ra)
    assert np.isclose(dec, initial_position.dec)

    # dt of 0.5 should match propagation to within 0.1 degrees
    t1 = t0 + 0.5
    ra, dec = np.rad2deg(
        spherical_geom.propagate_linearly(
            np.deg2rad(initial_position.ra),
            np.deg2rad(initial_position.dec),
            np.deg2rad(initial_position.ra_velocity),
            np.deg2rad(initial_position.dec_velocity),
            t1 - t0,
        )
    )
    expected_position = o.compute_ephemeris(obscode, [t1])[0]
    assert np.isclose(ra, expected_position.ra, atol=0.1)
    assert np.isclose(dec, expected_position.dec, atol=0.1)

    # dt of 1.0 should match propagation to within 0.2 degrees
    t1 = t0 + 1.0
    ra, dec = np.rad2deg(
        spherical_geom.propagate_linearly(
            np.deg2rad(initial_position.ra),
            np.deg2rad(initial_position.dec),
            np.deg2rad(initial_position.ra_velocity),
            np.deg2rad(initial_position.dec_velocity),
            t1 - t0,
        )
    )
    expected_position = o.compute_ephemeris(obscode, [t1])[0]
    assert np.isclose(ra, expected_position.ra, atol=0.2)
    assert np.isclose(dec, expected_position.dec, atol=0.2)

    # dt of 3.0 should match propagation to within 0.5 degrees
    t1 = t0 + 3.0
    ra, dec = np.rad2deg(
        spherical_geom.propagate_linearly(
            np.deg2rad(initial_position.ra),
            np.deg2rad(initial_position.dec),
            np.deg2rad(initial_position.ra_velocity),
            np.deg2rad(initial_position.dec_velocity),
            t1 - t0,
        )
    )
    expected_position = o.compute_ephemeris(obscode, [t1])[0]
    assert np.isclose(ra, expected_position.ra, atol=0.5)
    assert np.isclose(dec, expected_position.dec, atol=0.5)
