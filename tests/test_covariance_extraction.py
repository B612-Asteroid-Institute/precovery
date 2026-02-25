from __future__ import annotations

import numpy as np

from adam_core.coordinates.covariances import CoordinateCovariances
from adam_core.coordinates.origin import Origin, OriginCodes
from adam_core.coordinates.spherical import SphericalCoordinates
from adam_core.orbits.ephemeris import Ephemeris
from adam_core.time import Timestamp

from precovery.search.covariance import (
    attach_cov_ll_to_ephemeris,
    cov_ll_elements_from_coordinate_covariances,
    cov_ll_from_ephemeris,
)


def test_cov_ll_elements_from_coordinate_covariances_extracts_lonlat_block() -> None:
    cov = np.zeros((2, 6, 6), dtype=np.float64)
    cov[:, 1, 1] = [1.0, 2.0]
    cov[:, 1, 2] = [3.0, 4.0]
    cov[:, 2, 1] = [3.0, 4.0]
    cov[:, 2, 2] = [5.0, 6.0]

    cc = CoordinateCovariances.from_matrix(cov)
    c00, c01, c11 = cov_ll_elements_from_coordinate_covariances(cc)
    assert c00.tolist() == [1.0, 2.0]
    assert c01.tolist() == [3.0, 4.0]
    assert c11.tolist() == [5.0, 6.0]


def test_attach_cov_ll_to_ephemeris_round_trips_via_cov_ll_from_ephemeris() -> None:
    t = Timestamp.from_mjd([60000.0, 60000.1], scale="utc")
    coords = SphericalCoordinates.from_kwargs(
        rho=[1.0, 1.0],
        lon=[10.0, 20.0],
        lat=[0.0, 1.0],
        vrho=[0.0, 0.0],
        vlon=[0.0, 0.0],
        vlat=[0.0, 0.0],
        time=t,
        origin=Origin.from_kwargs(code=[OriginCodes.SUN.name, OriginCodes.SUN.name]),
        frame="equatorial",
    )
    ephem = Ephemeris.from_kwargs(orbit_id=["o1", "o1"], object_id=["o1", "o1"], coordinates=coords)

    cov_ll = np.array([[[1.0, 2.0], [2.0, 3.0]], [[4.0, 5.0], [5.0, 6.0]]], dtype=np.float64)
    ephem2 = attach_cov_ll_to_ephemeris(ephem=ephem, cov_ll_deg2=cov_ll)

    got = cov_ll_from_ephemeris(ephem2)
    np.testing.assert_allclose(got, cov_ll)

