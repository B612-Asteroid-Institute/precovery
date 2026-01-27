import numpy as np
import pyarrow as pa
from adam_core.coordinates.covariances import CoordinateCovariances
from adam_core.coordinates.origin import Origin
from adam_core.coordinates.spherical import SphericalCoordinates
from adam_core.orbits.ephemeris import Ephemeris
from adam_core.time import Timestamp

from experiments.covariance_precovery.methods.detection_filtering import (
    FootprintThenChi2Filter,
    score_mog_tangent_plane,
)
from experiments.covariance_precovery.methods.footprints import EllipseFootprint
from precovery.observation import ObservationsTable


def test_score_mog_returns_finite_for_near_points() -> None:
    obs_ra = np.array([10.0, 10.01])
    obs_dec = np.array([20.0, 20.0])
    samp_ra = np.array([10.0, 10.005])
    samp_dec = np.array([20.0, 20.0])
    ll = score_mog_tangent_plane(
        obs_ra_deg=obs_ra,
        obs_dec_deg=obs_dec,
        sample_ra_deg=samp_ra,
        sample_dec_deg=samp_dec,
        sigma_arcsec=10.0,
    )
    assert np.isfinite(ll).all()


def test_footprint_prefilter_reduces_observations() -> None:
    # Build synthetic observations: one near, one far.
    t = Timestamp.from_mjd([60000.0, 60000.0], scale="utc")
    obs = ObservationsTable.from_kwargs(
        id=[b"a", b"b"],
        time=t,
        ra=[10.0, 100.0],
        dec=[20.0, -20.0],
        ra_sigma=[1.0 / 3600.0, 1.0 / 3600.0],
        dec_sigma=[1.0 / 3600.0, 1.0 / 3600.0],
        mag=[20.0, 20.0],
        mag_sigma=[0.1, 0.1],
    )

    # Synthetic ephem aligned with obs, with tiny predicted covariance so chi2 gate passes near one.
    cov = np.zeros((2, 6, 6), dtype=np.float64)
    cov[:, 0, 0] = 1.0
    cov[:, 3, 3] = 1.0
    cov[:, 4, 4] = 1.0
    cov[:, 5, 5] = 1.0
    cov[:, 1, 1] = (1.0 / 3600.0) ** 2
    cov[:, 2, 2] = (1.0 / 3600.0) ** 2

    coords = SphericalCoordinates.from_kwargs(
        rho=np.full(2, np.nan),
        lon=pa.array([10.0, 100.0]),
        lat=pa.array([20.0, -20.0]),
        vrho=np.full(2, np.nan),
        vlon=np.full(2, np.nan),
        vlat=np.full(2, np.nan),
        time=t,
        covariance=CoordinateCovariances.from_matrix(cov),
        origin=Origin.from_kwargs(code=["I41", "I41"]),
        frame="equatorial",
    )
    eph = Ephemeris.from_kwargs(orbit_id=["o1", "o1"], coordinates=coords)

    fp = EllipseFootprint(
        lon0_deg=10.0,
        lat0_deg=20.0,
        cov_ll_deg2=np.array([[1e-6, 0.0], [0.0, 1e-6]]),
        n_sigma=3.0,
    )

    filt = FootprintThenChi2Filter()
    obs2, _, metrics = filt.run(observations=obs, ephem=eph, footprint=fp, n_sigma=3.0)
    assert metrics.n_in == 2
    assert metrics.n_after_footprint == 1
    assert len(obs2) == 1

