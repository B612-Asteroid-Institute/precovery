import os
import shutil
import sqlite3 as sql

import numpy as np
import pandas as pd
import pytest

from precovery.ingest import index
from precovery.main import precover
from precovery.orbit import EpochTimescale, Orbit

SAMPLE_ORBITS_FILE = os.path.join(
    os.path.dirname(__file__), "data", "sample_orbits.csv"
)
TEST_OBSERVATION_FILE = os.path.join(
    os.path.dirname(__file__), "data/index", "observations.csv"
)
MILLIARCSECOND = 1 / 3600 / 1000


@pytest.fixture
def test_db_dir():
    out_dir = os.path.join(os.path.dirname(__file__), "database")
    yield out_dir
    shutil.rmtree(out_dir)


precovery_algorithms = ["nelson", "bruteforce"]


@pytest.mark.parametrize("algorithm", precovery_algorithms)
def test_precovery(test_db_dir, algorithm):
    """
    Given a properly formatted h5 file, ensure the observations are indexed properly
    """
    index(
        out_dir=test_db_dir,
        dataset_id="test_dataset",
        dataset_name="Test Dataset",
        data_dir=os.path.join(os.path.dirname(__file__), "data/index"),
    )

    # Initialize orbits from sample orbits file
    orbits_df = pd.read_csv(SAMPLE_ORBITS_FILE)
    orbit_name_mapping = {}
    orbits_keplerian = []
    for i in range(len(orbits_df)):
        orbit_name_mapping[i] = orbits_df["orbit_name"].values[i]
        orbit = Orbit.keplerian(
            i,
            orbits_df["a"].values[i],
            orbits_df["e"].values[i],
            orbits_df["i"].values[i],
            orbits_df["om"].values[i],
            orbits_df["w"].values[i],
            orbits_df["ma"].values[i],
            orbits_df["mjd_tt"].values[i],
            EpochTimescale.TT,
            orbits_df["H"].values[i],
            orbits_df["G"].values[i],
        )
        orbits_keplerian.append(orbit)

    # Load observations from csv file
    observations_df = pd.read_csv(TEST_OBSERVATION_FILE, float_precision="round_trip")

    # Test that the number of frames is equal to the number of observations
    # Note that this will only be true for a small enough number of objects that are not near
    # each other on the sky, which is fine for our test data set
    con = sql.connect(os.path.join(test_db_dir, "index.db"))
    frames = pd.read_sql("""SELECT * FROM frames""", con)
    assert len(frames) == len(observations_df)
    con.close()

    # For each sample orbit, validate we get all the observations we planted
    for orbit in orbits_keplerian:
        results = precover(
            orbit, test_db_dir, tolerance=1 / 3600, window_size=1, algorithm=algorithm
        )

        object_observations = observations_df[
            observations_df["object_id"] == orbit_name_mapping[orbit.orbit_id]
        ]
        assert len(results) == len(object_observations)
        assert len(results) == len(object_observations)
        assert len(results) == len(object_observations)
        assert len(results) > 0

        results.rename(
            columns={
                "ra_deg": "ra",
                "dec_deg": "dec",
                "ra_sigma_arcsec": "ra_sigma",
                "dec_sigma_arcsec": "dec_sigma",
                "observation_id": "obs_id",
                "obscode": "observatory_code",
            },
            inplace=True,
        )

        results["ra_sigma"] /= 3600.0
        results["dec_sigma"] /= 3600.0

        # We are assuming that both the test observation file and the results
        # are sorted by mjd
        for col in [
            "mjd",
            "ra",
            "ra_sigma",
            "dec",
            "dec_sigma",
            "mag",
            "mag_sigma",
            "exposure_mjd_start",
            "exposure_mjd_mid",
            "exposure_duration"
            # "filter", # can't do string comparisons this way
        ]:
            np.testing.assert_array_equal(
                object_observations[col].values,
                results[col].values,
                err_msg=f"Column {col} does not match",
            )

        # Test that the observation_id, exposure_id, observatory_code, and filter
        # are identical to the test observations
        for col in [
            "obs_id",
            "exposure_id",
            "observatory_code",
            "filter",
        ]:
            assert (results[col].values == object_observations[col].values).all()

        # Test that the predicted location of each object in each exposure is
        # close to the actual location of the object in that exposure (we did
        # not add any errors to the test observations)
        # Note that the predicted location is sensitive to accumulating float point arithmetic
        # errors since orbits in precovery are propagated, then stored, then propagated again, and so on.
        # The number of propagations will have an effect on the consistency of the predicted
        # location when compared to the single propagation required to create the test observations.
        # Additionally, the observations in the test data are not defined at the same as the midpoint of
        # the exposure. Internally precovery will use 2-body propagation to adjust the predicted location
        # of the orbit within the frame in the cases, which will also introduce some error.
        np.testing.assert_allclose(
            results[["pred_ra_deg", "pred_dec_deg"]].values,
            object_observations[["ra", "dec"]].values,
            atol=MILLIARCSECOND / 10,
            rtol=0,
        )

        # Test that the calculated distance is within 1 millarcsecond (need additional order of magnitude
        # tolerance to account for errors added in quadrature)
        np.testing.assert_allclose(
            results["distance_arcsec"].values / 3600.0,
            np.zeros(len(results), dtype=np.float64),
            atol=MILLIARCSECOND,
            rtol=0,
        )
