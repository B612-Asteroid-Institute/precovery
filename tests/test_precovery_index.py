import glob
import logging
import os
import shutil

import numpy as np
import pandas as pd
import pytest

from precovery.ingest import index
from precovery.main import precover
from precovery.orbit import EpochTimescale, Orbit

SAMPLE_ORBITS_FILE = os.path.join(
    os.path.dirname(__file__), "data", "sample_orbits.csv"
)
TEST_OBSERVATIONS_DIR = os.path.join(os.path.dirname(__file__), "data/index")
MILLIARCSECOND = 1 / 3600 / 1000
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

logger = logging.getLogger("precovery-test")


@pytest.fixture
def test_db_dir():
    out_dir = os.path.join(os.path.dirname(__file__), "database")
    yield out_dir
    shutil.rmtree(out_dir)


def test_precovery(test_db_dir):
    """
    Given a properly formatted h5 file, ensure the observations are indexed properly
    """
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

    # Load observations from csv files and index each one (one observation file
    # per dataset)
    observation_files = glob.glob(
        os.path.join(TEST_OBSERVATIONS_DIR, "dataset_*", "*.csv")
    )
    logger.info(f"observation files: {observation_files}")
    observations_dfs = []
    for observation_file in observation_files:
        logger.info(f"reading file {observation_file}")
        observations_df_i = pd.read_csv(
            observation_file,
            float_precision="round_trip",
            dtype={
                "dataset_id": str,
                "observatory_code": str,
                "filter": str,
                "exposure_duration": np.float64,
            },
        )
        observations_dfs.append(observations_df_i)

        dataset_id = observations_df_i["dataset_id"].values[0]

        data_dir = os.path.join(os.path.dirname(__file__), f"data/index/{dataset_id}/")
        logger.info(
            f"indexing, out_dir={test_db_dir}, \
            dataset_id={dataset_id}, \
            dataset_name={dataset_id}, \
            data_dir={data_dir}, \
            nside=16"
        )
        index(
            out_dir=test_db_dir,
            dataset_id=dataset_id,
            dataset_name=dataset_id,
            data_dir=data_dir,
            nside=16,
        )
    observations_df = pd.concat(observations_dfs, ignore_index=True)
    observations_df.sort_values(
        by=["mjd", "observatory_code"], inplace=True, ignore_index=True
    )

    logger.info(f"index glob: {glob.glob('*index.db')}")
    logger.info(f"index double glob: {glob.glob('**index.db')}")

    # For each sample orbit, validate we get all the observations we planted
    for orbit in orbits_keplerian:
        logger.info(f"precovering {orbit}")
        results = precover(orbit, test_db_dir, tolerance=1 / 3600, window_size=1)

        orbit_name = orbit_name_mapping[orbit.orbit_id]
        object_observations = observations_df[
            observations_df["object_id"] == orbit_name
        ]
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
                err_msg=f"Column {col} does not match for {orbit_name}.",
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

        # test that the predicted location of each object in each exposure is
        # close to the actual location of the object in that exposure (we did
        # not add any errors to the test observations)
        # Note that the predicted location is sensitive to accumulating float point arithmetic
        # errors since orbits in precovery are propagated, then stored, then propagated again, and so on.
        # The number of propagations will have an effect on the consistency of the predicted
        # location when compared to the single propagation required to create the test observations.
        # Additionally, the observations in the test data are not defined at the same as the midpoint of
        # the exposure. Internally precovery will use 2-body propagation to adjust the predicted location
        # of the orbit within the frame in the cases, which will also introduce some error.
        try:
            np.testing.assert_allclose(
                results[["pred_ra_deg", "pred_dec_deg"]].values,
                object_observations[["ra", "dec"]].values,
                atol=MILLIARCSECOND,
                rtol=0,
                err_msg=(
                    f"Predicted location does match actual location for {orbit_name}.",
                ),
            )

            # Test that the calculated distance is within 1 millarcsecond (need additional order of magnitude
            # tolerance to account for errors added in quadrature)
            np.testing.assert_allclose(
                results["distance_arcsec"].values / 3600.0,
                np.zeros(len(results), dtype=np.float64),
                atol=MILLIARCSECOND,
                rtol=0,
                err_msg=f"Distance for {orbit_name} is not within tolerance.",
            )

        except AssertionError as e:
            os.makedirs(RESULTS_DIR, exist_ok=True)
            result_file = os.path.join(RESULTS_DIR, f"results_{orbit_name}.csv")
            results.to_csv(result_file, float_format="%.16f", index=False)

            object_observations_file = os.path.join(
                RESULTS_DIR,
                f"object_observations_{orbit_name}.csv",
            )
            object_observations.to_csv(
                object_observations_file, float_format="%.16f", index=False
            )

            print(f"Results written to: {result_file}")
            print(f"Object observations written to: {object_observations_file}")
            raise e
