import os
import numpy as np
import pandas as pd
import shutil

from precovery.ingest import index
from precovery.main import precover
from precovery.orbit import EpochTimescale, Orbit

SAMPLE_ORBITS_FILE = os.path.join(os.path.dirname(__file__), "data", "sample_orbits.csv")
TEST_OBSERVATION_FILE =  os.path.join(os.path.dirname(__file__), "data", "observations.h5")

def test_precovery():
    """
    Given a properly formatted h5 file, ensure the observations are indexed properly
    """
    try:
        out_dir = os.path.join(os.path.dirname(__file__), "database")
        index(
            out_dir=out_dir,
            dataset_id="test_dataset",
            dataset_name="Test Dataset",
            data_dir=os.path.join(os.path.dirname(__file__), "data"),
        )

        # TODO: Cross-check the total number of observations in each frame file
        # against the total number in the input hdf5 file

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

        # Load observations from h5 file
        observations_df = pd.read_hdf(TEST_OBSERVATION_FILE)

        # For each sample orbit, validate we get all the observations we planted
        for orbit in orbits_keplerian:
            results = precover(orbit, out_dir)

            object_observations = observations_df[
                observations_df["object_id"] == orbit_name_mapping[orbit.orbit_id]
            ]
            assert len(results) == len(object_observations) 
            assert len(results) > 0

            results.rename(
                columns={
                    "ra_deg": "ra",
                    "dec_deg": "dec",
                    "ra_sigma_arcsec": "ra_sigma",
                    "dec_sigma_arcsec": "dec_sigma",
                    "observation_id" : "obs_id",
                    "obscode" : "observatory_code",
                },
                inplace=True,
            )

            results["ra_sigma"] /= 3600.0
            results["dec_sigma"] /= 3600.0

            # We are assuming that both the test observation file and the results
            # are sorted by mjd_utc
            for col in [
                "mjd_utc",
                "ra",
                "ra_sigma",
                "dec",
                "dec_sigma",
                "mag",
                "mag_sigma",
                # "filter", # can't do string comparisons this way
            ]:
                np.testing.assert_array_equal(
                    object_observations[col].values, results[col].values
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

            # Test that the predicted location of each objet in each exposure is 
            # identical to the actual location of the object in that exposure (we did
            # not add any errors to the test observations)
            np.testing.assert_array_equal(
                results[["pred_ra_deg", "pred_dec_deg"]].values,
                object_observations[["ra", "dec"]].values,
            )

            # Test that the calculated distance is all zeros (again, we did not add
            # any errors to the test observations)
            np.testing.assert_array_equal(results["distance_arcsec"].values, np.zeros(len(results), dtype=np.float64))

            
        # TODO: validate length of frames table against what we would expect
    
    finally:

        # Clean up
        shutil.rmtree("database")