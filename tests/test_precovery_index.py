import glob
import os
import shutil

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest
import quivr as qv

from precovery.healpix_geom import radec_to_healpixel
from precovery.ingest import index
from precovery.observation import ObservationsTable
from precovery.precovery_db import PrecoveryDatabase

SAMPLE_ORBITS_FILE = os.path.join(
    os.path.dirname(__file__), "data", "sample_orbits.parquet"
)
TEST_OBSERVATIONS_DIR = os.path.join(os.path.dirname(__file__), "data/index")
MILLIARCSECOND = 1 / 3600 / 1000
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


@pytest.fixture
def test_db_dir():
    out_dir = os.path.join(os.path.dirname(__file__), "database")
    yield out_dir
    shutil.rmtree(out_dir)


def test_precovery_index(test_db_dir):
    """
    Test that frames and observations are being added to db and fs correctly.
    """
    observation_files = glob.glob(
        os.path.join(TEST_OBSERVATIONS_DIR, "dataset_*", "*.csv")
    )
    observations_dfs = []
    for observation_file in observation_files:
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

        index(
            out_dir=test_db_dir,
            dataset_id=dataset_id,
            dataset_name=dataset_id,
            data_dir=os.path.join(
                os.path.dirname(__file__), f"data/index/{dataset_id}/"
            ),
            nside=32,
        )

    observations_df = pd.concat(observations_dfs, ignore_index=True)
    observations_df.rename(columns={"obs_id": "id"}, inplace=True)
    observations_df = observations_df.sort_values("exposure_mjd_mid")
    input_observations_healpix = radec_to_healpixel(
        observations_df["ra"], observations_df["dec"], nside=32
    )
    observations_df["healpix"] = input_observations_healpix

    # Read in the frames and observations from the database
    db = PrecoveryDatabase.from_dir(test_db_dir, mode="r")
    frames = db.frames.idx.all_frames()
    frames = frames.sort_by("exposure_mjd_mid")

    # We should have a frame for every unique
    # combination of obscode, exposure_mjd_mid and healpix

    # generate unique grouping of healpix, obscode, exposure_mjd_mid
    # in the observation dataframe
    observation_grouping = observations_df.groupby(
        ["healpix", "observatory_code", "exposure_mjd_mid"]
    )
    assert len(frames) == len(observation_grouping)

    all_observations: ObservationsTable = ObservationsTable.empty()
    for frame in frames:
        frame_observations = db.frames.get_observations(frame)

        # Check that the frame observations contains the same list
        # as those in the observation dataframe with matching
        # healpix, obscode, exposure_mjd_mid
        observation_subset = observation_grouping.get_group(
            (
                frame.healpixel[0].as_py(),
                frame.obscode[0].as_py(),
                frame.exposure_mjd_mid[0].as_py(),
            )
        )

        for column in ["ra", "dec", "ra_sigma", "dec_sigma", "mag", "mag_sigma"]:
            npt.assert_array_equal(
                frame_observations.table[column].to_numpy(),
                observation_subset[column].values,
            )

        db_ids = frame_observations.table["id"].to_numpy().astype(str)
        npt.assert_array_equal(
            db_ids,
            observation_subset["id"].values,
        )

        db_mjds = frame_observations.time.mjd().to_numpy()
        npt.assert_array_equal(
            db_mjds,
            observation_subset["mjd"].values,
        )

        all_observations = qv.concatenate([all_observations, frame_observations])

    assert len(all_observations) == len(observations_df)

    # Make sure we have all the datasets
    all_datasets = db.all_datasets()
    input_datasets = set(observations_df["dataset_id"].unique())
    assert all_datasets == input_datasets
