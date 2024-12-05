import glob
import os

import numpy as np
import pandas as pd
import pytest
from adam_core.orbits import Orbits

from precovery.frame_db import FrameDB, FrameIndex
from precovery.ingest import index
from precovery.precovery_db import PrecoveryDatabase

SAMPLE_ORBITS_FILE = os.path.join(
    os.path.dirname(__file__), "data", "sample_orbits.parquet"
)
TEST_OBSERVATIONS_DIR = os.path.join(os.path.dirname(__file__), "data/index")


@pytest.fixture
def test_db():
    out_db = os.path.join(os.path.dirname(__file__), "test.db")
    yield out_db
    os.remove(out_db)


@pytest.fixture
def frame_index(tmp_path):
    fidx = FrameIndex("sqlite:///" + str(tmp_path) + "/test.db", mode="w")
    yield fidx
    fidx.close()


@pytest.fixture
def frame_db(tmp_path, frame_index):
    data_root = tmp_path / "data"
    # remove the folder first to avoid any conflicts
    if os.path.exists(data_root):
        os.removedirs(name=data_root)
    os.makedirs(name=data_root)
    fdb = FrameDB(idx=frame_index, data_root=str(data_root), mode="w")
    yield fdb
    fdb.close()


@pytest.fixture
def precovery_db(tmp_path, frame_db):
    yield PrecoveryDatabase.create(str(tmp_path), nside=32)


@pytest.fixture
def precovery_db_with_data(tmp_path):
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
            out_dir=tmp_path,
            dataset_id=dataset_id,
            dataset_name=dataset_id,
            data_dir=os.path.join(
                os.path.dirname(__file__), f"data/index/{dataset_id}/"
            ),
            nside=32,
        )

    # Read in the frames and observations from the database
    db = PrecoveryDatabase.from_dir(tmp_path, mode="r")
    return db


@pytest.fixture
def sample_orbits():
    sample_orbits_file = os.path.join(
        os.path.dirname(__file__), "data", "sample_orbits.parquet"
    )
    orbits = Orbits.from_parquet(sample_orbits_file)

    return orbits
