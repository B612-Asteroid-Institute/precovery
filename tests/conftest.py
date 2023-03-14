import os

import pandas as pd
import pytest

from precovery.frame_db import FrameDB, FrameIndex
from precovery.orbit import EpochTimescale, Orbit
from precovery.precovery_db import PrecoveryDatabase


@pytest.fixture
def test_db():
    out_db = os.path.join(os.path.dirname(__file__), "test.db")
    yield out_db
    os.remove(out_db)


@pytest.fixture
def frame_index(tmp_path):
    fidx = FrameIndex.open("sqlite:///" + str(tmp_path) + "/test.db", mode="r")
    fidx.initialize_tables()
    yield fidx
    fidx.close()


@pytest.fixture
def frame_db(tmp_path, frame_index):
    data_root = tmp_path / "data"
    os.makedirs(name=data_root)
    fdb = FrameDB(idx=frame_index, data_root=str(data_root), mode="w")
    yield fdb
    fdb.close()


@pytest.fixture
def precovery_db(tmp_path, frame_db):
    yield PrecoveryDatabase.from_dir(str(tmp_path), create=True)


@pytest.fixture
def sample_orbits():
    sample_orbits_file = os.path.join(
        os.path.dirname(__file__), "data", "sample_orbits.csv"
    )
    df = pd.read_csv(sample_orbits_file)
    orbits = []
    for i in range(len(df)):
        orbit = Orbit.keplerian(
            i,
            df["a"].values[i],
            df["e"].values[i],
            df["i"].values[i],
            df["om"].values[i],
            df["w"].values[i],
            df["ma"].values[i],
            df["mjd_tt"].values[i],
            EpochTimescale.TT,
            df["H"].values[i],
            df["G"].values[i],
        )
        orbits.append(orbit)
    return orbits
