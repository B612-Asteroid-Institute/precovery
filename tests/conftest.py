import os

import pytest
from adam_core.orbits import Orbits

from precovery.frame_db import FrameDB, FrameIndex
from precovery.precovery_db import PrecoveryDatabase


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
    os.makedirs(name=data_root)
    fdb = FrameDB(idx=frame_index, data_root=str(data_root), mode="w")
    yield fdb
    fdb.close()


@pytest.fixture
def precovery_db(tmp_path, frame_db):
    yield PrecoveryDatabase.from_dir(str(tmp_path), mode="w", create=True)


@pytest.fixture
def sample_orbits():
    sample_orbits_file = os.path.join(
        os.path.dirname(__file__), "data", "sample_orbits.parquet"
    )
    orbits = Orbits.from_parquet(sample_orbits_file)

    return orbits
