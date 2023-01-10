import os
import sqlite3 as sql

import pytest

from precovery.frame_db import FrameIndex


@pytest.fixture
def test_db():
    out_db = os.path.join(os.path.dirname(__file__), "test.db")
    yield out_db
    os.remove(out_db)


def test_fast_query_warning(test_db):
    """Test that a warning is raised when the fast_query index does not exist."""
    con = sql.connect(test_db)
    con.execute(
        """
        CREATE TABLE frames (
            id INTEGER PRIMARY KEY,
            dataset_id TEXT,
            obscode TEXT,
            exposure_id TEXT,
            filter TEXT,
            exposure_mjd_start FLOAT,
            exposure_mjd_mid FLOAT,
            exposure_duration FLOAT,
            healpixel INTEGER,
            data_uri TEXT,
            data_offset INTEGER,
            data_length INTEGER
            );"""
    )

    with pytest.warns(UserWarning):
        FrameIndex.open("sqlite:///" + test_db, mode="r")

    return
