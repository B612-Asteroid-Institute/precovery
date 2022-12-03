import os
import pytest
import sqlite3 as sql

from precovery.frame_db import FrameIndex

TEST_DB = os.path.join(os.path.dirname(__file__), 'test.db')

def test_fast_query_warning():
    """Test that a warning is raised when the fast_query index does not exist."""
    con = sql.connect(TEST_DB)
    con.execute("""
        CREATE TABLE frames (
            id INTEGER PRIMARY KEY, 
            dataset_id TEXT, 
            obscode TEXT,
            exposure_id TEXT,
            filter TEXT,
            mjd FLOAT,
            healpixel INTEGER,
            data_uri TEXT,
            data_offset INTEGER,
            data_length INTEGER
            );"""
    )

    with pytest.warns(UserWarning):
        frame_db = FrameIndex.open("sqlite:///" + TEST_DB, mode="r")
    
    # Clean up
    os.remove(TEST_DB)
    return