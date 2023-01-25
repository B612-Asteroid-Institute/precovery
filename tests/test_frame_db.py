import os
import sqlite3 as sql

import pytest
from astropy.time import Time

from precovery.frame_db import FrameDB, FrameIndex
from precovery.sourcecatalog import SourceFrame, SourceObservation


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


def test_year_month_str_key():
    t_start = Time("2006-01-02 15:04:05")
    t_mid = Time("2007-02-02 15:04:05")

    frame = SourceFrame(
        exposure_id="id",
        obscode="n/a",
        filter="n/a",
        exposure_mjd_start=t_start.mjd,
        exposure_mjd_mid=t_mid.mjd,
        exposure_duration=1,
        healpixel=1,
        observations=[],
    )
    assert FrameDB._compute_year_month_str(frame) == "2007-02"


def test_add_dataset_creates_if_missing(frame_db):
    frame_db.add_dataset(
        dataset_id="added_in_test__dataset_id",
        name="added_in_test__name",
        reference_doi="added_in_test__reference_doi",
        documentation_url="added_in_test__documentation_url",
        sia_url="added_in_test__sia_url",
    )
    assert frame_db.has_dataset("added_in_test__dataset_id")


def test_add_frames_without_observations(frame_db):
    """Adding a frame without observations should not crash."""
    frame_db.add_dataset(
        dataset_id="test_dataset",
    )
    t_start = Time("2006-01-02 15:04:05").mjd
    t_mid = Time("2006-01-02 15:04:35").mjd

    frame1 = SourceFrame(
        exposure_id="id",
        obscode="n/a",
        filter="n/a",
        exposure_mjd_start=t_start,
        exposure_mjd_mid=t_mid,
        exposure_duration=30,
        healpixel=1,
        observations=[],
    )
    frame_db.add_frames("test_dataset", [frame1])


def test_add_frames_empty_list(frame_db):
    """Adding an empty list should not crash."""
    frame_db.add_dataset(
        dataset_id="test_dataset",
    )
    frame_db.add_frames("test_dataset", [])


def test_add_frames(frame_db):
    """End-to-end test of adding frames."""
    frame_db.add_dataset(
        dataset_id="test_dataset",
    )
    t1_start = Time("2006-01-02 15:04:05").mjd
    t1_mid = Time("2006-01-02 15:04:35").mjd

    # A frame with two observations in it.
    frame1 = SourceFrame(
        exposure_id="e1",
        obscode="obs",
        filter="filter",
        exposure_mjd_start=t1_start,
        exposure_mjd_mid=t1_mid,
        exposure_duration=30,
        healpixel=1,
        observations=[
            SourceObservation(
                exposure_id="e1",
                obscode="obs",
                id=b"1",
                mjd=t1_mid,
                ra=1.0,
                dec=2.0,
                ra_sigma=3.0,
                dec_sigma=4.0,
                mag=5.0,
                mag_sigma=6.0,
                filter="filter",
                exposure_mjd_start=t1_start,
                exposure_mjd_mid=t1_mid,
                exposure_duration=30,
            ),
            SourceObservation(
                exposure_id="e1",
                obscode="obs",
                id=b"2",
                mjd=t1_mid,
                ra=21.0,
                dec=22.0,
                ra_sigma=23.0,
                dec_sigma=24.0,
                mag=25.0,
                mag_sigma=26.0,
                filter="filter",
                exposure_mjd_start=t1_start,
                exposure_mjd_mid=t1_mid,
                exposure_duration=30,
            ),
        ],
    )

    # Another frame from the same exposure with one observation.
    frame2 = SourceFrame(
        exposure_id="e1",
        obscode="obs",
        filter="filter",
        exposure_mjd_start=t1_start,
        exposure_mjd_mid=t1_mid,
        exposure_duration=30,
        healpixel=2,
        observations=[
            SourceObservation(
                exposure_id="e1",
                obscode="obs",
                id=b"3",
                mjd=t1_mid,
                ra=31.0,
                dec=32.0,
                ra_sigma=33.0,
                dec_sigma=34.0,
                mag=35.0,
                mag_sigma=36.0,
                filter="filter",
                exposure_mjd_start=t1_start,
                exposure_mjd_mid=t1_mid,
                exposure_duration=30,
            ),
        ],
    )

    t2_start = Time("2006-01-02 15:05:05").mjd
    t2_mid = Time("2006-01-02 15:05:35").mjd
    # A third frame from a different exposure.
    frame3 = SourceFrame(
        exposure_id="e2",
        obscode="obs",
        filter="filter",
        exposure_mjd_start=t2_start,
        exposure_mjd_mid=t2_mid,
        exposure_duration=30,
        healpixel=2,
        observations=[
            SourceObservation(
                exposure_id="e2",
                obscode="obs",
                id=b"4",
                mjd=t2_mid,
                ra=41.0,
                dec=42.0,
                ra_sigma=43.0,
                dec_sigma=44.0,
                mag=45.0,
                mag_sigma=46.0,
                filter="filter",
                exposure_mjd_start=t2_start,
                exposure_mjd_mid=t2_mid,
                exposure_duration=30,
            ),
        ],
    )
    frame_db.add_frames("test_dataset", [frame1, frame2, frame3])

    # Okay, setup is done - assertions begin here.
    stored_frames = list(frame_db.idx.all_frames())
    assert len(stored_frames) == 3

    # Find frame 1
    stored_frame_1 = None
    for f in stored_frames:
        if f.exposure_id == "e1" and f.healpixel == 1:
            stored_frame_1 = f
            break
    assert stored_frame_1 is not None, "frame1 was not stored, or not retrieved"

    # Get associated observations
    stored_obs1 = list(frame_db.iterate_observations(stored_frame_1))
    assert len(stored_obs1) == 2
