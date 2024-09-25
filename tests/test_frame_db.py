import sqlite3 as sql

import numpy as np
import pytest
from astropy.time import Time

from precovery.frame_db import FrameDB, FrameIndex
from precovery.sourcecatalog import SourceFrame, SourceObservation, bundle_into_frames

from .testutils import make_sourceframe_with_observations, make_sourceobs


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

    con.execute(
        """
        CREATE TABLE datasets (
            id INTEGER PRIMARY KEY,
            name TEXT
            );"""
    )

    with pytest.warns(UserWarning):
        FrameIndex("sqlite:///" + test_db, mode="r")

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
    stored_frames = list(frame_db.idx.all_frames_by_key())
    assert len(stored_frames) == 3

    # Find frame 1
    stored_frame_1 = None
    for key, frames in stored_frames:
        for f in frames:
            if f.exposure_id == "e1" and f.healpixel == 1:
                stored_frame_1 = f
                break
    assert stored_frame_1 is not None, "frame1 was not stored, or not retrieved"

    # Get associated observations
    stored_obs1 = list(frame_db.iterate_observations(stored_frame_1))
    assert len(stored_obs1) == 2


def test_add_frames_optional(frame_db):
    """Add a frame with an observation that contains optional quantities (represented as NaN)
    and ensure that they are stored and retrieved correctly.

    Optional quantities are currently limited to ra_sigma, dec_sigma, and mag_sigma.
    """
    frame_db.add_dataset(
        dataset_id="test_dataset_optional",
    )

    start_time = 60000.0
    exposure_duration = 60
    midpoint_time = start_time + exposure_duration / 86400

    # A frame with two observations in it.
    frame = SourceFrame(
        exposure_id="n1",
        obscode="obs_optional",
        filter="filter",
        exposure_mjd_start=start_time,
        exposure_mjd_mid=midpoint_time,
        exposure_duration=exposure_duration,
        healpixel=1,
        observations=[
            SourceObservation(
                exposure_id="n1",
                obscode="obs_optional",
                id=b"1",
                mjd=midpoint_time,
                ra=1.0,
                dec=2.0,
                ra_sigma=np.nan,
                dec_sigma=np.nan,
                mag=5.0,
                mag_sigma=np.nan,
                filter="filter",
                exposure_mjd_start=start_time,
                exposure_mjd_mid=midpoint_time,
                exposure_duration=exposure_duration,
            ),
        ],
    )

    frame_db.add_frames("test_dataset_optional", [frame])

    frame_stored = list(frame_db.idx.get_frames("obs_optional", midpoint_time, 1))
    assert len(frame_stored) == 1

    # Let's extract the observation and check that the nans were
    # correctly stored.
    for obs in frame_db.iterate_observations(frame_stored[0]):
        assert np.isnan(obs.ra_sigma)
        assert np.isnan(obs.dec_sigma)
        assert np.isnan(obs.mag_sigma)


def test_add_frames_dedupes_sorted_exposures(frame_db):
    """When the database ingests sorted exposure data, observations
    from the same frame should just generate one row in the index for
    the frame, not multiple.

    """
    observations = [
        make_sourceobs(
            exposure_id=b"exp0",
            id=b"obs1",
            healpixel=1,
        ),
        make_sourceobs(
            exposure_id=b"exp1",
            id=b"obs2",
            healpixel=1,
        ),
        make_sourceobs(
            exposure_id=b"exp1",
            id=b"obs3",
            healpixel=1,
        ),
        make_sourceobs(
            exposure_id=b"exp1",
            id=b"obs4",
            healpixel=2,
        ),
        make_sourceobs(
            exposure_id=b"exp1",
            id=b"obs5",
            healpixel=2,
        ),
        make_sourceobs(
            exposure_id=b"exp2",
            id=b"obs6",
            healpixel=1,
        ),
    ]
    frames = list(bundle_into_frames(observations))

    assert len(frames) == 4

    dataset_id = "test_dataset"
    frame_db.add_dataset(dataset_id)
    frame_db.add_frames(dataset_id, frames)

    assert frame_db.idx.n_frames() == 4


def test_get_frames_for_ra_dec(frame_db):
    observations = [
        make_sourceobs(
            exposure_id=b"exp0",
            id=b"obs1",
            ra=1,
            dec=2,
            obscode="testobs",
        ),
        make_sourceobs(
            exposure_id=b"exp0",
            id=b"obs1",
            ra=1,
            dec=2,
            obscode="testobs",
        ),
        make_sourceobs(
            exposure_id=b"exp0",
            id=b"obs1",
            ra=40,
            dec=41,
            obscode="testobs",
        ),
    ]
    frames = list(bundle_into_frames(observations))
    assert len(frames) == 2

    dataset_id = "test_dataset"
    frame_db.add_dataset(dataset_id)
    frame_db.add_frames(dataset_id, frames)

    results = list(frame_db.get_frames_for_ra_dec(1, 2, "testobs"))
    assert len(results) == 1


def test_window_centers_dataset_filter(frame_db):
    mjd = 10
    frames_1 = [
        make_sourceframe_with_observations(
            obscode="obs1", n_observations=1, exposure_mjd_mid=mjd
        )
    ]
    dataset_1 = "test_dataset_1"
    frame_db.add_dataset(dataset_1)
    frame_db.add_frames(dataset_1, frames_1)

    frames_2 = [
        make_sourceframe_with_observations(
            obscode="obs2", n_observations=1, exposure_mjd_mid=mjd
        )
    ]
    dataset_2 = "test_dataset_2"
    frame_db.add_dataset(dataset_2)
    frame_db.add_frames(dataset_2, frames_2)

    # When using no dataset filter, we should return windows for all datasets
    centers = frame_db.idx.window_centers(
        start_mjd=mjd - 1, end_mjd=mjd + 1, window_size_days=1
    )
    print(centers)

    assert len(centers) == 2
    seen_obscodes = set()
    for obscode, _ in centers:
        seen_obscodes.add(obscode)

    assert seen_obscodes == {"obs1", "obs2"}

    # When using a dataset filter, we should return windows for only
    # the specified datasets
    centers = frame_db.idx.window_centers(
        start_mjd=mjd - 1,
        end_mjd=mjd + 1,
        window_size_days=1,
        datasets={dataset_1},
    )

    assert len(centers) == 1
    seen_obscodes = set()
    for obscode, _ in centers:
        seen_obscodes.add(obscode)

    assert seen_obscodes == {"obs1"}


def test_propagation_targets_dataset_filter(frame_db):
    mjd = 10
    frames_1 = [
        make_sourceframe_with_observations(
            exposure_id="exp1",
            obscode="obs1",
            n_observations=1,
            exposure_mjd_mid=mjd,
            healpixel=1,
        )
    ]
    dataset_1 = "test_dataset_1"
    frame_db.add_dataset(dataset_1)
    frame_db.add_frames(dataset_1, frames_1)

    # Note: second dataset has same obscode as the first dataset,
    # which makes sure we test the filtering behavior.
    frames_2 = [
        make_sourceframe_with_observations(
            exposure_id="exp2",
            obscode="obs1",
            n_observations=1,
            exposure_mjd_mid=mjd,
            healpixel=2,
        )
    ]
    dataset_2 = "test_dataset_2"
    frame_db.add_dataset(dataset_2)
    frame_db.add_frames(dataset_2, frames_2)

    targets = list(
        frame_db.idx.propagation_targets(
            start_mjd=mjd - 1,
            end_mjd=mjd + 1,
            obscode="obs1",
            datasets={dataset_1},
        )
    )

    assert len(targets) == 1
    target_healpixels = targets[0][1]
    assert target_healpixels == {1}


def test_get_frames_dataset_filter(frame_db):
    mjd = 10
    frames_1 = [
        make_sourceframe_with_observations(
            exposure_id="exp1",
            obscode="obs1",
            n_observations=1,
            exposure_mjd_mid=mjd,
            healpixel=1,
        )
    ]
    dataset_1 = "test_dataset_1"
    frame_db.add_dataset(dataset_1)
    frame_db.add_frames(dataset_1, frames_1)

    # Note: second dataset has same obscode as the first dataset,
    # which makes sure we test the filtering behavior.
    frames_2 = [
        make_sourceframe_with_observations(
            exposure_id="exp2",
            obscode="obs1",
            n_observations=1,
            exposure_mjd_mid=mjd,
            healpixel=1,
        )
    ]
    dataset_2 = "test_dataset_2"
    frame_db.add_dataset(dataset_2)
    frame_db.add_frames(dataset_2, frames_2)

    returned_frames = list(
        frame_db.idx.get_frames(
            obscode="obs1",
            mjd=mjd,
            healpixel=1,
            datasets={dataset_1},
        )
    )

    assert len(returned_frames) == 1
    assert returned_frames[0].dataset_id == dataset_1
