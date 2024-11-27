import sqlite3 as sql

import numpy as np
import pyarrow.compute as pc
import pytest
from adam_core.time import Timestamp
from astropy.time import Time

from precovery.frame_db import FrameDB, FrameIndex, WindowCenters
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
    stored_frames = frame_db.idx.all_frames()
    assert len(stored_frames) == 3

    # Find frame 1
    stored_frame_1 = stored_frames.select("exposure_id", "e1").select("healpixel", 1)
    assert len(stored_frame_1) == 1, "frame1 was not stored, or not retrieved"

    # Get associated observations
    stored_obs1 = frame_db.get_observations(stored_frame_1)
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
    obs = frame_db.get_observations(frame_stored[0])
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


def test_window_centers(frame_db):
    """
    Ensure we are getting the correct number and size of window centers
    """
    # Create one observation per week over 3 months
    observation_times = np.arange(23.3, 110, 7)

    # Delete all times between 50 and 70 to provide a gap to test
    observation_times = observation_times[
        np.logical_or(observation_times < 50, observation_times > 70)
    ]
    obs = [
        make_sourceobs(mjd=obs_time, exposure_duration=1)
        for obs_time in observation_times
    ]
    frames = list(bundle_into_frames(obs))
    frame_db.add_dataset("test_dataset")
    frame_db.add_frames("test_dataset", frames)

    window_centers_7 = frame_db.idx.window_centers(
        start_mjd=0, end_mjd=200, window_size_days=7
    )
    assert len(window_centers_7) == 10
    assert window_centers_7.time.mjd().to_pylist() == [
        24.5,
        31.5,
        38.5,
        45.5,
        73.5,
        80.5,
        87.5,
        94.5,
        101.5,
        108.5,
    ]

    window_centers_2 = frame_db.idx.window_centers(
        start_mjd=0, end_mjd=200, window_size_days=2
    )
    assert len(window_centers_2) == 10
    assert window_centers_2.time.mjd().to_pylist() == [
        23.0,
        31.0,
        37.0,
        45.0,
        73.0,
        79.0,
        87.0,
        93.0,
        101.0,
        107.0,
    ]

    window_centers_30 = frame_db.idx.window_centers(
        start_mjd=0, end_mjd=200, window_size_days=30
    )
    assert len(window_centers_30) == 4
    assert window_centers_30.time.mjd().to_pylist() == [15.0, 45.0, 75.0, 105.0]


def test_window_centers_dataset_filter(frame_db):
    mjd = 10
    frames_1 = [
        make_sourceframe_with_observations(obscode="obs1", n_observations=1, mjd=mjd)
    ]
    dataset_1 = "test_dataset_1"
    frame_db.add_dataset(dataset_1)
    frame_db.add_frames(dataset_1, frames_1)

    frames_2 = [
        make_sourceframe_with_observations(obscode="obs2", n_observations=1, mjd=mjd)
    ]
    dataset_2 = "test_dataset_2"
    frame_db.add_dataset(dataset_2)
    frame_db.add_frames(dataset_2, frames_2)

    # When using no dataset filter, we should return windows for all datasets
    centers = frame_db.idx.window_centers(
        start_mjd=mjd - 1, end_mjd=mjd + 1, window_size_days=1
    )

    assert len(centers) == 2
    assert set(pc.unique(centers.obscode).to_pylist()) == {"obs1", "obs2"}

    # When using a dataset filter, we should return windows for only
    # the specified datasets
    centers = frame_db.idx.window_centers(
        start_mjd=mjd - 1,
        end_mjd=mjd + 1,
        window_size_days=1,
        datasets={dataset_1},
    )

    assert len(centers) == 1
    assert set(pc.unique(centers.obscode).to_pylist()) == {"obs1"}


def test_propagation_targets_dataset_filter(frame_db):
    mjd = 10
    frames_1 = [
        make_sourceframe_with_observations(
            exposure_id="exp1",
            obscode="obs1",
            n_observations=1,
            mjd=mjd,
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
            mjd=mjd,
            healpixel=2,
        )
    ]
    dataset_2 = "test_dataset_2"
    frame_db.add_dataset(dataset_2)
    frame_db.add_frames(dataset_2, frames_2)
    window: WindowCenters = WindowCenters.from_kwargs(
        obscode=["obs1"],
        time=Timestamp.from_mjd([mjd], scale="utc"),
        window_size_days=[1],
    )
    targets = frame_db.idx.propagation_targets(
        window,
        datasets={dataset_1},
    )

    assert len(targets) == 1
    assert targets.healpixel.to_pylist() == [1]


def test_get_frames_dataset_filter(frame_db):
    mjd = 10
    frames_1 = [
        make_sourceframe_with_observations(
            exposure_id="exp1",
            obscode="obs1",
            n_observations=1,
            mjd=mjd,
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
            mjd=mjd,
            healpixel=1,
        )
    ]
    dataset_2 = "test_dataset_2"
    frame_db.add_dataset(dataset_2)
    frame_db.add_frames(dataset_2, frames_2)

    returned_frames = frame_db.idx.get_frames(
        obscode="obs1",
        mjd=mjd,
        healpixel=1,
        datasets={dataset_1},
    )

    assert len(returned_frames) == 1
    assert returned_frames.dataset_id.to_pylist() == [dataset_1]


def test_get_frames_by_id(frame_db):
    mjd = 10
    frames = [
        make_sourceframe_with_observations(
            exposure_id="exp1",
            obscode="obs1",
            n_observations=1,
            mjd=mjd,
            healpixel=1,
        ),
        make_sourceframe_with_observations(
            exposure_id="exp2",
            obscode="obs1",
            n_observations=1,
            mjd=mjd,
            healpixel=1,
        ),
    ]
    dataset = "test_dataset"
    frame_db.add_dataset(dataset)
    frame_db.add_frames(dataset, frames)

    frames_in_db = frame_db.idx.all_frames()
    frame_1_id = frames_in_db.id[0].as_py()

    returned_frames = frame_db.idx.get_frames_by_id([frame_1_id])

    assert len(returned_frames) == 1
    assert returned_frames.dataset_id.to_pylist() == [dataset]


def test_n_frames(frame_db):
    mjd = 10
    frames = [
        make_sourceframe_with_observations(
            exposure_id="exp1",
            obscode="obs1",
            n_observations=1,
            mjd=mjd,
            healpixel=1,
        ),
        make_sourceframe_with_observations(
            exposure_id="exp2",
            obscode="obs1",
            n_observations=1,
            mjd=mjd,
            healpixel=1,
        ),
    ]
    dataset = "test_dataset"
    frame_db.add_dataset(dataset)
    frame_db.add_frames(dataset, frames)

    assert frame_db.idx.n_frames() == 2

    # Add a frame to a different dataset
    frames = [
        make_sourceframe_with_observations(
            exposure_id="exp3",
            obscode="obs1",
            n_observations=1,
            mjd=mjd,
            healpixel=1,
        ),
    ]
    dataset = "test_dataset_2"
    frame_db.add_dataset(dataset)
    frame_db.add_frames(dataset, frames)

    assert frame_db.idx.n_frames() == 3


def test_n_bytes(frame_db):
    mjd = 10
    frames = [
        make_sourceframe_with_observations(
            exposure_id="exp1",
            obscode="obs1",
            n_observations=1,
            mjd=mjd,
            healpixel=1,
        ),
        make_sourceframe_with_observations(
            exposure_id="exp2",
            obscode="obs1",
            n_observations=1,
            mjd=mjd,
            healpixel=1,
        ),
    ]
    dataset = "test_dataset"
    frame_db.add_dataset(dataset)
    frame_db.add_frames(dataset, frames)

    assert frame_db.idx.n_bytes() > 0


def test_n_bytes_empty(frame_db):
    assert frame_db.idx.n_bytes() == 0


def test_n_unique_frames(frame_db):
    mjd = 10
    frames = [
        make_sourceframe_with_observations(
            exposure_id="exp1",
            obscode="obs1",
            n_observations=1,
            mjd=mjd,
            healpixel=1,
        ),
        make_sourceframe_with_observations(
            exposure_id="exp2",
            obscode="obs1",
            n_observations=1,
            mjd=mjd,
            healpixel=1,
        ),
    ]
    dataset = "test_dataset"
    frame_db.add_dataset(dataset)
    frame_db.add_frames(dataset, frames)

    assert (
        frame_db.idx.n_unique_frames() == 1
    ), "Frames with identical obscode, time, healpix should only count as one"

    # Add a frame to a different dataset
    frames = [
        make_sourceframe_with_observations(
            exposure_id="exp3",
            obscode="obs1",
            n_observations=1,
            mjd=mjd,
            healpixel=2,
        ),
    ]
    dataset = "test_dataset_2"
    frame_db.add_dataset(dataset)
    frame_db.add_frames(dataset, frames)

    assert frame_db.idx.n_unique_frames() == 2


def test_mjd_bounds(frame_db):
    mjd = 10
    frames = [
        make_sourceframe_with_observations(
            exposure_id="exp1",
            obscode="obs1",
            n_observations=1,
            mjd=mjd,
            healpixel=1,
        ),
        make_sourceframe_with_observations(
            exposure_id="exp2",
            obscode="obs1",
            n_observations=1,
            mjd=mjd,
            healpixel=1,
        ),
    ]
    dataset = "test_dataset"
    frame_db.add_dataset(dataset)
    frame_db.add_frames(dataset, frames)

    assert frame_db.idx.mjd_bounds() == (10, 10)

    # Add a frame to a different dataset
    frames = [
        make_sourceframe_with_observations(
            exposure_id="exp3",
            obscode="obs1",
            n_observations=1,
            mjd=mjd + 100,
            healpixel=2,
        ),
    ]
    dataset = "test_dataset_2"
    frame_db.add_dataset(dataset)
    frame_db.add_frames(dataset, frames)

    assert frame_db.idx.mjd_bounds() == (10, 110)


def test_mjd_bounds_dataset_filter(frame_db):
    mjd = 10
    frames = [
        make_sourceframe_with_observations(
            exposure_id="exp1",
            obscode="obs1",
            n_observations=1,
            mjd=mjd,
            healpixel=1,
        ),
        make_sourceframe_with_observations(
            exposure_id="exp2",
            obscode="obs1",
            n_observations=1,
            mjd=mjd,
            healpixel=1,
        ),
    ]
    dataset = "test_dataset"
    frame_db.add_dataset(dataset)
    frame_db.add_frames(dataset, frames)

    # Add a frame to a different dataset
    frames = [
        make_sourceframe_with_observations(
            exposure_id="exp3",
            obscode="obs1",
            n_observations=1,
            mjd=mjd + 100,
            healpixel=2,
        ),
    ]
    other_dataset = "test_dataset_2"
    frame_db.add_dataset(other_dataset)
    frame_db.add_frames(other_dataset, frames)

    assert frame_db.idx.mjd_bounds(datasets={dataset}) == (10, 10)
    assert frame_db.idx.mjd_bounds(datasets={other_dataset}) == (110, 110)
