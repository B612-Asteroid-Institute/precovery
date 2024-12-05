from adam_assist import ASSISTPropagator

from precovery.main import precover
from precovery.precovery_db import PrecoveryDatabase
from precovery.sourcecatalog import bundle_into_frames

from .testutils import make_sourceobs, make_sourceobs_of_orbit


def test_precover(precovery_db, sample_orbits):

    # Make dataset which contains something we're looking for.
    orbit = sample_orbits[0]
    timestamps = [50000.0, 50001.0, 50002.0]

    object_observations = [
        make_sourceobs_of_orbit(orbit, "I41", mjd) for mjd in timestamps
    ]

    # Include some stuff we're not looking for.
    extra_observations = [
        make_sourceobs(obscode="I41", mjd=mjd, exposure_duration=30)
        for mjd in timestamps
    ]

    frames = list(bundle_into_frames(object_observations + extra_observations))

    ds_id = "test_dataset_1"

    precovery_db.frames.add_dataset(ds_id)
    precovery_db.frames.add_frames(ds_id, frames)

    # Do the search. We should find the three observations we inserted.
    matches, misses = precovery_db.precover(orbit, propagator_class=ASSISTPropagator)
    assert len(matches) == 3
    assert len(misses) == 0

    have_ids = set(matches.observation_id.to_pylist())
    want_ids = set(o.id.decode("utf8") for o in object_observations)
    assert have_ids == want_ids


def test_precover_dataset_filter(precovery_db, sample_orbits):
    # Make two datasets which contain something we're looking for.

    orbit = sample_orbits[0]
    timestamps = [50000.0, 50001.0, 50002.0]

    ds1_observations = [
        make_sourceobs_of_orbit(orbit, "I41", mjd) for mjd in timestamps
    ]
    ds2_observations = [
        make_sourceobs_of_orbit(orbit, "I41", mjd) for mjd in timestamps
    ]

    ds1_id = "test_dataset_1"
    precovery_db.frames.add_dataset(ds1_id)
    precovery_db.frames.add_frames(ds1_id, bundle_into_frames(ds1_observations))
    ds2_id = "test_dataset_2"
    precovery_db.frames.add_dataset(ds2_id)
    precovery_db.frames.add_frames(ds2_id, bundle_into_frames(ds2_observations))

    # Do the search with no dataset filters. We should find all six
    # observations we inserted.
    matches, _ = precovery_db.precover(orbit, propagator_class=ASSISTPropagator)
    assert len(matches) == 6

    have_ids = set(matches.observation_id.to_pylist())
    want_ids = set(o.id.decode("utf8") for o in (ds1_observations + ds2_observations))
    assert have_ids == want_ids

    # Now repeat the search, but filter to just one dataset. We should
    # only find that dataset's observations.
    matches, _ = list(
        precovery_db.precover(
            orbit, datasets={ds1_id}, propagator_class=ASSISTPropagator
        )
    )
    assert len(matches) == 3

    have_ids = set(matches.observation_id.to_pylist())
    want_ids = set(o.id.decode("utf8") for o in ds1_observations)
    assert have_ids == want_ids


def test_multiple_workers(precovery_db_with_data, sample_orbits):
    """
    Make sure we have no issues when running precovery with multiple workers.
    """
    # Do the search. We should find the three observations we inserted.
    matches, misses = precover(
        sample_orbits[:2],
        precovery_db_with_data.directory,
        max_processes=4,
        propagator_class=ASSISTPropagator,
    )
    assert len(matches) == 896
    assert len(misses) == 8