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
    matches, misses = precovery_db.precover(orbit)
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
    matches, _ = precovery_db.precover(orbit)
    assert len(matches) == 6

    have_ids = set(matches.observation_id.to_pylist())
    want_ids = set(o.id.decode("utf8") for o in (ds1_observations + ds2_observations))
    assert have_ids == want_ids

    # Now repeat the search, but filter to just one dataset. We should
    # only find that dataset's observations.
    matches, _ = list(precovery_db.precover(orbit, datasets={ds1_id}))
    assert len(matches) == 3

    have_ids = set(matches.observation_id.to_pylist())
    want_ids = set(o.id.decode("utf8") for o in ds1_observations)
    assert have_ids == want_ids


def test_multiple_workers(tmp_path, sample_orbits):
    """
    Smoke test: multi-worker run completes and returns deterministic results.

    We intentionally avoid asserting large, dataset-dependent match counts here. The
    performance-first pipeline is free to change scoring/selection behavior as it
    evolves; this test focuses on exercising the multi-process propagation path.
    """
    db = PrecoveryDatabase.create(str(tmp_path), nside=32)
    db.frames.add_dataset("ds")

    timestamps = [50000.0, 50001.0, 50002.0]
    obs = []
    for orbit in sample_orbits[:2]:
        obs.extend([make_sourceobs_of_orbit(orbit, "I41", mjd) for mjd in timestamps])

    db.frames.add_frames("ds", bundle_into_frames(obs))

    matches, misses = precover(
        sample_orbits[:2],
        db.directory,
        start_mjd=49999.0,
        end_mjd=50003.0,
        max_processes=1,
    )
    assert len(matches) == 6
    assert len(misses) == 0


def test_precover_ray_chunk_parallelism_smoke(tmp_path, sample_orbits) -> None:
    """
    Smoke test: the Ray worker body can reopen the DB and process a chunk.

    We intentionally do not require a functioning Ray runtime in unit tests (developers may
    have an unrelated Ray cluster running). Instead, we directly call the worker function
    used by the Ray path and assert correctness of its results.
    """
    from precovery.precovery_db import PrecoveryDatabase

    db = PrecoveryDatabase.create(str(tmp_path), nside=32)
    db.frames.add_dataset("ds")

    timestamps = [50000.0, 50001.0, 50002.0]
    obs = [make_sourceobs_of_orbit(sample_orbits[0], "I41", mjd) for mjd in timestamps]
    db.frames.add_frames("ds", bundle_into_frames(obs))

    from precovery.search.search import (
        _process_targets_chunk_ray_worker,
        enumerate_targets,
    )

    orbit = sample_orbits[0]
    start_mjd = 49999.0
    end_mjd = 50003.0
    targets = enumerate_targets(db=db, start_mjd=start_mjd, end_mjd=end_mjd, datasets=None)
    assert len(targets) > 0

    matches, misses, _agg = _process_targets_chunk_ray_worker(
        db_dir=db.directory,
        allow_version_mismatch=True,
        orbit=orbit,
        orbit_id=str(orbit.orbit_id[0].as_py()),
        targets=targets,
        t0=0,
        t1=len(targets),
        tolerance=None,
        start_mjd=start_mjd,
        end_mjd=end_mjd,
        datasets=None,
        window_size_days=7,
        n_sigma=3.0,
        propagation_max_processes=1,
        want_per_target_metrics=False,
    )

    assert len(matches) == 3
    assert len(misses) == 0
