from precovery.db import PrecoveryDatabase
from precovery.observation import Observation


def test_precovery_database_add_observations():
    # Test that inserting observations maintains the proper sorted order and
    # updates the minimum_epoch and maximum_epoch attributes correctly.
    db = PrecoveryDatabase()

    obscode = "568"

    # Add a single batch.
    batch1_epoch = 10
    batch1 = [
        Observation(0, 0, batch1_epoch, 1),
        Observation(0, 0, batch1_epoch, 2),
    ]
    db.add_observations(batch1_epoch, obscode, batch1)

    # It should be the only thing in the DB.
    assert len(db.observations) == 1
    assert len(db.observations[0]) == 2

    # Bounds should be correctly set.
    assert db.minimum_epoch == batch1_epoch
    assert db.maximum_epoch == batch1_epoch

    # Add another batch before the first one.
    batch2_epoch = 8
    batch2 = [
        Observation(0, 0, batch2_epoch, 3),
        Observation(0, 0, batch2_epoch, 4),
        Observation(0, 0, batch2_epoch, 5),
    ]
    db.add_observations(batch2_epoch, obscode, batch2)

    # Check DB counts, values, and bounds.
    assert len(db.observations) == 2
    assert len(db.observations[0]) == 3
    assert len(db.observations[1]) == 2
    assert db.minimum_epoch == batch2_epoch
    assert db.maximum_epoch == batch1_epoch

    # Add a third after the last one.
    batch3_epoch = 12
    batch3 = [
        Observation(0, 0, batch3_epoch, 6),
    ]
    db.add_observations(batch3_epoch, obscode, batch3)

    # Check DB counts, values, and bounds.
    assert len(db.observations) == 3
    assert len(db.observations[0]) == 3
    assert len(db.observations[1]) == 2
    assert len(db.observations[2]) == 1
    assert db.minimum_epoch == batch2_epoch
    assert db.maximum_epoch == batch3_epoch

    # Finally, add one that should squeeze between the first and second
    # observation batches.q
    batch4_epoch = 9
    batch4 = [
        Observation(0, 0, batch4_epoch, 7),
        Observation(0, 0, batch4_epoch, 8),
        Observation(0, 0, batch4_epoch, 9),
        Observation(0, 0, batch4_epoch, 10),
    ]
    db.add_observations(batch4_epoch, obscode, batch4)

    # Check DB counts, values, and bounds.
    assert len(db.observations) == 4
    assert len(db.observations[0]) == 3
    assert len(db.observations[1]) == 4
    assert len(db.observations[2]) == 2
    assert len(db.observations[3]) == 1
    assert db.minimum_epoch == batch2_epoch
    assert db.maximum_epoch == batch3_epoch
