from precovery.precovery_db import PrecoveryDatabase
from precovery.sourcecatalog import bundle_into_frames

from .testutils import make_sourceobs


def test_find_observations_in_regions(tmp_path):
    db = PrecoveryDatabase.create(str(tmp_path), nside=32)

    db.frames.add_dataset("test_dataset")
    # Make 3 observations. Two are within 2arcsec of each other, the
    # other is far away.
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
            id=b"obs2",
            ra=1.0 + 2 / 3600.0,
            dec=2,
            obscode="testobs",
        ),
        make_sourceobs(
            exposure_id=b"exp0",
            id=b"obs3",
            ra=40,
            dec=41,
            obscode="testobs",
        ),
    ]
    frames = list(bundle_into_frames(observations))
    assert len(frames) == 2
    dataset_id = "test_dataset"
    db.frames.add_dataset(dataset_id)
    db.frames.add_frames(dataset_id, frames)

    # Should get two results within 10 arcsec
    results = list(db.find_observations_in_radius(1, 2, 10 / 3600.0, "testobs"))
    assert len(results) == 2

    # Should get one result within 1 arcsec
    results = db.find_observations_in_radius(1, 2, 1 / 3600.0, "testobs")
    assert len(list(results)) == 1
