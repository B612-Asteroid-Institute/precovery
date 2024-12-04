import numpy as np
import numpy.testing as npt
import pyarrow as pa
from adam_assist import ASSISTPropagator
from adam_core.coordinates import CartesianCoordinates, Origin, SphericalCoordinates
from adam_core.orbits import Orbits
from adam_core.orbits.ephemeris import Ephemeris
from adam_core.time import Timestamp

from precovery.frame_db import GenericFrame, HealpixFrame
from precovery.healpix_geom import radec_to_healpixel
from precovery.observation import ObservationsTable
from precovery.precovery_db import (
    CANDIDATE_NSIDE,
    PrecoveryCandidates,
    PrecoveryDatabase,
    candidates_from_ephem,
    find_healpixel_matches,
    find_observation_matches,
)
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


def test_find_healpixel_matches():
    ra = np.linspace(0, 360, 6)
    dec = np.linspace(-90, 90, 6)
    healpixels = radec_to_healpixel(ra, dec, nside=32)
    times = Timestamp.from_mjd(
        [50000.0, 50001.0, 50002.0, 50003.0, 50004.0, 50005.0], scale="utc"
    )

    propagation_targets = GenericFrame.from_kwargs(
        # let's only match half the healpixels
        healpixel=[healpixels[0], 1, healpixels[2], 1, healpixels[4], 1],
        time=times,
    )
    ephems = Ephemeris.from_kwargs(
        orbit_id=[f"test_orbit_{i}" for i in range(6)],
        coordinates=SphericalCoordinates.from_kwargs(
            lat=dec,
            lon=ra,
            vlon=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            vlat=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            # Ephem times should tolerate some small differences
            time=times.add_micros(100),
            origin=Origin.from_kwargs(code=pa.repeat("EARTH", 6)),
        ),
    )
    matches = find_healpixel_matches(propagation_targets, ephems, 32)
    assert len(matches) == 3
    # Check that matches match the first, third and fifth times
    assert matches.time.mjd().to_pylist() == [50000.0, 50002.0, 50004.0]
    assert matches.healpixel.to_pylist() == [
        healpixels[0],
        healpixels[2],
        healpixels[4],
    ]


def test_find_matches_in_frame(tmp_path, mocker):
    get_obs_mock = mocker.patch("precovery.frame_db.FrameDB.get_observations")
    get_obs_mock.return_value = ObservationsTable.from_kwargs(
        id=["obs1", "obs2", "obs3"],
        ra=[1, 1.0 + 2 / 3600.0, 40],
        dec=[2, 2, 41],
        ra_sigma=[3.0, 3.0, 3.0],
        dec_sigma=[4.0, 4.0, 4.0],
        mag=[5.0, 5.0, 5.0],
        mag_sigma=[6.0, 6.0, 6.0],
        time=Timestamp.from_mjd(pa.repeat(50000.0, 3), scale="utc"),
    )

    frame_ephem = Ephemeris.from_kwargs(
        orbit_id=["test_orbit"],
        object_id=["test_object"],
        coordinates=SphericalCoordinates.from_kwargs(
            lat=[2.0],
            lon=[1.0 + 2 / 3600.0],
            vlon=[0.1],
            vlat=[0.1],
            origin=Origin.from_kwargs(code=["EARTH"]),
            time=Timestamp.from_mjd([50000.00000001], scale="utc"),
        ),
    )

    frame = HealpixFrame.from_kwargs(
        healpixel=[1],
        dataset_id=["test_dataset"],
        obscode=["testobs"],
        exposure_id=["exp0"],
        exposure_mjd_start=[49999.9],
        exposure_mjd_mid=[50000.0],
        exposure_duration=[(50000 - 49999.9) / 86400],
        filter=["filter"],
        data_uri=[""],
        data_offset=[0],
        data_length=[0],
    )

    db = PrecoveryDatabase.create(str(tmp_path), nside=32)

    # we don't use during the test
    orbit = Orbits.from_kwargs(
        orbit_id=["test_orbit"],
        coordinates=CartesianCoordinates.from_kwargs(
            x=[1.0],
            y=[0.0],
            z=[0.0],
            vx=[0.0],
            vy=[1.0],
            vz=[0.0],
            time=Timestamp.from_mjd([50000.0], scale="utc"),
        ),
    )

    candidates = db.find_matches_in_frame(
        frame, orbit, frame_ephem, 1 / 3600.0, ASSISTPropagator()
    )
    assert len(candidates) == 1


def test_find_matches_in_frame_per_obs_timestamps(tmp_path, mocker):
    get_obs_mock = mocker.patch("precovery.frame_db.FrameDB.get_observations")
    get_obs_mock.return_value = ObservationsTable.from_kwargs(
        id=["obs1", "obs2", "obs3"],
        ra=[1, 1.0 + 2 / 3600.0, 40],
        dec=[2, 2, 41],
        ra_sigma=[3.0, 3.0, 3.0],
        dec_sigma=[4.0, 4.0, 4.0],
        mag=[5.0, 5.0, 5.0],
        mag_sigma=[6.0, 6.0, 6.0],
        time=Timestamp.from_mjd([50000.0, 50000.01, 50000.03], scale="utc"),
    )

    gen_ephem = mocker.patch(
        "precovery.precovery_db.generate_ephem_for_per_obs_timestamps"
    )
    gen_ephem.return_value = Ephemeris.from_kwargs(
        orbit_id=pa.repeat("test_orbit", 3),
        object_id=pa.repeat("test_object", 3),
        coordinates=SphericalCoordinates.from_kwargs(
            lat=[2.0, 2.0, 2.0],
            # only middle one matches our ra above
            lon=[1.0 + 2 / 3600.0, 1.0 + 2 / 3600.0, 1.0 + 3 / 3600.0],
            vlon=[0.1, 0.1, 0.1],
            vlat=[0.1, 0.1, 0.1],
            origin=Origin.from_kwargs(code=pa.repeat("EARTH", 3)),
            time=Timestamp.from_mjd([50000.0, 50000.01, 50000.03], scale="utc"),
        ),
    )

    frame_ephem = Ephemeris.from_kwargs(
        orbit_id=["test_orbit"],
        object_id=["test_object"],
        coordinates=SphericalCoordinates.from_kwargs(
            lat=[2.0],
            lon=[1.0 + 2 / 3600.0],
            vlon=[0.1],
            vlat=[0.1],
            origin=Origin.from_kwargs(code=["EARTH"]),
            time=Timestamp.from_mjd([50000.02], scale="utc"),
        ),
    )

    frame = HealpixFrame.from_kwargs(
        healpixel=[1],
        dataset_id=["test_dataset"],
        obscode=["testobs"],
        exposure_id=["exp0"],
        exposure_mjd_start=[50000.0],
        exposure_mjd_mid=[50000.02],
        exposure_duration=[(50000.04 - 50000.0) / 86400],
        filter=["filter"],
        data_uri=[""],
        data_offset=[0],
        data_length=[0],
    )

    db = PrecoveryDatabase.create(str(tmp_path), nside=32)

    # we don't use during the test
    orbit = Orbits.from_kwargs(
        orbit_id=["test_orbit"],
        coordinates=CartesianCoordinates.from_kwargs(
            x=[1.0],
            y=[0.0],
            z=[0.0],
            vx=[0.0],
            vy=[1.0],
            vz=[0.0],
            time=Timestamp.from_mjd([50000.0], scale="utc"),
        ),
    )

    candidates = db.find_matches_in_frame(
        frame, orbit, frame_ephem, 1 / 3600.0, ASSISTPropagator()
    )

    assert len(candidates) == 1
    assert gen_ephem.call_count == 1


def test_find_observation_matches():
    obs = ObservationsTable.from_kwargs(
        id=["obs1", "obs2", "obs3"],
        ra=[1, 1.0 + 2 / 3600.0, 1.0 + 4 / 3600.0],
        dec=[2, 2, 2],
        ra_sigma=[3.0, 3.0, 3.0],
        dec_sigma=[4.0, 4.0, 4.0],
        mag=[5.0, 5.0, 5.0],
        mag_sigma=[6.0, 6.0, 6.0],
        time=Timestamp.from_mjd([50000.0, 50000.01, 50000.03], scale="utc"),
    )

    ephem = Ephemeris.from_kwargs(
        orbit_id=pa.repeat("test_orbit", 3),
        object_id=pa.repeat("test_object", 3),
        coordinates=SphericalCoordinates.from_kwargs(
            lon=[1.0 + 3 / 3600.0, 1.0 + 3 / 3600.0, 1.0 + 8 / 3600.0],
            lat=[2.0, 2.0, 2.0],
            vlon=[0.1, 0.1, 0.1],
            vlat=[0.1, 0.1, 0.1],
            origin=Origin.from_kwargs(code=pa.repeat("EARTH", 3)),
            time=Timestamp.from_mjd([50000.0, 50000.01, 50000.03], scale="utc"),
        ),
    )
    matching_obs, _ = find_observation_matches(obs, ephem, 1 / 3600.0)
    assert len(matching_obs) == 1

    matching_obs, _ = find_observation_matches(obs, ephem, 3 / 3600.0)
    assert len(matching_obs) == 2

    matching_obs, _ = find_observation_matches(obs, ephem, 10 / 3600.0)
    assert len(matching_obs) == 3


def test_candidates_from_ephem():
    obs = ObservationsTable.from_kwargs(
        id=["obs1", "obs2", "obs3"],
        ra=[1, 1.0 + 2 / 3600.0, 1.0 + 4 / 3600.0],
        dec=[2, 2, 2],
        ra_sigma=[3.0, 3.0, 3.0],
        dec_sigma=[4.0, 4.0, 4.0],
        mag=[5.0, 5.0, 5.0],
        mag_sigma=[6.0, 6.0, 6.0],
        time=Timestamp.from_mjd([50000.0, 50000.01, 50000.03], scale="utc"),
    )

    ephem = Ephemeris.from_kwargs(
        orbit_id=pa.repeat("test_orbit", 3),
        object_id=pa.repeat("test_object", 3),
        coordinates=SphericalCoordinates.from_kwargs(
            lon=[1.0 + 3 / 3600.0, 1.0 + 3 / 3600.0, 1.0 + 8 / 3600.0],
            lat=[2.0, 2.0, 2.0],
            vlon=[0.1, 0.1, 0.1],
            vlat=[0.1, 0.1, 0.1],
            origin=Origin.from_kwargs(code=pa.repeat("EARTH", 3)),
            time=Timestamp.from_mjd([50000.0, 50000.01, 50000.03], scale="utc"),
        ),
    )

    frame = HealpixFrame.from_kwargs(
        healpixel=[1],
        dataset_id=["test_dataset"],
        obscode=["testobs"],
        exposure_id=["exp0"],
        exposure_mjd_start=[50000.0],
        exposure_mjd_mid=[50000.02],
        exposure_duration=[(50000.04 - 50000.0) / 86400],
        filter=["filter"],
        data_uri=[""],
        data_offset=[0],
        data_length=[0],
    )

    candidates = candidates_from_ephem(obs, ephem, frame)

    assert len(candidates) == 3
    assert candidates.exposure_id.to_pylist() == ["exp0", "exp0", "exp0"]
    assert candidates.distance_arcsec.to_pylist() == [2.9981724810565917, 0.9993908270183969, 3.99756330807566]