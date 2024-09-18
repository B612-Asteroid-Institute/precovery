import dataclasses
import logging
import os
from typing import Iterator, List, Optional, Tuple

import numpy as np
import pyarrow as pa
import quivr as qv
from adam_core.coordinates import CoordinateCovariances
from adam_core.dynamics.ephemeris import generate_ephemeris_2body
from adam_core.dynamics.propagation import propagate_2body
from adam_core.observers import Observers
from adam_core.orbits import Orbits as AdamOrbits
from adam_core.orbits.ephemeris import Ephemeris as AdamEphemeris
from adam_core.propagator import Propagator
from adam_core.time import Timestamp

from ._version import __version__
from .config import Config, DefaultConfig
from .frame_db import FrameDB, FrameIndex, HealpixFrame
from .healpix_geom import radec_to_healpixel
from .observation import Observation, ObservationArray
from .precovery_db import FrameCandidatesQv, PrecoveryCandidatesQv
from .spherical_geom import haversine_distance_deg

DEGREE = 1.0
ARCMIN = DEGREE / 60
ARCSEC = ARCMIN / 60

CANDIDATE_K = 15
CANDIDATE_NSIDE = 2**CANDIDATE_K

logging.basicConfig()
logger = logging.getLogger("precovery")


# TODO this may nbeed  to be ported to the PrecoveryCandidatesQV class
def candidates_from_ephem(obs: Observation, frame: HealpixFrame, ephem: AdamEphemeris):
    # Calculate the HEALpixel ID for the predicted ephemeris of
    # the orbit with a high nside value (k=15, nside=2**15) The
    # indexed observations are indexed to a much lower nside but
    # we may decide in the future to re-index the database using
    # different values for that parameter. As long as we return a
    # Healpix ID generated with nside greater than the indexed
    # database then we can always down-sample the ID to a lower
    # nside value
    healpix_id = int(
        radec_to_healpixel(
            ephem.coordinates.lon[0].as_py(),
            ephem.coordinates.lat[0].as_py(),
            nside=CANDIDATE_NSIDE,
        )
    )

    return PrecoveryCandidatesQv.from_kwargs(
        time=Timestamp.from_mjd([obs.mjd], scale="utc"),
        ra_deg=[obs.ra],
        dec_deg=[obs.dec],
        ra_sigma_arcsec=[obs.ra_sigma / ARCSEC],
        dec_sigma_arcsec=[obs.dec_sigma / ARCSEC],
        mag=[obs.mag],
        mag_sigma=[obs.mag_sigma],
        exposure_time_start=Timestamp.from_mjd([frame.exposure_mjd_start], scale="utc"),
        exposure_time_mid=Timestamp.from_mjd([frame.exposure_mjd_mid], scale="utc"),
        filter=[frame.filter],
        obscode=[frame.obscode],
        exposure_id=[frame.exposure_id],
        exposure_duration=[frame.exposure_duration],
        observation_id=[obs.id.decode()],
        healpix_id=[healpix_id],
        pred_ra_deg=ephem.coordinates.lon,
        pred_dec_deg=ephem.coordinates.lat,
        pred_vra_degpday=ephem.coordinates.vlon,
        pred_vdec_degpday=ephem.coordinates.vlat,
        delta_ra_arcsec=[(obs.ra - ephem.coordinates.lon[0].as_py()) / ARCSEC],
        delta_dec_arcsec=[(obs.dec - ephem.coordinates.lat[0].as_py()) / ARCSEC],
        distance_arcsec=[
            haversine_distance_deg(
                obs.ra,
                ephem.coordinates.lon[0].as_py(),
                obs.dec,
                ephem.coordinates.lat[0].as_py(),
            )
            / ARCSEC
        ],
        dataset_id=[frame.dataset_id],
        orbit_id=ephem.orbit_id,
    )


def to_dict(self):
    return dataclasses.asdict(self)


def frame_candidates_from_frame(frame: HealpixFrame, ephem: AdamEphemeris):
    # Calculate the HEALpixel ID for the predicted ephemeris of
    # the orbit with a high nside value (k=15, nside=2**15) The
    # indexed observations are indexed to a much lower nside but
    # we may decide in the future to re-index the database using
    # different values for that parameter. As long as we return a
    # Healpix ID generated with nside greater than the indexed
    # database then we can always down-sample the ID to a lower
    # nside value
    healpix_id = int(
        radec_to_healpixel(
            ephem.coordinates.lon[0].as_py(),
            ephem.coordinates.lat[0].as_py(),
            nside=CANDIDATE_NSIDE,
        )
    )
    return FrameCandidatesQv.from_kwargs(
        exposure_time_start=Timestamp.from_mjd([frame.exposure_mjd_start], scale="utc"),
        exposure_time_mid=Timestamp.from_mjd([frame.exposure_mjd_mid], scale="utc"),
        filter=[frame.filter],
        obscode=[frame.obscode],
        exposure_id=[frame.exposure_id],
        exposure_duration=[frame.exposure_duration],
        dataset_id=[frame.dataset_id],
        healpix_id=[healpix_id],
        pred_ra_deg=ephem.coordinates.lon,
        pred_dec_deg=ephem.coordinates.lat,
        pred_vra_degpday=ephem.coordinates.vlon,
        pred_vdec_degpday=ephem.coordinates.vlat,
        orbit_id=ephem.orbit_id,
    )


class PrecoveryDatabaseQv:
    def __init__(self, frames: FrameDB, config: Config = DefaultConfig):
        self.frames = frames
        self._exposures_by_obscode: dict = {}
        self.config = config

    @classmethod
    def from_dir(
        cls,
        directory: str,
        create: bool = False,
        mode: str = "r",
        allow_version_mismatch: bool = False,
    ):
        if not os.path.exists(directory):
            if create:
                return cls.create(directory)

        try:
            config = Config.from_json(os.path.join(directory, "config.json"))
        except FileNotFoundError:
            if not create:
                raise Exception("No config file found and create=False")
            config = DefaultConfig
            config.to_json(os.path.join(directory, "config.json"))

        if config.build_version != __version__:
            if not allow_version_mismatch:
                raise Exception(
                    f"Version mismatch: \nRunning version: {__version__}\nDatabase"
                    f" version: {config.build_version}\nUse allow_version_mismatch=True"
                    " to ignore this error."
                )
            else:
                logger.warning(
                    f"Version mismatch: \nRunning version: {__version__}\nDatabase"
                    f" version: {config.build_version}\nallow_version_mismatch=True, so"
                    " continuing."
                )

        frame_idx_db = "sqlite:///" + os.path.join(directory, "index.db")
        frame_idx = FrameIndex(frame_idx_db, mode=mode)

        data_path = os.path.join(directory, "data")
        frame_db = FrameDB(
            frame_idx, data_path, config.data_file_max_size, config.nside, mode=mode
        )
        return cls(frame_db, config)

    @classmethod
    def create(
        cls,
        directory: str,
        nside: int = DefaultConfig.nside,
        data_file_max_size: int = DefaultConfig.data_file_max_size,
    ):
        """
        Create a new database on disk in the given directory.
        """
        os.makedirs(directory, exist_ok=True)

        frame_idx_db = "sqlite:///" + os.path.join(directory, "index.db")
        frame_idx = FrameIndex(frame_idx_db, mode="w")

        config = Config(nside=nside, data_file_max_size=data_file_max_size)
        config.to_json(os.path.join(directory, "config.json"))

        data_path = os.path.join(directory, "data")
        os.makedirs(data_path, exist_ok=True)

        frame_db = FrameDB(frame_idx, data_path, data_file_max_size, nside)

        return cls(frame_db, config)

    def precover(
        self,
        orbit: AdamOrbits,
        tolerance: float = 30 * ARCSEC,
        start_mjd: Optional[float] = None,
        end_mjd: Optional[float] = None,
        window_size: int = 7,
        datasets: Optional[set[str]] = None,
        propagator: Optional[
            Propagator
        ] = None,  # should we initialize an assist propagator?
    ) -> Tuple[PrecoveryCandidatesQv, FrameCandidatesQv]:
        """
        Find observations which match orbit in the database. Observations are
        searched in descending order by mjd.

        orbit: The orbit to match.

        start_mjd: Only consider observations from after this epoch
        (inclusive). If None, find all.

        end_mjd: Only consider observations from before this epoch (inclusive).
        If None, find all.

        datasets: Only consider observations from the indicated datasets.

        Returns
        -------
        Tuple[List[PrecoveryCandidate], List[FrameCandidate]]
            Precovery candidate observations and frame candidates.
        """
        # basically:
        """
        find all windows between start and end of given size
        for each window:
            propagate to window center
            for each unique epoch,obscode in window:
                propagate to epoch
                find frames which match healpix of propagation
                for each matching frame
                    find matching observations
                    for each matching observation
                        yield match
        """
        orbit_id = orbit.orbit_id[0].as_py()

        if datasets is not None:
            self._warn_for_missing_datasets(datasets)

        if start_mjd is None or end_mjd is None:
            first, last = self.frames.idx.mjd_bounds(datasets=datasets)
            if start_mjd is None:
                start_mjd = first
            if end_mjd is None:
                end_mjd = last

        logger.info(
            "precovering orbit %s from %.5f to %.5f, window=%d, datasets=%s",
            orbit_id,
            start_mjd,
            end_mjd,
            window_size,
            datasets or "all",
        )

        windows = self.frames.idx.window_centers_grouped(
            start_mjd, end_mjd, window_size, datasets=datasets
        )
        # group windows by obscodes so that many windows can be searched at once
        candidates = []
        frame_candidates = []
        for obscode, mjds in windows:
            logger.info("searching windows for obscode %s", obscode)
            # create our Timestamp ghere
            times = Timestamp.from_mjd(mjds, scale="utc")

            candidates_obscode, frame_candidates_obscode = self._check_windows(
                times,
                obscode,
                orbit,
                tolerance,
                propagator,
                start_mjd=start_mjd,
                end_mjd=end_mjd,
                window_size=window_size,
                datasets=datasets,
            )
            candidates.append(candidates_obscode)
            frame_candidates.append(frame_candidates_obscode)

        # convert these to our new output formats
        return qv.concatenate(candidates), qv.concatenate(frame_candidates)

    def _check_windows(
        self,
        window_midpoints: Timestamp,
        obscode: str,
        orbit: AdamOrbits,
        tolerance: float,
        propagator: Propagator,
        start_mjd: Optional[float] = None,
        end_mjd: Optional[float] = None,
        window_size: int = 7,
        datasets: Optional[set[str]] = None,
    ) -> Tuple[PrecoveryCandidatesQv, FrameCandidatesQv]:
        """
        Find all observations that match orbit within a list of windows
        """
        # Propagate the orbit with n-body to every window center
        orbit_propagated = propagator.propagate_orbits(orbit, window_midpoints)
        cov = CoordinateCovariances.from_kwargs(
            values=pa.repeat(None, len(orbit_propagated)),
        )
        orbit_propagated = orbit_propagated.set_column("coordinates.covariance", cov)

        # Using the propagated orbits, check each window. Propagate the orbit from the center of
        # window using 2-body to find any HealpixFrames where a detection could have occured
        precovery_candidates = [PrecoveryCandidatesQv.empty()]
        frame_candidates = [FrameCandidatesQv.empty()]
        for orbit_window in orbit_propagated:
            # fmt: off
            window_midpoint = orbit_window.coordinates.time[0].rescale("utc").mjd().to_numpy(False)[0]
            # fmt: on

            start_mjd_window = window_midpoint - (window_size / 2)
            end_mjd_window = window_midpoint + (window_size / 2)

            # Check if start_mjd_window is not earlier than start_mjd (if defined)
            # If start_mjd_window is earlier, then set start_mjd_window to start_mjd
            if (start_mjd is not None) and (start_mjd_window < start_mjd):
                logger.debug(
                    f"Window start MJD [UTC] ({start_mjd_window}) is earlier than"
                    f" desired start MJD [UTC] ({start_mjd})."
                )
                start_mjd_window = start_mjd

            # Check if end_mjd_window is not later than end_mjd (if defined)
            # If end_mjd_window is later, then set end_mjd_window to end_mjd
            if (end_mjd is not None) and (end_mjd_window > end_mjd):
                logger.debug(
                    f"Window end MJD [UTC] ({end_mjd_window}) is later than desired end"
                    f" MJD [UTC] ({end_mjd})."
                )
                end_mjd_window = end_mjd

            candidates_window, frame_candidates_window = self._check_window(
                window_start=start_mjd_window,
                window_end=end_mjd_window,
                orbit=orbit_window,
                obscode=obscode,
                tolerance=tolerance,
                propagator=propagator,
                datasets=datasets,
            )
            precovery_candidates.append(candidates_window)
            frame_candidates.append(frame_candidates_window)
        return qv.concatenate(precovery_candidates), qv.concatenate(frame_candidates)

    def _check_window(
        self,
        window_start: float,
        window_end: float,
        orbit: AdamOrbits,
        obscode: str,
        tolerance: float,
        propagator: Propagator,
        datasets: Optional[set[str]],
    ):
        # Gather all MJDs and their associated healpixels in the window.
        timestamps: list[float] = []
        observation_healpixels: list[set[int]] = []
        for timestamp, healpixels in self.frames.idx.propagation_targets(
            window_start,
            window_end,
            obscode,
            datasets,
        ):
            logger.debug("mjd=%.6f:\thealpixels with data: %r", timestamp, healpixels)
            timestamps.append(timestamp)
            observation_healpixels.append(healpixels)

        # Propagate the orbit with 2-body dynamics to every timestamp in the
        # window.
        times = Timestamp.from_mjd(timestamps, scale="utc")
        # create our observers
        observers = Observers.from_code(obscode, times)
        # first propagate with 2_body
        propagated_orbits = propagate_2body(orbit, times)
        # generate ephemeris
        ephems = generate_ephemeris_2body(propagated_orbits, observers)

        # Convert those ephemerides to healpixels.
        ephem_healpixels = radec_to_healpixel(
            ra=ephems.coordinates.lon.to_numpy(),
            dec=ephems.coordinates.lat.to_numpy(),
            nside=self.frames.healpix_nside,
        ).astype(int)

        # Check which ephemerides land within data

        mjds_to_check = []
        healpixels_to_check = []
        for timestamp, observation_healpixel_set, ephem_healpixel in zip(
            timestamps, observation_healpixels, ephem_healpixels
        ):
            if ephem_healpixel in observation_healpixel_set:
                # We have a match! add mjds to list to check
                # these are coming in reverse order so we need to reverse them
                mjds_to_check.append(timestamp)
                healpixels_to_check.append(ephem_healpixel)
        if len(mjds_to_check) != 0:
            return self._check_frames(
                orbit=orbit,
                healpixels=list(reversed(healpixels_to_check)),
                obscode=obscode,
                mjds=list(reversed(mjds_to_check)),
                tolerance=tolerance,
                datasets=datasets,
                propagator=propagator,
            )
        else:
            return PrecoveryCandidatesQv.empty(), FrameCandidatesQv.empty()

    def _check_frames(
        self,
        orbit: AdamOrbits,
        healpixels: list[int],
        obscode: str,
        mjds: List[float],
        tolerance: float,
        datasets: Optional[set[str]],
        propagator: Propagator,
    ) -> Tuple[PrecoveryCandidatesQv, FrameCandidatesQv]:
        """
        Deeply inspect all frames that match the given obscode, mjd, and healpix to
        see if they contain observations which match the ephemeris.
        """
        times = Timestamp.from_mjd(mjds, scale="utc")
        # create observers
        observers = Observers.from_code(obscode, times)
        # Compute the position of the ephem carefully.
        ephemeris = propagator.generate_ephemeris(orbit, observers)
        frames = []
        for mjd, healpixel in zip(mjds, healpixels):
            frames.append(self.frames.idx.get_frames(obscode, mjd, healpixel, datasets))
        # logger.debug(
        #     "checking frames for healpix=%d obscode=%s mjd=%f",
        #     healpixel,
        #     obscode,
        #     mjd,
        # )
        precovery_candidates = [PrecoveryCandidatesQv.empty()]
        frame_candidates = [FrameCandidatesQv.empty()]
        for i, frames_i in enumerate(frames):
            for f in frames_i:
                matches = self._find_matches_in_frame(
                    f, orbit, ephemeris[i], tolerance, propagator
                )
                # If no observations were found in this frame then we
                # return frame candidates
                # Note that for the frame candidate we report the predicted
                # ephemeris at the exposure midpoint not at the observation
                # times which may differ from the exposure midpoint time
                if len(matches) == 0:
                    logger.debug("no observations found in this frame")
                    frame_candidate = frame_candidates_from_frame(f, ephemeris[i])
                    frame_candidates.append(frame_candidate)
                else:
                    precovery_candidates.append(matches)
        return qv.concatenate(precovery_candidates), qv.concatenate(frame_candidates)

    def _find_matches_in_frame(
        self,
        frame: HealpixFrame,
        orbit: AdamOrbits,
        ephem: AdamEphemeris,
        tolerance: float,
        propagator: Propagator,
    ) -> PrecoveryCandidatesQv:
        """
        Find all sources in a single frame which match ephem.
        """
        logger.debug("checking frame: %s", frame)

        # Gather all observations.
        observations = ObservationArray(list(self.frames.iterate_observations(frame)))
        n_obs = len(observations)

        mjd = ephem.coordinates.time[0].mjd().to_numpy(False)[0]
        if not np.all(observations.values["mjd"] == mjd):
            # Data has per-observation MJDs. We'll need propagate to every observation's time.
            matches = self._find_matches_using_per_obs_timestamps(
                frame, orbit, observations, tolerance, propagator
            )
            return matches

        # Compute distance between our single ephemeris and all observations
        # We need to do this in quivr
        distances = haversine_distance_deg(
            ephem.coordinates.lon[0].as_py(),
            observations.values["ra"],
            ephem.coordinates.lat[0].as_py(),
            observations.values["dec"],
        )

        # Gather ones within the tolerance
        observations.values = observations.values[distances < tolerance]
        matching_observations = observations.to_list()

        logger.debug(
            f"checked {n_obs} observations in frame {frame} and found {len(matching_observations)}"
        )

        # We can probably do without this loop
        candidates = [PrecoveryCandidatesQv.empty()]
        for obs in matching_observations:
            candidates.append(candidates_from_ephem(obs, frame, ephem))
        return qv.concatenate(candidates)

    def _find_matches_using_per_obs_timestamps(
        self,
        frame: HealpixFrame,
        orbit: AdamOrbits,
        observations: ObservationArray,
        tolerance: float,
        propagator: Propagator,
    ) -> PrecoveryCandidatesQv:
        """If a frame has per-source observation times, we need to
        propagate to every time in the frame, so

        """
        # First, propagate to the mean MJD using N-body propagation.
        mean_mjd = observations.values["mjd"].mean()

        times = Timestamp.from_mjd([mean_mjd], scale="utc")

        mean_orbit_state = propagator.propagate_orbits(orbit, times)

        # Next, compute an ephemeris for every observation in the
        # frame. Use 2-body propagation from the mean position to the
        # individual timestamps.
        times = Timestamp.from_mjd(observations.values["mjd"], scale="utc")
        observers = Observers.from_code(frame.obscode, times)
        ephemeris = generate_ephemeris_2body(mean_orbit_state, observers)

        # Now check distances.
        ephem_position_ra = ephemeris.coordinates.lon.to_numpy()
        ephem_position_dec = ephemeris.coordinates.lat.to_numpy()

        distances = haversine_distance_deg(
            ephem_position_ra,
            observations.values["ra"],
            ephem_position_dec,
            observations.values["dec"],
        )

        # Gather any matches.
        candidates = [PrecoveryCandidatesQv.empty()]
        for index in np.where(distances < tolerance)[0]:
            obs = Observation(*observations.values[index])
            ephem = ephemeris[index]
            candidates.append(candidates_from_ephem(obs, frame, ephem))
        return qv.concatenate(candidates)

    def find_observations_in_region(
        self, ra: float, dec: float, obscode: str
    ) -> Iterator[Observation]:
        """Gets all the Observations within the same Healpixel as a
        given RA, Dec for a particular observatory (specified as an obscode).

        """
        frames = self.frames.get_frames_for_ra_dec(ra, dec, obscode)
        for f in frames:
            yield from self.frames.iterate_observations(f)

    def find_observations_in_radius(
        self, ra: float, dec: float, tolerance: float, obscode: str
    ) -> Iterator[Observation]:
        """Gets all the Observations within a radius (in degrees) of
        a particular RA and Dec at a specific observatory.

        This method is approximate, and does not correctly find
        Observations that are within the radius, but on a different
        healpixel.

        """
        for o in self.find_observations_in_region(ra, dec, obscode):
            if haversine_distance_deg(o.ra, ra, o.dec, dec) <= tolerance:
                yield o

    def all_datasets(self) -> set[str]:
        """
        Returns the set of all dataset ID strings loaded in the database.
        """
        return set(self.frames.idx.get_dataset_ids())

    def _warn_for_missing_datasets(self, datasets: set[str]):
        """Log some warning messages if the given set includes
        dataset IDs which are not present in the database.

        """
        any_missing = False
        for ds in datasets:
            if not self.frames.has_dataset(ds):
                any_missing = True
                logger.warn(f'dataset "{ds}" is not in the database')
        if any_missing:
            logger.warn(f"datasets in the databse: {self.all_datasets()}")
