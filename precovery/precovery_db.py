import dataclasses
import itertools
import logging
import os
from typing import Iterable, Iterator, List, Optional, Union

import numpy as np

from .config import Config, DefaultConfig
from .frame_db import FrameDB, FrameIndex, HealpixFrame
from .healpix_geom import radec_to_healpixel
from .observation import Observation, ObservationArray
from .orbit import Ephemeris, EpochTimescale, Orbit, PropagationIntegrator
from .spherical_geom import haversine_distance_deg
from .version import __version__

DEGREE = 1.0
ARCMIN = DEGREE / 60
ARCSEC = ARCMIN / 60

CANDIDATE_K = 15
CANDIDATE_NSIDE = 2**CANDIDATE_K

logging.basicConfig()
logger = logging.getLogger("precovery")


@dataclasses.dataclass
class PrecoveryCandidate:
    mjd: float
    ra_deg: float
    dec_deg: float
    ra_sigma_arcsec: float
    dec_sigma_arcsec: float
    mag: float
    mag_sigma: float
    exposure_mjd_start: float
    exposure_mjd_mid: float
    filter: str
    obscode: str
    exposure_id: str
    exposure_duration: float
    observation_id: str
    healpix_id: int
    pred_ra_deg: float
    pred_dec_deg: float
    pred_vra_degpday: float
    pred_vdec_degpday: float
    delta_ra_arcsec: float
    delta_dec_arcsec: float
    distance_arcsec: float
    dataset_id: str

    @classmethod
    def from_observed_ephem(
        cls, obs: Observation, frame: HealpixFrame, ephem: Ephemeris
    ):
        # Calculate the HEALpixel ID for the predicted ephemeris of
        # the orbit with a high nside value (k=15, nside=2**15) The
        # indexed observations are indexed to a much lower nside but
        # we may decide in the future to re-index the database using
        # different values for that parameter. As long as we return a
        # Healpix ID generated with nside greater than the indexed
        # database then we can always down-sample the ID to a lower
        # nside value
        healpix_id = int(radec_to_healpixel(ephem.ra, ephem.dec, nside=CANDIDATE_NSIDE))

        return cls(
            # Observation data:
            observation_id=obs.id.decode(),
            mjd=obs.mjd,
            ra_deg=obs.ra,
            dec_deg=obs.dec,
            ra_sigma_arcsec=obs.ra_sigma / ARCSEC,
            dec_sigma_arcsec=obs.dec_sigma / ARCSEC,
            mag=obs.mag,
            mag_sigma=obs.mag_sigma,
            # Exposure data:
            exposure_mjd_start=frame.exposure_mjd_start,
            exposure_mjd_mid=frame.exposure_mjd_mid,
            filter=frame.filter,
            obscode=frame.obscode,
            exposure_id=frame.exposure_id,
            exposure_duration=frame.exposure_duration,
            dataset_id=frame.dataset_id,
            # Ephemeris data:
            healpix_id=healpix_id,
            pred_ra_deg=ephem.ra,
            pred_dec_deg=ephem.dec,
            pred_vra_degpday=ephem.ra_velocity,
            pred_vdec_degpday=ephem.dec_velocity,
            # Data on the distance between the observation and ephemeris:
            delta_ra_arcsec=(obs.ra - ephem.ra) / ARCSEC,
            delta_dec_arcsec=(obs.dec - ephem.dec) / ARCSEC,
            distance_arcsec=haversine_distance_deg(obs.ra, ephem.ra, obs.dec, ephem.dec)
            / ARCSEC,
        )


@dataclasses.dataclass
class FrameCandidate:
    exposure_mjd_start: float
    exposure_mjd_mid: float
    filter: str
    obscode: str
    exposure_id: str
    exposure_duration: float
    healpix_id: int
    pred_ra_deg: float
    pred_dec_deg: float
    pred_vra_degpday: float
    pred_vdec_degpday: float
    dataset_id: str

    @classmethod
    def from_frame(cls, frame: HealpixFrame, ephem: Ephemeris):
        # Calculate the HEALpixel ID for the predicted ephemeris of
        # the orbit with a high nside value (k=15, nside=2**15) The
        # indexed observations are indexed to a much lower nside but
        # we may decide in the future to re-index the database using
        # different values for that parameter. As long as we return a
        # Healpix ID generated with nside greater than the indexed
        # database then we can always down-sample the ID to a lower
        # nside value
        healpix_id = int(radec_to_healpixel(ephem.ra, ephem.dec, nside=CANDIDATE_NSIDE))
        return cls(
            # Exposure data:
            exposure_mjd_start=frame.exposure_mjd_start,
            exposure_mjd_mid=frame.exposure_mjd_mid,
            filter=frame.filter,
            obscode=frame.obscode,
            exposure_id=frame.exposure_id,
            exposure_duration=frame.exposure_duration,
            dataset_id=frame.dataset_id,
            healpix_id=healpix_id,
            # Ephemeris data:
            pred_ra_deg=ephem.ra,
            pred_dec_deg=ephem.dec,
            pred_vra_degpday=ephem.ra_velocity,
            pred_vdec_degpday=ephem.dec_velocity,
        )


def sort_candidates(
    candidates: List[Union[PrecoveryCandidate, FrameCandidate]]
) -> List[Union[PrecoveryCandidate, FrameCandidate]]:
    """
    Sort candidates by ascending MJD. For precovery candidates, use the MJD of the observation.
    For frame candidates, use the MJD at the midpoint of the exposure.

    Parameters
    ----------
    candidates : List[Union[PrecoveryCandidate, FrameCandidate]]
        List of candidates to sort.

    Returns
    -------
    List[Union[PrecoveryCandidate, FrameCandidate]]
        Sorted list of candidates.
    """
    return sorted(
        candidates,
        key=lambda c: (c.mjd, c.observation_id)
        if isinstance(c, PrecoveryCandidate)
        else (c.exposure_mjd_mid, ""),
    )


class PrecoveryDatabase:
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
        frame_idx = FrameIndex.open(frame_idx_db, mode=mode)

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
        frame_idx = FrameIndex.open(frame_idx_db)

        config = Config(nside=nside, data_file_max_size=data_file_max_size)
        config.to_json(os.path.join(directory, "config.json"))

        data_path = os.path.join(directory, "data")
        os.makedirs(data_path, exist_ok=True)

        frame_db = FrameDB(frame_idx, data_path, data_file_max_size, nside)

        return cls(frame_db, config)

    def precover(
        self,
        orbit: Orbit,
        tolerance: float = 30 * ARCSEC,
        start_mjd: Optional[float] = None,
        end_mjd: Optional[float] = None,
        window_size: int = 7,
        include_frame_candidates: bool = False,
        datasets: Optional[set[str]] = None,
    ) -> List[Union[PrecoveryCandidate, FrameCandidate]]:
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
        list : List[PrecoveryCandidates, FrameCandidates]
            Precovery candidate observations, and optionally frame candidates.
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
        if datasets is not None:
            self._warn_for_missing_datasets(datasets)

        if start_mjd is None or end_mjd is None:
            first, last = self.frames.idx.mjd_bounds()
            if start_mjd is None:
                start_mjd = first
            if end_mjd is None:
                end_mjd = last

        logger.info(
            "precovering orbit %s from %.5f to %.5f, window=%d, datasets=%s",
            orbit.orbit_id,
            start_mjd,
            end_mjd,
            window_size,
            datasets or "all",
        )

        windows = self.frames.idx.window_centers(
            start_mjd, end_mjd, window_size, datasets=datasets
        )

        # group windows by obscodes so that many windows can be searched at once
        matches = []
        for obscode, obs_windows in itertools.groupby(
            windows, key=lambda pair: pair[1]
        ):
            mjds = [window[0] for window in obs_windows]
            matches_window = self._check_windows(
                mjds,
                obscode,
                orbit,
                tolerance,
                start_mjd=start_mjd,
                end_mjd=end_mjd,
                window_size=window_size,
                include_frame_candidates=include_frame_candidates,
                datasets=datasets,
            )
            matches += list(matches_window)

        # Sort matches by mjd
        if len(matches) > 0:
            matches = sort_candidates(matches)

        return matches

    def _check_windows(
        self,
        window_midpoints: Iterable[float],
        obscode: str,
        orbit: Orbit,
        tolerance: float,
        start_mjd: Optional[float] = None,
        end_mjd: Optional[float] = None,
        window_size: int = 7,
        include_frame_candidates: bool = False,
        datasets: Optional[set[str]] = None,
    ):
        """
        Find all observations that match orbit within a list of windows
        """
        # Propagate the orbit with n-body to every window center
        # Since the window midpoints are calculated from the observations
        # in the database then they are in the UTC timescale so let's use that
        orbit_propagated = orbit.propagate(
            window_midpoints,
            PropagationIntegrator.N_BODY,
            time_scale=EpochTimescale.UTC,
        )

        # Calculate the location of the orbit on the sky with n-body propagation
        # Again, we do this in the UTC timescale to match the observations in the database
        window_ephems = orbit.compute_ephemeris(
            obscode,
            window_midpoints,
            PropagationIntegrator.N_BODY,
            time_scale=EpochTimescale.UTC,
        )
        window_healpixels = radec_to_healpixel(
            np.array([w.ra for w in window_ephems]),
            np.array([w.dec for w in window_ephems]),
            self.frames.healpix_nside,
        ).astype(int)

        # Using the propagated orbits, check each window. Propagate the orbit from the center of
        # window using 2-body to find any HealpixFrames where a detection could have occured
        for window_midpoint, window_ephem, window_healpixel, orbit_window in zip(
            window_midpoints, window_ephems, window_healpixels, orbit_propagated
        ):
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

            yield from self._check_window(
                window_start=start_mjd_window,
                window_end=end_mjd_window,
                orbit=orbit_window,
                obscode=obscode,
                tolerance=tolerance,
                include_frame_candidates=include_frame_candidates,
                datasets=datasets,
            )

    def _check_window(
        self,
        window_start: float,
        window_end: float,
        orbit: Orbit,
        obscode: str,
        tolerance: float,
        include_frame_candidates: bool,
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
        ephems = orbit.compute_ephemeris(
            obscode,
            timestamps,
            PropagationIntegrator.TWO_BODY,
            time_scale=EpochTimescale.UTC,
        )

        # Convert those ephemerides to healpixels.
        ephem_healpixels = radec_to_healpixel(
            ra=np.array([w.ra for w in ephems]),
            dec=np.array([w.dec for w in ephems]),
            nside=self.frames.healpix_nside,
        ).astype(int)

        # Check which ephemerides land within data
        for timestamp, observation_healpixel_set, ephem_healpixel in zip(
            timestamps, observation_healpixels, ephem_healpixels
        ):
            if ephem_healpixel in observation_healpixel_set:
                # We have a match! Check the frame.
                yield from self._check_frame(
                    orbit=orbit,
                    healpixel=ephem_healpixel,
                    obscode=obscode,
                    mjd=timestamp,
                    tolerance=tolerance,
                    include_frame_candidates=include_frame_candidates,
                    datasets=datasets,
                )

    def _check_frame(
        self,
        orbit: Orbit,
        healpixel: int,
        obscode: str,
        mjd: float,
        tolerance: float,
        include_frame_candidates: bool,
        datasets: Optional[set[str]],
    ) -> Iterator[Union[PrecoveryCandidate, FrameCandidate]]:
        """
        Deeply inspect all frames that match the given obscode, mjd, and healpix to
        see if they contain observations which match the ephemeris.
        """
        # Compute the position of the ephem carefully.
        ephem = orbit.compute_ephemeris(
            obscode,
            [mjd],
            PropagationIntegrator.N_BODY,
            time_scale=EpochTimescale.UTC,
        )[0]

        frames = self.frames.idx.get_frames(obscode, mjd, healpixel, datasets)
        logger.debug(
            "checking frames for healpix=%d obscode=%s mjd=%f",
            healpixel,
            obscode,
            mjd,
        )
        for f in frames:
            any_matches = False
            for match in self._find_matches_in_frame(f, orbit, ephem, tolerance):
                any_matches = True
                yield match
            # If no observations were found in this frame then we
            # yield a frame candidate if desired
            # Note that for the frame candidate we report the predicted
            # ephemeris at the exposure midpoint not at the observation
            # times which may differ from the exposure midpoint time
            if not any_matches and (include_frame_candidates):
                logger.debug("no observations found in this frame")
                frame_candidate = FrameCandidate.from_frame(f, ephem)
                yield frame_candidate

    def _find_matches_in_frame(
        self, frame: HealpixFrame, orbit: Orbit, ephem: Ephemeris, tolerance: float
    ) -> Iterator[PrecoveryCandidate]:
        """
        Find all sources in a single frame which match ephem.
        """
        logger.debug("checking frame: %s", frame)

        # Gather all observations.
        observations = ObservationArray(list(self.frames.iterate_observations(frame)))
        n_obs = len(observations)

        if not np.all(observations.values["mjd"] == ephem.mjd):
            # Data has per-observation MJDs. We'll need propagate to every observation's time.
            for match in self._find_matches_using_per_obs_timestamps(
                frame, orbit, observations, tolerance
            ):
                yield (match)
            return

        # Compute distance between our single ephemeris and all observations
        distances = ephem.distance(observations)

        # Gather ones within the tolerance
        observations.values = observations.values[distances < tolerance]
        matching_observations = observations.to_list()

        logger.debug(
            f"checked {n_obs} observations in frame {frame} and found {len(matching_observations)}"
        )

        for obs in matching_observations:
            yield PrecoveryCandidate.from_observed_ephem(obs, frame, ephem)

    def _find_matches_using_per_obs_timestamps(
        self,
        frame: HealpixFrame,
        orbit: Orbit,
        observations: ObservationArray,
        tolerance: float,
    ) -> Iterator[PrecoveryCandidate]:
        """If a frame has per-source observation times, we need to
        propagate to every time in the frame, so

        """
        # First, propagate to the mean MJD using N-body propagation.
        mean_mjd = observations.values["mjd"].mean()
        mean_orbit_state = orbit.propagate(
            epochs=[mean_mjd],
            method=PropagationIntegrator.N_BODY,
            time_scale=EpochTimescale.UTC,
        )[0]

        # Next, compute an ephemeris for every observation in the
        # frame. Use 2-body propagation from the mean position to the
        # individual timestamps.
        ephemerides = mean_orbit_state.compute_ephemeris(
            obscode=frame.obscode,
            epochs=observations.values["mjd"],
            method=PropagationIntegrator.TWO_BODY,
            time_scale=EpochTimescale.UTC,
        )

        # Now check distances.
        ephem_position_ra = np.array([e.ra for e in ephemerides])
        ephem_position_dec = np.array([e.dec for e in ephemerides])

        distances = haversine_distance_deg(
            ephem_position_ra,
            observations.values["ra"],
            ephem_position_dec,
            observations.values["dec"],
        )

        # Gather any matches.
        for index in np.where(distances < tolerance)[0]:
            obs = Observation(*observations.values[index])
            ephem = ephemerides[index]
            yield PrecoveryCandidate.from_observed_ephem(obs, frame, ephem)

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
