import dataclasses
import itertools
import logging
import os
from typing import Iterable, Iterator, Optional, Union

import numpy as np
import pandas as pd

from .config import Config, DefaultConfig
from .frame_db import FrameDB, FrameIndex, HealpixFrame
from .healpix_geom import radec_to_healpixel
from .orbit import EpochTimescale, Orbit, PropagationIntegrator
from .spherical_geom import haversine_distance_deg
from .version import __version__

DEGREE = 1.0
ARCMIN = DEGREE / 60
ARCSEC = ARCMIN / 60

CANDIDATE_K = 15
CANDIDATE_NSIDE = 2 ** CANDIDATE_K

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
        max_matches: Optional[int] = None,
        start_mjd: Optional[float] = None,
        end_mjd: Optional[float] = None,
        window_size: int = 7,
        include_frame_candidates: bool = False,
    ):
        """
        Find observations which match orbit in the database. Observations are
        searched in descending order by mjd.

        orbit: The orbit to match.

        max_matches: End once this many matches have been found. If None, find
        all matches.

        start_mjd: Only consider observations from after this epoch
        (inclusive). If None, find all.

        end_mjd: Only consider observations from before this epoch (inclusive).
        If None, find all.
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
        if start_mjd is None or end_mjd is None:
            first, last = self.frames.idx.mjd_bounds()
            if start_mjd is None:
                start_mjd = first
            if end_mjd is None:
                end_mjd = last

        n = 0
        logger.info(
            "precovering orbit %s from %f.6f to %f.5f, window=%d",
            orbit.orbit_id,
            start_mjd,
            end_mjd,
            window_size,
        )

        windows = self.frames.idx.window_centers(start_mjd, end_mjd, window_size)

        # group windows by obscodes so that many windows can be searched at once
        for obscode, obs_windows in itertools.groupby(
            windows, key=lambda pair: pair[1]
        ):
            mjds = [window[0] for window in obs_windows]
            matches = self._check_windows(
                mjds,
                obscode,
                orbit,
                tolerance,
                start_mjd=start_mjd,
                end_mjd=end_mjd,
                window_size=window_size,
                include_frame_candidates=include_frame_candidates,
            )
            for result in matches:
                yield result
                n += 1
                if max_matches is not None and n >= max_matches:
                    return

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
                logger.info(
                    f"Window start MJD [UTC] ({start_mjd_window}) is earlier than"
                    f" desired start MJD [UTC] ({start_mjd})."
                )
                start_mjd_window = start_mjd

            # Check if end_mjd_window is not later than end_mjd (if defined)
            # If end_mjd_window is later, then set end_mjd_window to end_mjd
            if (end_mjd is not None) and (end_mjd_window > end_mjd):
                logger.info(
                    f"Window end MJD [UTC] ({end_mjd_window}) is later than desired end"
                    f" MJD [UTC] ({end_mjd})."
                )
                end_mjd_window = end_mjd

            test_mjds = []
            test_healpixels = []
            for mjd, healpixels in self.frames.idx.propagation_targets(
                start_mjd_window, end_mjd_window, obscode
            ):
                logger.debug("mjd=%.6f:\thealpixels with data: %r", mjd, healpixels)
                test_mjds.append(mjd)
                test_healpixels.append(healpixels)

            # Propagate the orbit with 2-body dynamics to the propagation
            # targets
            approx_ephems = orbit_window.compute_ephemeris(
                obscode,
                test_mjds,
                PropagationIntegrator.TWO_BODY,
                time_scale=EpochTimescale.UTC,
            )
            approx_ras = np.array([w.ra for w in approx_ephems])
            approx_decs = np.array([w.dec for w in approx_ephems])

            approx_healpixels = radec_to_healpixel(
                approx_ras, approx_decs, self.frames.healpix_nside
            ).astype(int)

            keep_mjds = []
            keep_approx_healpixels = []
            for mjd, healpixels, approx_ra, approx_dec, approx_healpix in zip(
                test_mjds, test_healpixels, approx_ras, approx_decs, approx_healpixels
            ):
                logger.debug("mjd=%.6f:\thealpixels with data: %r", mjd, healpixels)
                logger.debug(
                    "mjd=%.6f:\tephemeris at ra=%.3f\tdec=%.3f\thealpix=%d",
                    mjd,
                    approx_ra,
                    approx_dec,
                    approx_healpix,
                )

                if approx_healpix not in healpixels:
                    # No exposures anywhere near the ephem, so move on.
                    continue
                logger.debug("mjd=%.6f: healpixel collision, checking frames", mjd)
                keep_mjds.append(mjd)
                keep_approx_healpixels.append(approx_healpix)

            if len(keep_mjds) > 0:
                matches = self._check_frames(
                    orbit_window,
                    keep_approx_healpixels,
                    obscode,
                    keep_mjds,
                    tolerance,
                    include_frame_candidates,
                )
                for m in matches:
                    yield m

    def _check_frames(
        self,
        orbit: Orbit,
        healpixels: Iterable[int],
        obscode: str,
        mjds: Iterable[float],
        tolerance: float,
        include_frame_candidates: bool,
    ) -> Iterator[Union[PrecoveryCandidate, FrameCandidate]]:
        """
        Deeply inspect all frames that match the given obscode, mjd, and healpix to
        see if they contain observations which match the ephemeris.
        """
        # Compute the position of the ephem carefully.
        exact_ephems = orbit.compute_ephemeris(
            obscode,
            mjds,
            PropagationIntegrator.N_BODY,
            time_scale=EpochTimescale.UTC,
        )

        for exact_ephem, mjd, healpix in zip(exact_ephems, mjds, healpixels):
            frames = self.frames.idx.get_frames(obscode, mjd, healpix)
            logger.info(
                "checking frames for healpix=%d obscode=%s mjd=%f",
                healpix,
                obscode,
                mjd,
            )
            n_frame = 0

            # Calculate the HEALpixel ID for the predicted ephemeris
            # of the orbit with a high nside value (k=15, nside=2**15)
            # The indexed observations are indexed to a much lower nside but
            # we may decide in the future to re-index the database using different
            # values for that parameter. As long as we return a Healpix ID generated with
            # nside greater than the indexed database then we can always down-sample the
            # ID to a lower nside value
            healpix_id = int(
                radec_to_healpixel(
                    exact_ephem.ra, exact_ephem.dec, nside=CANDIDATE_NSIDE
                )
            )

            for f in frames:
                logger.info("checking frame: %s", f)
                obs = np.array(list(self.frames.iterate_observations(f)))
                n_obs = len(obs)

                # Read observation mjds, ra, decs into arrays
                obs_mjds = np.array([o.mjd for o in obs])
                obs_ras = np.array([o.ra for o in obs])
                obs_decs = np.array([o.dec for o in obs])

                # If any observation MJD does not match the mjd of the ephemeris
                # to then adjust the predicted ephemeris accordingly
                time_delta = obs_mjds - exact_ephem.mjd
                if np.any(np.abs(time_delta) > 0):
                    pred_ephems = orbit.compute_ephemeris(
                        obscode,
                        obs_mjds,
                        PropagationIntegrator.TWO_BODY,
                        time_scale=EpochTimescale.UTC,
                    )
                    pred_ras = np.array([w.ra for w in pred_ephems])
                    pred_decs = np.array([w.dec for w in pred_ephems])
                    pred_vras = np.array([w.ra_velocity for w in pred_ephems])
                    pred_vdecs = np.array([w.dec_velocity for w in pred_ephems])
                else:
                    pred_ras = np.array([exact_ephem.ra for _ in range(n_obs)])
                    pred_decs = np.array([exact_ephem.dec for _ in range(n_obs)])
                    pred_vras = np.array(
                        [exact_ephem.ra_velocity for _ in range(n_obs)]
                    )
                    pred_vdecs = np.array(
                        [exact_ephem.dec_velocity for _ in range(n_obs)]
                    )

                distances = haversine_distance_deg(
                    pred_ras,
                    obs_ras,
                    pred_decs,
                    obs_decs,
                )
                dras = pred_ras - obs_ras
                ddecs = pred_decs - obs_decs
                # filter to observations with distance below tolerance
                idx = distances < tolerance
                distances = distances[idx]
                dras = dras[idx]
                ddecs = ddecs[idx]
                obs = obs[idx]
                for (
                    o,
                    pred_ra,
                    pred_dec,
                    pred_vra,
                    pred_vdec,
                    distance,
                    dra,
                    ddec,
                ) in zip(
                    obs,
                    pred_ras,
                    pred_decs,
                    pred_vras,
                    pred_vdecs,
                    distances,
                    dras,
                    ddecs,
                ):
                    candidate = PrecoveryCandidate(
                        mjd=o.mjd,
                        ra_deg=o.ra,
                        dec_deg=o.dec,
                        ra_sigma_arcsec=o.ra_sigma / ARCSEC,
                        dec_sigma_arcsec=o.dec_sigma / ARCSEC,
                        mag=o.mag,
                        mag_sigma=o.mag_sigma,
                        exposure_mjd_start=f.exposure_mjd_start,
                        exposure_mjd_mid=f.exposure_mjd_mid,
                        filter=f.filter,
                        obscode=f.obscode,
                        exposure_id=f.exposure_id,
                        exposure_duration=f.exposure_duration,
                        observation_id=o.id.decode(),
                        healpix_id=healpix_id,
                        pred_ra_deg=pred_ra,
                        pred_dec_deg=pred_dec,
                        pred_vra_degpday=pred_vra,
                        pred_vdec_degpday=pred_vdec,
                        delta_ra_arcsec=dra / ARCSEC,
                        delta_dec_arcsec=ddec / ARCSEC,
                        distance_arcsec=distance / ARCSEC,
                        dataset_id=f.dataset_id,
                    )
                    yield candidate

                logger.info("checked %d observations in frame", n_obs)
                if (n_obs == 0) and (include_frame_candidates):
                    frame_candidate = FrameCandidate(
                        exposure_mjd_start=f.exposure_mjd_start,
                        exposure_mjd_mid=f.exposure_mjd_mid,
                        filter=f.filter,
                        obscode=f.obscode,
                        exposure_id=f.exposure_id,
                        exposure_duration=f.exposure_duration,
                        healpix_id=healpix_id,
                        pred_ra_deg=pred_ra,
                        pred_dec_deg=pred_dec,
                        pred_vra_degpday=pred_vra,
                        pred_vdec_degpday=pred_vdec,
                        dataset_id=f.dataset_id,
                    )
                    yield frame_candidate

                    logger.info("no observations found in this frame")

                n_frame += 1
            logger.info("checked %d frames", n_frame)

    def extract_observations_by_frames(self, frames: Iterable[HealpixFrame]):
        # consider warnings for available memory
        obs_out = pd.DataFrame(
            columns=[
                "observatory_code",
                "healpixel",
                "obs_id",
                "RA_deg",
                "Dec_deg",
                "RA_sigma_deg",
                "Dec_sigma_deg",
                "mjd_utc",
            ]
        )
        frame_dfs = []
        # iterate over frames, initially accumulating observations in numpy arrays for speed
        # over loop, build dataframe of observations within each frame with shared frame scalars
        for frame in frames:
            # can't mix numpy array types, so two accumulators: one for floats, one for obs_id strings
            inc_arr = np.empty((0, 5), float)
            obs_ids = np.empty((0, 1), object)
            for obs in self.frames.iterate_observations(frame):
                inc_arr = np.append(
                    inc_arr,
                    np.array(
                        [[obs.ra, obs.dec, obs.ra_sigma, obs.dec_sigma, frame.mjd]]
                    ),
                    axis=0,
                )
                obs_ids = np.append(
                    obs_ids,
                    np.array([[obs.id.decode()]]),
                    axis=0,
                )
            if np.any(inc_arr):
                frame_obs = pd.DataFrame(
                    inc_arr,
                    columns=[
                        "RA_deg",
                        "Dec_deg",
                        "RA_sigma_deg",
                        "Dec_sigma_deg",
                        "mjd_utc",
                    ],
                )
                frame_obs.insert(0, "observatory_code", frame.obscode)
                frame_obs.insert(1, "healpixel", frame.healpixel)
                frame_obs.insert(2, "obs_id", obs_ids)
                frame_dfs.append(frame_obs)
        obs_out = pd.concat(frame_dfs)
        return obs_out

    def extract_observations_by_date(
        self,
        mjd_start: float,
        mjd_end: float,
    ):
        # consider warnings for available memory

        frames = self.frames.idx.frames_by_date(mjd_start, mjd_end)

        return self.extract_observations_by_frames(frames)

