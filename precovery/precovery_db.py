import os
import dataclasses
import itertools
import logging
import numpy as np
from typing import (
    Iterable,
    Iterator,
    Optional,
    Union
)

from .config import (
    Config,
    DefaultConfig
)
from .frame_db import (
    FrameDB,
    FrameIndex
)
from .healpix_geom import radec_to_healpixel
from .orbit import Orbit
from .spherical_geom import haversine_distance_deg

DEGREE = 1.0
ARCMIN = DEGREE / 60
ARCSEC = ARCMIN / 60

CANDIDATE_K = 15
CANDIDATE_NSIDE = 2**CANDIDATE_K

logging.basicConfig()
logger = logging.getLogger("precovery")

@dataclasses.dataclass
class PrecoveryCandidate:
    mjd_utc: float
    ra_deg: float
    dec_deg: float
    ra_sigma_arcsec: float
    dec_sigma_arcsec: float
    mag: float
    mag_sigma: float
    filter: str
    obscode: str
    exposure_id: str
    observation_id: str
    healpix_id: int
    pred_ra_deg: float
    pred_dec_deg: float
    pred_vra_degpday: float
    pred_vdec_degpday: float
    delta_ra_arcsec: float
    delta_dec_arcsec: float
    distance_arcsec: float

@dataclasses.dataclass
class FrameCandidate:
    mjd_utc: float
    filter: str
    obscode: str
    exposure_id: str
    healpix_id: int
    pred_ra_deg: float
    pred_dec_deg: float
    pred_vra_degpday: float
    pred_vdec_degpday: float

class PrecoveryDatabase:
    def __init__(self, frames: FrameDB):
        self.frames = frames
        self._exposures_by_obscode: dict = {}

    @classmethod
    def from_dir(cls, directory: str, create: bool = False, mode: str = "r"):
        if not os.path.exists(directory):
            if create:
                return cls.create(directory)

        try:
            config = Config.from_json(
                os.path.join(directory, "config.json")
            )
        except FileNotFoundError:
            config = DefaultConfig
            if not create:
                logger.warning("No configuration file found. Adopting configuration defaults.")

        frame_idx_db = "sqlite:///" + os.path.join(directory, "index.db")
        frame_idx = FrameIndex.open(frame_idx_db, mode=mode)

        data_path = os.path.join(directory, "data")
        frame_db = FrameDB(
            frame_idx, data_path, config.data_file_max_size, config.nside
        )
        return cls(frame_db)

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
        os.makedirs(directory)

        frame_idx_db = "sqlite:///" + os.path.join(directory, "index.db")
        frame_idx = FrameIndex.open(frame_idx_db)

        config = Config(
            nside=nside,
            data_file_max_size=data_file_max_size
        )
        config.to_json(os.path.join(directory, "config.json"))

        data_path = os.path.join(directory, "data")
        os.makedirs(data_path)

        frame_db = FrameDB(frame_idx, data_path, data_file_max_size, nside)
        return cls(frame_db)

    def precover(
        self,
        orbit: Orbit,
        tolerance: float = 30 * ARCSEC,
        max_matches: Optional[int] = None,
        start_mjd: Optional[float] = None,
        end_mjd: Optional[float] = None,
        window_size: int = 7,
        include_frame_candidates: bool = False
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
                window_size,
                include_frame_candidates
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
        window_size: int,
        include_frame_candidates: bool
    ):
        """
        Find all observations that match orbit within a list of windows
        """
        window_ephems = orbit.compute_ephemeris(obscode, window_midpoints)
        window_healpixels = radec_to_healpixel(
            np.array([w.ra for w in window_ephems]),
            np.array([w.dec for w in window_ephems]),
            self.frames.healpix_nside,
        ).astype(int)

        for window_midpoint, window_ephem, window_healpixel in zip(
            window_midpoints, window_ephems, window_healpixels
        ):
            start_mjd = window_midpoint - (window_size / 2)
            end_mjd = window_midpoint + (window_size / 2)
            timedeltas = []
            test_mjds = []
            test_healpixels = []
            for mjd, healpixels in self.frames.idx.propagation_targets(
                start_mjd, end_mjd, obscode
            ):
                logger.debug("mjd=%.6f:\thealpixels with data: %r", mjd, healpixels)
                timedelta = mjd - window_midpoint
                timedeltas.append(timedelta)
                test_mjds.append(mjd)
                test_healpixels.append(healpixels)

            approx_ras, approx_decs = window_ephem.approximately_propagate(
                obscode,
                orbit,
                timedeltas,
            )
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
                    orbit,
                    keep_approx_healpixels,
                    obscode,
                    keep_mjds,
                    tolerance,
                    include_frame_candidates
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
        include_frame_candidates: bool
    ) -> Iterator[Union[PrecoveryCandidate, FrameCandidate]]:
        """
        Deeply inspect all frames that match the given obscode, mjd, and healpix to
        see if they contain observations which match the ephemeris.
        """
        # Compute the position of the ephem carefully.
        exact_ephems = orbit.compute_ephemeris(obscode, mjds)
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
            healpix_id = radec_to_healpixel(
                exact_ephem.ra,
                exact_ephem.dec,
                nside=CANDIDATE_NSIDE
            )

            for f in frames:
                logger.info("checking frame: %s", f)
                obs = np.array(list(self.frames.iterate_observations(f)))
                n = len(obs)
                obs_ras = np.array([o.ra for o in obs])
                obs_decs = np.array([o.dec for o in obs])
                distances = haversine_distance_deg(
                    exact_ephem.ra,
                    obs_ras,
                    exact_ephem.dec,
                    obs_decs,
                )
                dras = exact_ephem.ra - obs_ras
                ddecs = exact_ephem.dec - obs_decs
                # filter to observations with distance below tolerance
                idx = distances < tolerance
                distances = distances[idx]
                dras = dras[idx]
                ddecs = ddecs[idx]
                obs = obs[idx]
                for o, distance, dra, ddec in zip(obs, distances, dras, ddecs):
                    candidate = PrecoveryCandidate(
                        mjd_utc=f.mjd,
                        ra_deg=o.ra,
                        dec_deg=o.dec,
                        ra_sigma_arcsec=o.ra_sigma/ARCSEC,
                        dec_sigma_arcsec=o.dec_sigma/ARCSEC,
                        mag=o.mag,
                        mag_sigma=o.mag_sigma,
                        filter=f.filter,
                        obscode=f.obscode,
                        exposure_id=f.exposure_id,
                        observation_id=o.id.decode(),
                        healpix_id=healpix_id,
                        pred_ra_deg=exact_ephem.ra,
                        pred_dec_deg=exact_ephem.dec,
                        pred_vra_degpday=exact_ephem.ra_velocity,
                        pred_vdec_degpday=exact_ephem.dec_velocity,
                        delta_ra_arcsec=dra/ARCSEC,
                        delta_dec_arcsec=ddec/ARCSEC,
                        distance_arcsec=distance/ARCSEC
                    )
                    yield candidate

                logger.info("checked %d observations in frame", n)
                if (len(obs) == 0) & (include_frame_candidates):
                    frame_candidate = FrameCandidate(
                        mjd_utc=f.mjd,
                        filter=f.filter,
                        obscode=f.obscode,
                        exposure_id=f.exposure_id,
                        healpix_id=healpix_id,
                        pred_ra_deg=exact_ephem.ra,
                        pred_dec_deg=exact_ephem.dec,
                        pred_vra_degpday=exact_ephem.ra_velocity,
                        pred_vdec_degpday=exact_ephem.dec_velocity,
                    )
                    yield frame_candidate

                    logger.info(f"no observations found in this frame")

                n_frame += 1
            logger.info("checked %d frames", n_frame)
