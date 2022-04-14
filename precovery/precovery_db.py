import dataclasses
import itertools
import logging
import os.path
from typing import Dict, Iterable, Iterator, Optional, Tuple

import numpy as np

from .frame_db import FrameDB, FrameIndex
from .healpix_geom import radec_to_healpixel
from .orbit import Orbit
from .spherical_geom import haversine_distance_deg

DEGREE = 1.0
ARCMIN = DEGREE / 60
ARCSEC = ARCMIN / 60

logging.basicConfig()
logger = logging.getLogger("precovery")


@dataclasses.dataclass
class PrecoveryCandidate:
    ra: float
    dec: float
    ra_sigma: float
    dec_sigma: float
    mag: float
    mag_sigma: float
    filter: str
    obscode: str
    mjd: float
    exposure_id: str
    id: str
    dra: float
    ddec: float
    distance: float


class PrecoveryDatabase:
    def __init__(self, frames: FrameDB, window_size: int):
        self.frames = frames
        self.window_size = window_size
        self._exposures_by_obscode: dict = {}

    @classmethod
    def from_dir(cls, directory: str, create: bool = False, mode: str = "r"):
        if not os.path.exists(directory):
            if create:
                return cls.create(directory)

        # todo: serialize config into file
        config: Dict[str, int] = {
            "healpix_nside": 32,
            "data_file_max_size": int(1e9),
            "window_size": 7,
        }

        frame_idx_db = "sqlite:///" + os.path.join(directory, "index.db")
        frame_idx = FrameIndex.open(frame_idx_db, mode=mode)

        data_path = os.path.join(directory, "data")
        frame_db = FrameDB(
            frame_idx, data_path, config["data_file_max_size"], config["healpix_nside"]
        )
        return cls(frame_db, config["window_size"])

    @classmethod
    def create(
        cls,
        directory: str,
        healpix_nside: int = 32,
        data_file_max_size: int = int(1e9),
        window_size: int = 7,
    ):
        """
        Create a new database on disk in the given directory.
        """
        os.makedirs(directory)

        frame_idx_db = "sqlite:///" + os.path.join(directory, "index.db")
        frame_idx = FrameIndex.open(frame_idx_db)

        data_path = os.path.join(directory, "data")
        os.makedirs(data_path)

        frame_db = FrameDB(frame_idx, data_path, data_file_max_size, healpix_nside)
        return cls(frame_db, window_size)

    def precover(
        self,
        orbit: Orbit,
        tolerance: float = 30 * ARCSEC,
        max_matches: Optional[int] = None,
        start_mjd: Optional[float] = None,
        end_mjd: Optional[float] = None,
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
            self.window_size,
        )
        windows = self.frames.idx.window_centers(start_mjd, end_mjd, self.window_size)

        # group windows by obscodes so that many windows can be searched at once
        for obscode, obs_windows in itertools.groupby(
            windows, key=lambda pair: pair[1]
        ):
            mjds = [window[0] for window in obs_windows]
            matches = self._check_windows(mjds, obscode, orbit, tolerance)
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
            start_mjd = window_midpoint - (self.window_size / 2)
            end_mjd = window_midpoint + (self.window_size / 2)
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
                    orbit, keep_approx_healpixels, obscode, keep_mjds, tolerance
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
    ) -> Iterator[PrecoveryCandidate]:
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
                        ra=o.ra,
                        dec=o.dec,
                        ra_sigma=o.ra_sigma,
                        dec_sigma=o.dec_sigma,
                        mag=o.mag,
                        mag_sigma=o.mag_sigma,
                        filter=f.filter,
                        obscode=f.obscode,
                        mjd=f.mjd,
                        exposure_id=f.exposure_id,
                        id=o.id.decode(),
                        dra=dra,
                        ddec=ddec,
                        distance=distance,
                    )
                    yield candidate
                logger.info("checked %d observations in frame", n)
                n_frame += 1
            logger.info("checked %d frames", n_frame)
