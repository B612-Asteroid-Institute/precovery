import dataclasses
import logging
import os.path
import numpy as np
from typing import Dict, Iterator, Optional

from .frame_db import FrameDB, FrameIndex
from .healpix_geom import radec_to_healpixel
from .orbit import Orbit

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
    obscode: str
    mjd: float
    catalog_id: str
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
    def from_dir(cls, directory: str, create: bool = False):
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
        frame_idx = FrameIndex.open(frame_idx_db)

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
        for mjd, obscode in windows:
            matches = self._check_window(mjd, obscode, orbit, tolerance)
            for result in matches:
                yield result
                n += 1
                if max_matches is not None and n >= max_matches:
                    return

    def _check_window(
        self, window_midpoint: float, obscode: str, orbit: Orbit, tolerance: float
    ):
        """
        Find all observations that match orbit within a single window
        """
        logger.debug("checking window %.2f in obs=%s", window_midpoint, obscode)
        window_ephem = orbit.compute_ephemeris(obscode, window_midpoint)
        window_healpixel = radec_to_healpixel(
            window_ephem.ra, window_ephem.dec, self.frames.healpix_nside
        )
        logger.debug(
            "propagated window midpoint to %s (healpixel: %d)",
            window_ephem,
            window_healpixel,
        )

        start_mjd = window_midpoint - (self.window_size / 2)
        end_mjd = window_midpoint + (self.window_size / 2)
        for mjd, healpixels in self.frames.idx.propagation_targets(
            start_mjd, end_mjd, obscode
        ):
            logger.debug("mjd=%.6f:\thealpixels with data: %r", mjd, healpixels)
            timedelta = mjd - window_midpoint
            approx_ra, approx_dec = window_ephem.approximately_propagate(
                obscode,
                orbit,
                timedelta,
            )
            approx_healpix = int(
                radec_to_healpixel(approx_ra, approx_dec, self.frames.healpix_nside)
            )
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
            matches = self._check_frames(orbit, approx_healpix, obscode, mjd, tolerance)
            for m in matches:
                yield m

    def _check_frames(
        self,
        orbit: Orbit,
        healpix: int,
        obscode: str,
        mjd: float,
        tolerance: float,
    ) -> Iterator[PrecoveryCandidate]:
        """
        Deeply inspect all frames that match the given obscode, mjd, and healpix to
        see if they contain observations which match the ephemeris.
        """
        # Compute the position of the ephem carefully.
        exact_ephem = orbit.compute_ephemeris(obscode, mjd)
        frames = self.frames.idx.get_frames(obscode, mjd, healpix)
        logger.info(
            "checking frames for healpix=%d obscode=%s mjd=%f", healpix, obscode, mjd
        )
        n_frame = 0
        exact_ephem.ra = np.deg2rad(exact_ephem.ra)
        exact_ephem.dec = np.deg2rad(exact_ephem.dec)
        for f in frames:
            logger.info("checking frame: %s", f)
            n = 0
            for obs in self.frames.iterate_observations(f):
                n += 1
                distance, dra, ddec = obs.distance(exact_ephem)
                if distance < tolerance:
                    candidate = PrecoveryCandidate(
                        ra=obs.ra,
                        dec=obs.dec,
                        ra_sigma=obs.ra_sigma,
                        dec_sigma=obs.dec_sigma,
                        obscode=f.obscode,
                        mjd=f.mjd,
                        catalog_id=f.catalog_id,
                        id=obs.id.decode(),
                        dra=dra,
                        ddec=ddec,
                        distance=distance,
                    )
                    yield candidate
            logger.info("checked %d observations in frame", n)
            n_frame += 1
        logger.info("checked %d frames", n_frame)
