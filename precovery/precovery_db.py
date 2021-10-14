import dataclasses
from typing import Iterator, Optional

from .frame_db import FrameBundleDescription, FrameDB
from .healpix_geom import radec_to_healpixel
from .orbit import Orbit

DEGREE = 1.0
ARCMIN = DEGREE / 60
ARCSEC = ARCMIN / 60


@dataclasses.dataclass
class PrecoveryCandidate:
    ra: float
    dec: float
    obscode: str
    mjd: float
    catalog_id: str
    id: str


class PrecoveryDatabase:
    def __init__(self, frames: FrameDB, window_size: int):
        self.frames = frames
        self.window_size = window_size
        self._exposures_by_obscode: dict = {}

    def precover(
        self,
        orbit: Orbit,
        tolerance: float = 1.0 * ARCSEC,
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
        if start_mjd is None or end_mjd is None:
            first, last = self.frames.idx.mjd_bounds()
            if start_mjd is None:
                start_mjd = first
            if end_mjd is None:
                end_mjd = last

        n = 0
        bundles = self.frames.idx.frame_bundles(self.window_size, start_mjd, end_mjd)
        for bundle in bundles:
            matches = self._check_bundle(bundle, orbit, tolerance)
            for result in matches:
                yield result
                n += 1
                if max_matches is not None and n >= max_matches:
                    return

    def _check_bundle(
        self, bundle: FrameBundleDescription, orbit: Orbit, tolerance: float
    ):
        """
        Find all observations that match orbit within a single FrameBundle.
        """
        bundle_epoch = bundle.epoch_midpoint()
        bundle_ephem = orbit.compute_ephemeris(bundle.obscode, bundle_epoch)

        for mjd, healpixels in self.frames.idx.propagation_targets(bundle):
            timedelta = mjd - bundle_epoch
            approx_ra, approx_dec = bundle_ephem.approximately_propagate(
                bundle.obscode,
                orbit,
                timedelta,
            )
            approx_healpix = radec_to_healpixel(
                approx_ra, approx_dec, self.frames.healpix_nside
            )

            if approx_healpix not in healpixels:
                # No exposures anywhere near the ephem, so move on.
                continue

            matches = self._check_frames(
                orbit, approx_healpix, bundle.obscode, mjd, tolerance
            )
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
        for f in frames:
            for obs in self.frames.iterate_observations(f):
                if obs.matches(exact_ephem, tolerance):
                    candidate = PrecoveryCandidate(
                        ra=obs.ra,
                        dec=obs.dec,
                        obscode=f.obscode,
                        mjd=f.mjd,
                        catalog_id=f.catalog_id,
                        id=obs.id.decode(),
                    )
                    yield candidate
