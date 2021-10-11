from typing import Iterator, Sequence

import numpy as np

from .observation import Observation
from .spherical_geom import haversine_distance


class _Exposure:
    """
    A collection of observations with a common epoch.

    TODO: This should mostly exist on disk, not in memory.
    """

    def __init__(self, epoch: float, obscode: str, observations: Sequence[Observation]):
        self.epoch = epoch
        self.obscode = obscode
        self.observations = list(observations)

        # Keep track of the bounding box of ra, dec. This lets us quickly
        # determine whether an ephemeris appears within the exposure.
        self.min_ra = self.max_ra = self.observations[0].ra
        self.min_dec = self.max_dec = self.observations[0].dec
        for o in self.observations:
            if o.ra < self.min_ra:
                self.min_ra = o.ra
            if o.ra > self.max_ra:
                self.max_ra = o.ra

            if o.dec < self.min_dec:
                self.min_dec = o.dec
            if o.dec > self.max_dec:
                self.max_dec = o.dec

    def contains(self, ra, dec, tolerance) -> bool:
        return (
            (ra + tolerance) > self.min_ra
            and (ra - tolerance) < self.max_ra
            and (dec + tolerance) > self.min_dec
            and (dec - tolerance) < self.max_dec
        )

    def __lt__(self, other) -> bool:
        return self.epoch < other.epoch

    def __len__(self) -> int:
        return self.n_observations()

    def n_observations(self) -> int:
        return len(self.observations)

    def cone_search(
        self, ra: float, dec: float, tolerance: float
    ) -> Iterator[Observation]:
        ra_rad = np.deg2rad(ra)
        dec_rad = np.deg2rad(dec)
        tol = np.deg2rad(tolerance)
        for o in self.observations:
            dist = haversine_distance(o.ra_rad, o.dec_rad, ra_rad, dec_rad)
            if dist <= tol:
                yield o
