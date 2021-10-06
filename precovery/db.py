import bisect
from typing import Iterator, List, Sequence

from .observation import Observation
from .orbit import Orbit


def haversine_distance(ra1, dec1, ra2, dec2) -> float:
    # TODO
    return 0


class _Exposure:
    """
    A collection of observations with a common epoch.

    TODO: This should mostly exist on disk, not in memory.
    """

    def __init__(self, epoch: int, obscode: str, observations: Sequence[Observation]):
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
        return len(self.observations)

    def cone_search(
        self, ra: float, dec: float, tolerance: float
    ) -> Iterator[Observation]:
        for o in self.observations:
            if haversine_distance(o.ra, o.dec, ra, dec) <= tolerance:
                yield o


class PrecoveryDatabase:
    def __init__(self):
        # invariant: self.observations is sorted by epoch
        self.observations: List[_Exposure] = []
        self.minimum_epoch: int = 1 << 31
        self.maximum_epoch: int = -(1 << 32)

    def precover(self, orbit: Orbit):
        # v1 is naive: Compute ephem every time
        for o in self.observations:
            ephem = orbit.compute_ephemeris(o.obscode, o.epoch)
            if o.contains(ephem.ra, ephem.dec, 0.5):
                candidates = o.cone_search(ephem.ra, ephem.dec, 0.01)
                for c in candidates:
                    yield c

    def add_observations(
        self, epoch: int, obscode: str, observations: Sequence[Observation]
    ):
        eo = _Exposure(epoch, obscode, observations)

        bisect.insort_left(self.observations, eo)

        if epoch < self.minimum_epoch:
            self.minimum_epoch = epoch

        if epoch > self.maximum_epoch:
            self.maximum_epoch = epoch
