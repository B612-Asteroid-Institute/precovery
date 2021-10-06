import bisect
from typing import List, Sequence

from .observation import Observation
from .orbit import Orbit


class _EpochalObservations:
    """
    A collection of observations with a common epoch.
    """

    def __init__(self, epoch: int, observations: Sequence[Observation]):
        self.epoch = epoch
        self.observations = list(observations)

    def __lt__(self, other) -> bool:
        return self.epoch < other.epoch

    def __len__(self) -> int:
        return len(self.observations)


class PrecoveryDatabase:
    def __init__(self):
        # invariant: self.observations is sorted by epoch
        self.observations: List[_EpochalObservations] = []
        self.minimum_epoch: int = 1 << 31
        self.maximum_epoch: int = -(1 << 32)

    def precover(self, orbit: Orbit):
        pass

    def add_observations(self, epoch: int, observations: Sequence[Observation]):
        eo = _EpochalObservations(epoch, observations)

        bisect.insort_left(self.observations, eo)

        if eo.epoch < self.minimum_epoch:
            self.minimum_epoch = eo.epoch

        if eo.epoch > self.maximum_epoch:
            self.maximum_epoch = eo.epoch
