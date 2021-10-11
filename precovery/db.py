import bisect
from typing import Dict, List, Sequence

from ._exposure import _Exposure
from .observation import Observation
from .orbit import Orbit, PropagationIntegrator


class PrecoveryDatabase:
    def __init__(self, days_per_bundle: int):
        # invariant: self.observations is sorted by epoch
        self._exposures_by_obscode: Dict[str, _ExposureDB] = {}
        self.minimum_epoch: float = 1 << 31
        self.maximum_epoch: float = -(1 << 32)
        self.days_per_bundle = days_per_bundle

    def precover(self, orbit: Orbit):
        # Hand the workload down to each observatory's worth of data
        for edb in self._exposures_by_obscode.values():
            for result in edb.precover(orbit):
                yield result

    def add_observations(
        self, epoch: float, obscode: str, observations: Sequence[Observation]
    ):
        edb = self._exposures_by_obscode.get(obscode)
        if edb is None:
            edb = _ExposureDB(obscode, self.days_per_bundle)
            self._exposures_by_obscode[obscode] = edb

        edb.add_observations(epoch, observations)

        if epoch < self.minimum_epoch:
            self.minimum_epoch = epoch

        if epoch > self.maximum_epoch:
            self.maximum_epoch = epoch

    def n_observations(self) -> int:
        return sum(edb.n_observations() for edb in self._exposures_by_obscode.values())

    def n_exposures(self) -> int:
        return sum(edb.n_exposures() for edb in self._exposures_by_obscode.values())

    def n_exposure_bundles(self) -> int:
        return sum(
            edb.n_exposure_bundles() for edb in self._exposures_by_obscode.values()
        )


class _ExposureDB:
    def __init__(self, obscode: str, days_per_bundle: int):
        """
        A collection of _ExposureBundles which are from the same observatory.

        The _ExposureBundles are groups of _Exposures which are nearby in time.
        "Nearby" is given a precise meaning through the "days_per_bundle"
        parameter.

        obscode: The MPC observatory code for all observations in the bundle.
        days_per_bundle: How many days of data should be included in one ExposureBundle?
        """
        self.obscode = obscode
        self.exposure_bundles: Dict[int, _ExposureBundle] = {}
        self.days_per_bundle = days_per_bundle

    def _truncate_epoch(self, epoch: float):
        return int(epoch) - (int(epoch) % self.days_per_bundle)

    def add_observations(self, epoch: float, observations: Sequence[Observation]):
        epoch_idx = self._truncate_epoch(epoch)
        eb = self.exposure_bundles.get(epoch_idx)
        if eb is None:
            eb = _ExposureBundle(self.obscode, epoch)
            self.exposure_bundles[epoch_idx] = eb

        eb.add_observations(epoch, observations)

    def precover(
        self, orbit: Orbit, rough_tolerance: float = 0.5, fine_tolerance: float = 0.01
    ):
        # Find all the observations in the _ExposureDB which might be the
        # orbit's ephemerides.
        for eb in self.exposure_bundles.values():
            for candidate in eb.precover(orbit, rough_tolerance, fine_tolerance):
                yield candidate

    def n_observations(self) -> int:
        return sum(eb.n_observations() for eb in self.exposure_bundles.values())

    def n_exposures(self) -> int:
        return sum(eb.n_exposures() for eb in self.exposure_bundles.values())

    def n_exposure_bundles(self) -> int:
        return len(self.exposure_bundles)


class _ExposureBundle:
    """
    A collection of _Exposures which are relatively nearby in time.
    """

    def __init__(self, obscode: str, epoch: float):
        self.obscode = obscode
        self.exposures: List[_Exposure] = []
        self.epoch = epoch

    def precover(
        self,
        orbit: Orbit,
        rough_tolerance: float,
        fine_tolerance: float,
    ):
        # Calculate ephemeris for the _ExposureBundle's epoch, which is nearly
        # that of all of its exposures.
        bundle_ephem = orbit.compute_ephemeris(self.obscode, self.epoch)

        for exposure in self.exposures:
            exposure_timedelta_in_days = exposure.epoch - self.epoch

            approx_ra, approx_dec = bundle_ephem.approximately_propagate(
                self.obscode,
                orbit,
                exposure_timedelta_in_days,
            )

            if exposure.contains(approx_ra, approx_dec, rough_tolerance):
                # It looks like the approximate location is within the
                # exposure, so it's worthwhile to compute the ephemeris
                # more accurately, and do a cone search.
                exact_ephem = orbit.compute_ephemeris(
                    self.obscode,
                    exposure.epoch,
                    method=PropagationIntegrator.N_BODY,
                )
                candidates = exposure.cone_search(
                    exact_ephem.ra, exact_ephem.dec, fine_tolerance
                )
                for c in candidates:
                    yield c

    def add_observations(self, epoch: float, observations: Sequence[Observation]):
        eo = _Exposure(epoch, self.obscode, observations)
        bisect.insort_left(self.exposures, eo)

    def n_observations(self) -> int:
        return sum(e.n_observations() for e in self.exposures)

    def n_exposures(self) -> int:
        return len(self.exposures)
