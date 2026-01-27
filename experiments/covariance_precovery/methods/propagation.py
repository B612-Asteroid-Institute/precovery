from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np
from adam_assist import ASSISTPropagator
from adam_core.dynamics.ephemeris import generate_ephemeris_2body
from adam_core.dynamics.propagation import propagate_2body
from adam_core.observers import Observers
from adam_core.orbits import Orbits
from adam_core.orbits.ephemeris import Ephemeris
from adam_core.orbits.variants import VariantOrbits
from adam_core.propagator import Propagator
from adam_core.time import Timestamp


@dataclass(frozen=True)
class PropagationInputs:
    orbit: Orbits  # (N)
    obscode: str
    times: Timestamp  # (M)


@dataclass(frozen=True)
class PropagationOutput:
    mean_ephem: Ephemeris  # (M)
    variants_ephem: Ephemeris | None  # (M * K) depending on propagator behavior
    variant_kind: str | None


class PropagationStrategy(Protocol):
    name: str

    def run(self, inp: PropagationInputs) -> PropagationOutput: ...


def _observers(obscode: str, times: Timestamp) -> Observers:
    return Observers.from_code(obscode, times)


class TwoBodyOnly(PropagationStrategy):
    name = "2body_only"

    def run(self, inp: PropagationInputs) -> PropagationOutput:
        # Propagate mean (with covariance if present) using 2-body.
        prop = propagate_2body(inp.orbit, inp.times)
        ephem = generate_ephemeris_2body(prop, _observers(inp.obscode, inp.times))
        return PropagationOutput(mean_ephem=ephem, variants_ephem=None, variant_kind=None)


class AssistToWindowThen2Body(PropagationStrategy):
    """
    Mirrors precovery's approach: n-body to a reference epoch, then 2-body within the window.
    """

    name = "assist_window_then_2body"

    def __init__(self, propagator: Propagator | None = None):
        self._prop = propagator or ASSISTPropagator()

    def run(self, inp: PropagationInputs) -> PropagationOutput:
        # Choose a reference epoch: mean of the times.
        mean_mjd = float(np.mean(inp.times.mjd().to_numpy(zero_copy_only=False)))
        t_ref = Timestamp.from_mjd([mean_mjd], scale="utc")
        use_cov = inp.orbit.coordinates.covariance is not None and not inp.orbit.coordinates.covariance.is_all_nan()
        orb_ref = self._prop.propagate_orbits(inp.orbit, t_ref, covariance=use_cov)
        prop = propagate_2body(orb_ref, inp.times)
        ephem = generate_ephemeris_2body(prop, _observers(inp.obscode, inp.times))
        return PropagationOutput(mean_ephem=ephem, variants_ephem=None, variant_kind=None)


class AssistVariants(PropagationStrategy):
    """
    Generate a cloud of variant orbits (sigma points or MC) and run n-body ephemeris generation.
    """

    name = "assist_variants"

    def __init__(
        self,
        *,
        variant_mode: Literal["sigma_points", "mc"] = "sigma_points",
        mc_num_samples: int = 256,
        seed: int = 0,
        scale: float = 1.0,
        propagator: Propagator | None = None,
    ):
        self.variant_mode = variant_mode
        self.mc_num_samples = int(mc_num_samples)
        self.seed = int(seed)
        self.scale = float(scale)
        self._prop = propagator or ASSISTPropagator()

    def run(self, inp: PropagationInputs) -> PropagationOutput:
        obs = _observers(inp.obscode, inp.times)

        use_cov = inp.orbit.coordinates.covariance is not None and not inp.orbit.coordinates.covariance.is_all_nan()
        mean_ephem = self._prop.generate_ephemeris(inp.orbit, obs, covariance=use_cov)

        if not use_cov:
            raise ValueError("AssistVariants requires an input orbit covariance to generate variants.")

        if self.variant_mode == "sigma_points":
            if self.scale != 1.0:
                raise ValueError("scale != 1.0 not supported for sigma-point variants in this benchmark")
            variants = VariantOrbits.create(inp.orbit, method="sigma-point")
            kind = "sigma_points"
        else:
            if self.scale != 1.0:
                raise ValueError("scale != 1.0 not supported for monte-carlo variants in this benchmark")
            variants = VariantOrbits.create(
                inp.orbit,
                method="monte-carlo",
                num_samples=int(self.mc_num_samples),
                seed=int(self.seed),
            )
            kind = f"mc_{self.mc_num_samples}"

        variants_ephem = self._prop.generate_ephemeris(variants, obs, covariance=False)
        return PropagationOutput(mean_ephem=mean_ephem, variants_ephem=variants_ephem, variant_kind=kind)


class TwoBodyWithRegimeTrigger(PropagationStrategy):
    """
    2-body by default, with an upgrade path when the regime looks risky.

    Initial trigger (cheap): if max |Δt| from orbit epoch exceeds a threshold, use
    the hybrid ASSIST-to-window + 2-body-in-window strategy.

    This is deliberately conservative; we will refine triggers later using the
    stress-test suite (close approaches, perturbation-sensitive objects).
    """

    name = "2body_with_trigger"

    def __init__(self, *, max_dt_days: float = 30.0, propagator: Propagator | None = None):
        self.max_dt_days = float(max_dt_days)
        self._fallback = TwoBodyOnly()
        self._upgrade = AssistToWindowThen2Body(propagator=propagator)

    def run(self, inp: PropagationInputs) -> PropagationOutput:
        t0s = inp.orbit.coordinates.time.mjd().to_numpy(zero_copy_only=False).astype(np.float64)
        ts = inp.times.mjd().to_numpy(zero_copy_only=False).astype(np.float64)
        dt = np.max(np.abs(ts[None, :] - t0s[:, None]))
        if float(dt) > self.max_dt_days:
            return self._upgrade.run(inp)
        return self._fallback.run(inp)

