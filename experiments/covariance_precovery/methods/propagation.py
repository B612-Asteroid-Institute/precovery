from __future__ import annotations

from typing import Literal, NamedTuple

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


class PropagationOutput(NamedTuple):
    mean_ephem: Ephemeris  # (M)
    variants_ephem: Ephemeris | None  # (M * K) depending on propagator behavior
    variant_kind: str | None


def _observers(obscode: str, times: Timestamp) -> Observers:
    return Observers.from_code(obscode, times)


def run_2body_only(*, orbit: Orbits, obscode: str, times: Timestamp) -> PropagationOutput:
    """
    Propagate mean orbit(s) with 2-body and generate ephemeris for (orbit × times).
    """
    prop = propagate_2body(orbit, times)
    ephem = generate_ephemeris_2body(prop, _observers(obscode, times))
    return PropagationOutput(mean_ephem=ephem, variants_ephem=None, variant_kind=None)


def run_assist_window_then_2body(
    *,
    orbit: Orbits,
    obscode: str,
    times: Timestamp,
    propagator: Propagator | None = None,
) -> PropagationOutput:
    """
    Mirror precovery's approach: n-body to a reference epoch, then 2-body within the window.
    """
    prop_assist = propagator or ASSISTPropagator()
    mean_mjd = float(np.mean(times.mjd().to_numpy(zero_copy_only=False)))
    t_ref = Timestamp.from_mjd([mean_mjd], scale="utc")
    use_cov = orbit.coordinates.covariance is not None and not orbit.coordinates.covariance.is_all_nan()
    orb_ref = prop_assist.propagate_orbits(orbit, t_ref, covariance=use_cov)
    prop = propagate_2body(orb_ref, times)
    ephem = generate_ephemeris_2body(prop, _observers(obscode, times))
    return PropagationOutput(mean_ephem=ephem, variants_ephem=None, variant_kind=None)


def run_assist_variants(
    *,
    orbit: Orbits,
    obscode: str,
    times: Timestamp,
    variant_mode: Literal["sigma_points", "mc"] = "sigma_points",
    mc_num_samples: int = 256,
    seed: int = 0,
    propagator: Propagator | None = None,
) -> PropagationOutput:
    """
    Generate a cloud of variant orbits (sigma points or MC) and run n-body ephemeris generation.
    """
    prop_assist = propagator or ASSISTPropagator()
    obs = _observers(obscode, times)

    use_cov = orbit.coordinates.covariance is not None and not orbit.coordinates.covariance.is_all_nan()
    mean_ephem = prop_assist.generate_ephemeris(orbit, obs, covariance=use_cov)

    if not use_cov:
        raise ValueError("run_assist_variants requires an input orbit covariance to generate variants.")

    if variant_mode == "sigma_points":
        variants = VariantOrbits.create(orbit, method="sigma-point")
        kind = "sigma_points"
    else:
        variants = VariantOrbits.create(
            orbit,
            method="monte-carlo",
            num_samples=int(mc_num_samples),
            seed=int(seed),
        )
        kind = f"mc_{int(mc_num_samples)}"

    variants_ephem = prop_assist.generate_ephemeris(variants, obs, covariance=False)
    return PropagationOutput(mean_ephem=mean_ephem, variants_ephem=variants_ephem, variant_kind=kind)


def run_2body_with_trigger(
    *,
    orbit: Orbits,
    obscode: str,
    times: Timestamp,
    max_dt_days: float = 30.0,
    propagator: Propagator | None = None,
) -> PropagationOutput:
    """
    2-body by default, with an upgrade path when the regime looks risky.

    Trigger (cheap): if max |Δt| from orbit epoch exceeds threshold, use
    the hybrid ASSIST-to-window + 2-body-in-window strategy.
    """
    t0s = orbit.coordinates.time.mjd().to_numpy(zero_copy_only=False).astype(np.float64)
    ts = times.mjd().to_numpy(zero_copy_only=False).astype(np.float64)
    dt = np.max(np.abs(ts[None, :] - t0s[:, None]))
    if float(dt) > float(max_dt_days):
        return run_assist_window_then_2body(
            orbit=orbit,
            obscode=obscode,
            times=times,
            propagator=propagator,
        )
    return run_2body_only(orbit=orbit, obscode=obscode, times=times)

