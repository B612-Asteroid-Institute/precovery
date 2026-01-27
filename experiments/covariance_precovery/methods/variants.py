from __future__ import annotations

from adam_core.orbits import Orbits
from adam_core.orbits.variants import VariantOrbits


def make_mc_variants(
    orbit: Orbits,
    *,
    num_samples: int,
    seed: int = 0,
    scale: float = 1.0,
) -> VariantOrbits:
    """
    Create Monte Carlo variants using `adam_core.orbits.variants.VariantOrbits.create`.
    """
    if int(num_samples) <= 0:
        raise ValueError("num_samples must be > 0")
    variants = VariantOrbits.create(
        orbit,
        method="monte-carlo",
        num_samples=int(num_samples),
        seed=int(seed),
    )
    # Apply "scale" by inflating covariance prior to sampling (not supported directly here).
    # For now, keep scale=1.0 behavior; callers should adjust upstream if needed.
    if float(scale) != 1.0:
        raise ValueError("scale != 1.0 is not supported when delegating to VariantOrbits.create")
    return variants


def make_sigma_point_variants(orbit: Orbits, *, scale: float = 1.0) -> VariantOrbits:
    """
    Create sigma-point variants using `adam_core.orbits.variants.VariantOrbits.create`.
    """
    variants = VariantOrbits.create(orbit, method="sigma-point")
    if float(scale) != 1.0:
        raise ValueError("scale != 1.0 is not supported when delegating to VariantOrbits.create")
    return variants

