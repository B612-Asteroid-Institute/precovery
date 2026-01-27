import numpy as np
from adam_core.orbits import Orbits

from experiments.covariance_precovery.methods.variants import (
    make_mc_variants,
    make_sigma_point_variants,
)


def test_make_sigma_point_variants_shape() -> None:
    orbit = Orbits.from_parquet("tests/data/sample_orbits.parquet")[0]
    variants = make_sigma_point_variants(orbit)
    # adam_core sigma-point variants are 13 samples for 6D coords
    assert len(variants) == 13


def test_make_mc_variants_includes_nominal() -> None:
    orbit = Orbits.from_parquet("tests/data/sample_orbits.parquet")[0]
    variants = make_mc_variants(orbit, num_samples=10, seed=0)
    assert len(variants) == 10


