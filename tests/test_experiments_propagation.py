import pytest
from adam_core.orbits import Orbits
from adam_core.time import Timestamp

from experiments.covariance_precovery.methods.propagation import (
    run_2body_only,
    run_2body_with_trigger,
    run_assist_variants,
    run_assist_window_then_2body,
)


def test_two_body_only_runs() -> None:
    orbit = Orbits.from_parquet("tests/data/sample_orbits.parquet")[0]
    t0 = orbit.coordinates.time.mjd()[0].as_py()
    times = Timestamp.from_mjd([t0, t0 + 1.0], scale="utc")
    out = run_2body_only(orbit=orbit, obscode="I41", times=times)
    assert len(out.mean_ephem) == 2


@pytest.mark.xfail(reason="Requires ASSIST runtime data; intended for manual/CI environments that include it.")
def test_assist_to_window_then_2body_runs() -> None:
    orbit = Orbits.from_parquet("tests/data/sample_orbits.parquet")[0]
    t0 = orbit.coordinates.time.mjd()[0].as_py()
    times = Timestamp.from_mjd([t0, t0 + 1.0], scale="utc")
    out = run_assist_window_then_2body(orbit=orbit, obscode="I41", times=times)
    assert len(out.mean_ephem) == 2


@pytest.mark.xfail(reason="Requires ASSIST runtime data; intended for manual/CI environments that include it.")
def test_assist_variants_sigma_points_runs() -> None:
    orbit = Orbits.from_parquet("tests/data/sample_orbits.parquet")[0]
    t0 = orbit.coordinates.time.mjd()[0].as_py()
    times = Timestamp.from_mjd([t0, t0 + 1.0], scale="utc")
    out = run_assist_variants(orbit=orbit, obscode="I41", times=times, variant_mode="sigma_points")
    assert len(out.mean_ephem) == 2
    assert out.variants_ephem is not None
    assert out.variant_kind == "sigma_points"


def test_two_body_with_trigger_selects_two_body_for_short_dt() -> None:
    orbit = Orbits.from_parquet("tests/data/sample_orbits.parquet")[0]
    t0 = orbit.coordinates.time.mjd()[0].as_py()
    times = Timestamp.from_mjd([t0, t0 + 1.0], scale="utc")
    out = run_2body_with_trigger(orbit=orbit, obscode="I41", times=times, max_dt_days=10.0)
    assert len(out.mean_ephem) == 2

