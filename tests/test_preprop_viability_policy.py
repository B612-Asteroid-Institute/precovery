from __future__ import annotations

import numpy as np
import pytest

from adam_core.coordinates.covariances import CoordinateCovariances
from adam_core.dynamics.exceptions import DynamicsNumericalError

from precovery.search.propagation import evaluate_preprop_viability_policy


def test_preprop_viability_policy_off_returns_full(sample_orbits) -> None:
    orbit = sample_orbits.take([0])
    out = evaluate_preprop_viability_policy(
        orbit=orbit,
        policy_mode="off",
        short_arc_days_threshold=14.0,
        time_limit_days_short_arc=30.0,
        time_limit_days_default=90.0,
        max_sigma_r_over_r=None,
        max_covariance_condition=None,
        fail_open_on_scoring_error=False,
    )
    assert out.decision == "full"
    assert out.reason == "policy_off"


def test_preprop_viability_policy_strict_scoring_error_raises(sample_orbits) -> None:
    orbit = sample_orbits.take([0]).set_column(
        "coordinates.covariance", CoordinateCovariances.nulls(1)
    )
    with pytest.raises(DynamicsNumericalError, match="policy_scoring_error"):
        evaluate_preprop_viability_policy(
            orbit=orbit,
            policy_mode="static",
            short_arc_days_threshold=14.0,
            time_limit_days_short_arc=30.0,
            time_limit_days_default=90.0,
            max_sigma_r_over_r=None,
            max_covariance_condition=None,
            fail_open_on_scoring_error=False,
        )


def test_preprop_viability_policy_large_covariance_not_full(sample_orbits) -> None:
    orbit = sample_orbits.take([0])
    huge_cov = np.eye(6, dtype=np.float64) * 1e2
    orbit = orbit.set_column(
        "coordinates.covariance",
        CoordinateCovariances.from_matrix(huge_cov.reshape(1, 6, 6)),
    )
    out = evaluate_preprop_viability_policy(
        orbit=orbit,
        policy_mode="dynamic_short_arc",
        short_arc_days_threshold=60.0,
        time_limit_days_short_arc=7.0,
        time_limit_days_default=30.0,
        max_sigma_r_over_r=1e-9,
        max_covariance_condition=None,
        fail_open_on_scoring_error=False,
    )
    assert out.decision == "time_limited"


def test_preprop_viability_policy_covariance_condition_independent_skip(sample_orbits) -> None:
    orbit = sample_orbits.take([0])
    # Positive-definite but very ill-conditioned covariance.
    ill = np.diag(np.asarray([1.0e-16, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64))
    orbit = orbit.set_column(
        "coordinates.covariance",
        CoordinateCovariances.from_matrix(ill.reshape(1, 6, 6)),
    )
    out = evaluate_preprop_viability_policy(
        orbit=orbit,
        policy_mode="dynamic_short_arc",
        short_arc_days_threshold=1.0,
        time_limit_days_short_arc=7.0,
        time_limit_days_default=30.0,
        max_sigma_r_over_r=None,
        max_covariance_condition=1.0e10,
        fail_open_on_scoring_error=False,
    )
    assert out.decision == "skip"
    assert out.reason == "covariance_condition_exceeds_max"
    assert out.trigger_metric == "covariance_condition"


def test_preprop_viability_policy_short_arc_independent_time_limited(sample_orbits) -> None:
    orbit = sample_orbits.take([0])
    out = evaluate_preprop_viability_policy(
        orbit=orbit,
        policy_mode="dynamic_short_arc",
        short_arc_days_threshold=1.0e9,
        time_limit_days_short_arc=11.0,
        time_limit_days_default=30.0,
        max_sigma_r_over_r=None,
        max_covariance_condition=None,
        fail_open_on_scoring_error=False,
    )
    assert out.decision == "time_limited"
    assert out.reason == "short_arc_proxy_days_below_threshold"
    assert out.trigger_metric == "short_arc_proxy_days"
    assert out.time_limit_days == 11.0


def test_preprop_viability_policy_sigma_r_over_r_independent_time_limited(sample_orbits) -> None:
    orbit = sample_orbits.take([0])
    out = evaluate_preprop_viability_policy(
        orbit=orbit,
        policy_mode="dynamic_short_arc",
        short_arc_days_threshold=1.0e-12,
        time_limit_days_short_arc=7.0,
        time_limit_days_default=22.0,
        max_sigma_r_over_r=0.0,
        max_covariance_condition=None,
        fail_open_on_scoring_error=False,
    )
    assert out.decision == "time_limited"
    assert out.reason == "sigma_r_over_r_exceeds_max"
    assert out.trigger_metric == "sigma_r_over_r"
    assert out.time_limit_days == 22.0
