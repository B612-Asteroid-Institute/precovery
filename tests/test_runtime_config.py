from __future__ import annotations

from precovery.config import Config
from precovery.search.runtime_config import (
    load_preprop_viability_policy_config,
    load_stage3_uncertainty_budget_config,
)


def test_load_stage3_uncertainty_budget_config_positive() -> None:
    cfg = Config(max_on_sky_sigma_major_arcsec=42.0)
    got = load_stage3_uncertainty_budget_config(config=cfg)
    assert got == 42.0


def test_load_stage3_uncertainty_budget_config_invalid_values_disabled() -> None:
    assert load_stage3_uncertainty_budget_config(config=Config(max_on_sky_sigma_major_arcsec=None)) is None
    assert load_stage3_uncertainty_budget_config(config=Config(max_on_sky_sigma_major_arcsec=0.0)) is None
    assert load_stage3_uncertainty_budget_config(config=Config(max_on_sky_sigma_major_arcsec=-1.0)) is None


def test_load_preprop_viability_policy_config_defaults_and_validation() -> None:
    cfg = Config(
        preprop_viability_policy="dynamic_short_arc",
        preprop_short_arc_days_threshold=21.0,
        preprop_time_limit_days_short_arc=14.0,
        preprop_time_limit_days_default=60.0,
        preprop_max_sigma_r_over_r=0.02,
        preprop_max_covariance_condition=1.0e8,
        preprop_fail_open_on_scoring_error=True,
    )
    out = load_preprop_viability_policy_config(config=cfg)
    assert out.policy == "dynamic_short_arc"
    assert out.short_arc_days_threshold == 21.0
    assert out.time_limit_days_short_arc == 14.0
    assert out.time_limit_days_default == 60.0
    assert out.max_sigma_r_over_r == 0.02
    assert out.max_covariance_condition == 1.0e8
    assert out.fail_open_on_scoring_error is True

    bad = Config(
        preprop_viability_policy="bad-policy",
        preprop_short_arc_days_threshold=-1.0,
        preprop_time_limit_days_short_arc=0.0,
        preprop_time_limit_days_default=float("nan"),
        preprop_max_sigma_r_over_r=-2.0,
        preprop_max_covariance_condition=0.0,
        preprop_fail_open_on_scoring_error=False,
    )
    out_bad = load_preprop_viability_policy_config(config=bad)
    assert out_bad.policy == "off"
    assert out_bad.short_arc_days_threshold == 14.0
    assert out_bad.time_limit_days_short_arc == 30.0
    assert out_bad.time_limit_days_default == 90.0
    assert out_bad.max_sigma_r_over_r is None
    assert out_bad.max_covariance_condition is None
    assert out_bad.fail_open_on_scoring_error is False
