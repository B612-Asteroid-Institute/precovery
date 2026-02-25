from __future__ import annotations

from precovery.search.backends.bigquery_virtual import _gate_expr_sql, _preds_unnest_sql
from precovery.search.backends.protocols import GateParams
from precovery.search.pipeline_types import PredictedTargets


def _one_pred() -> PredictedTargets:
    return PredictedTargets.from_kwargs(
        orbit_id=["o1"],
        target_idx=[0],
        obscode=["I41"],
        exposure_mjd_mid_utc=[60000.0],
        exposure_mjd_mid_key_us=[1],
        canonical_filter_id=["r"],
        pred_lon_deg=[10.0],
        pred_lat_deg=[5.0],
        cov_ll_00=[1.0e-8],
        cov_ll_01=[0.0],
        cov_ll_11=[1.0e-8],
        pred_mag=[20.0],
    )


def test_gate_expr_fills_only_invalid_sigmas_and_avoids_greatest() -> None:
    gate = GateParams(
        innovation_gate_n_sigma=3.0,
        invalid_sigma_fill_floor_arcsec_global=0.1,
        invalid_sigma_fill_floor_arcsec_by_obscode={"I41": 0.2},
    )
    expr = _gate_expr_sql(gate=gate)
    assert "GREATEST" not in expr
    assert "ra_sigma_deg IS NOT NULL AND ra_sigma_deg > 0.0" in expr
    assert "dec_sigma_deg IS NOT NULL AND dec_sigma_deg > 0.0" in expr
    assert "WHEN obscode = 'I41' THEN" in expr
    assert "mag - pred_mag" not in expr


def test_gate_expr_includes_systematic_only_for_sys_trigger_overlap() -> None:
    gate = GateParams(
        innovation_gate_n_sigma=3.0,
        invalid_sigma_fill_floor_arcsec_global=0.1,
        sigma_systematic_arcsec_by_obscode={"T05": 0.3, "T08": 0.3},
        apply_systematic_if_reported_rms_lt_arcsec_by_obscode={"T05": 0.4},
    )
    expr = _gate_expr_sql(gate=gate)
    assert "WHEN obscode = 'T05' THEN 0.3" in expr
    assert "WHEN obscode = 'T05' THEN 0.4" in expr
    assert "WHEN obscode = 'T08' THEN 0.3" not in expr
    assert "SQRT(POWER(" in expr


def test_gate_expr_adds_mag_residual_thresholds() -> None:
    gate = GateParams(
        innovation_gate_n_sigma=3.0,
        invalid_sigma_fill_floor_arcsec_global=0.1,
        max_mag_residual_fainter_mag=1.5,
        max_mag_residual_brighter_mag=0.7,
    )
    expr = _gate_expr_sql(gate=gate)
    assert "(mag - pred_mag) <= 1.5" in expr
    assert "(mag - pred_mag) >= -0.7" in expr


def test_preds_unnest_sql_can_skip_pred_mag_column() -> None:
    preds = _one_pred()
    sql_with_mag = _preds_unnest_sql(preds, include_pred_mag=True)
    sql_no_mag = _preds_unnest_sql(preds, include_pred_mag=False)

    assert "AS pred_mag" in sql_with_mag
    assert "AS pred_mag" not in sql_no_mag
