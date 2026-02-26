from __future__ import annotations

from precovery.search.gate_defaults import (
    DEFAULT_APPLY_SYSTEMATIC_IF_REPORTED_RMS_LT_ARCSEC_BY_OBSCODE,
    DEFAULT_INVALID_SIGMA_FILL_FLOOR_ARCSEC_BY_OBSCODE,
    DEFAULT_INVALID_SIGMA_FILL_FLOOR_ARCSEC_GLOBAL,
    DEFAULT_SIGMA_SYSTEMATIC_ARCSEC_BY_OBSCODE,
)
from precovery.search.gate_params_factory import build_gate_params


def test_build_gate_params_uses_production_defaults_by_default() -> None:
    gate = build_gate_params(innovation_gate_n_sigma=3.0)
    assert gate.innovation_gate_n_sigma == 3.0
    assert gate.invalid_sigma_fill_floor_arcsec_global == DEFAULT_INVALID_SIGMA_FILL_FLOOR_ARCSEC_GLOBAL
    assert gate.invalid_sigma_fill_floor_arcsec_by_obscode == DEFAULT_INVALID_SIGMA_FILL_FLOOR_ARCSEC_BY_OBSCODE
    assert gate.sigma_systematic_arcsec_by_obscode == DEFAULT_SIGMA_SYSTEMATIC_ARCSEC_BY_OBSCODE
    assert (
        gate.apply_systematic_if_reported_rms_lt_arcsec_by_obscode
        == DEFAULT_APPLY_SYSTEMATIC_IF_REPORTED_RMS_LT_ARCSEC_BY_OBSCODE
    )


def test_build_gate_params_respects_explicit_overrides() -> None:
    gate = build_gate_params(
        innovation_gate_n_sigma=4.0,
        invalid_sigma_fill_floor_arcsec_global=0.2,
        invalid_sigma_fill_floor_arcsec_by_obscode={"I41": 0.2},
        sigma_systematic_arcsec_by_obscode={"I41": 0.1},
        apply_systematic_if_reported_rms_lt_arcsec_by_obscode={"I41": 0.3},
        max_mag_residual_fainter_mag=1.2,
        max_mag_residual_brighter_mag=0.8,
    )
    assert gate.innovation_gate_n_sigma == 4.0
    assert gate.invalid_sigma_fill_floor_arcsec_global == 0.2
    assert gate.invalid_sigma_fill_floor_arcsec_by_obscode == {"I41": 0.2}
    assert gate.sigma_systematic_arcsec_by_obscode == {"I41": 0.1}
    assert gate.apply_systematic_if_reported_rms_lt_arcsec_by_obscode == {"I41": 0.3}
    assert gate.max_mag_residual_fainter_mag == 1.2
    assert gate.max_mag_residual_brighter_mag == 0.8
