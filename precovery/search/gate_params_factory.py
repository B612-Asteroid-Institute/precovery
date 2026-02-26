from __future__ import annotations

from collections.abc import Mapping

from .backends.protocols import GateParams
from .gate_defaults import (
    DEFAULT_APPLY_SYSTEMATIC_IF_REPORTED_RMS_LT_ARCSEC_BY_OBSCODE,
    DEFAULT_INVALID_SIGMA_FILL_FLOOR_ARCSEC_BY_OBSCODE,
    DEFAULT_INVALID_SIGMA_FILL_FLOOR_ARCSEC_GLOBAL,
    DEFAULT_SIGMA_SYSTEMATIC_ARCSEC_BY_OBSCODE,
)


def build_gate_params(
    *,
    innovation_gate_n_sigma: float,
    invalid_sigma_fill_floor_arcsec_global: float | None = None,
    invalid_sigma_fill_floor_arcsec_by_obscode: Mapping[str, float] | None = None,
    sigma_systematic_arcsec_by_obscode: Mapping[str, float] | None = None,
    apply_systematic_if_reported_rms_lt_arcsec_by_obscode: Mapping[str, float] | None = None,
    max_mag_residual_fainter_mag: float | None = None,
    max_mag_residual_brighter_mag: float | None = None,
) -> GateParams:
    """
    Build GateParams using production defaults unless explicit overrides are provided.
    """
    return GateParams(
        innovation_gate_n_sigma=float(innovation_gate_n_sigma),
        invalid_sigma_fill_floor_arcsec_global=(
            float(DEFAULT_INVALID_SIGMA_FILL_FLOOR_ARCSEC_GLOBAL)
            if invalid_sigma_fill_floor_arcsec_global is None
            else float(invalid_sigma_fill_floor_arcsec_global)
        ),
        invalid_sigma_fill_floor_arcsec_by_obscode=(
            dict(DEFAULT_INVALID_SIGMA_FILL_FLOOR_ARCSEC_BY_OBSCODE)
            if invalid_sigma_fill_floor_arcsec_by_obscode is None
            else {str(k): float(v) for k, v in invalid_sigma_fill_floor_arcsec_by_obscode.items()}
        ),
        sigma_systematic_arcsec_by_obscode=(
            dict(DEFAULT_SIGMA_SYSTEMATIC_ARCSEC_BY_OBSCODE)
            if sigma_systematic_arcsec_by_obscode is None
            else {str(k): float(v) for k, v in sigma_systematic_arcsec_by_obscode.items()}
        ),
        apply_systematic_if_reported_rms_lt_arcsec_by_obscode=(
            dict(DEFAULT_APPLY_SYSTEMATIC_IF_REPORTED_RMS_LT_ARCSEC_BY_OBSCODE)
            if apply_systematic_if_reported_rms_lt_arcsec_by_obscode is None
            else {
                str(k): float(v)
                for k, v in apply_systematic_if_reported_rms_lt_arcsec_by_obscode.items()
            }
        ),
        max_mag_residual_fainter_mag=max_mag_residual_fainter_mag,
        max_mag_residual_brighter_mag=max_mag_residual_brighter_mag,
    )
