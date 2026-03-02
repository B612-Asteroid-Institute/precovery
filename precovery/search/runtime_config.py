from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
from dataclasses import dataclass

from precovery.config import Config
from precovery.limiting_magnitude_presets import build_default_limiting_magnitudes_table


def _empty_kv() -> tuple[pa.Array, pa.Array]:
    return (
        pa.array([], type=pa.large_string()),
        pa.array([], type=pa.float64()),
    )


def _sorted_kv_from_map(m: dict[str, float]) -> tuple[pa.Array, pa.Array]:
    if not m:
        return _empty_kv()
    items = sorted(((str(k), float(v)) for k, v in m.items()), key=lambda kv: kv[0])
    keys = pa.array([k for k, _ in items], type=pa.large_string())
    vals = pa.array([v for _, v in items], type=pa.float64())
    return keys, vals


def load_mag_gate_config(
    *,
    subset_dir: Path,
    obscodes: set[str] | None,
    config: Config,
) -> tuple[pa.Array, pa.Array, float, float | None, float | None]:
    """
    Load limiting-magnitude + magnitude-residual gate configuration.

    Returns
    -------
    (limit_codefid_keys, limit_codefid_vals, faint_margin_mag, max_faint, max_bright)
    """
    faint_margin = float(getattr(config, "faint_frame_skip_margin_mag", 0.0) or 0.0)
    max_faint = getattr(config, "max_mag_residual_fainter_mag", None)
    max_bright = getattr(config, "max_mag_residual_brighter_mag", None)

    # Presets (filtered to window obscodes when provided).
    preset_map = build_default_limiting_magnitudes_table().static_code_filter_map()
    if obscodes:
        allow = {str(x).strip() for x in obscodes if str(x).strip()}
        preset_map = {
            k: v for k, v in preset_map.items() if k.split("|", 1)[0] in allow
        }

    # Policy: use centralized presets only (no subset-local parquet overrides).
    keys, vals = _sorted_kv_from_map(preset_map)
    return keys, vals, faint_margin, max_faint, max_bright


def load_stage3_uncertainty_budget_config(*, config: Config) -> float | None:
    """
    Load optional Stage-3 on-sky uncertainty budget from config.

    Returns
    -------
    max_on_sky_sigma_major_arcsec
        Positive finite arcsec threshold, or None when disabled.
    """
    raw = getattr(config, "max_on_sky_sigma_major_arcsec", None)
    if raw is None:
        return None
    try:
        v = float(raw)
    except Exception:
        return None
    if (not np.isfinite(v)) or v <= 0.0:
        return None
    return float(v)


@dataclass(frozen=True)
class PrepropViabilityPolicyConfig:
    policy: str
    short_arc_days_threshold: float
    time_limit_days_short_arc: float
    time_limit_days_default: float
    max_sigma_r_over_r: float | None
    max_covariance_condition: float | None
    fail_open_on_scoring_error: bool


def load_preprop_viability_policy_config(*, config: Config) -> PrepropViabilityPolicyConfig:
    raw_policy = str(getattr(config, "preprop_viability_policy", "off") or "off").strip().lower()
    if raw_policy not in {"off", "static", "dynamic_short_arc"}:
        raw_policy = "off"

    def _finite_positive(value: object, default: float) -> float:
        try:
            v = float(value)
        except Exception:
            return float(default)
        if (not np.isfinite(v)) or v <= 0.0:
            return float(default)
        return float(v)

    def _finite_nonnegative_or_none(value: object) -> float | None:
        if value is None:
            return None
        try:
            v = float(value)
        except Exception:
            return None
        if (not np.isfinite(v)) or v < 0.0:
            return None
        return float(v)

    def _finite_at_least_one_or_none(value: object) -> float | None:
        if value is None:
            return None
        try:
            v = float(value)
        except Exception:
            return None
        if (not np.isfinite(v)) or v < 1.0:
            return None
        return float(v)

    return PrepropViabilityPolicyConfig(
        policy=raw_policy,
        short_arc_days_threshold=_finite_positive(
            getattr(config, "preprop_short_arc_days_threshold", 14.0), 14.0
        ),
        time_limit_days_short_arc=_finite_positive(
            getattr(config, "preprop_time_limit_days_short_arc", 30.0), 30.0
        ),
        time_limit_days_default=_finite_positive(
            getattr(config, "preprop_time_limit_days_default", 90.0), 90.0
        ),
        max_sigma_r_over_r=_finite_nonnegative_or_none(
            getattr(config, "preprop_max_sigma_r_over_r", None)
        ),
        max_covariance_condition=_finite_at_least_one_or_none(
            getattr(config, "preprop_max_covariance_condition", None)
        ),
        fail_open_on_scoring_error=bool(
            getattr(config, "preprop_fail_open_on_scoring_error", False)
        ),
    )
