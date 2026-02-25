from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc

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

