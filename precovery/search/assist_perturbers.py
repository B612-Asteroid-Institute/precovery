"""
Perturber detection for ASSIST propagation: attach warnings to results.

Uses the canonical perturber set from adam_assist so the list is defined in one place.
When building result tables (e.g. per-orbit benchmark results), call
perturber_warnings_for_orbit_ids(...) so we can use adam_assist's vectorized matcher.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

try:
    from adam_assist import is_perturber as _is_perturber
except ImportError:
    _is_perturber = None  # type: ignore[misc, assignment]

# Fallback set if adam_assist not installed (e.g. tests without full deps).
_FALLBACK_PERTURBER_IDS: frozenset[str] = frozenset({
    "sun", "mercury", "venus", "earth", "mars", "jupiter", "saturn", "uranus", "neptune",
    "moon", "pluto", "134340",
    "1", "ceres", "2", "pallas", "3", "juno", "4", "vesta", "5", "astraea",
    "6", "hebe", "7", "iris", "8", "flora", "9", "metis", "10", "hygiea",
    "11", "parthenope", "12", "victoria", "13", "egeria", "14", "irene",
    "15", "eunomia", "16", "psyche",
})

_ASSIST_PERTURBER_WARNING = (
    "orbit_id matches an ASSIST perturber; propagation with ASSIST is not valid for this object"
)


def _normalize_for_lookup(value: object) -> str:
    """
    Mirror adam_assist's scalar normalization:
    strip + lowercase + first whitespace token.
    """
    s = str(value).strip().lower()
    if not s:
        return s
    parts = s.split()
    return parts[0] if parts else s


def _fallback_match_one(orbit_id: object, *, normalize: bool) -> str | None:
    if orbit_id is None:
        return None
    if isinstance(orbit_id, float) and np.isnan(orbit_id):
        return None
    s = _normalize_for_lookup(orbit_id) if normalize else str(orbit_id).strip()
    if not s:
        return None
    return s if s in _FALLBACK_PERTURBER_IDS else None


def _matches_for_orbit_ids(
    orbit_ids: Sequence[object] | np.ndarray,
    *,
    normalize: bool,
) -> np.ndarray:
    arr = np.asarray(orbit_ids, dtype=object).ravel()
    if arr.size == 0:
        return np.empty(0, dtype=object)
    if _is_perturber is not None:
        try:
            out = _is_perturber(arr, normalize=normalize)
            out_arr = np.asarray(out, dtype=object)
            if out_arr.shape == arr.shape:
                return out_arr
        except Exception:
            pass
        # Compatibility path if installed adam_assist only supports scalar calls.
        return np.array([_is_perturber(x, normalize=normalize) for x in arr], dtype=object)
    return np.array([_fallback_match_one(x, normalize=normalize) for x in arr], dtype=object)


def is_assist_perturber(orbit_id: str, *, normalize: bool = True) -> bool:
    """Return True if orbit_id (by name or number) matches an ASSIST perturber."""
    return _matches_for_orbit_ids([orbit_id], normalize=normalize)[0] is not None


def perturber_warnings_for_orbit_ids(
    orbit_ids: Sequence[object] | np.ndarray,
    *,
    normalize: bool = True,
) -> list[str | None]:
    """
    Return per-row warning strings aligned to `orbit_ids`.

    Uses adam_assist's vectorized API when available.
    """
    matched = _matches_for_orbit_ids(orbit_ids, normalize=normalize)
    return [
        _ASSIST_PERTURBER_WARNING if x is not None else None
        for x in matched
    ]


def perturber_warning_or_none(orbit_id: str) -> str | None:
    """
    If orbit_id matches an ASSIST perturber, return the warning message; else None.
    Attach this to per-orbit or run results when using ASSIST propagation.
    """
    if is_assist_perturber(orbit_id):
        return _ASSIST_PERTURBER_WARNING
    return None
