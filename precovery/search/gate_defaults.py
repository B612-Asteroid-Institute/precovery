from __future__ import annotations

# Hardcoded per-obscode gate defaults used by production + benchmarks.
#
# These are *not* loaded from config.json; they’re part of the algorithm contract and
# intended to be stable unless explicitly changed in code review.

DEFAULT_INVALID_SIGMA_FILL_FLOOR_ARCSEC_GLOBAL: float = 0.10

DEFAULT_INVALID_SIGMA_FILL_FLOOR_ARCSEC_BY_OBSCODE: dict[str, float] = {
    # Heuristic mapping for the standard experiment subsets where obscodes imply survey:
    # - I41: ZTF
    # - T05/T08: ATLAS
    "I41": 0.10,
    "T05": 0.30,
    "T08": 0.30,
}

# Heuristic per-obscode astrometric sigma systematic terms (arcsec).
# These are applied only when the *reported* per-detection sigma RMS is unusually small.
DEFAULT_SIGMA_SYSTEMATIC_ARCSEC_BY_OBSCODE: dict[str, float] = {
    "T05": 0.30,
    "T08": 0.30,
}
DEFAULT_APPLY_SYSTEMATIC_IF_REPORTED_RMS_LT_ARCSEC_BY_OBSCODE: dict[str, float] = {
    # Trigger thresholds chosen to target the “too-optimistic sigma” tail without
    # inflating the entire station population.
    "T05": 0.40,
    "T08": 0.45,
}

# Backward-compatible aliases.
DEFAULT_DET_SIGMA_FLOOR_ARCSEC_BY_OBSCODE = DEFAULT_INVALID_SIGMA_FILL_FLOOR_ARCSEC_BY_OBSCODE
DEFAULT_DET_SIGMA_SYS_ARCSEC_BY_OBSCODE = DEFAULT_SIGMA_SYSTEMATIC_ARCSEC_BY_OBSCODE
DEFAULT_DET_SIGMA_SYS_APPLY_RMS_LT_ARCSEC_BY_OBSCODE = (
    DEFAULT_APPLY_SYSTEMATIC_IF_REPORTED_RMS_LT_ARCSEC_BY_OBSCODE
)
