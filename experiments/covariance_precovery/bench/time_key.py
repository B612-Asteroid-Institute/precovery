from __future__ import annotations

import numpy as np


MJD0_UTC: float = 0.0
SECONDS_PER_DAY: float = 86_400.0


def mjd_to_time_key_us(mjd_utc: float | np.ndarray) -> np.ndarray:
    """
    Convert MJD (UTC) to an integer microsecond key.

    Notes
    -----
    - We use an integer key to avoid float equality joins across backends.
    - Definition: key_us = round(mjd_days * 86400 * 1e6)
    - This key is *not* a Unix timestamp; it is only intended for internal equality joins.
    """
    x = np.asarray(mjd_utc, dtype=np.float64)
    us = np.rint(x * SECONDS_PER_DAY * 1e6).astype(np.int64)
    return us


def time_key_us_to_mjd(key_us: int | np.ndarray) -> np.ndarray:
    """
    Inverse of `mjd_to_time_key_us`.
    """
    k = np.asarray(key_us, dtype=np.int64).astype(np.float64)
    return k / (SECONDS_PER_DAY * 1e6)

