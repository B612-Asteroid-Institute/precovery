from __future__ import annotations

import numpy as np
from adam_core.time import Timestamp

SECONDS_PER_DAY: float = 86_400.0


def mjd_to_time_key_us(mjd_utc: float | np.ndarray) -> np.ndarray:
    """
    Convert MJD (UTC) to an integer microsecond key.

    Notes
    -----
    - We use an integer key to avoid float equality joins across backends.
    - Definition: key_us = round(mjd_days * 86400 * 1e6)
    - This key is intended for internal equality joins, not as a Unix timestamp.
    """
    x = np.asarray(mjd_utc, dtype=np.float64)
    return np.rint(x * SECONDS_PER_DAY * 1e6).astype(np.int64)


def time_key_us_to_mjd(key_us: int | np.ndarray) -> np.ndarray:
    """
    Inverse of `mjd_to_time_key_us`.
    """
    k = np.asarray(key_us, dtype=np.int64).astype(np.float64)
    return k / (SECONDS_PER_DAY * 1e6)


def time_key_us_to_timestamp_utc(key_us: int | np.ndarray) -> Timestamp:
    """
    Convert the integer microsecond key (MJD-based) to a UTC `Timestamp` without float round-tripping.
    """
    k = np.asarray(key_us, dtype=np.int64)
    us_per_day = int(SECONDS_PER_DAY * 1e6)
    days = (k // us_per_day).astype(np.int64, copy=False)
    rem_us = (k - days * us_per_day).astype(np.int64, copy=False)
    nanos = (rem_us * 1000).astype(np.int64, copy=False)
    return Timestamp.from_kwargs(days=days, nanos=nanos, scale="utc")


def timestamp_utc_to_time_key_us(t: Timestamp) -> np.ndarray:
    """
    Convert a UTC Timestamp to the integer microsecond key.
    """
    tt = t.rescale("utc")
    mjd = tt.mjd().to_numpy(zero_copy_only=False).astype(np.float64)
    return mjd_to_time_key_us(mjd)

