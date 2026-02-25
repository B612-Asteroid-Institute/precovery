from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from adam_core.time import Timestamp

from .pipeline_types import BenchTargets
from .time_key import timestamp_utc_to_time_key_us


def map_times_to_target_idx_by_obscode(
    *,
    obscode: np.ndarray,
    time_utc: Timestamp,
    targets: BenchTargets,
    dt_sec: float,
) -> np.ndarray:
    """
    Map each (obscode, time_utc) row to a Stage-2 target_idx.

    Fast path: exact join on (obscode, exposure_mjd_mid_key_us).
    Fallback: for any unmatched rows, do nearest-neighbor by obscode within dt_sec.
    """
    N = int(len(time_utc))
    if N == 0:
        return np.zeros(0, dtype=np.int64)
    if len(targets) == 0:
        return np.full(N, -1, dtype=np.int64)

    codes = np.asarray(obscode, dtype=object)
    if codes.shape != (N,):
        raise ValueError("obscode must be a 1D array aligned to time_utc.")

    # Canonical equality join key for times (microsecond precision).
    key_us = timestamp_utc_to_time_key_us(time_utc)

    src = pa.table(
        {
            "row_idx": pa.array(np.arange(N, dtype=np.int64), pa.int64()),
            "obscode": pa.array(codes, pa.large_string()),
            "exposure_mjd_mid_key_us": pa.array(key_us.astype(np.int64, copy=False), pa.int64()),
        }
    )
    # Targets table already has the key_us we want.
    targ = targets.table.select(["obscode", "exposure_mjd_mid_key_us"])
    targ = targ.append_column(
        "target_idx", pa.array(np.arange(len(targets), dtype=np.int64), pa.int64())
    )

    joined = src.join(
        targ, keys=["obscode", "exposure_mjd_mid_key_us"], join_type="inner"
    )
    out = np.full(N, -1, dtype=np.int64)
    if joined.num_rows > 0:
        row = np.asarray(joined["row_idx"].to_numpy(zero_copy_only=False), dtype=np.int64)
        tidx = np.asarray(joined["target_idx"].to_numpy(zero_copy_only=False), dtype=np.int64)
        out[row] = tidx

    # Fallback nearest-neighbor within dt_sec for any remaining misses.
    miss = out < 0
    if (not miss.any()) or (not np.isfinite(float(dt_sec))) or float(dt_sec) <= 0.0:
        return out

    out2 = _map_times_to_target_idx_by_obscode_nearest(
        obscode=codes[miss],
        time_utc=time_utc.take(np.nonzero(miss)[0].tolist()),
        targets=targets,
        dt_sec=float(dt_sec),
    )
    out[np.nonzero(miss)[0]] = out2
    return out


def _map_times_to_target_idx_by_obscode_nearest(
    *,
    obscode: np.ndarray,
    time_utc: Timestamp,
    targets: BenchTargets,
    dt_sec: float,
) -> np.ndarray:
    """
    Nearest-neighbor per obscode within dt_sec.

    This is only used for the small fraction of rows that miss the exact key_us join.
    """
    N = int(len(time_utc))
    if N == 0:
        return np.zeros(0, dtype=np.int64)

    # Targets: group by obscode, sort by key_us.
    targ_code = np.asarray(targets.obscode.to_pylist(), dtype=object)
    targ_key = np.asarray(
        targets.exposure_mjd_mid_key_us.to_numpy(zero_copy_only=False), dtype=np.int64
    )
    targ_idx = np.arange(len(targets), dtype=np.int64)

    # Source keys.
    src_code = np.asarray(obscode, dtype=object)
    src_key = timestamp_utc_to_time_key_us(time_utc).astype(np.int64, copy=False)

    dt_us = int(round(float(dt_sec) * 1e6))
    if dt_us < 0:
        dt_us = 0

    out = np.full(N, -1, dtype=np.int64)
    src_code_s = np.asarray([str(x) for x in src_code.tolist()], dtype=object)
    targ_code_s = np.asarray([str(x) for x in targ_code.tolist()], dtype=object)
    for code in np.unique(src_code_s):
        m_src = src_code_s == code
        if not bool(np.any(m_src)):
            continue
        m_targ = targ_code_s == code
        if not bool(np.any(m_targ)):
            continue

        tk = targ_key[m_targ]
        ti = targ_idx[m_targ]
        order = np.argsort(tk)
        tk = tk[order]
        ti = ti[order]

        sk = src_key[m_src]
        pos = np.searchsorted(tk, sk, side="left")
        pos0 = np.clip(pos - 1, 0, len(tk) - 1)
        pos1 = np.clip(pos, 0, len(tk) - 1)
        k0 = tk[pos0]
        k1 = tk[pos1]
        d0 = np.abs(sk - k0)
        d1 = np.abs(sk - k1)
        use1 = d1 < d0
        best_pos = np.where(use1, pos1, pos0)
        best_d = np.where(use1, d1, d0)
        best_idx = ti[best_pos]
        best_idx = np.where(best_d <= dt_us, best_idx, -1)

        out[np.nonzero(m_src)[0]] = best_idx.astype(np.int64, copy=False)

    return out

