from __future__ import annotations

import numpy as np
import pyarrow as pa

from bench.benchmarks.runner import _filter_truth_to_targets, _with_truth_exposure_key
from precovery.search.pipeline_types import BenchTargets
from precovery.search.time_key import mjd_to_time_key_us


def _targets_for_keys(*, obscodes: list[str], keys: list[int]) -> BenchTargets:
    mids = np.asarray(keys, dtype=np.float64) / (86400.0 * 1_000_000.0)
    return BenchTargets.from_kwargs(
        obscode=obscodes,
        exposure_mjd_mid_utc=mids,
        filter=["V"] * len(keys),
        exposure_mjd_mid_key_us=keys,
    )


def test_filter_truth_to_targets_uses_existing_exposure_key() -> None:
    truth = pa.table(
        {
            "orbit_id": pa.array(["O1", "O1", "O2"], type=pa.large_string()),
            "observation_id": pa.array(["A", "B", "C"], type=pa.large_string()),
            "obscode": pa.array(["X05", "X05", "X05"], type=pa.large_string()),
            "time_mjd_utc": pa.array([61040.1, 61040.2, 61040.3], type=pa.float64()),
            "exposure_mjd_mid_key_us": pa.array([10, 20, 30], type=pa.int64()),
        }
    )
    targets = _targets_for_keys(obscodes=["X05", "X05"], keys=[10, 30])

    out = _filter_truth_to_targets(truth=truth, targets=targets)
    assert out.num_rows == 2
    assert set(out["observation_id"].to_pylist()) == {"A", "C"}


def test_filter_truth_to_targets_derives_exposure_key_from_time() -> None:
    t1 = 61040.25
    t2 = 61040.75
    keys = mjd_to_time_key_us(np.asarray([t1, t2], dtype=np.float64))
    truth = pa.table(
        {
            "orbit_id": pa.array(["O1", "O1", "O1"], type=pa.large_string()),
            "observation_id": pa.array(["A", "B", "C"], type=pa.large_string()),
            "obscode": pa.array(["X05", "X05", "W84"], type=pa.large_string()),
            "time_mjd_utc": pa.array([t1, t2, t1], type=pa.float64()),
        }
    )
    targets = _targets_for_keys(obscodes=["X05"], keys=[int(keys[1])])

    out = _filter_truth_to_targets(truth=truth, targets=targets)
    assert out.num_rows == 1
    assert out["observation_id"][0].as_py() == "B"
    assert "exposure_mjd_mid_key_us" in out.column_names


def test_with_truth_exposure_key_adds_column_when_missing() -> None:
    t = 61040.125
    truth = pa.table(
        {
            "orbit_id": pa.array(["O1"], type=pa.large_string()),
            "observation_id": pa.array(["A"], type=pa.large_string()),
            "obscode": pa.array(["X05"], type=pa.large_string()),
            "time_mjd_utc": pa.array([t], type=pa.float64()),
        }
    )
    out = _with_truth_exposure_key(truth=truth)
    assert "exposure_mjd_mid_key_us" in out.column_names
    assert out["exposure_mjd_mid_key_us"][0].as_py() == int(
        mjd_to_time_key_us(np.asarray([t], dtype=np.float64))[0]
    )
