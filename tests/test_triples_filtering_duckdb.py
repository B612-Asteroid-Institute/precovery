from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from precovery.search.backends.duckdb_parquet import DuckDbParquetBackend
from precovery.search.pipeline_types import PredictedTriples, SubsetPaths


def test_duckdb_filter_triples_to_existing_frames(tmp_path: Path) -> None:
    # Build a tiny detections parquet.
    det = pa.table(
        {
            "obscode": pa.array(["I41"], type=pa.large_string()),
            "exposure_mjd_mid_utc": pa.array([60000.0], type=pa.float64()),
            "exposure_mjd_mid_key_us": pa.array([int(round(60000.0 * 86400 * 1e6))], type=pa.int64()),
            "filter": pa.array(["r"], type=pa.large_string()),
            "healpixel": pa.array([123], type=pa.int64()),
            "observation_id": pa.array(["obs1"], type=pa.large_string()),
            "obstime_mjd_utc": pa.array([60000.0], type=pa.float64()),
            "ra_deg": pa.array([10.0], type=pa.float64()),
            "dec_deg": pa.array([20.0], type=pa.float64()),
            "ra_sigma_deg": pa.array([0.0], type=pa.float64()),
            "dec_sigma_deg": pa.array([0.0], type=pa.float64()),
            "mag": pa.array([None], type=pa.float64()),
            "mag_sigma": pa.array([None], type=pa.float64()),
        }
    )
    p = tmp_path / "detections.parquet"
    pq.write_table(det, str(p))

    backend = DuckDbParquetBackend(parquet_path=p)

    key_us = int(round(60000.0 * 86400 * 1e6))
    triples = PredictedTriples.from_kwargs(
        orbit_id=["o1", "o1"],
        target_idx=[0, 1],
        obscode=["I41", "I41"],
        exposure_mjd_mid_utc=[60000.0, 60000.0],
        exposure_mjd_mid_key_us=[key_us, key_us],
        healpixel=[123, 999],
    )
    out = backend.filter_triples_to_existing_frames(subset=SubsetPaths(subset_dir=tmp_path), triples=triples)
    assert len(out) == 1
    assert out.healpixel[0].as_py() == 123

