from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from precovery.search.backends.bigquery_virtual import BqDetectionsTableConfig, BigQueryVirtualBackend
from precovery.search.backends.clickhouse_local import (
    ClickHouseLocalBackend,
    clickhouse_create_detections_table_sql,
)
from precovery.search.backends.duckdb_parquet import DuckDbParquetBackend
from precovery.search.backends.protocols import GateParams
from precovery.search.pipeline_types import PredictedTargets, PredictedTriples, SubsetPaths


def _write_detections_parquet(path: Path) -> None:
    table = pa.table(
        {
            "obscode": pa.array(["I41", "I41", "I41", "T05"], type=pa.large_string()),
            "exposure_mjd_mid_utc": pa.array([60000.0, 60000.0, 60000.5, 60000.0], type=pa.float64()),
            "exposure_mjd_mid_key_us": pa.array([1, 1, 2, 1], type=pa.int64()),
            "filter": pa.array(["r", "r", "r", "o"], type=pa.large_string()),
            "healpixel": pa.array([32, 64, 32, 32], type=pa.int64()),
            "observation_id": pa.array(["obs-a", "obs-b", "obs-c", "obs-d"], type=pa.large_string()),
            "obstime_mjd_utc": pa.array([60000.01, 60000.01, 60000.51, 60000.01], type=pa.float64()),
            "ra_deg": pa.array([10.0, 10.1, 10.2, 11.0], type=pa.float64()),
            "dec_deg": pa.array([5.0, 5.1, 5.2, 6.0], type=pa.float64()),
            "ra_sigma_deg": pa.array([1e-5, 1e-5, 1e-5, 1e-5], type=pa.float64()),
            "dec_sigma_deg": pa.array([1e-5, 1e-5, 1e-5, 1e-5], type=pa.float64()),
            "mag": pa.array([20.0, 20.1, 20.2, 19.0], type=pa.float64()),
            "mag_sigma": pa.array([0.1, 0.1, 0.1, 0.1], type=pa.float64()),
        }
    )
    pq.write_table(table, path)


def test_duckdb_contract_target_and_candidate_keys(tmp_path: Path) -> None:
    parquet = tmp_path / "detections.parquet"
    _write_detections_parquet(parquet)

    backend = DuckDbParquetBackend(parquet_path=parquet)
    subset = SubsetPaths(subset_dir=tmp_path)

    targets = backend.enumerate_targets(
        subset=subset,
        start_mjd_utc=59999.0,
        end_mjd_utc=60001.0,
        obscodes=("I41",),
    )
    target_keys = list(zip(targets.obscode.to_pylist(), targets.exposure_mjd_mid_key_us.to_pylist()))
    assert target_keys == [("I41", 1), ("I41", 2)]

    triples = PredictedTriples.from_kwargs(
        orbit_id=["o1", "o1"],
        target_idx=[0, 1],
        obscode=["I41", "I41"],
        exposure_mjd_mid_utc=[60000.0, 60000.0],
        exposure_mjd_mid_key_us=[1, 1],
        healpixel=[32, 999],
    )
    cands = backend.fetch_candidates(subset=subset, triples=triples, limit=None)
    assert len(cands) == 1
    assert cands.observation_id[0].as_py() == "obs-a"
    assert cands.healpixel[0].as_py() == 32
    assert cands.exposure_mjd_mid_key_us[0].as_py() == 1


def test_bigquery_virtual_contract_generates_key_join_sql(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[str] = []

    def _fake_estimate_bq_bytes(query: str) -> int:
        captured.append(query)
        return 12345

    monkeypatch.setattr("precovery.search.backends.bigquery_virtual._estimate_bq_bytes", _fake_estimate_bq_bytes)

    backend = BigQueryVirtualBackend(
        cfg=BqDetectionsTableConfig(table="project.dataset.detections"),
        allow_execute=False,
    )
    subset = SubsetPaths(subset_dir=Path("."))
    preds = PredictedTargets.from_kwargs(
        orbit_id=["o1"],
        target_idx=[0],
        obscode=["I41"],
        exposure_mjd_mid_utc=[60000.0],
        exposure_mjd_mid_key_us=[1],
        canonical_filter_id=["r"],
        pred_lon_deg=[10.0],
        pred_lat_deg=[5.0],
        cov_ll_00=[1e-8],
        cov_ll_01=[0.0],
        cov_ll_11=[1e-8],
        pred_mag=[20.0],
    )
    triples = PredictedTriples.from_kwargs(
        orbit_id=["o1"],
        target_idx=[0],
        obscode=["I41"],
        exposure_mjd_mid_utc=[60000.0],
        exposure_mjd_mid_key_us=[1],
        healpixel=[32],
    )
    counts = backend.count_accepted(
        subset=subset,
        triples=triples,
        preds=preds,
        gate=GateParams(),
    )
    assert len(counts) == 0
    assert backend.last_bytes_estimate == 12345
    assert captured
    sql = captured[-1]
    assert "d.exposure_mjd_mid_key_us = t.exposure_mjd_mid_key_us" in sql
    assert "d.healpixel = t.healpixel" in sql
    assert "d.obscode = t.obscode" in sql


def test_clickhouse_contract_target_and_candidate_keys_if_available() -> None:
    table = f"tmp_precovery_contract_{uuid4().hex[:12]}"
    backend = ClickHouseLocalBackend(table=table)

    try:
        client = backend._client()
        client.query("SELECT 1")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"ClickHouse service unavailable: {exc}")

    try:
        client.command(f"DROP TABLE IF EXISTS {table}")
        client.command(clickhouse_create_detections_table_sql(table=table))
        rows = [
            (
                "I41",
                "2021-09",
                60000.0,
                1,
                "r",
                32,
                "obs-a",
                60000.01,
                10.0,
                5.0,
                1e-5,
                1e-5,
                20.0,
                0.1,
            ),
            (
                "I41",
                "2021-09",
                60000.0,
                1,
                "r",
                64,
                "obs-b",
                60000.01,
                10.1,
                5.1,
                1e-5,
                1e-5,
                20.1,
                0.1,
            ),
            (
                "I41",
                "2021-09",
                60000.5,
                2,
                "r",
                32,
                "obs-c",
                60000.51,
                10.2,
                5.2,
                1e-5,
                1e-5,
                20.2,
                0.1,
            ),
        ]
        client.insert(
            table,
            rows,
            column_names=[
                "obscode",
                "year_month",
                "exposure_mjd_mid_utc",
                "exposure_mjd_mid_key_us",
                "filter",
                "healpixel",
                "observation_id",
                "obstime_mjd_utc",
                "ra_deg",
                "dec_deg",
                "ra_sigma_deg",
                "dec_sigma_deg",
                "mag",
                "mag_sigma",
            ],
        )

        subset = SubsetPaths(subset_dir=Path("."))
        targets = backend.enumerate_targets(
            subset=subset,
            start_mjd_utc=59999.0,
            end_mjd_utc=60001.0,
            obscodes=("I41",),
        )
        target_keys = list(zip(targets.obscode.to_pylist(), targets.exposure_mjd_mid_key_us.to_pylist()))
        assert target_keys == [("I41", 1), ("I41", 2)]

        triples = PredictedTriples.from_kwargs(
            orbit_id=["o1", "o2"],
            target_idx=[0, 1],
            obscode=["I41", "I41"],
            exposure_mjd_mid_utc=[60000.0, 60000.0],
            exposure_mjd_mid_key_us=[1, 1],
            healpixel=[32, 999],
        )
        cands = backend.fetch_candidates(subset=subset, triples=triples, limit=None)
        assert len(cands) == 1
        assert cands.observation_id[0].as_py() == "obs-a"
    finally:
        client.command(f"DROP TABLE IF EXISTS {table}")
