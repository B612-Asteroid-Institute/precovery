from __future__ import annotations

from pathlib import Path

from precovery.config import Config

from .bigquery_virtual import BigQueryVirtualBackend, BqDetectionsTableConfig
from .clickhouse_local import ClickHouseLocalBackend
from .duckdb_parquet import DuckDbParquetBackend
from .protocols import SearchBackend


def backend_from_config(*, config: Config, subset_dir: Path) -> SearchBackend:
    kind = str(getattr(config, "backend", "duckdb_parquet")).strip().lower()

    if kind == "duckdb_parquet":
        p = getattr(config, "detections_parquet", None)
        if p is None or not str(p).strip():
            raise ValueError(
                "Config missing required `detections_parquet` for backend='duckdb_parquet'"
            )
        parquet_path = Path(subset_dir) / str(p) if not Path(str(p)).is_absolute() else Path(str(p))
        return DuckDbParquetBackend(parquet_path=parquet_path)

    if kind == "clickhouse_local":
        table = getattr(config, "clickhouse_table", None)
        if table is None or not str(table).strip():
            raise ValueError("Config missing required `clickhouse_table` for clickhouse_local")
        return ClickHouseLocalBackend(
            table=str(table).strip(),
            host=str(getattr(config, "clickhouse_host", "localhost")),
            port=int(getattr(config, "clickhouse_port", 8123)),
            user=str(getattr(config, "clickhouse_user", "default")),
            password=str(getattr(config, "clickhouse_password", "")),
            database=str(getattr(config, "clickhouse_database", "default")),
        )

    if kind == "bigquery_virtual":
        table = getattr(config, "bigquery_table", None)
        cfg = BqDetectionsTableConfig(table=str(table).strip()) if table else BqDetectionsTableConfig()
        return BigQueryVirtualBackend(
            cfg=cfg,
            allow_execute=bool(getattr(config, "bigquery_allow_execute", False)),
            maximum_bytes_billed=(
                None
                if getattr(config, "bigquery_maximum_bytes_billed", None) is None
                else int(getattr(config, "bigquery_maximum_bytes_billed"))
            ),
        )

    raise ValueError(f"Unknown backend kind: {kind!r}")

