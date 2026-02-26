from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ..selection.bq_select import estimate_bq_bytes


@dataclass(frozen=True)
class BqDetectionsDesign:
    """
    Candidate BigQuery table/MV design for detections optimized for precovery-style joins.
    """

    table: str
    partition_expr: str | None = None
    cluster_cols: tuple[str, ...] = ()

    def describe(self) -> str:
        p = "none" if self.partition_expr is None else self.partition_expr
        c = "none" if not self.cluster_cols else ",".join(self.cluster_cols)
        return f"table={self.table} partition={p} cluster={c}"


def create_table_as_select_sql(
    *,
    destination_table: str,
    select_sql: str,
    partition_expr: str | None,
    cluster_cols: tuple[str, ...],
    replace: bool = True,
) -> str:
    """
    Return a DDL statement for a CTAS-style optimized table.

    This is intended for later execution when credits arrive; the harness can still
    dry-run downstream queries against the design's `table` once it exists.
    """
    repl = "OR REPLACE " if bool(replace) else ""
    part = "" if partition_expr is None else f"\nPARTITION BY {partition_expr}"
    clust = "" if not cluster_cols else f"\nCLUSTER BY {', '.join(cluster_cols)}"
    return f"""
    CREATE {repl}TABLE `{destination_table}`
    {part}{clust}
    AS
    {select_sql}
    """.strip()


def create_exposures_mv_sql(
    *,
    destination_mv: str,
    source_table: str,
    exposure_time_key_col: str,
    exposure_mjd_mid_col: str,
    obscode_col: str,
    replace: bool = True,
) -> str:
    repl = "OR REPLACE " if bool(replace) else ""
    return f"""
    CREATE {repl}MATERIALIZED VIEW `{destination_mv}` AS
    SELECT
      CAST({obscode_col} AS STRING) AS obscode,
      CAST({exposure_time_key_col} AS INT64) AS exposure_time_key_us,
      CAST({exposure_mjd_mid_col} AS FLOAT64) AS exposure_mjd_mid
    FROM `{source_table}`
    GROUP BY obscode, exposure_time_key_us, exposure_mjd_mid
    """.strip()


def estimate_query_bytes_for_designs(
    *,
    base_query_template: str,
    designs: Iterable[BqDetectionsDesign],
) -> list[tuple[BqDetectionsDesign, int]]:
    """
    Given a query template that includes a `{table}` placeholder, estimate bytes for each design.

    Example
    -------
    base_query_template = "SELECT COUNT(1) FROM `{table}` WHERE ..."
    """
    out: list[tuple[BqDetectionsDesign, int]] = []
    for d in designs:
        q = base_query_template.format(table=d.table)
        out.append((d, int(estimate_bq_bytes(q))))
    return out

