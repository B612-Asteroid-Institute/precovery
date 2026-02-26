from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from mpcq.client import BigQueryMPCClient

from .bq_select import BqConfig

ResolutionStatus = Literal["resolved", "partially_resolved", "unresolved"]


@dataclass(frozen=True)
class ResolutionResult:
    table: pa.Table
    selected_designations_parquet: Path
    out_parquet: Path


def _dedupe_keep_order(values: Sequence[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for v in values:
        s = str(v).strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _empty_resolution_table() -> pa.Table:
    return pa.table(
        {
            "selected_designation": pa.array([], type=pa.large_string()),
            "permid_norm": pa.array([], type=pa.large_string()),
            "primary_provid": pa.array([], type=pa.large_string()),
            "secondary_provids": pa.array([], type=pa.list_(pa.large_string())),
            "mpcq_request_provids": pa.array([], type=pa.list_(pa.large_string())),
            "resolution_status": pa.array([], type=pa.large_string()),
        }
    )


def _resolution_status(
    *, permid_norm: str | None, primary_provid: str | None, secondary_provids: list[str]
) -> ResolutionStatus:
    if primary_provid is not None:
        return "resolved"
    if permid_norm is not None or bool(secondary_provids):
        return "partially_resolved"
    return "unresolved"


def _resolve_chunk(
    *,
    client: BigQueryMPCClient,
    cfg: BqConfig,
    selected_designations: list[str],
) -> pa.Table:
    if not selected_designations:
        return _empty_resolution_table()

    des_list = ", ".join("'" + str(d).replace("'", "''") + "'" for d in selected_designations)
    query = f"""
    WITH requested AS (
      SELECT designation
      FROM UNNEST([{des_list}]) AS designation
    ),
    parsed AS (
      SELECT
        designation,
        SAFE_CAST(designation AS INT64) AS permid_req
      FROM requested
    ),
    base AS (
      SELECT
        p.designation AS selected_designation,
        COALESCE(
          ni_req.permid,
          ni_ci.permid,
          CASE
            WHEN p.permid_req IS NOT NULL THEN CAST(p.permid_req AS STRING)
            ELSE NULL
          END
        ) AS permid_norm,
        COALESCE(
          ni_req.unpacked_primary_provisional_designation,
          ni_ci.unpacked_primary_provisional_designation,
          ci.unpacked_primary_provisional_designation,
          ci_self.unpacked_primary_provisional_designation
        ) AS primary_provid
      FROM parsed AS p
      LEFT JOIN `{cfg.numbered_identifications_table}` AS ni_req
        ON ni_req.permid = p.designation
      LEFT JOIN `{cfg.current_identifications_table}` AS ci
        ON ci.unpacked_secondary_provisional_designation = p.designation
      LEFT JOIN `{cfg.current_identifications_table}` AS ci_self
        ON ci_self.unpacked_primary_provisional_designation = p.designation
      LEFT JOIN `{cfg.numbered_identifications_table}` AS ni_ci
        ON ni_ci.unpacked_primary_provisional_designation = COALESCE(
          ci.unpacked_primary_provisional_designation,
          ci_self.unpacked_primary_provisional_designation
        )
    )
    SELECT
      b.selected_designation,
      b.permid_norm,
      b.primary_provid,
      ARRAY_AGG(
        DISTINCT ci2.unpacked_secondary_provisional_designation IGNORE NULLS
      ) AS secondary_provids
    FROM base AS b
    LEFT JOIN `{cfg.current_identifications_table}` AS ci2
      ON ci2.unpacked_primary_provisional_designation = b.primary_provid
    GROUP BY b.selected_designation, b.permid_norm, b.primary_provid
    ORDER BY b.selected_designation
    """
    raw = client.client.query(query).result().to_arrow(
        progress_bar_type=None, create_bqstorage_client=True
    )
    if raw.num_rows == 0:
        return _empty_resolution_table()

    selected = [str(x) if x is not None else "" for x in raw["selected_designation"].to_pylist()]
    permid_norm = [None if x is None else str(x) for x in raw["permid_norm"].to_pylist()]
    primary_provid = [None if x is None else str(x) for x in raw["primary_provid"].to_pylist()]
    secondary_lists_raw = raw["secondary_provids"].to_pylist()

    secondary_lists: list[list[str]] = []
    request_lists: list[list[str]] = []
    statuses: list[str] = []
    for i in range(len(selected)):
        secondaries = []
        raw_secondaries = secondary_lists_raw[i]
        if raw_secondaries is not None:
            secondaries = _dedupe_keep_order([str(x) for x in raw_secondaries if x is not None])
        secondary_lists.append(secondaries)

        request_lists.append(
            _dedupe_keep_order(
                [
                    selected[i],
                    "" if permid_norm[i] is None else permid_norm[i],
                    "" if primary_provid[i] is None else primary_provid[i],
                    *secondaries,
                ]
            )
        )
        statuses.append(
            _resolution_status(
                permid_norm=permid_norm[i],
                primary_provid=primary_provid[i],
                secondary_provids=secondaries,
            )
        )

    return pa.table(
        {
            "selected_designation": pa.array(selected, type=pa.large_string()),
            "permid_norm": pa.array(permid_norm, type=pa.large_string()),
            "primary_provid": pa.array(primary_provid, type=pa.large_string()),
            "secondary_provids": pa.array(secondary_lists, type=pa.list_(pa.large_string())),
            "mpcq_request_provids": pa.array(request_lists, type=pa.list_(pa.large_string())),
            "resolution_status": pa.array(statuses, type=pa.large_string()),
        }
    )


def resolve_designations_via_bigquery(
    *,
    selected_designations: Sequence[str],
    cfg: BqConfig | None = None,
    chunk_size: int = 2000,
) -> pa.Table:
    cfg_use = BqConfig() if cfg is None else cfg
    selected = _dedupe_keep_order([str(x) for x in selected_designations if str(x).strip()])
    if not selected:
        return _empty_resolution_table()

    client = BigQueryMPCClient(dataset_id=cfg_use.dataset_id, views_dataset_id=cfg_use.views_dataset_id)
    chunks: list[pa.Table] = []
    for i0 in range(0, len(selected), int(chunk_size)):
        chunks.append(
            _resolve_chunk(
                client=client,
                cfg=cfg_use,
                selected_designations=selected[i0 : i0 + int(chunk_size)],
            )
        )
    out = pa.concat_tables(chunks, promote_options="default").combine_chunks()

    # Enforce one row per selected_designation.
    grouped = out.group_by(["selected_designation"]).aggregate([("selected_designation", "count")])
    dup = grouped.filter(pc.greater(pc.cast(grouped["selected_designation_count"], pa.int64()), 1))
    if dup.num_rows > 0:
        vals = [str(x) for x in dup["selected_designation"].to_pylist()]
        raise ValueError(f"duplicate selected_designation rows in resolution output: {vals[:20]}")
    return out


def materialize_designation_resolution_for_selected_designations(
    *,
    selected_designations_parquet: Path,
    out_parquet: Path | None = None,
    cfg: BqConfig | None = None,
) -> ResolutionResult:
    selected_designations_parquet = Path(selected_designations_parquet).resolve()
    if not selected_designations_parquet.exists():
        raise FileNotFoundError(f"selected_designations parquet not found: {selected_designations_parquet}")
    t_sel = pq.read_table(selected_designations_parquet, columns=["designation"]).combine_chunks()
    selected_designations = [str(x) for x in t_sel["designation"].to_pylist() if x is not None and str(x).strip()]
    table = resolve_designations_via_bigquery(selected_designations=selected_designations, cfg=cfg)
    out = (
        Path(out_parquet).resolve()
        if out_parquet is not None
        else (selected_designations_parquet.parent / "designation_resolution.parquet").resolve()
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out)
    return ResolutionResult(
        table=table,
        selected_designations_parquet=selected_designations_parquet,
        out_parquet=out,
    )


def read_designation_resolution(path: Path) -> pa.Table:
    t = pq.read_table(Path(path), columns=[
        "selected_designation",
        "permid_norm",
        "primary_provid",
        "secondary_provids",
        "mpcq_request_provids",
        "resolution_status",
    ]).combine_chunks()
    need = {
        "selected_designation",
        "permid_norm",
        "primary_provid",
        "secondary_provids",
        "mpcq_request_provids",
        "resolution_status",
    }
    missing = sorted(need - set(t.column_names))
    if missing:
        raise ValueError(f"designation_resolution missing required columns: {missing}")
    return t


__all__ = [
    "ResolutionResult",
    "materialize_designation_resolution_for_selected_designations",
    "read_designation_resolution",
    "resolve_designations_via_bigquery",
]

