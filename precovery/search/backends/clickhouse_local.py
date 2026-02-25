from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..gate_counts import accepted_counts_python
from ..pipeline_types import (
    AcceptedCounts,
    BenchTargets,
    CandidateDetections,
    PredictedTargets,
    PredictedTriples,
    SubsetPaths,
)
from .protocols import BackendCapabilities, GateParams, SearchBackend


def _require_clickhouse_connect():
    try:
        import clickhouse_connect  # type: ignore

        return clickhouse_connect
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "ClickHouse backend requires `clickhouse-connect`. Install with `pdm add clickhouse-connect`."
        ) from e


@dataclass
class ClickHouseLocalBackend(SearchBackend):
    """
    ClickHouse backend intended for local Docker evaluation.

    This backend assumes a ClickHouse server is running and a detections table exists
    (loaded from parquet export). It uses a temp table join for predicted triples.
    """

    table: str
    host: str = "localhost"
    port: int = 8123
    user: str = "default"
    password: str = ""
    database: str = "default"

    name: str = "clickhouse_local"
    capabilities: BackendCapabilities = BackendCapabilities(
        supports_enumerate_targets=True,
        supports_sql_gate_counts=False,
        supports_sql_gate_rows=False,
        supports_filter_triples_to_existing_frames=False,
    )

    def _client(self):
        ch = _require_clickhouse_connect()
        return ch.get_client(
            host=str(self.host),
            port=int(self.port),
            username=str(self.user),
            password=str(self.password),
            database=str(self.database),
        )

    def enumerate_targets(
        self,
        *,
        subset: SubsetPaths,  # unused
        start_mjd_utc: float,
        end_mjd_utc: float,
        obscodes: tuple[str, ...],
    ) -> BenchTargets:
        client = self._client()
        stn_sql = ""
        if obscodes:
            stn_list = ", ".join(f"'{s}'" for s in obscodes)
            stn_sql = f" AND obscode IN ({stn_list})"
        q = f"""
        SELECT
          obscode,
          exposure_mjd_mid_utc,
          exposure_mjd_mid_key_us,
          filter
        FROM {self.table}
        WHERE exposure_mjd_mid_utc >= {float(start_mjd_utc)}
          AND exposure_mjd_mid_utc <  {float(end_mjd_utc)}
          {stn_sql}
        GROUP BY obscode, exposure_mjd_mid_utc, exposure_mjd_mid_key_us, filter
        ORDER BY exposure_mjd_mid_utc ASC, obscode ASC, filter ASC
        """
        res = client.query(q)
        rows = res.result_rows
        if not rows:
            return BenchTargets.empty()
        obsc = [str(r[0]) for r in rows]
        mjd = [float(r[1]) for r in rows]
        key = [int(r[2]) for r in rows]
        filt = [str(r[3]) for r in rows]
        return BenchTargets.from_kwargs(
            obscode=obsc,
            exposure_mjd_mid_utc=mjd,
            exposure_mjd_mid_key_us=key,
            filter=filt,
        )

    def _create_temp_triples(self, *, client, triples: PredictedTriples) -> str:
        tmp = "tmp_triples"
        client.command(f"DROP TABLE IF EXISTS {tmp}")
        client.command(
            f"""
            CREATE TEMPORARY TABLE {tmp} (
              orbit_id String,
              target_idx Int64,
              obscode String,
              exposure_mjd_mid_key_us Int64,
              healpixel Int64
            )
            """
        )
        data = list(
            zip(
                [str(x) for x in triples.orbit_id.to_pylist()],
                triples.target_idx.to_numpy(zero_copy_only=False).astype(np.int64).tolist(),
                [str(x) for x in triples.obscode.to_pylist()],
                triples.exposure_mjd_mid_key_us.to_numpy(zero_copy_only=False).astype(np.int64).tolist(),
                triples.healpixel.to_numpy(zero_copy_only=False).astype(np.int64).tolist(),
            )
        )
        if data:
            client.insert(
                tmp,
                data,
                column_names=[
                    "orbit_id",
                    "target_idx",
                    "obscode",
                    "exposure_mjd_mid_key_us",
                    "healpixel",
                ],
            )
        return tmp

    def fetch_candidates(
        self,
        *,
        subset: SubsetPaths,  # unused
        triples: PredictedTriples,
        limit: int | None = None,
    ) -> CandidateDetections:
        if len(triples) == 0:
            return CandidateDetections.empty()
        client = self._client()
        tmp = self._create_temp_triples(client=client, triples=triples)
        lim = "" if limit is None else f"LIMIT {int(limit)}"
        q = f"""
        SELECT
          t.orbit_id,
          t.target_idx,
          d.obscode,
          d.exposure_mjd_mid_key_us,
          d.healpixel,
          d.filter,
          d.observation_id,
          d.obstime_mjd_utc,
          d.ra_deg,
          d.dec_deg,
          d.ra_sigma_deg,
          d.dec_sigma_deg,
          d.mag,
          d.mag_sigma
        FROM {tmp} AS t
        INNER JOIN {self.table} AS d
          ON d.obscode = t.obscode
         AND d.exposure_mjd_mid_key_us = t.exposure_mjd_mid_key_us
         AND d.healpixel = t.healpixel
        {lim}
        """
        res = client.query(q)
        rows = res.result_rows
        if not rows:
            return CandidateDetections.empty()
        return CandidateDetections.from_kwargs(
            orbit_id=[str(r[0]) for r in rows],
            target_idx=[int(r[1]) for r in rows],
            obscode=[str(r[2]) for r in rows],
            exposure_mjd_mid_key_us=[int(r[3]) for r in rows],
            healpixel=[int(r[4]) for r in rows],
            filter=[str(r[5]) if r[5] is not None else None for r in rows],
            observation_id=[str(r[6]) for r in rows],
            obstime_mjd_utc=[float(r[7]) for r in rows],
            ra_deg=[float(r[8]) for r in rows],
            dec_deg=[float(r[9]) for r in rows],
            ra_sigma_deg=[float(r[10]) for r in rows],
            dec_sigma_deg=[float(r[11]) for r in rows],
            mag=[None if r[12] is None else float(r[12]) for r in rows],
            mag_sigma=[None if r[13] is None else float(r[13]) for r in rows],
        )

    def count_accepted(
        self,
        *,
        subset: SubsetPaths,
        triples: PredictedTriples,
        preds: PredictedTargets,
        gate: GateParams,
    ) -> AcceptedCounts:
        cands = self.fetch_candidates(subset=subset, triples=triples, limit=None)
        return accepted_counts_python(candidates=cands, preds=preds, gate=gate)

    def filter_triples_to_existing_frames(
        self,
        *,
        subset: SubsetPaths,
        triples: PredictedTriples,
    ) -> PredictedTriples:
        # Default no-op for ClickHouse for now (can be implemented with a DISTINCT join).
        _ = subset
        return triples

    def frame_pixels_by_target(
        self,
        *,
        subset: SubsetPaths,  # unused
        targets: BenchTargets,  # unused
    ) -> dict[tuple[str, int], np.ndarray]:
        # Not yet implemented for ClickHouse.
        _ = subset
        _ = targets
        return {}


def clickhouse_create_detections_table_sql(*, table: str) -> str:
    """
    Helper SQL for a local ClickHouse detections table (MergeTree).
    """
    return f"""
    CREATE TABLE IF NOT EXISTS {table} (
      obscode String,
      year_month String,
      exposure_mjd_mid_utc Float64,
      exposure_mjd_mid_key_us Int64,
      filter String,
      healpixel Int64,
      observation_id String,
      obstime_mjd_utc Float64,
      ra_deg Float64,
      dec_deg Float64,
      ra_sigma_deg Float64,
      dec_sigma_deg Float64,
      mag Nullable(Float64),
      mag_sigma Nullable(Float64)
    )
    ENGINE = MergeTree
    PARTITION BY year_month
    ORDER BY (obscode, exposure_mjd_mid_key_us, healpixel)
    """

