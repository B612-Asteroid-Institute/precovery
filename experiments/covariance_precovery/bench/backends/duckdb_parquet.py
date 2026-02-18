from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa

from ..gate import accepted_counts_python
from ..types import AcceptedCounts, BenchTargets, CandidateDetections, PredictedTargets, PredictedTriples, SubsetPaths
from ..time_key import mjd_to_time_key_us
from .protocols import BackendCapabilities, BenchBackend, GateParams


def _require_duckdb():
    try:
        import duckdb  # type: ignore

        return duckdb
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "DuckDB backend requires `duckdb` dependency. Install with `pdm add duckdb`."
        ) from e


def _sql_quote(s: str) -> str:
    # DuckDB uses single quotes for string literals; escape by doubling.
    return "'" + str(s).replace("'", "''") + "'"


@dataclass
class DuckDbParquetBackend(BenchBackend):
    """
    DuckDB backend over a parquet dataset.

    Expected parquet columns (minimum):
      obscode: STRING
      exposure_mjd_mid_utc: DOUBLE
      exposure_mjd_mid_key_us: BIGINT
      filter: STRING
      healpixel: BIGINT
      observation_id: STRING
      obstime_mjd_utc: DOUBLE
      ra_deg/dec_deg: DOUBLE
      ra_sigma_deg/dec_sigma_deg: DOUBLE
      mag/mag_sigma: DOUBLE (nullable)
    """

    parquet_path: Path
    name: str = "duckdb_parquet"
    capabilities: BackendCapabilities = BackendCapabilities(
        supports_enumerate_targets=True,
        supports_sql_gate_counts=False,
        supports_sql_gate_rows=False,
    )
    duckdb_path: Path | None = None

    def _conn(self):
        duckdb = _require_duckdb()
        if self.duckdb_path is None:
            return duckdb.connect(database=":memory:")
        self.duckdb_path.parent.mkdir(parents=True, exist_ok=True)
        return duckdb.connect(database=str(self.duckdb_path))

    def enumerate_targets(
        self,
        *,
        subset: SubsetPaths,  # unused (parquet-backed)
        start_mjd_utc: float,
        end_mjd_utc: float,
        obscodes: tuple[str, ...],
    ) -> BenchTargets:
        if not obscodes:
            raise ValueError("obscodes cannot be empty")
        con = self._conn()
        try:
            p = _sql_quote(str(self.parquet_path))
            con.execute(f"CREATE OR REPLACE VIEW det AS SELECT * FROM read_parquet({p})")
            qs = ",".join(["?"] * len(obscodes))
            q = f"""
            SELECT DISTINCT obscode, exposure_mjd_mid_utc, exposure_mjd_mid_key_us, filter
            FROM det
            WHERE exposure_mjd_mid_utc >= ?
              AND exposure_mjd_mid_utc < ?
              AND obscode IN ({qs})
            ORDER BY exposure_mjd_mid_utc ASC, obscode ASC
            """
            params = [float(start_mjd_utc), float(end_mjd_utc), *list(map(str, obscodes))]
            t = con.execute(q, params).fetch_arrow_table()
        finally:
            con.close()
        if t.num_rows == 0:
            return BenchTargets.empty()
        return BenchTargets.from_pyarrow(t)

    def fetch_candidates(
        self,
        *,
        subset: SubsetPaths,  # unused
        triples: PredictedTriples,
        limit: int | None = None,
    ) -> CandidateDetections:
        if len(triples) == 0:
            return CandidateDetections.empty()

        con = self._conn()
        try:
            p = _sql_quote(str(self.parquet_path))
            con.execute(f"CREATE OR REPLACE VIEW det AS SELECT * FROM read_parquet({p})")
            con.register("triples", triples.table)
            lim = "" if limit is None else f"LIMIT {int(limit)}"
            q = f"""
            SELECT
              t.orbit_id,
              t.target_idx,
              d.observation_id,
              d.obstime_mjd_utc,
              d.ra_deg,
              d.dec_deg,
              d.ra_sigma_deg,
              d.dec_sigma_deg,
              d.mag,
              d.mag_sigma
            FROM triples t
            INNER JOIN det d
              ON d.obscode = t.obscode
             AND d.exposure_mjd_mid_key_us = t.exposure_mjd_mid_key_us
             AND d.healpixel = t.healpixel
            {lim}
            """
            out = con.execute(q).fetch_arrow_table()
        finally:
            con.close()

        if out.num_rows == 0:
            return CandidateDetections.empty()
        return CandidateDetections.from_pyarrow(out)

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

