from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc

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
class DuckDbParquetBackend(SearchBackend):
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
        supports_filter_triples_to_existing_frames=True,
        supports_frame_pixels_by_target=True,
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
        con = self._conn()
        try:
            p = _sql_quote(str(self.parquet_path))
            con.execute(f"CREATE OR REPLACE VIEW det AS SELECT * FROM read_parquet({p})")
            where = "WHERE exposure_mjd_mid_utc >= ? AND exposure_mjd_mid_utc < ?"
            params: list[object] = [float(start_mjd_utc), float(end_mjd_utc)]
            if obscodes:
                qs = ",".join(["?"] * len(obscodes))
                where += f" AND obscode IN ({qs})"
                params.extend(list(map(str, obscodes)))
            q = f"""
            SELECT DISTINCT obscode, exposure_mjd_mid_utc, exposure_mjd_mid_key_us, filter
            FROM det
            {where}
            ORDER BY exposure_mjd_mid_utc ASC, obscode ASC
            """
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

    def filter_triples_to_existing_frames(
        self,
        *,
        subset: SubsetPaths,  # unused
        triples: PredictedTriples,
    ) -> PredictedTriples:
        if len(triples) == 0:
            return PredictedTriples.empty()

        con = self._conn()
        try:
            p = _sql_quote(str(self.parquet_path))
            con.execute(f"CREATE OR REPLACE VIEW det AS SELECT * FROM read_parquet({p})")
            con.register(
                "triples",
                triples.table.select(
                    [
                        "orbit_id",
                        "target_idx",
                        "obscode",
                        "exposure_mjd_mid_utc",
                        "exposure_mjd_mid_key_us",
                        "healpixel",
                    ]
                ),
            )
            q = """
            SELECT DISTINCT
              t.orbit_id,
              t.target_idx,
              t.obscode,
              t.exposure_mjd_mid_utc,
              t.exposure_mjd_mid_key_us,
              t.healpixel
            FROM triples t
            INNER JOIN det d
              ON d.obscode = t.obscode
             AND d.exposure_mjd_mid_key_us = t.exposure_mjd_mid_key_us
             AND d.healpixel = t.healpixel
            """
            out = con.execute(q).fetch_arrow_table()
        finally:
            con.close()

        if out.num_rows == 0:
            return PredictedTriples.empty()
        return PredictedTriples.from_pyarrow(out)

    def count_accepted(
        self,
        *,
        subset: SubsetPaths,  # unused
        triples: PredictedTriples,
        preds: PredictedTargets,
        gate: GateParams,
    ) -> AcceptedCounts:
        cands = self.fetch_candidates(subset=subset, triples=triples, limit=None)
        return accepted_counts_python(candidates=cands, preds=preds, gate=gate)

    # Optional dataset-level helpers (used by the benchmark harness; harmless in production).
    def frame_pixels_by_target(
        self,
        *,
        subset: SubsetPaths,  # unused
        targets: BenchTargets,
    ) -> dict[tuple[str, int], "np.ndarray"]:
        if len(targets) == 0:
            return {}

        import numpy as np

        con = self._conn()
        try:
            p = _sql_quote(str(self.parquet_path))
            con.execute(
                f"CREATE OR REPLACE VIEW det AS "
                f"SELECT obscode, exposure_mjd_mid_key_us, healpixel FROM read_parquet({p})"
            )
            con.register(
                "targets",
                targets.table.select(["obscode", "exposure_mjd_mid_key_us"]),
            )
            q = """
            SELECT
              d.obscode,
              d.exposure_mjd_mid_key_us,
              LIST(DISTINCT d.healpixel) AS healpixels
            FROM det d
            INNER JOIN targets t
              ON d.obscode = t.obscode
             AND d.exposure_mjd_mid_key_us = t.exposure_mjd_mid_key_us
            GROUP BY d.obscode, d.exposure_mjd_mid_key_us
            """
            frames = con.execute(q).fetch_arrow_table()
        finally:
            con.close()

        if frames.num_rows == 0:
            return {}

        oc = pc.cast(frames["obscode"], pa.large_string()).to_pylist()
        key = pc.cast(frames["exposure_mjd_mid_key_us"], pa.int64()).to_pylist()
        hp_lists = frames["healpixels"].to_pylist()

        out: dict[tuple[str, int], np.ndarray] = {}
        for o, k, hps in zip(oc, key, hp_lists):
            if o is None or k is None or not hps:
                continue
            arr = np.asarray(hps, dtype=np.int64)
            if arr.size == 0:
                continue
            out[(str(o), int(k))] = arr
        return out

    def count_total_window_frames(
        self,
        *,
        start_mjd_utc: float,
        end_mjd_utc: float,
        obscodes: tuple[str, ...],
    ) -> int:
        con = self._conn()
        try:
            p = _sql_quote(str(self.parquet_path))
            con.execute(f"CREATE OR REPLACE VIEW det AS SELECT * FROM read_parquet({p})")
            where = "WHERE exposure_mjd_mid_utc >= ? AND exposure_mjd_mid_utc < ?"
            params: list[object] = [float(start_mjd_utc), float(end_mjd_utc)]
            if obscodes:
                qs = ",".join(["?"] * len(obscodes))
                where += f" AND obscode IN ({qs})"
                params.extend(list(map(str, obscodes)))
            q = f"""
            SELECT
              COUNT(DISTINCT (obscode, exposure_mjd_mid_key_us, healpixel)) AS n
            FROM det
            {where}
            """
            row = con.execute(q, params).fetchone()
        finally:
            con.close()
        if row is None or row[0] is None:
            return 0
        return int(row[0])

    def count_matched_frame_keys(self, *, triples: PredictedTriples) -> int:
        if len(triples) == 0:
            return 0
        con = self._conn()
        try:
            p = _sql_quote(str(self.parquet_path))
            con.execute(f"CREATE OR REPLACE VIEW det AS SELECT * FROM read_parquet({p})")
            con.register(
                "triples",
                triples.table.select(["obscode", "exposure_mjd_mid_key_us", "healpixel"]),
            )
            q = """
            SELECT
              COUNT(DISTINCT (t.obscode, t.exposure_mjd_mid_key_us, t.healpixel)) AS n
            FROM triples t
            INNER JOIN det d
              ON d.obscode = t.obscode
             AND d.exposure_mjd_mid_key_us = t.exposure_mjd_mid_key_us
             AND d.healpixel = t.healpixel
            """
            row = con.execute(q).fetchone()
        finally:
            con.close()
        if row is None or row[0] is None:
            return 0
        return int(row[0])

    def truth_frame_keys_from_truth_table(self, *, truth: pa.Table) -> pa.Table:
        if truth.num_rows == 0:
            return pa.table(
                {
                    "orbit_id": pa.array([], type=pa.large_string()),
                    "obscode": pa.array([], type=pa.large_string()),
                    "exposure_mjd_mid_key_us": pa.array([], type=pa.int64()),
                    "healpixel": pa.array([], type=pa.int64()),
                    "observation_id": pa.array([], type=pa.large_string()),
                }
            )

        need = {"orbit_id", "observation_id", "obscode"}
        missing = sorted(need - set(truth.column_names))
        if missing:
            raise ValueError(f"truth table missing required columns: {missing}")

        con = self._conn()
        try:
            p = _sql_quote(str(self.parquet_path))
            con.execute(f"CREATE OR REPLACE VIEW det AS SELECT * FROM read_parquet({p})")
            con.register("truth", truth.select(["orbit_id", "obscode", "observation_id"]))
            q = """
            SELECT
              t.orbit_id,
              t.obscode,
              d.exposure_mjd_mid_key_us,
              d.healpixel,
              d.observation_id
            FROM truth t
            INNER JOIN det d
              ON d.obscode = t.obscode
             AND d.observation_id = t.observation_id
            """
            out = con.execute(q).fetch_arrow_table()
        finally:
            con.close()
        if out.num_rows == 0:
            return pa.table(
                {
                    "orbit_id": pa.array([], type=pa.large_string()),
                    "obscode": pa.array([], type=pa.large_string()),
                    "exposure_mjd_mid_key_us": pa.array([], type=pa.int64()),
                    "healpixel": pa.array([], type=pa.int64()),
                    "observation_id": pa.array([], type=pa.large_string()),
                }
            )
        # Ensure stable column types/order.
        return pa.table(
            {
                "orbit_id": pc.cast(out["orbit_id"], pa.large_string()),
                "obscode": pc.cast(out["obscode"], pa.large_string()),
                "exposure_mjd_mid_key_us": pc.cast(out["exposure_mjd_mid_key_us"], pa.int64()),
                "healpixel": pc.cast(out["healpixel"], pa.int64()),
                "observation_id": pc.cast(out["observation_id"], pa.large_string()),
            }
        )

