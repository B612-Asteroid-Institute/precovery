from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from typing import Any

from ...selection.bq_select import estimate_bq_bytes
from ..types import AcceptedCounts, BenchTargets, CandidateDetections, PredictedTargets, PredictedTriples, SubsetPaths
from .protocols import BackendCapabilities, BenchBackend, GateParams


@dataclass(frozen=True)
class BqDetectionsTableConfig:
    """
    BigQuery table/view containing detection rows.

    The backend assumes the table already contains precomputed join keys:
      - obscode
      - exposure_mjd_mid_key_us (INT64)  (derived from exposure_mjd_mid_utc)
      - healpixel (INT64) at the chosen nside
    and the detection-level columns required for innovation gating.
    """

    table: str = "moeyens-thor-dev.production_aims.aims"

    col_obscode: str = "obscode"
    col_exposure_mjd_mid_utc: str = "exposure_mjd_mid_utc"
    col_exposure_mjd_mid_key_us: str = "exposure_mjd_mid_key_us"
    col_filter: str = "filter"
    col_healpixel: str = "healpixel"

    col_observation_id: str = "observation_id"
    col_obstime_mjd_utc: str = "obstime_mjd_utc"
    col_ra_deg: str = "ra_deg"
    col_dec_deg: str = "dec_deg"
    col_ra_sigma_deg: str = "ra_sigma_deg"
    col_dec_sigma_deg: str = "dec_sigma_deg"
    col_mag: str = "mag"
    col_mag_sigma: str = "mag_sigma"


def _run_bq_query_json(
    query: str, *, max_rows: int = 100000, maximum_bytes_billed: int | None = None
) -> list[dict[str, Any]]:
    cmd = [
        "bq",
        "query",
        "--nouse_legacy_sql",
        "--format=json",
        f"--max_rows={int(max_rows)}",
    ]
    if maximum_bytes_billed is not None:
        cmd.append(f"--maximum_bytes_billed={int(maximum_bytes_billed)}")
    cmd.append(str(query))
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    out = proc.stdout.strip()
    if not out:
        return []
    return json.loads(out)


def _wrap_delta_lon_deg_sql(lon_o: str, lon_p: str) -> str:
    # Equivalent to: MOD((lon_o - lon_p + 180), 360) - 180
    return f"(MOD(({lon_o}) - ({lon_p}) + 180.0, 360.0) - 180.0)"


def _gate_expr_sql(*, gate: GateParams) -> str:
    """
    Returns a SQL boolean expression `chi2 <= n_sigma^2` in terms of column aliases:
      ra_deg, dec_deg, ra_sigma_deg, dec_sigma_deg,
      pred_lon_deg, pred_lat_deg, cov_ll_00, cov_ll_01, cov_ll_11
    """
    floor_deg = float(gate.det_sigma_floor_arcsec) / 3600.0
    n2 = float(gate.n_sigma) ** 2

    dlon = _wrap_delta_lon_deg_sql("ra_deg", "pred_lon_deg")
    cos_lat = "GREATEST(ABS(COS(RADIANS(pred_lat_deg))), 1e-12) * SIGN(COS(RADIANS(pred_lat_deg)))"
    # Use a stable cos(lat) with floor.
    cos_lat = f"IF(ABS(COS(RADIANS(pred_lat_deg))) > 1e-12, COS(RADIANS(pred_lat_deg)), 1e-12)"

    x = f"(({dlon}) * ({cos_lat}))"
    y = "(dec_deg - pred_lat_deg)"

    sig_lon = f"GREATEST(COALESCE(NULLIF(ra_sigma_deg, 0.0), 0.0), {floor_deg})"
    sig_lat = f"GREATEST(COALESCE(NULLIF(dec_sigma_deg, 0.0), 0.0), {floor_deg})"

    a_p = f"(({cos_lat}) * ({cos_lat}) * cov_ll_00)"
    d_p = "cov_ll_11"
    b_p = f"(({cos_lat}) * cov_ll_01)"

    var_x = f"POWER(({sig_lon}) * ({cos_lat}), 2)"
    var_y = f"POWER(({sig_lat}), 2)"

    a = f"(({a_p}) + ({var_x}))"
    d = f"(({d_p}) + ({var_y}))"
    b = f"({b_p})"
    det = f"(({a})*({d}) - ({b})*({b}))"
    chi2 = f"((({x})*({x})*({d}) + ({y})*({y})*({a}) - 2.0*({x})*({y})*({b})) / NULLIF(({det}), 0.0))"
    geo_ok = f"(({chi2}) <= {n2})"

    # Optional magnitude residual rejection.
    # mag_residual = mag - pred_mag
    max_faint = gate.max_mag_residual_fainter_mag
    max_bright = gate.max_mag_residual_brighter_mag
    mag_ok = "TRUE"
    if max_faint is not None:
        mag_ok = (
            f"({mag_ok} AND (mag IS NULL OR pred_mag IS NULL OR (mag - pred_mag) <= {float(max_faint)}))"
        )
    if max_bright is not None:
        mag_ok = (
            f"({mag_ok} AND (mag IS NULL OR pred_mag IS NULL OR (mag - pred_mag) >= {-float(max_bright)}))"
        )
    return f"(({geo_ok}) AND ({mag_ok}))"


def _preds_unnest_sql(preds: PredictedTargets) -> str:
    rows = []
    for i in range(len(preds)):
        pm = preds.pred_mag[i].as_py()
        pm_sql = "CAST(NULL AS FLOAT64)" if pm is None else str(float(pm))
        rows.append(
            f"STRUCT('{preds.orbit_id[i].as_py()}' AS orbit_id, {int(preds.target_idx[i].as_py())} AS target_idx, "
            f"{float(preds.pred_lon_deg[i].as_py())} AS pred_lon_deg, {float(preds.pred_lat_deg[i].as_py())} AS pred_lat_deg, "
            f"{float(preds.cov_ll_00[i].as_py())} AS cov_ll_00, {float(preds.cov_ll_01[i].as_py())} AS cov_ll_01, {float(preds.cov_ll_11[i].as_py())} AS cov_ll_11, "
            f"{pm_sql} AS pred_mag)"
        )
    if not rows:
        return "SELECT * FROM UNNEST([])"
    return "SELECT * FROM UNNEST([\n" + ",\n".join(rows) + "\n])"


def _triples_unnest_sql(triples: PredictedTriples) -> str:
    rows = []
    for i in range(len(triples)):
        rows.append(
            f"STRUCT('{triples.orbit_id[i].as_py()}' AS orbit_id, {int(triples.target_idx[i].as_py())} AS target_idx, "
            f"'{triples.obscode[i].as_py()}' AS obscode, {int(triples.exposure_mjd_mid_key_us[i].as_py())} AS exposure_mjd_mid_key_us, "
            f"{int(triples.healpixel[i].as_py())} AS healpixel)"
        )
    if not rows:
        return "SELECT * FROM UNNEST([])"
    return "SELECT * FROM UNNEST([\n" + ",\n".join(rows) + "\n])"


@dataclass
class BigQueryVirtualBackend(BenchBackend):
    """
    BigQuery "virtual" backend:
      - always generates SQL for each operation
      - supports dry-run byte estimates
      - supports executing *small* aggregated queries (counts) when allowed
    """

    cfg: BqDetectionsTableConfig
    allow_execute: bool = False
    maximum_bytes_billed: int | None = None
    name: str = "bigquery_virtual"
    capabilities: BackendCapabilities = BackendCapabilities(
        supports_enumerate_targets=True,
        supports_sql_gate_counts=True,
        supports_sql_gate_rows=False,
    )
    last_bytes_estimate: int = 0

    def enumerate_targets(
        self,
        *,
        subset: SubsetPaths,  # unused
        start_mjd_utc: float,
        end_mjd_utc: float,
        obscodes: tuple[str, ...],
    ) -> BenchTargets:
        if not obscodes:
            raise ValueError("obscodes cannot be empty")
        stn_list = ", ".join(f"'{s}'" for s in obscodes)
        q = f"""
        SELECT
          CAST({self.cfg.col_obscode} AS STRING) AS obscode,
          CAST({self.cfg.col_exposure_mjd_mid_utc} AS FLOAT64) AS exposure_mjd_mid_utc,
          CAST({self.cfg.col_exposure_mjd_mid_key_us} AS INT64) AS exposure_mjd_mid_key_us,
          CAST({self.cfg.col_filter} AS STRING) AS filter
        FROM `{self.cfg.table}`
        WHERE {self.cfg.col_exposure_mjd_mid_utc} >= {float(start_mjd_utc)}
          AND {self.cfg.col_exposure_mjd_mid_utc} <  {float(end_mjd_utc)}
          AND {self.cfg.col_obscode} IN ({stn_list})
        GROUP BY obscode, exposure_mjd_mid_utc, exposure_mjd_mid_key_us, filter
        ORDER BY exposure_mjd_mid_utc ASC, obscode ASC, filter ASC
        """
        self.last_bytes_estimate = int(estimate_bq_bytes(q))
        if not self.allow_execute:
            return BenchTargets.empty()
        rows = _run_bq_query_json(q, max_rows=200000, maximum_bytes_billed=self.maximum_bytes_billed)
        if not rows:
            return BenchTargets.empty()
        return BenchTargets.from_kwargs(
            obscode=[str(r["obscode"]) for r in rows],
            exposure_mjd_mid_utc=[float(r["exposure_mjd_mid_utc"]) for r in rows],
            exposure_mjd_mid_key_us=[int(r["exposure_mjd_mid_key_us"]) for r in rows],
            filter=[str(r["filter"]) for r in rows],
        )

    def fetch_candidates(
        self,
        *,
        subset: SubsetPaths,  # unused
        triples: PredictedTriples,
        limit: int | None = None,
    ) -> CandidateDetections:
        # Intentionally unsupported for now; fetching rows is expensive and unnecessary early.
        raise NotImplementedError("BigQuery virtual backend does not fetch candidate rows (counts only).")

    def count_accepted(
        self,
        *,
        subset: SubsetPaths,  # unused
        triples: PredictedTriples,
        preds: PredictedTargets,
        gate: GateParams,
    ) -> AcceptedCounts:
        if len(triples) == 0:
            return AcceptedCounts.empty()
        if len(preds) == 0:
            raise ValueError("preds cannot be empty when triples is non-empty")

        if len(triples) > 50_000:
            raise ValueError("Refusing to inline >50k triples into SQL; materialize to a temp table instead.")
        if len(preds) > 10_000:
            raise ValueError("Refusing to inline >10k preds into SQL; materialize to a temp table instead.")

        preds_sql = _preds_unnest_sql(preds)
        triples_sql = _triples_unnest_sql(triples)
        gate_expr = _gate_expr_sql(gate=gate)

        q = f"""
        WITH preds AS (
          {preds_sql}
        ),
        triples AS (
          {triples_sql}
        ),
        cands AS (
          SELECT
            t.orbit_id,
            t.target_idx,
            d.{self.cfg.col_observation_id} AS observation_id,
            CAST(d.{self.cfg.col_ra_deg} AS FLOAT64) AS ra_deg,
            CAST(d.{self.cfg.col_dec_deg} AS FLOAT64) AS dec_deg,
            CAST(d.{self.cfg.col_ra_sigma_deg} AS FLOAT64) AS ra_sigma_deg,
            CAST(d.{self.cfg.col_dec_sigma_deg} AS FLOAT64) AS dec_sigma_deg,
            CAST(d.{self.cfg.col_mag} AS FLOAT64) AS mag,
            p.pred_lon_deg AS pred_lon_deg,
            p.pred_lat_deg AS pred_lat_deg,
            p.cov_ll_00 AS cov_ll_00,
            p.cov_ll_01 AS cov_ll_01,
            p.cov_ll_11 AS cov_ll_11,
            p.pred_mag AS pred_mag
          FROM triples t
          INNER JOIN `{self.cfg.table}` d
            ON d.{self.cfg.col_obscode} = t.obscode
           AND d.{self.cfg.col_exposure_mjd_mid_key_us} = t.exposure_mjd_mid_key_us
           AND d.{self.cfg.col_healpixel} = t.healpixel
          INNER JOIN preds p
            ON p.orbit_id = t.orbit_id
           AND p.target_idx = t.target_idx
        )
        SELECT
          orbit_id,
          target_idx,
          COUNT(1) AS n_candidates,
          SUM(CASE WHEN {gate_expr} THEN 1 ELSE 0 END) AS n_accepted
        FROM cands
        GROUP BY orbit_id, target_idx
        """

        self.last_bytes_estimate = int(estimate_bq_bytes(q))
        if not self.allow_execute:
            return AcceptedCounts.empty()
        rows = _run_bq_query_json(q, max_rows=200000, maximum_bytes_billed=self.maximum_bytes_billed)
        if not rows:
            return AcceptedCounts.empty()
        return AcceptedCounts.from_kwargs(
            orbit_id=[str(r["orbit_id"]) for r in rows],
            target_idx=[int(r["target_idx"]) for r in rows],
            n_candidates=[int(r["n_candidates"]) for r in rows],
            n_accepted=[int(r["n_accepted"]) for r in rows],
        )

