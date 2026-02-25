from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from typing import Any

from ..pipeline_types import (
    AcceptedCounts,
    BenchTargets,
    CandidateDetections,
    PredictedTargets,
    PredictedTriples,
    SubsetPaths,
)
from .protocols import BackendCapabilities, GateParams, SearchBackend


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


_BQ_BYTES_RE = re.compile(r"process(?: upper bound of)?\s+([0-9,]+)\s+bytes", re.IGNORECASE)


def _estimate_bq_bytes(query: str) -> int:
    cmd = ["bq", "query", "--nouse_legacy_sql", "--dry_run", str(query)]
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    text = proc.stderr + "\n" + proc.stdout
    for line in str(text).splitlines():
        m = _BQ_BYTES_RE.search(line)
        if m:
            return int(m.group(1).replace(",", ""))
    return 0


@dataclass(frozen=True)
class BqDetectionsTableConfig:
    """
    BigQuery table/view containing detection rows.
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


def _wrap_delta_lon_deg_sql(lon_o: str, lon_p: str) -> str:
    return f"(MOD(({lon_o}) - ({lon_p}) + 180.0, 360.0) - 180.0)"


def _gate_expr_sql(*, gate: GateParams) -> str:
    default_floor_deg = float(gate.invalid_sigma_fill_floor_arcsec_global) / 3600.0
    by_code = gate.invalid_sigma_fill_floor_arcsec_by_obscode or {}
    if by_code:
        # BigQuery CASE ... WHEN obscode='X' THEN <deg> ... ELSE <default> END
        parts = [f"WHEN obscode = '{str(k)}' THEN {float(v)/3600.0}" for k, v in by_code.items()]
        floor_deg = "(CASE " + " ".join(parts) + f" ELSE {default_floor_deg} END)"
    else:
        floor_deg = str(default_floor_deg)
    n2 = float(gate.innovation_gate_n_sigma) ** 2

    dlon = _wrap_delta_lon_deg_sql("ra_deg", "pred_lon_deg")
    cos_lat = "IF(ABS(COS(RADIANS(pred_lat_deg))) > 1e-12, COS(RADIANS(pred_lat_deg)), 1e-12)"
    x = f"(({dlon}) * ({cos_lat}))"
    y = "(dec_deg - pred_lat_deg)"

    # Fill floor applies only when reported sigma is missing/invalid.
    sig_lon0 = f"(CASE WHEN ra_sigma_deg IS NOT NULL AND ra_sigma_deg > 0.0 THEN ra_sigma_deg ELSE {floor_deg} END)"
    sig_lat0 = f"(CASE WHEN dec_sigma_deg IS NOT NULL AND dec_sigma_deg > 0.0 THEN dec_sigma_deg ELSE {floor_deg} END)"

    sys_by_code = gate.sigma_systematic_arcsec_by_obscode or {}
    trig_by_code = gate.apply_systematic_if_reported_rms_lt_arcsec_by_obscode or {}
    if sys_by_code and trig_by_code:
        sys_parts = []
        trig_parts = []
        for code, sys_arc in sys_by_code.items():
            if code not in trig_by_code:
                continue
            sys_parts.append(f"WHEN obscode = '{str(code)}' THEN {float(sys_arc)}")
            trig_parts.append(f"WHEN obscode = '{str(code)}' THEN {float(trig_by_code[code])}")
        if sys_parts and trig_parts:
            sys_arcsec = "(CASE " + " ".join(sys_parts) + " ELSE 0.0 END)"
            trig_arcsec = "(CASE " + " ".join(trig_parts) + " ELSE -1e99 END)"
            rms_arcsec = f"(SQRT(POWER({sig_lon0}, 2) + POWER({sig_lat0}, 2)) * 3600.0)"
            apply_sys = f"(({sys_arcsec}) > 0.0 AND ({rms_arcsec}) < ({trig_arcsec}))"
            sig_sys_deg = f"(({sys_arcsec}) / 3600.0)"
            sig_lon = f"(CASE WHEN {apply_sys} THEN SQRT(POWER({sig_lon0}, 2) + POWER({sig_sys_deg}, 2)) ELSE {sig_lon0} END)"
            sig_lat = f"(CASE WHEN {apply_sys} THEN SQRT(POWER({sig_lat0}, 2) + POWER({sig_sys_deg}, 2)) ELSE {sig_lat0} END)"
        else:
            sig_lon = sig_lon0
            sig_lat = sig_lat0
    else:
        sig_lon = sig_lon0
        sig_lat = sig_lat0

    a_p = f"(({cos_lat}) * ({cos_lat}) * cov_ll_00)"
    d_p = "cov_ll_11"
    b_p = f"(({cos_lat}) * cov_ll_01)"

    var_x = f"POWER(({sig_lon}) * ({cos_lat}), 2)"
    var_y = f"POWER(({sig_lat}), 2)"

    a = f"(({a_p}) + ({var_x}))"
    d = f"(({d_p}) + ({var_y}))"
    b = f"({b_p})"
    det = f"(({a})*({d}) - ({b})*({b}))"
    chi2_num = f"(({x})*({x})*({d}) + ({y})*({y})*({a}) - 2.0*({x})*({y})*({b}))"
    chi2 = f"SAFE_DIVIDE(({chi2_num}), ({det}))"
    geo_ok = f"(({det}) > 0.0 AND IS_FINITE({chi2}) AND ({chi2}) <= {n2})"

    max_faint = gate.max_mag_residual_fainter_mag
    max_bright = gate.max_mag_residual_brighter_mag
    mag_ok = "TRUE"
    if max_faint is not None:
        mag_ok = f"({mag_ok} AND (mag IS NULL OR pred_mag IS NULL OR (mag - pred_mag) <= {float(max_faint)}))"
    if max_bright is not None:
        mag_ok = f"({mag_ok} AND (mag IS NULL OR pred_mag IS NULL OR (mag - pred_mag) >= {-float(max_bright)}))"
    return f"(({geo_ok}) AND ({mag_ok}))"


def _preds_unnest_sql(preds: PredictedTargets, *, include_pred_mag: bool = True) -> str:
    rows = []
    for i in range(len(preds)):
        head = (
            f"STRUCT('{preds.orbit_id[i].as_py()}' AS orbit_id, {int(preds.target_idx[i].as_py())} AS target_idx, "
            f"{float(preds.pred_lon_deg[i].as_py())} AS pred_lon_deg, {float(preds.pred_lat_deg[i].as_py())} AS pred_lat_deg, "
            f"{float(preds.cov_ll_00[i].as_py())} AS cov_ll_00, {float(preds.cov_ll_01[i].as_py())} AS cov_ll_01, {float(preds.cov_ll_11[i].as_py())} AS cov_ll_11"
        )
        if include_pred_mag:
            pm = preds.pred_mag[i].as_py()
            pm_sql = "CAST(NULL AS FLOAT64)" if pm is None else str(float(pm))
            rows.append(f"{head}, {pm_sql} AS pred_mag)")
        else:
            rows.append(f"{head})")
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
class BigQueryVirtualBackend(SearchBackend):
    """
    BigQuery "virtual" backend:
      - always generates SQL for each operation
      - supports dry-run byte estimates
      - optionally supports executing small aggregated queries (counts)
    """

    cfg: BqDetectionsTableConfig
    allow_execute: bool = False
    maximum_bytes_billed: int | None = None
    name: str = "bigquery_virtual"
    capabilities: BackendCapabilities = BackendCapabilities(
        supports_enumerate_targets=True,
        supports_sql_gate_counts=True,
        supports_sql_gate_rows=False,
        supports_filter_triples_to_existing_frames=False,
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
        stn_sql = ""
        if obscodes:
            stn_list = ", ".join(f"'{s}'" for s in obscodes)
            stn_sql = f" AND {self.cfg.col_obscode} IN ({stn_list})"
        q = f"""
        SELECT
          CAST({self.cfg.col_obscode} AS STRING) AS obscode,
          CAST({self.cfg.col_exposure_mjd_mid_utc} AS FLOAT64) AS exposure_mjd_mid_utc,
          CAST({self.cfg.col_exposure_mjd_mid_key_us} AS INT64) AS exposure_mjd_mid_key_us,
          CAST({self.cfg.col_filter} AS STRING) AS filter
        FROM `{self.cfg.table}`
        WHERE {self.cfg.col_exposure_mjd_mid_utc} >= {float(start_mjd_utc)}
          AND {self.cfg.col_exposure_mjd_mid_utc} <  {float(end_mjd_utc)}
          {stn_sql}
        GROUP BY obscode, exposure_mjd_mid_utc, exposure_mjd_mid_key_us, filter
        ORDER BY exposure_mjd_mid_utc ASC, obscode ASC, filter ASC
        """
        self.last_bytes_estimate = int(_estimate_bq_bytes(q))
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
        raise NotImplementedError(
            "BigQuery virtual backend does not fetch candidate rows (counts only)."
        )

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

        need_mag_residual_gate = (
            gate.max_mag_residual_fainter_mag is not None
            or gate.max_mag_residual_brighter_mag is not None
        )

        preds_sql = _preds_unnest_sql(preds, include_pred_mag=need_mag_residual_gate)
        triples_sql = _triples_unnest_sql(triples)
        gate_expr = _gate_expr_sql(gate=gate)

        mag_cols = ""
        if need_mag_residual_gate:
            mag_cols = (
                f",\n            CAST(d.{self.cfg.col_mag} AS FLOAT64) AS mag"
                ",\n            p.pred_mag AS pred_mag"
            )

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
            p.pred_lon_deg AS pred_lon_deg,
            p.pred_lat_deg AS pred_lat_deg,
            p.cov_ll_00 AS cov_ll_00,
            p.cov_ll_01 AS cov_ll_01,
            p.cov_ll_11 AS cov_ll_11{mag_cols}
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
        ORDER BY orbit_id ASC, target_idx ASC
        """
        self.last_bytes_estimate = int(_estimate_bq_bytes(q))
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

    def filter_triples_to_existing_frames(
        self,
        *,
        subset: SubsetPaths,  # unused
        triples: PredictedTriples,
    ) -> PredictedTriples:
        # No-op: this backend does not fetch rows and should not execute large DISTINCT joins.
        return triples

    def frame_pixels_by_target(
        self,
        *,
        subset: SubsetPaths,  # unused
        targets: BenchTargets,  # unused
    ) -> dict[tuple[str, int], np.ndarray]:
        # Deliberate no-op: this backend avoids executing large queries.
        _ = subset
        _ = targets
        return {}
