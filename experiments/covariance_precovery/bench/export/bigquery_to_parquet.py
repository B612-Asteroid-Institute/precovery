from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Sequence

from ...selection.bq_select import (
    estimate_bq_bytes,
    export_bq_table_to_gcs_parquet,
    run_bq_query_to_table,
)


@dataclass(frozen=True)
class BqDetectionsExportConfig:
    """
    Configuration for exporting detections from BigQuery into parquet for local backends.

    Notes
    -----
    This is intentionally flexible because the exact schema of
    `moeyens-thor-dev.production_aims.aims` may evolve.
    """

    source_table: str = "moeyens-thor-dev.production_aims.aims"

    # Column names in the source table.
    col_obscode: str = "obscode"
    col_obstime: str = "obstime"  # TIMESTAMP
    col_obsid: str = "observation_id"
    col_ra_deg: str = "ra_deg"
    col_dec_deg: str = "dec_deg"
    col_ra_sigma_deg: str = "ra_sigma_deg"
    col_dec_sigma_deg: str = "dec_sigma_deg"
    col_mag: str = "mag"
    col_mag_sigma: str = "mag_sigma"
    col_healpixel: str = "healpix_n32"  # expected precomputed nested healpix at nside=32
    col_exposure_mjd_mid: str = "exposure_mjd_mid"  # float, if available; else will be derived upstream


def _utc_ts(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt = dt.astimezone(timezone.utc)
    return dt.isoformat().replace("+00:00", "Z")


def build_export_query(
    *,
    cfg: BqDetectionsExportConfig,
    start_utc: datetime,
    end_utc: datetime,
    obscodes: Sequence[str],
) -> str:
    if not obscodes:
        raise ValueError("obscodes cannot be empty")
    stn_list = ", ".join(f"'{str(s)}'" for s in obscodes)
    t0 = _utc_ts(start_utc)
    t1 = _utc_ts(end_utc)

    # We compute an integer time key in SQL to avoid float equality joins downstream.
    # key_us = round(exposure_mjd_mid * 86400 * 1e6)
    return f"""
    SELECT
      CAST({cfg.col_obscode} AS STRING) AS obscode,
      CAST({cfg.col_exposure_mjd_mid} AS FLOAT64) AS exposure_mjd_mid_utc,
      CAST(ROUND(CAST({cfg.col_exposure_mjd_mid} AS FLOAT64) * 86400.0 * 1e6) AS INT64) AS exposure_mjd_mid_key_us,
      CAST({cfg.col_healpixel} AS INT64) AS healpixel,
      CAST({cfg.col_obsid} AS STRING) AS observation_id,
      -- time in MJD UTC (for Python gating); requires obstime present
      (UNIX_MICROS({cfg.col_obstime}) / 86400.0 / 1e6 + 40587.0) AS obstime_mjd_utc,
      CAST({cfg.col_ra_deg} AS FLOAT64) AS ra_deg,
      CAST({cfg.col_dec_deg} AS FLOAT64) AS dec_deg,
      CAST({cfg.col_ra_sigma_deg} AS FLOAT64) AS ra_sigma_deg,
      CAST({cfg.col_dec_sigma_deg} AS FLOAT64) AS dec_sigma_deg,
      CAST({cfg.col_mag} AS FLOAT64) AS mag,
      CAST({cfg.col_mag_sigma} AS FLOAT64) AS mag_sigma
    FROM `{cfg.source_table}`
    WHERE {cfg.col_obscode} IN ({stn_list})
      AND {cfg.col_obstime} >= TIMESTAMP('{t0}')
      AND {cfg.col_obstime} <  TIMESTAMP('{t1}')
    """


def export_bigquery_detections_to_parquet(
    *,
    cfg: BqDetectionsExportConfig,
    start_utc: datetime,
    end_utc: datetime,
    obscodes: Sequence[str],
    destination_table: str,
    gcs_parquet_uri: str,
    replace: bool = True,
    max_bytes_estimate: int | None = None,
) -> int:
    """
    Export a window of detections to parquet via:
      (1) CTAS to a destination table
      (2) `bq extract` to parquet in GCS

    Returns the dry-run bytes estimate.

    This function does not enforce `maximumBytesBilled` directly because the `bq` CLI
    flag differs across environments; the harness should call `estimate_bq_bytes` first
    and abort if it is too large.
    """
    q = build_export_query(cfg=cfg, start_utc=start_utc, end_utc=end_utc, obscodes=obscodes)
    est = int(estimate_bq_bytes(q))
    if max_bytes_estimate is not None and int(est) > int(max_bytes_estimate):
        raise RuntimeError(
            f"Refusing export: estimated bytes {est} exceeds max_bytes_estimate={int(max_bytes_estimate)}"
        )

    run_bq_query_to_table(query=q, destination_table=str(destination_table), replace=bool(replace))
    export_bq_table_to_gcs_parquet(source_table=str(destination_table), destination_uri=str(gcs_parquet_uri))
    return int(est)

