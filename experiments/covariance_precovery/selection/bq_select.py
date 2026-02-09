from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Sequence

import quivr as qv


@dataclass(frozen=True)
class BqConfig:
    dataset_id: str = "moeyens-thor-dev.mpc_sbn_aurora"
    views_dataset_id: str = "moeyens-thor-dev.mpc_sbn_aurora_views"

    @property
    def obs_table(self) -> str:
        # Has columns: obstime (TIMESTAMP), stn (STRING), obsid (STRING), id (INTEGER), ra_f64, dec_f64, st_geo
        return f"{self.views_dataset_id}.public_obs_sbn_clustered"

    @property
    def orbits_table(self) -> str:
        # Has orbit summary columns + designation strings. The mpc_orb_jsonb can be used later for full details.
        return f"{self.dataset_id}.public_mpc_orbits"

    @property
    def obs_sbn_table(self) -> str:
        # Raw MPC observation table (includes permid/provid/orbit_id, but ra/dec as strings).
        return f"{self.dataset_id}.public_obs_sbn"

    @property
    def current_identifications_table(self) -> str:
        return f"{self.dataset_id}.public_current_identifications"

    @property
    def numbered_identifications_table(self) -> str:
        return f"{self.dataset_id}.public_numbered_identifications"


def _run_bq_json(
    query: str, *, dry_run: bool = False, max_rows: int = 100000
) -> list[dict[str, Any]]:
    """
    Run a BigQuery SQL query using the `bq` CLI and return JSON rows.
    """
    cmd = [
        "bq",
        "query",
        "--nouse_legacy_sql",
        "--format=json",
    ]
    if dry_run:
        cmd.append("--dry_run")
    else:
        # `bq query` defaults to 100 rows; experiments frequently need more.
        cmd.append(f"--max_rows={int(max_rows)}")
    cmd.append(query)

    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    out = proc.stdout.strip()
    if dry_run:
        # `bq query --dry_run` does not return rows in JSON format; callers should not request rows.
        return []
    if not out:
        return []
    return json.loads(out)


def estimate_bq_bytes(query: str) -> int:
    """
    Return BigQuery bytes processed estimate for a query (via `bq query --dry_run`).
    """
    cmd = ["bq", "query", "--nouse_legacy_sql", "--dry_run", query]
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return _parse_bq_bytes_processed(proc.stderr + "\n" + proc.stdout)


_BQ_BYTES_RE = re.compile(r"process(?: upper bound of)?\s+([0-9,]+)\s+bytes", re.IGNORECASE)


def _parse_bq_bytes_processed(text: str) -> int:
    """
    Parse the `bq query --dry_run` human output and extract bytes processed.

    Example formats observed:
    - "running this query will process 0 bytes of data."
    - "running this query will process upper bound of 2637618380 bytes of data."
    """
    for line in str(text).splitlines():
        m = _BQ_BYTES_RE.search(line)
        if m:
            return int(m.group(1).replace(",", ""))
    return 0


def bq_table_ref(*, project_id: str, dataset_id: str, table_id: str) -> str:
    """
    Return a `bq` CLI table reference like "project:dataset.table".
    """
    project_id = str(project_id).strip()
    dataset_id = str(dataset_id).strip()
    table_id = str(table_id).strip()
    if not project_id or not dataset_id or not table_id:
        raise ValueError("project_id, dataset_id, and table_id must be non-empty")
    return f"{project_id}:{dataset_id}.{table_id}"


def run_bq_query_to_table(
    *,
    query: str,
    destination_table: str,
    replace: bool = True,
) -> None:
    """
    Execute a Standard SQL query and write results to `destination_table`.

    `destination_table` should be a `bq` reference like "project:dataset.table".
    """
    cmd = [
        "bq",
        "query",
        "--nouse_legacy_sql",
        f"--destination_table={str(destination_table)}",
    ]
    if bool(replace):
        cmd.append("--replace")
    cmd.append(str(query))
    subprocess.run(cmd, check=True)


def export_bq_table_to_gcs_parquet(
    *,
    source_table: str,
    destination_uri: str,
    compression: str = "SNAPPY",
) -> None:
    """
    Export a BigQuery table to Parquet in GCS.

    `destination_uri` may include a wildcard (recommended), e.g.
    "gs://bucket/prefix/part-*.parquet".
    """
    cmd = [
        "bq",
        "extract",
        "--destination_format=PARQUET",
        f"--compression={str(compression)}",
        str(source_table),
        str(destination_uri),
    ]
    subprocess.run(cmd, check=True)


def _utc_ts(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt = dt.astimezone(timezone.utc)
    return dt.isoformat().replace("+00:00", "Z")


class ObservedObjectSummary(qv.Table):
    mpc_id = qv.Int64Column()  # MPC internal integer id in mpc_sbn_aurora
    stn = qv.LargeStringColumn()
    t_min_utc = qv.LargeStringColumn()
    t_max_utc = qv.LargeStringColumn()
    n_obs = qv.Int64Column()


def find_objects_observed_in_window(
    *,
    cfg: BqConfig,
    start_utc: datetime,
    end_utc: datetime,
    obscodes: Sequence[str],
    limit_per_stn: int | None = None,
    max_rows: int = 100000,
) -> ObservedObjectSummary:
    """
    Find MPC object IDs (`public_obs_sbn_clustered.id`) observed by any of the given stations
    in the UTC time window. Returns per-(id, stn) summaries.
    """
    if not obscodes:
        raise ValueError("obscodes cannot be empty")

    stn_list = ", ".join(f"'{s}'" for s in obscodes)
    t0 = _utc_ts(start_utc)
    t1 = _utc_ts(end_utc)

    limit_sql = f"QUALIFY ROW_NUMBER() OVER (PARTITION BY stn ORDER BY n_obs DESC) <= {int(limit_per_stn)}" if limit_per_stn else ""

    query = f"""
    WITH base AS (
      SELECT
        id AS mpc_id,
        stn,
        MIN(obstime) AS t_min,
        MAX(obstime) AS t_max,
        COUNT(1) AS n_obs
      FROM `{cfg.obs_table}`
      WHERE stn IN ({stn_list})
        AND obstime >= TIMESTAMP('{t0}')
        AND obstime <  TIMESTAMP('{t1}')
      GROUP BY mpc_id, stn
    )
    SELECT mpc_id, stn, CAST(t_min AS STRING) AS t_min_utc, CAST(t_max AS STRING) AS t_max_utc, n_obs
    FROM base
    {limit_sql}
    ORDER BY stn, n_obs DESC
    """

    rows = _run_bq_json(query, max_rows=max_rows)
    if not rows:
        return ObservedObjectSummary.empty()

    return ObservedObjectSummary.from_kwargs(
        mpc_id=[int(r["mpc_id"]) for r in rows],
        stn=[str(r["stn"]) for r in rows],
        t_min_utc=[str(r["t_min_utc"]) for r in rows],
        t_max_utc=[str(r["t_max_utc"]) for r in rows],
        n_obs=[int(r["n_obs"]) for r in rows],
    )


class ObservedDesignationSummary(qv.Table):
    """
    A more directly useful selector: returns MPC designations (permid/provid) per station.

    This avoids the ambiguity in the clustered view's integer `id` column, which is not
    the same ID space as `public_mpc_orbits.id`.
    """

    designation = qv.LargeStringColumn()
    stn = qv.LargeStringColumn()
    t_min_utc = qv.LargeStringColumn()
    t_max_utc = qv.LargeStringColumn()
    n_obs = qv.Int64Column()


def find_designations_observed_in_window(
    *,
    cfg: BqConfig,
    start_utc: datetime,
    end_utc: datetime,
    obscodes: Sequence[str],
    limit_per_stn: int | None = None,
    max_rows: int = 100000,
) -> ObservedDesignationSummary:
    """
    Find designations observed by station(s) in a time window using the raw MPC observation table.

    Uses `COALESCE(permid, provid)` as the object key.
    """
    if not obscodes:
        raise ValueError("obscodes cannot be empty")

    stn_list = ", ".join(f"'{s}'" for s in obscodes)
    t0 = _utc_ts(start_utc)
    t1 = _utc_ts(end_utc)

    limit_sql = (
        f"QUALIFY ROW_NUMBER() OVER (PARTITION BY stn ORDER BY n_obs DESC) <= {int(limit_per_stn)}"
        if limit_per_stn
        else ""
    )

    query = f"""
    WITH base AS (
      SELECT
        COALESCE(permid, provid) AS designation,
        stn,
        MIN(obstime) AS t_min,
        MAX(obstime) AS t_max,
        COUNT(1) AS n_obs
      FROM `{cfg.obs_sbn_table}`
      WHERE stn IN ({stn_list})
        AND obstime >= TIMESTAMP('{t0}')
        AND obstime <  TIMESTAMP('{t1}')
        AND COALESCE(permid, provid) IS NOT NULL
      GROUP BY designation, stn
    )
    SELECT designation, stn, CAST(t_min AS STRING) AS t_min_utc, CAST(t_max AS STRING) AS t_max_utc, n_obs
    FROM base
    {limit_sql}
    ORDER BY stn, n_obs DESC
    """

    rows = _run_bq_json(query, max_rows=max_rows)
    if not rows:
        return ObservedDesignationSummary.empty()

    return ObservedDesignationSummary.from_kwargs(
        designation=[str(r["designation"]) for r in rows],
        stn=[str(r["stn"]) for r in rows],
        t_min_utc=[str(r["t_min_utc"]) for r in rows],
        t_max_utc=[str(r["t_max_utc"]) for r in rows],
        n_obs=[int(r["n_obs"]) for r in rows],
    )


class OrbitSummary(qv.Table):
    mpc_id = qv.Int64Column()
    packed_primary_provisional_designation = qv.LargeStringColumn(nullable=True)
    unpacked_primary_provisional_designation = qv.LargeStringColumn(nullable=True)
    orbit_type_int = qv.Int64Column(nullable=True)
    u_param = qv.Int64Column(nullable=True)
    nopp = qv.Int64Column(nullable=True)
    arc_length_total = qv.Float64Column(nullable=True)
    nobs_total = qv.Int64Column(nullable=True)
    a = qv.Float64Column(nullable=True)
    e = qv.Float64Column(nullable=True)
    i = qv.Float64Column(nullable=True)
    q = qv.Float64Column(nullable=True)
    epoch_mjd = qv.Float64Column(nullable=True)


def fetch_orbit_summaries(cfg: BqConfig, mpc_ids: Sequence[int]) -> OrbitSummary:
    if not mpc_ids:
        return OrbitSummary.empty()

    # BigQuery IN lists have practical limits; chunk upstream if needed.
    id_list = ", ".join(str(int(i)) for i in mpc_ids)
    query = f"""
    SELECT
      id AS mpc_id,
      packed_primary_provisional_designation,
      unpacked_primary_provisional_designation,
      orbit_type_int,
      u_param,
      nopp,
      arc_length_total,
      nobs_total,
      a, e, i, q,
      epoch_mjd
    FROM `{cfg.orbits_table}`
    WHERE id IN ({id_list})
    """

    rows = _run_bq_json(query)
    if not rows:
        return OrbitSummary.empty()

    def _maybe_int(v: Any) -> int | None:
        return None if v is None else int(v)

    def _maybe_float(v: Any) -> float | None:
        return None if v is None else float(v)

    return OrbitSummary.from_kwargs(
        mpc_id=[int(r["mpc_id"]) for r in rows],
        packed_primary_provisional_designation=[
            r.get("packed_primary_provisional_designation") for r in rows
        ],
        unpacked_primary_provisional_designation=[
            r.get("unpacked_primary_provisional_designation") for r in rows
        ],
        orbit_type_int=[_maybe_int(r.get("orbit_type_int")) for r in rows],
        u_param=[_maybe_int(r.get("u_param")) for r in rows],
        nopp=[_maybe_int(r.get("nopp")) for r in rows],
        arc_length_total=[_maybe_float(r.get("arc_length_total")) for r in rows],
        nobs_total=[_maybe_int(r.get("nobs_total")) for r in rows],
        a=[_maybe_float(r.get("a")) for r in rows],
        e=[_maybe_float(r.get("e")) for r in rows],
        i=[_maybe_float(r.get("i")) for r in rows],
        q=[_maybe_float(r.get("q")) for r in rows],
        epoch_mjd=[_maybe_float(r.get("epoch_mjd")) for r in rows],
    )


class ObservedDesignationWindowSummary(qv.Table):
    """
    Per-designation observation summary over a time window, aggregated across stations.

    This is usually what we want for building a "wide" designation list to test precovery:
    it avoids multiplying rows by station while still retaining which stations were involved.
    """

    designation = qv.LargeStringColumn()
    t_min_utc = qv.LargeStringColumn()
    t_max_utc = qv.LargeStringColumn()
    n_obs = qv.Int64Column()
    n_stn = qv.Int64Column()
    stn_csv = qv.LargeStringColumn()


def find_designations_observed_any_station_in_window(
    *,
    cfg: BqConfig,
    start_utc: datetime,
    end_utc: datetime,
    obscodes: Sequence[str],
    max_rows: int = 100000,
    partition_mod: int | None = None,
    partition_idx: int | None = None,
) -> ObservedDesignationWindowSummary:
    """
    Find unique designations observed by the given station(s) in a time window, aggregated
    across stations.

    Uses `COALESCE(permid, provid)` as the object key.
    """
    if not obscodes:
        raise ValueError("obscodes cannot be empty")

    stn_list = ", ".join(f"'{s}'" for s in obscodes)
    t0 = _utc_ts(start_utc)
    t1 = _utc_ts(end_utc)

    partition_sql = ""
    if partition_mod is not None or partition_idx is not None:
        if partition_mod is None or partition_idx is None:
            raise ValueError("partition_mod and partition_idx must be provided together")
        if int(partition_mod) <= 0:
            raise ValueError("partition_mod must be > 0")
        if not (0 <= int(partition_idx) < int(partition_mod)):
            raise ValueError("partition_idx must satisfy 0 <= idx < mod")
        # Deterministic partitioning to avoid `bq query` row-limit truncation.
        partition_sql = (
            f"AND MOD(ABS(FARM_FINGERPRINT(COALESCE(permid, provid))), {int(partition_mod)}) = {int(partition_idx)}"
        )

    query = f"""
    WITH base AS (
      SELECT
        COALESCE(permid, provid) AS designation,
        MIN(obstime) AS t_min,
        MAX(obstime) AS t_max,
        COUNT(1) AS n_obs,
        COUNT(DISTINCT stn) AS n_stn,
        STRING_AGG(DISTINCT stn, ',' ORDER BY stn) AS stn_csv
      FROM `{cfg.obs_sbn_table}`
      WHERE stn IN ({stn_list})
        AND obstime >= TIMESTAMP('{t0}')
        AND obstime <  TIMESTAMP('{t1}')
        AND COALESCE(permid, provid) IS NOT NULL
        {partition_sql}
      GROUP BY designation
    )
    SELECT
      designation,
      CAST(t_min AS STRING) AS t_min_utc,
      CAST(t_max AS STRING) AS t_max_utc,
      n_obs,
      n_stn,
      stn_csv
    FROM base
    ORDER BY n_obs DESC
    """

    rows = _run_bq_json(query, max_rows=max_rows)
    if not rows:
        return ObservedDesignationWindowSummary.empty()

    return ObservedDesignationWindowSummary.from_kwargs(
        designation=[str(r["designation"]) for r in rows],
        t_min_utc=[str(r["t_min_utc"]) for r in rows],
        t_max_utc=[str(r["t_max_utc"]) for r in rows],
        n_obs=[int(r["n_obs"]) for r in rows],
        n_stn=[int(r["n_stn"]) for r in rows],
        stn_csv=[str(r["stn_csv"]) for r in rows],
    )


def count_designations_observed_any_station_in_window(
    *,
    cfg: BqConfig,
    start_utc: datetime,
    end_utc: datetime,
    obscodes: Sequence[str],
    partition_mod: int | None = None,
    partition_idx: int | None = None,
) -> int:
    """
    Return the number of unique designations observed by the given stations in the time window.

    This is intended to support deterministic partitioning so we can fetch full result sets
    without relying on `bq query` row limits.
    """
    if not obscodes:
        raise ValueError("obscodes cannot be empty")

    stn_list = ", ".join(f"'{s}'" for s in obscodes)
    t0 = _utc_ts(start_utc)
    t1 = _utc_ts(end_utc)

    partition_sql = ""
    if partition_mod is not None or partition_idx is not None:
        if partition_mod is None or partition_idx is None:
            raise ValueError("partition_mod and partition_idx must be provided together")
        if int(partition_mod) <= 0:
            raise ValueError("partition_mod must be > 0")
        if not (0 <= int(partition_idx) < int(partition_mod)):
            raise ValueError("partition_idx must satisfy 0 <= idx < mod")
        partition_sql = (
            f"AND MOD(ABS(FARM_FINGERPRINT(COALESCE(permid, provid))), {int(partition_mod)}) = {int(partition_idx)}"
        )

    query = f"""
    WITH base AS (
      SELECT
        COALESCE(permid, provid) AS designation
      FROM `{cfg.obs_sbn_table}`
      WHERE stn IN ({stn_list})
        AND obstime >= TIMESTAMP('{t0}')
        AND obstime <  TIMESTAMP('{t1}')
        AND COALESCE(permid, provid) IS NOT NULL
        {partition_sql}
      GROUP BY designation
    )
    SELECT COUNT(1) AS n
    FROM base
    """

    rows = _run_bq_json(query, max_rows=10)
    if not rows:
        return 0
    return int(rows[0]["n"])


class DesignationOrbitWindowFeatures(qv.Table):
    """
    Window-level designation features joined with orbit-summary metadata (when available).

    Designation normalization follows the same idea as `mpcq`:
    - use `public_current_identifications` to map secondary provisional designations to the
      primary provisional designation
    - use `public_numbered_identifications` to map a primary provisional designation to a
      numbered `permid` when available

    We then join `public_mpc_orbits` on `unpacked_primary_provisional_designation` (the
    primary provisional designation).
    """

    designation = qv.LargeStringColumn()
    t_min_utc = qv.LargeStringColumn()
    t_max_utc = qv.LargeStringColumn()
    n_obs_window = qv.Int64Column()
    n_stn_window = qv.Int64Column()
    stn_csv = qv.LargeStringColumn()

    mpc_id = qv.Int64Column(nullable=True)
    packed_primary_provisional_designation = qv.LargeStringColumn(nullable=True)
    unpacked_primary_provisional_designation = qv.LargeStringColumn(nullable=True)
    orbit_type_int = qv.Int64Column(nullable=True)
    u_param = qv.Int64Column(nullable=True)
    nopp = qv.Int64Column(nullable=True)
    arc_length_total = qv.Float64Column(nullable=True)
    nobs_total = qv.Int64Column(nullable=True)
    a = qv.Float64Column(nullable=True)
    e = qv.Float64Column(nullable=True)
    i = qv.Float64Column(nullable=True)
    q = qv.Float64Column(nullable=True)
    epoch_mjd = qv.Float64Column(nullable=True)


def _designation_orbit_features_query(
    *,
    cfg: BqConfig,
    start_utc: datetime,
    end_utc: datetime,
    obscodes: Sequence[str],
    partition_mod: int | None = None,
    partition_idx: int | None = None,
) -> str:
    if not obscodes:
        raise ValueError("obscodes cannot be empty")

    stn_list = ", ".join(f"'{s}'" for s in obscodes)
    t0 = _utc_ts(start_utc)
    t1 = _utc_ts(end_utc)

    partition_sql = ""
    if partition_mod is not None or partition_idx is not None:
        if partition_mod is None or partition_idx is None:
            raise ValueError("partition_mod and partition_idx must be provided together")
        if int(partition_mod) <= 0:
            raise ValueError("partition_mod must be > 0")
        if not (0 <= int(partition_idx) < int(partition_mod)):
            raise ValueError("partition_idx must satisfy 0 <= idx < mod")
        partition_sql = f"""
        WHERE MOD(ABS(FARM_FINGERPRINT(designation)), {int(partition_mod)}) = {int(partition_idx)}
        """

    return f"""
    WITH obs AS (
      SELECT
        permid,
        provid,
        stn,
        obstime
      FROM `{cfg.obs_sbn_table}`
      WHERE stn IN ({stn_list})
        AND obstime >= TIMESTAMP('{t0}')
        AND obstime <  TIMESTAMP('{t1}')
        AND (permid IS NOT NULL OR provid IS NOT NULL)
    ),
    mapped AS (
      SELECT
        o.*,
        ci.unpacked_primary_provisional_designation AS primary_provid_ci,
        ni_permid.unpacked_primary_provisional_designation AS primary_provid_num,
        ni_permid.permid AS permid_num,
        ni_ci.permid AS permid_from_ci,
        COALESCE(
          ni_permid.unpacked_primary_provisional_designation,
          ci.unpacked_primary_provisional_designation,
          o.provid
        ) AS primary_provid,
        COALESCE(ni_permid.permid, ni_ci.permid, o.permid) AS permid_norm,
        CASE
          WHEN COALESCE(ni_permid.permid, ni_ci.permid, o.permid) IS NOT NULL THEN COALESCE(ni_permid.permid, ni_ci.permid, o.permid)
          ELSE COALESCE(ci.unpacked_primary_provisional_designation, o.provid)
        END AS designation
      FROM obs o
      LEFT JOIN `{cfg.current_identifications_table}` AS ci
        ON ci.unpacked_secondary_provisional_designation = o.provid
        OR ci.unpacked_primary_provisional_designation = o.provid
      LEFT JOIN `{cfg.numbered_identifications_table}` AS ni_permid
        ON ni_permid.permid = o.permid
      LEFT JOIN `{cfg.numbered_identifications_table}` AS ni_ci
        ON ni_ci.unpacked_primary_provisional_designation = ci.unpacked_primary_provisional_designation
    ),
    win AS (
      SELECT
        designation,
        MIN(obstime) AS t_min,
        MAX(obstime) AS t_max,
        COUNT(1) AS n_obs_window,
        COUNT(DISTINCT stn) AS n_stn_window,
        STRING_AGG(DISTINCT stn, ',' ORDER BY stn) AS stn_csv,
        ANY_VALUE(primary_provid) AS primary_provid
      FROM mapped
      WHERE designation IS NOT NULL
      GROUP BY designation
    ),
    window_part AS (
      SELECT *
      FROM win
      {partition_sql}
    ),
    o AS (
      SELECT
        id AS mpc_id,
        packed_primary_provisional_designation,
        unpacked_primary_provisional_designation,
        orbit_type_int,
        u_param,
        nopp,
        arc_length_total,
        nobs_total,
        a, e, i, q,
        epoch_mjd
      FROM `{cfg.orbits_table}`
    )
    SELECT
      w.designation,
      CAST(w.t_min AS STRING) AS t_min_utc,
      CAST(w.t_max AS STRING) AS t_max_utc,
      w.n_obs_window,
      w.n_stn_window,
      w.stn_csv,
      o.mpc_id AS mpc_id,
      o.packed_primary_provisional_designation AS packed_primary_provisional_designation,
      o.unpacked_primary_provisional_designation AS unpacked_primary_provisional_designation,
      o.orbit_type_int AS orbit_type_int,
      o.u_param AS u_param,
      o.nopp AS nopp,
      COALESCE(o.arc_length_total, TIMESTAMP_DIFF(w.t_max, w.t_min, SECOND) / 86400.0) AS arc_length_total,
      o.nobs_total AS nobs_total,
      COALESCE(o.a, SAFE_DIVIDE(o.q, NULLIF(1.0 - o.e, 0.0))) AS a,
      o.e AS e,
      o.i AS i,
      o.q AS q,
      o.epoch_mjd AS epoch_mjd
    FROM window_part w
    LEFT JOIN o
      ON o.unpacked_primary_provisional_designation = w.primary_provid
    """


def materialize_designation_orbit_features_for_window_to_table(
    *,
    cfg: BqConfig,
    start_utc: datetime,
    end_utc: datetime,
    obscodes: Sequence[str],
    destination_table: str,
    replace: bool = True,
    partition_mod: int | None = None,
    partition_idx: int | None = None,
) -> None:
    """
    Materialize designation-orbit window features to a BigQuery destination table.
    """
    query = _designation_orbit_features_query(
        cfg=cfg,
        start_utc=start_utc,
        end_utc=end_utc,
        obscodes=obscodes,
        partition_mod=partition_mod,
        partition_idx=partition_idx,
    )
    run_bq_query_to_table(query=query, destination_table=destination_table, replace=replace)


def count_designation_orbit_features_for_window(
    *,
    cfg: BqConfig,
    start_utc: datetime,
    end_utc: datetime,
    obscodes: Sequence[str],
    partition_mod: int | None = None,
    partition_idx: int | None = None,
) -> int:
    if not obscodes:
        raise ValueError("obscodes cannot be empty")

    stn_list = ", ".join(f"'{s}'" for s in obscodes)
    t0 = _utc_ts(start_utc)
    t1 = _utc_ts(end_utc)

    partition_sql = ""
    if partition_mod is not None or partition_idx is not None:
        if partition_mod is None or partition_idx is None:
            raise ValueError("partition_mod and partition_idx must be provided together")
        if int(partition_mod) <= 0:
            raise ValueError("partition_mod must be > 0")
        if not (0 <= int(partition_idx) < int(partition_mod)):
            raise ValueError("partition_idx must satisfy 0 <= idx < mod")
        partition_sql = f"""
        WHERE MOD(ABS(FARM_FINGERPRINT(designation)), {int(partition_mod)}) = {int(partition_idx)}
        """

    query = f"""
    WITH obs AS (
      SELECT
        permid,
        provid
      FROM `{cfg.obs_sbn_table}`
      WHERE stn IN ({stn_list})
        AND obstime >= TIMESTAMP('{t0}')
        AND obstime <  TIMESTAMP('{t1}')
        AND (permid IS NOT NULL OR provid IS NOT NULL)
    ),
    mapped AS (
      SELECT DISTINCT
        CASE
          WHEN COALESCE(ni_permid.permid, ni_ci.permid, o.permid) IS NOT NULL THEN COALESCE(ni_permid.permid, ni_ci.permid, o.permid)
          ELSE COALESCE(ci.unpacked_primary_provisional_designation, o.provid)
        END AS designation
      FROM obs o
      LEFT JOIN `{cfg.current_identifications_table}` AS ci
        ON ci.unpacked_secondary_provisional_designation = o.provid
        OR ci.unpacked_primary_provisional_designation = o.provid
      LEFT JOIN `{cfg.numbered_identifications_table}` AS ni_permid
        ON ni_permid.permid = o.permid
      LEFT JOIN `{cfg.numbered_identifications_table}` AS ni_ci
        ON ni_ci.unpacked_primary_provisional_designation = ci.unpacked_primary_provisional_designation
      WHERE (o.permid IS NOT NULL OR o.provid IS NOT NULL)
    ),
    window_part AS (
      SELECT designation
      FROM mapped
      WHERE designation IS NOT NULL
      {partition_sql}
    )
    SELECT COUNT(1) AS n FROM window_part
    """

    rows = _run_bq_json(query, max_rows=10)
    if not rows:
        return 0
    return int(rows[0]["n"])


def fetch_designation_orbit_features_for_window(
    *,
    cfg: BqConfig,
    start_utc: datetime,
    end_utc: datetime,
    obscodes: Sequence[str],
    max_rows: int = 100000,
    partition_mod: int | None = None,
    partition_idx: int | None = None,
) -> DesignationOrbitWindowFeatures:
    if not obscodes:
        raise ValueError("obscodes cannot be empty")

    query = _designation_orbit_features_query(
        cfg=cfg,
        start_utc=start_utc,
        end_utc=end_utc,
        obscodes=obscodes,
        partition_mod=partition_mod,
        partition_idx=partition_idx,
    )

    rows = _run_bq_json(query, max_rows=max_rows)
    if not rows:
        return DesignationOrbitWindowFeatures.empty()

    def _maybe_int(v: Any) -> int | None:
        return None if v is None else int(v)

    def _maybe_float(v: Any) -> float | None:
        return None if v is None else float(v)

    return DesignationOrbitWindowFeatures.from_kwargs(
        designation=[str(r["designation"]) for r in rows],
        t_min_utc=[str(r["t_min_utc"]) for r in rows],
        t_max_utc=[str(r["t_max_utc"]) for r in rows],
        n_obs_window=[int(r["n_obs_window"]) for r in rows],
        n_stn_window=[int(r["n_stn_window"]) for r in rows],
        stn_csv=[str(r["stn_csv"]) for r in rows],
        mpc_id=[_maybe_int(r.get("mpc_id")) for r in rows],
        packed_primary_provisional_designation=[
            r.get("packed_primary_provisional_designation") for r in rows
        ],
        unpacked_primary_provisional_designation=[
            r.get("unpacked_primary_provisional_designation") for r in rows
        ],
        orbit_type_int=[_maybe_int(r.get("orbit_type_int")) for r in rows],
        u_param=[_maybe_int(r.get("u_param")) for r in rows],
        nopp=[_maybe_int(r.get("nopp")) for r in rows],
        arc_length_total=[_maybe_float(r.get("arc_length_total")) for r in rows],
        nobs_total=[_maybe_int(r.get("nobs_total")) for r in rows],
        a=[_maybe_float(r.get("a")) for r in rows],
        e=[_maybe_float(r.get("e")) for r in rows],
        i=[_maybe_float(r.get("i")) for r in rows],
        q=[_maybe_float(r.get("q")) for r in rows],
        epoch_mjd=[_maybe_float(r.get("epoch_mjd")) for r in rows],
    )

