from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Sequence

import pyarrow as pa
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
    # Output is not stable JSON; parse the known line containing bytes processed.
    for line in proc.stderr.splitlines() + proc.stdout.splitlines():
        if "bytes processed" in line.lower():
            # Example: "Query successfully validated. Assuming the tables are not modified, running this query will process 12345 bytes."
            tokens = [t for t in line.replace(",", "").split() if t.isdigit()]
            if tokens:
                return int(tokens[-1])
    return 0


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

    The join attempts multiple keys against `public_mpc_orbits`:
    - `unpacked_primary_provisional_designation == designation`
    - `packed_primary_provisional_designation == designation`
    - `CAST(id AS STRING) == designation` (handles numeric permids like "433")
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
    WITH win AS (
      SELECT
        COALESCE(permid, provid) AS designation,
        MIN(obstime) AS t_min,
        MAX(obstime) AS t_max,
        COUNT(1) AS n_obs_window,
        COUNT(DISTINCT stn) AS n_stn_window,
        STRING_AGG(DISTINCT stn, ',' ORDER BY stn) AS stn_csv
      FROM `{cfg.obs_sbn_table}`
      WHERE stn IN ({stn_list})
        AND obstime >= TIMESTAMP('{t0}')
        AND obstime <  TIMESTAMP('{t1}')
        AND COALESCE(permid, provid) IS NOT NULL
      GROUP BY designation
    ),
    window_part AS (
      SELECT *
      FROM win
      {partition_sql}
    )
    SELECT COUNT(1) AS n
    FROM window_part
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
    WITH win AS (
      SELECT
        COALESCE(permid, provid) AS designation,
        MIN(obstime) AS t_min,
        MAX(obstime) AS t_max,
        COUNT(1) AS n_obs_window,
        COUNT(DISTINCT stn) AS n_stn_window,
        STRING_AGG(DISTINCT stn, ',' ORDER BY stn) AS stn_csv
      FROM `{cfg.obs_sbn_table}`
      WHERE stn IN ({stn_list})
        AND obstime >= TIMESTAMP('{t0}')
        AND obstime <  TIMESTAMP('{t1}')
        AND COALESCE(permid, provid) IS NOT NULL
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
      COALESCE(ou.mpc_id, op.mpc_id, oi.mpc_id) AS mpc_id,
      COALESCE(ou.packed_primary_provisional_designation, op.packed_primary_provisional_designation, oi.packed_primary_provisional_designation) AS packed_primary_provisional_designation,
      COALESCE(ou.unpacked_primary_provisional_designation, op.unpacked_primary_provisional_designation, oi.unpacked_primary_provisional_designation) AS unpacked_primary_provisional_designation,
      COALESCE(ou.orbit_type_int, op.orbit_type_int, oi.orbit_type_int) AS orbit_type_int,
      COALESCE(ou.u_param, op.u_param, oi.u_param) AS u_param,
      COALESCE(ou.nopp, op.nopp, oi.nopp) AS nopp,
      COALESCE(ou.arc_length_total, op.arc_length_total, oi.arc_length_total) AS arc_length_total,
      COALESCE(ou.nobs_total, op.nobs_total, oi.nobs_total) AS nobs_total,
      COALESCE(ou.a, op.a, oi.a) AS a,
      COALESCE(ou.e, op.e, oi.e) AS e,
      COALESCE(ou.i, op.i, oi.i) AS i,
      COALESCE(ou.q, op.q, oi.q) AS q,
      COALESCE(ou.epoch_mjd, op.epoch_mjd, oi.epoch_mjd) AS epoch_mjd
    FROM window_part w
    LEFT JOIN o AS ou
      ON ou.unpacked_primary_provisional_designation = w.designation
    LEFT JOIN o AS op
      ON op.packed_primary_provisional_designation = w.designation
    LEFT JOIN o AS oi
      ON CAST(oi.mpc_id AS STRING) = w.designation
    """

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

