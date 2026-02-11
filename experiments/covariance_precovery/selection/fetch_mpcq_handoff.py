"""Fetch MPCOrbits and MPCObservations via mpcq for a designation list and persist to parquet for handoff."""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from mpcq.client import BigQueryMPCClient
from mpcq.observations import MPCObservations
from mpcq.orbits import MPCOrbits

from .bq_select import BqConfig
from .subset_sampling import SamplingConfig


@dataclass(frozen=True)
class HandoffResult:
    out_dir: Path
    designations_parquet: Path
    designations_copy: Path
    mpcq_orbits_parquet: Path
    mpcq_observations_parquet: Path
    meta_json: Path
    report_md: Path
    n_designations: int
    n_orbits: int
    n_observations: int


# Pattern for "Selected designations": `.../artifacts/<run_tag>/selected_designations.parquet`
_SELECTED_DESIGNATIONS_PATTERN = re.compile(
    r"selected designations[^\n]*`(\.\.\./)?(artifacts/[^`]+/selected_designations\.parquet)`",
    re.IGNORECASE,
)


def designations_parquet_path_from_report(report_path: Path) -> Path:
    """
    Resolve the selected_designations.parquet path from a covariance precovery report.

    The report uses paths like `.../artifacts/w_20200101_20240101__I41_T05_T08_W84/selected_designations.parquet`.
    We resolve "..." as two levels up from the report file (so when report is at subset_dir/harness/report.md,
    root is subset_dir; when report is at repo/.../harness/report.md, root is the experiment package).
    """
    report_path = report_path.resolve()
    if not report_path.is_file():
        raise FileNotFoundError(f"Report not found: {report_path}")
    text = report_path.read_text(encoding="utf-8")
    m = _SELECTED_DESIGNATIONS_PATTERN.search(text)
    if not m:
        raise ValueError(
            f"Report at {report_path} does not contain a 'Selected designations' path line "
            "(expected pattern: `.../artifacts/.../selected_designations.parquet`)"
        )
    suffix = m.group(2)  # artifacts/w_.../selected_designations.parquet
    # "..." = root that contains artifacts/. Try: (1) two levels up from report; (2) local_db/full_precovery_n32 under experiment pkg
    experiment_root = report_path.parent.parent  # covariance_precovery when report is in harness/
    candidates = [
        experiment_root / suffix,
        experiment_root / "local_db" / "full_precovery_n32" / suffix,
    ]
    for p in candidates:
        if p.resolve().exists():
            return p.resolve()
    return candidates[0].resolve()  # return first so caller gets a clear FileNotFoundError path


def _read_designations_from_parquet(path: Path) -> tuple[list[str], list[str] | None]:
    """Read designation column from parquet; return (designations, strata or None)."""
    tbl = pq.read_table(path)
    if "designation" not in tbl.column_names:
        raise ValueError(f"Parquet at {path} has no 'designation' column; got {tbl.column_names}")
    designations = [str(x.as_py()) for x in tbl.column("designation")]
    strata: list[str] | None = None
    if "stratum" in tbl.column_names:
        strata = [str(x.as_py()) for x in tbl.column("stratum")]
    return designations, strata


def _resolve_mpcq_provids_for_designations(
    *,
    selected_designations_parquet: Path,
    requested_designations: list[str],
) -> tuple[list[str], dict[str, object], list[str]]:
    """
    Resolve a list of selected "designation" strings (often COALESCE(permid, provid)) to
    mpcq-compatible provids (primary provisional designations).

    We prefer using the local BQ features parquet next to the selection, if present:
    <artifacts_dir>/bq_designation_orbit_features.parquet

    This avoids re-querying BigQuery and ensures we use the same primary-provid mapping
    used in the selection pipeline.
    """
    features_path = selected_designations_parquet.parent / "bq_designation_orbit_features.parquet"
    if not features_path.exists():
        return requested_designations, {
            "mpcq_provids_source": None,
            "n_requested": len(requested_designations),
            "n_mapped": 0,
            "n_missing_primary_provid": len(requested_designations),
        }, list(requested_designations)

    feat = pq.read_table(
        features_path,
        columns=["designation", "unpacked_primary_provisional_designation"],
    ).combine_chunks()

    des_col = feat.column("designation")
    prim_col = feat.column("unpacked_primary_provisional_designation")

    # Build mapping for non-null primary_provid rows.
    keep = pc.and_(pc.is_valid(des_col), pc.is_valid(prim_col))
    feat_kept = feat.filter(keep)

    mapping: dict[str, str] = {}
    for d, p in zip(
        feat_kept.column("designation").to_pylist(),
        feat_kept.column("unpacked_primary_provisional_designation").to_pylist(),
    ):
        if d is None or p is None:
            continue
        mapping[str(d)] = str(p)

    out: list[str] = []
    n_mapped = 0
    n_missing = 0
    missing: list[str] = []
    for d in requested_designations:
        mp = mapping.get(d)
        if mp is None:
            out.append(d)
            n_missing += 1
            missing.append(d)
            continue
        out.append(mp)
        if mp != d:
            n_mapped += 1

    return out, {
        "mpcq_provids_source": str(features_path),
        "n_requested": len(requested_designations),
        "n_mapped": int(n_mapped),
        "n_missing_primary_provid": int(n_missing),
    }, missing


def _mpcq_request_provids_from_bigquery(
    *,
    client: BigQueryMPCClient,
    cfg: BqConfig,
    requested_designations: list[str],
) -> dict[str, str]:
    """
    Map mixed designation keys (often COALESCE(permid, provid)) to primary provisional designations
    using the replicated identifications tables.

    This is intentionally a small query intended for a short list (e.g. the handful of unmapped
    designations after local parquet mapping).
    """
    if not requested_designations:
        return {}

    # BigQuery string literal escape: double quotes not needed; use single-quoted strings.
    des_list = ", ".join("'" + str(d).replace("'", "''") + "'" for d in requested_designations)

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
    )
    , base AS (
      SELECT
        p.designation AS requested_designation,
        COALESCE(
          ni_req.unpacked_primary_provisional_designation,
          ci.unpacked_primary_provisional_designation,
          p.designation
        ) AS primary_provid
      FROM parsed p
      LEFT JOIN `{cfg.numbered_identifications_table}` AS ni_req
        -- permid is commonly stored as a STRING in the replica; match on the original string key.
        ON p.permid_req IS NOT NULL AND ni_req.permid = p.designation
      LEFT JOIN `{cfg.current_identifications_table}` AS ci
        ON ci.unpacked_secondary_provisional_designation = p.designation
        OR ci.unpacked_primary_provisional_designation = p.designation
    ),
    secondary_choice AS (
      SELECT
        b.requested_designation,
        b.primary_provid,
        ANY_VALUE(ci2.unpacked_secondary_provisional_designation) AS some_secondary
      FROM base b
      LEFT JOIN `{cfg.current_identifications_table}` AS ci2
        ON ci2.unpacked_primary_provisional_designation = b.primary_provid
      GROUP BY b.requested_designation, b.primary_provid
    )
    SELECT
      requested_designation,
      COALESCE(some_secondary, primary_provid) AS mpcq_request_provid
    FROM secondary_choice
    """

    table = client.client.query(query).result().to_arrow(progress_bar_type=None, create_bqstorage_client=True)
    out: dict[str, str] = {}
    req = table["requested_designation"].to_pylist()
    prov = table["mpcq_request_provid"].to_pylist()
    for r, p in zip(req, prov):
        if r is None or p is None:
            continue
        out[str(r)] = str(p)
    return out


def _binning_config_for_report() -> dict[str, object]:
    """Binning/st stratum config used in selection (SamplingConfig + labels)."""
    cfg = SamplingConfig()
    return {
        "arc_bins_days": list(cfg.arc_bins_days),
        "dt_bins_days": list(cfg.dt_bins_days),
        "hi_i_deg": cfg.hi_i_deg,
        "regime_labels": {
            "NEO": "q < 1.3 au",
            "MCA": "1.3 <= q < 1.666 au",
            "MBA": "a >= 1.666 au",
            "JupiterPlus": "a >= 5 au",
            "TNO": "a >= 30 au",
            "Unknown": "otherwise",
        },
        "u_bins": {"u_0_2": "u <= 2", "u_3_5": "3 <= u <= 5", "u_6p": "u >= 6", "u_unknown": "null"},
        "stratum_format": "regime|arc_bin|dt_bin|u_bin|i_hi|i_norm",
    }


def _designation_key_columns(table: pa.Table) -> str | None:
    """Return column name to group by designation, or None."""
    for key in ("requested_provid", "provid", "primary_designation", "designation"):
        if key in table.column_names:
            return key
    return None


def _per_designation_counts(table: pa.Table) -> dict[str, int]:
    """Count rows per designation using the first available key column."""
    key_col = _designation_key_columns(table)
    if key_col is None:
        return {}
    col = table.column(key_col)
    counts: dict[str, int] = {}
    for i in range(len(col)):
        v = col[i]
        if v is None:
            k = "__null__"
        else:
            k = str(v.as_py()) if hasattr(v, "as_py") else str(v)
        counts[k] = counts.get(k, 0) + 1
    return counts


def _stratum_counts(strata: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for s in strata:
        k = str(s)
        counts[k] = counts.get(k, 0) + 1
    return dict(sorted(counts.items()))


def _split_stratum(stratum: str | None) -> tuple[str | None, str | None, str | None, str | None, str | None]:
    if stratum is None:
        return None, None, None, None, None
    parts = str(stratum).split("|")
    if len(parts) != 5:
        return str(stratum), None, None, None, None
    return parts[0], parts[1], parts[2], parts[3], parts[4]


def _filter_null_orbit_rows(
    orbits: pa.Table, observations: pa.Table
) -> tuple[pa.Table, pa.Table, dict[str, object]]:
    """
    Drop the mpcq placeholder orbit rows (no provid/id/q/e/i) and any observations associated
    with those requests (matching on observations.requested_provid).
    """
    required = ["requested_provid", "provid", "id", "q", "e", "i"]
    missing_cols = [c for c in required if c not in orbits.column_names]
    if missing_cols:
        # If schema changes, do nothing but record.
        return orbits, observations, {"drop_null_orbits": False, "reason": f"missing orbit columns: {missing_cols}"}

    valid = pc.and_(
        pc.and_(pc.and_(pc.is_valid(orbits["provid"]), pc.is_valid(orbits["id"])), pc.is_valid(orbits["q"])),
        pc.and_(pc.is_valid(orbits["e"]), pc.is_valid(orbits["i"])),
    )
    kept_orbits = orbits.filter(valid)
    dropped_orbits = orbits.filter(pc.invert(valid))

    kept_requests = [str(x) for x in kept_orbits["requested_provid"].to_pylist()]
    dropped_requests = [str(x) for x in dropped_orbits["requested_provid"].to_pylist()]
    keep_set = pa.array(sorted(set(kept_requests)), type=pa.string())

    if "requested_provid" in observations.column_names:
        obs_keep = observations.filter(pc.is_in(observations["requested_provid"], value_set=keep_set))
    else:
        obs_keep = observations

    meta = {
        "drop_null_orbits": True,
        "n_orbits_before": int(len(orbits)),
        "n_orbits_after": int(len(kept_orbits)),
        "n_orbits_dropped": int(len(dropped_orbits)),
        "dropped_requested_provids": sorted(set(dropped_requests)),
        "n_observations_before": int(len(observations)),
        "n_observations_after": int(len(obs_keep)),
        "n_observations_dropped": int(len(observations) - len(obs_keep)),
    }
    return kept_orbits, obs_keep, meta


def _enrich_orbits_with_bins(
    *, orbits: pa.Table, designation_mapping: pa.Table
) -> pa.Table:
    """
    Add selection/binning columns to the orbit table using designation_mapping.parquet.

    Join key: orbits.requested_provid == designation_mapping.mpcq_provid
    """
    if "requested_provid" not in orbits.column_names or "mpcq_provid" not in designation_mapping.column_names:
        return orbits

    # Build lookup dicts keyed by mpcq_provid.
    mpcq_provids = [str(x) for x in designation_mapping["mpcq_provid"].to_pylist()]
    req_des = designation_mapping["requested_designation"].to_pylist() if "requested_designation" in designation_mapping.column_names else [None] * len(mpcq_provids)
    strata = designation_mapping["stratum"].to_pylist() if "stratum" in designation_mapping.column_names else [None] * len(mpcq_provids)

    by_provid: dict[str, tuple[str | None, str | None]] = {}
    for p, r, s in zip(mpcq_provids, req_des, strata):
        if p not in by_provid:
            by_provid[p] = (None if r is None else str(r), None if s is None else str(s))

    out_req: list[str | None] = []
    out_stratum: list[str | None] = []
    out_regime: list[str | None] = []
    out_arc: list[str | None] = []
    out_dt: list[str | None] = []
    out_u: list[str | None] = []
    out_i: list[str | None] = []

    for p in orbits["requested_provid"].to_pylist():
        key = None if p is None else str(p)
        r, s = by_provid.get(key, (None, None))
        out_req.append(r)
        out_stratum.append(s)
        regime, arc, dt, u, ii = _split_stratum(s)
        out_regime.append(regime)
        out_arc.append(arc)
        out_dt.append(dt)
        out_u.append(u)
        out_i.append(ii)

    t = orbits
    t = t.append_column("selected_designation", pa.array(out_req, type=pa.string()))
    t = t.append_column("stratum", pa.array(out_stratum, type=pa.string()))
    t = t.append_column("regime", pa.array(out_regime, type=pa.string()))
    t = t.append_column("arc_bin", pa.array(out_arc, type=pa.string()))
    t = t.append_column("dt_bin", pa.array(out_dt, type=pa.string()))
    t = t.append_column("u_bin", pa.array(out_u, type=pa.string()))
    t = t.append_column("i_bin", pa.array(out_i, type=pa.string()))
    return t


def _write_report(
    *,
    out_path: Path,
    designations_path: Path,
    n_designations: int,
    strata: list[str] | None,
    mpcq_provids_meta: dict[str, object],
    filtering_meta: dict[str, object] | None,
    n_orbits: int,
    n_observations: int,
    orbits_per_des: dict[str, int],
    obs_per_des: dict[str, int],
    meta: dict[str, object],
    binning_config: dict[str, object],
) -> None:
    """Write markdown report of selection, serialization, and binning stats."""
    lines: list[str] = []
    lines.append("# MPCQ handoff report")
    lines.append("")
    lines.append(f"Generated: {meta.get('generated_at_utc', '')}")
    lines.append("")
    lines.append("## Source")
    lines.append(f"- **Designations parquet**: `{designations_path}`")
    lines.append(f"- **Designations requested**: {n_designations}")
    if mpcq_provids_meta.get("mpcq_provids_source") is not None:
        lines.append(f"- **mpcq provid mapping source**: `{mpcq_provids_meta.get('mpcq_provids_source')}`")
        lines.append(f"- **Designations mapped to primary provid**: {mpcq_provids_meta.get('n_mapped', 0)}")
        lines.append(
            f"- **Missing primary provid mapping (fallback to original designation)**: {mpcq_provids_meta.get('n_missing_primary_provid', 0)}"
        )
    lines.append("")
    lines.append("## Binning / stratum config")
    lines.append("Selection used the following bins (see `SamplingConfig` in `subset_sampling`):")
    lines.append("")
    lines.append("- **arc_bins_days** (days): " + str(binning_config.get("arc_bins_days")))
    lines.append("- **dt_bins_days** (days): " + str(binning_config.get("dt_bins_days")))
    lines.append("- **hi_i_deg**: " + str(binning_config.get("hi_i_deg")))
    lines.append("- **stratum format**: `" + str(binning_config.get("stratum_format", "")) + "`")
    lines.append("")
    lines.append("### Regime labels")
    for k, v in (binning_config.get("regime_labels") or {}).items():
        lines.append(f"- **{k}**: {v}")
    lines.append("")
    lines.append("### U (uncertainty) bins")
    for k, v in (binning_config.get("u_bins") or {}).items():
        lines.append(f"- **{k}**: {v}")
    lines.append("")

    # Stratum counts: prefer the kept-set counts if present (filtered product), otherwise fall back to full selection.
    stratum_counts_kept = meta.get("stratum_counts_kept")
    if isinstance(stratum_counts_kept, dict) and stratum_counts_kept:
        lines.append("## Stratum counts (kept orbits)")
        lines.append("")
        for stratum, count in dict(sorted(stratum_counts_kept.items())).items():
            lines.append(f"- `{stratum}`: {count}")
        lines.append("")
    elif strata is not None:
        stratum_counts = _stratum_counts(strata)
        lines.append("## Stratum counts (from designations parquet)")
        lines.append("")
        for stratum, count in stratum_counts.items():
            lines.append(f"- `{stratum}`: {count}")
        lines.append("")

    lines.append("## Serialization summary")
    lines.append("")
    if filtering_meta is not None and bool(filtering_meta.get("drop_null_orbits", False)):
        lines.append("### Filtering")
        lines.append(
            f"- **Dropped null-orbit rows**: {filtering_meta.get('n_orbits_dropped', 0)} "
            f"(kept {filtering_meta.get('n_orbits_after', n_orbits)} / {filtering_meta.get('n_orbits_before', n_orbits)})"
        )
        lines.append(
            f"- **Dropped observations (for dropped orbit requests)**: {filtering_meta.get('n_observations_dropped', 0)}"
        )
        dropped = filtering_meta.get("dropped_requested_provids") or []
        if dropped:
            lines.append(f"- **Dropped requested_provids**: {', '.join(str(x) for x in dropped)}")
        lines.append("")
    lines.append(f"- **MPC orbits returned**: {n_orbits}")
    lines.append(f"- **MPC observations returned**: {n_observations}")
    if orbits_per_des:
        vals = list(orbits_per_des.values())
        lines.append(f"- **Orbits per designation**: min={min(vals)}, max={max(vals)}, mean={sum(vals)/len(vals):.1f}")
    if obs_per_des:
        vals = list(obs_per_des.values())
        lines.append(f"- **Observations per designation**: min={min(vals)}, max={max(vals)}, mean={sum(vals)/len(vals):.1f}")
    lines.append("")

    lines.append("## Output files")
    lines.append("")
    lines.append("| File | Description |")
    lines.append("|------|-------------|")
    lines.append("| `designations.parquet` | Copy of input designation list (and stratum if present) |")
    lines.append("| `designation_mapping.parquet` | Mapping from requested designation to the provid used for mpcq queries |")
    lines.append("| `mpcq_orbits.parquet` | MPC orbit table (mpcq schema) |")
    lines.append("| `mpcq_orbits_with_bins.parquet` | MPC orbit table with selection/bin columns (stratum + split bins) |")
    lines.append("| `mpcq_observations.parquet` | MPC observations (mpcq schema) |")
    lines.append("| `meta.json` | Counts, paths, dataset ids, timestamp |")
    lines.append("| `handoff_report.md` | This report |")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def fetch_mpcq_handoff(
    *,
    designations_parquet: Path,
    out_dir: Path,
    cfg: BqConfig | None = None,
    drop_null_orbits: bool = False,
) -> HandoffResult:
    """
    Fetch MPCOrbits and MPCObservations for the given designations and persist to out_dir.

    Reads designations (and optional stratum column) from the parquet, queries mpcq,
    writes mpcq_orbits.parquet, mpcq_observations.parquet, meta.json, and a markdown report.
    """
    if cfg is None:
        cfg = BqConfig()
    designations_parquet = designations_parquet.resolve()
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    designations, strata = _read_designations_from_parquet(designations_parquet)
    if not designations:
        raise ValueError(f"No designations found in {designations_parquet}")

    requested_designations = [str(d) for d in designations]

    client = BigQueryMPCClient(dataset_id=cfg.dataset_id, views_dataset_id=cfg.views_dataset_id)

    mpcq_provids, mpcq_provids_meta, missing = _resolve_mpcq_provids_for_designations(
        selected_designations_parquet=designations_parquet,
        requested_designations=requested_designations,
    )

    # Fallback: for any designations that couldn't be mapped from the local features parquet,
    # map them via identifications tables directly (handles numeric permids and primary/secondary provids).
    if missing:
        bq_map = _mpcq_request_provids_from_bigquery(client=client, cfg=cfg, requested_designations=missing)
        if bq_map:
            filled = 0
            mpcq_provids = [bq_map.get(d, p) for d, p in zip(requested_designations, mpcq_provids)]
            for d in missing:
                if bq_map.get(d) is not None:
                    filled += 1
            mpcq_provids_meta = dict(mpcq_provids_meta)
            mpcq_provids_meta["bq_fallback_n_requested"] = int(len(missing))
            mpcq_provids_meta["bq_fallback_n_filled"] = int(filled)
            mpcq_provids_meta["bq_fallback_unfilled"] = int(len(missing) - filled)

    mpc_orbits: MPCOrbits = client.query_orbits(mpcq_provids)
    mpc_observations: MPCObservations = client.query_observations(mpcq_provids)

    orbits_tbl = mpc_orbits.table.combine_chunks()
    obs_tbl = mpc_observations.table.combine_chunks()
    filtering_meta: dict[str, object] | None = None
    if bool(drop_null_orbits):
        orbits_tbl, obs_tbl, filtering_meta = _filter_null_orbit_rows(orbits_tbl, obs_tbl)

    out_designations = out_dir / "designations.parquet"
    shutil.copy2(designations_parquet, out_designations)
    # Persist mapping used for mpcq queries (requested designation -> mpcq provid).
    out_mapping = out_dir / "designation_mapping.parquet"
    mapping_tbl = pa.table(
        {
            "requested_designation": requested_designations,
            "mpcq_provid": mpcq_provids,
        }
    )
    if strata is not None and len(strata) == len(requested_designations):
        mapping_tbl = mapping_tbl.append_column("stratum", pa.array(strata, type=pa.string()))

    # If we dropped null-orbit rows, also filter the mapping table down to the kept orbit requests
    # so bins/strata match the delivered orbit/observation set.
    kept_stratum_counts: dict[str, int] | None = None
    if filtering_meta is not None and bool(filtering_meta.get("drop_null_orbits", False)) and "requested_provid" in orbits_tbl.column_names:
        keep_set = pa.array(sorted(set(str(x) for x in orbits_tbl["requested_provid"].to_pylist())), type=pa.string())
        if "mpcq_provid" in mapping_tbl.column_names:
            mapping_tbl = mapping_tbl.filter(pc.is_in(mapping_tbl["mpcq_provid"], value_set=keep_set))
        if "stratum" in mapping_tbl.column_names:
            kept_stratum_counts = _stratum_counts([str(x) for x in mapping_tbl["stratum"].to_pylist() if x is not None])
    pq.write_table(mapping_tbl, out_mapping)

    out_orbits = out_dir / "mpcq_orbits.parquet"
    out_observations = out_dir / "mpcq_observations.parquet"
    # Write filtered (or unfiltered) results.
    try:
        MPCOrbits.from_pyarrow(orbits_tbl).to_parquet(str(out_orbits))
    except Exception:
        pq.write_table(orbits_tbl, out_orbits)
    try:
        MPCObservations.from_pyarrow(obs_tbl).to_parquet(str(out_observations))
    except Exception:
        pq.write_table(obs_tbl, out_observations)

    # Convenience: orbits denormalized with selection/bin columns for easy association.
    out_orbits_bins = out_dir / "mpcq_orbits_with_bins.parquet"
    orbits_with_bins = _enrich_orbits_with_bins(orbits=orbits_tbl, designation_mapping=mapping_tbl)
    pq.write_table(orbits_with_bins, out_orbits_bins)

    n_orbits = int(len(orbits_tbl))
    n_observations = int(len(obs_tbl))
    orbits_per_des = _per_designation_counts(orbits_tbl)
    obs_per_des = _per_designation_counts(obs_tbl)

    generated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    meta = {
        "designations_parquet": str(designations_parquet),
        "designations_copy": str(out_designations),
        "designation_mapping_parquet": str(out_mapping),
        "mpcq_orbits_with_bins_parquet": str(out_orbits_bins),
        "n_designations": len(designations),
        "n_orbits": n_orbits,
        "n_observations": n_observations,
        "dataset_id": cfg.dataset_id,
        "views_dataset_id": cfg.views_dataset_id,
        "generated_at_utc": generated_at,
        "binning_config": _binning_config_for_report(),
        "mpcq_provids_meta": mpcq_provids_meta,
    }
    if strata is not None:
        meta["stratum_counts"] = _stratum_counts(strata)
    if kept_stratum_counts is not None:
        meta["stratum_counts_kept"] = kept_stratum_counts
    if filtering_meta is not None:
        meta["filtering"] = filtering_meta
    out_meta = out_dir / "meta.json"
    out_meta.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report_path = out_dir / "handoff_report.md"
    _write_report(
        out_path=report_path,
        designations_path=designations_parquet,
        n_designations=len(designations),
        strata=strata,
        mpcq_provids_meta=mpcq_provids_meta,
        filtering_meta=filtering_meta,
        n_orbits=n_orbits,
        n_observations=n_observations,
        orbits_per_des=orbits_per_des,
        obs_per_des=obs_per_des,
        meta=meta,
        binning_config=_binning_config_for_report(),
    )

    return HandoffResult(
        out_dir=out_dir,
        designations_parquet=designations_parquet,
        designations_copy=out_designations,
        mpcq_orbits_parquet=out_orbits,
        mpcq_observations_parquet=out_observations,
        meta_json=out_meta,
        report_md=report_path,
        n_designations=len(designations),
        n_orbits=n_orbits,
        n_observations=n_observations,
    )


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(
        description="Fetch MPCOrbits and MPCObservations for designations and persist to parquet for handoff."
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--designations-parquet", type=str, help="Path to selected_designations.parquet")
    g.add_argument(
        "--report",
        type=str,
        help="Path to report markdown (e.g. report_20260211_population.md); designations path is resolved from it.",
    )
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument(
        "--drop-null-orbits",
        action="store_true",
        help="Drop mpcq placeholder orbit rows with null q/e/i/provid/id and drop their observations.",
    )
    args = p.parse_args()

    if args.designations_parquet is not None:
        designations_parquet = Path(args.designations_parquet).resolve()
    else:
        designations_parquet = designations_parquet_path_from_report(Path(args.report))

    result = fetch_mpcq_handoff(
        designations_parquet=designations_parquet,
        out_dir=Path(args.out_dir),
        drop_null_orbits=bool(args.drop_null_orbits),
    )
    print(f"out_dir={result.out_dir}")
    print(f"n_designations={result.n_designations} n_orbits={result.n_orbits} n_observations={result.n_observations}")
    print(f"report={result.report_md}")


if __name__ == "__main__":
    main()
