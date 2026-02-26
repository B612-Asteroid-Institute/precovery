"""Fetch MPCOrbits and MPCObservations via mpcq for a designation list and persist to parquet."""

from __future__ import annotations

import json
import re
import shutil
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from mpcq.client import BigQueryMPCClient
from mpcq.observations import MPCObservations
from mpcq.orbits import MPCOrbits

from .bq_select import BqConfig
from .designation_resolution import (
    materialize_designation_resolution_for_selected_designations,
    read_designation_resolution,
)

AliasRequestMode = Literal["all_linked", "selected_only"]


@dataclass(frozen=True)
class BinningConfig:
    arc_bins_days: tuple[float, float, float] = (7.0, 30.0, 180.0)
    dt_bins_days: tuple[float, float, float] = (30.0, 180.0, 1000.0)
    hi_i_deg: float = 30.0


@dataclass(frozen=True)
class HandoffResult:
    out_dir: Path
    designations_parquet: Path
    designations_copy: Path
    designation_resolution_parquet: Path
    designation_mapping_parquet: Path
    mpcq_orbits_parquet: Path
    mpcq_observations_parquet: Path
    meta_json: Path
    report_md: Path
    n_designations: int
    n_request_provids: int
    n_orbits: int
    n_observations: int


_SELECTED_DESIGNATIONS_PATTERN = re.compile(
    r"selected designations[^\n]*`(\.\.\./)?(artifacts/[^`]+/selected_designations\.parquet)`",
    re.IGNORECASE,
)


def designations_parquet_path_from_report(report_path: Path) -> Path:
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
    suffix = m.group(2)
    experiment_root = report_path.parent.parent
    candidates = [
        experiment_root / suffix,
        experiment_root / "local_db" / "full_precovery_n32" / suffix,
    ]
    for p in candidates:
        if p.resolve().exists():
            return p.resolve()
    return candidates[0].resolve()


def _read_designations_from_parquet(path: Path) -> tuple[list[str], list[str] | None]:
    tbl = pq.read_table(path)
    if "designation" not in tbl.column_names:
        raise ValueError(f"Parquet at {path} has no 'designation' column; got {tbl.column_names}")
    designations = [str(x.as_py()) for x in tbl.column("designation")]
    strata: list[str] | None = None
    if "stratum" in tbl.column_names:
        strata = [str(x.as_py()) for x in tbl.column("stratum")]
    return designations, strata


def _dedupe_keep_order(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for v in values:
        s = str(v).strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _build_designation_mapping_table(
    *,
    designations: list[str],
    strata: list[str] | None,
    designation_resolution: pa.Table,
    alias_request_mode: AliasRequestMode,
) -> pa.Table:
    if alias_request_mode not in ("all_linked", "selected_only"):
        raise ValueError(f"Unsupported alias_request_mode: {alias_request_mode!r}")

    by_selected: dict[str, tuple[str, list[str]]] = {}
    for selected, status, reqs in zip(
        designation_resolution["selected_designation"].to_pylist(),
        designation_resolution["resolution_status"].to_pylist(),
        designation_resolution["mpcq_request_provids"].to_pylist(),
    ):
        if selected is None:
            continue
        s_selected = str(selected)
        s_status = "unresolved" if status is None else str(status)
        req_list = []
        if reqs is not None:
            req_list = [str(x) for x in reqs if x is not None and str(x).strip()]
        by_selected[s_selected] = (s_status, _dedupe_keep_order(req_list))

    out_selected: list[str] = []
    out_requested: list[str] = []
    out_status: list[str] = []
    out_stratum: list[str | None] = []
    for i, selected in enumerate(designations):
        status, reqs = by_selected.get(str(selected), ("unresolved", []))
        if alias_request_mode == "selected_only":
            reqs_use = [str(selected)]
        else:
            reqs_use = reqs if reqs else [str(selected)]
        reqs_use = _dedupe_keep_order(reqs_use)
        stratum = None if strata is None else strata[i]
        for req in reqs_use:
            out_selected.append(str(selected))
            out_requested.append(str(req))
            out_status.append(str(status))
            out_stratum.append(None if stratum is None else str(stratum))

    if not out_selected:
        return pa.table(
            {
                "selected_designation": pa.array([], type=pa.string()),
                "requested_provid": pa.array([], type=pa.string()),
                "resolution_status": pa.array([], type=pa.string()),
                "matched_orbit_rows": pa.array([], type=pa.int64()),
                "matched_observation_rows": pa.array([], type=pa.int64()),
                "stratum": pa.array([], type=pa.string()),
            }
        )

    return pa.table(
        {
            "selected_designation": pa.array(out_selected, type=pa.string()),
            "requested_provid": pa.array(out_requested, type=pa.string()),
            "resolution_status": pa.array(out_status, type=pa.string()),
            "matched_orbit_rows": pa.array([0] * len(out_selected), type=pa.int64()),
            "matched_observation_rows": pa.array([0] * len(out_selected), type=pa.int64()),
            "stratum": pa.array(out_stratum, type=pa.string()),
        }
    )


def _split_stratum(stratum: str | None) -> tuple[str | None, str | None, str | None, str | None, str | None]:
    if stratum is None:
        return None, None, None, None, None
    parts = str(stratum).split("|")
    if len(parts) != 5:
        return str(stratum), None, None, None, None
    return parts[0], parts[1], parts[2], parts[3], parts[4]


def _stratum_counts(strata: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for s in strata:
        k = str(s)
        counts[k] = counts.get(k, 0) + 1
    return dict(sorted(counts.items()))


def _count_rows_by_requested_provid(
    *,
    table: pa.Table,
    key_column: str,
    require_valid_column: str | None,
) -> Counter[str]:
    if key_column not in table.column_names:
        return Counter()
    t = table
    if require_valid_column is not None and require_valid_column in t.column_names:
        t = t.filter(pc.is_valid(t[require_valid_column]))
    out: Counter[str] = Counter()
    for v in t[key_column].to_pylist():
        if v is None:
            continue
        out[str(v)] += 1
    return out


def _filter_null_orbit_rows(
    orbits: pa.Table, observations: pa.Table
) -> tuple[pa.Table, pa.Table, dict[str, object]]:
    required = ["requested_provid", "provid", "id", "q", "e", "i"]
    missing_cols = [c for c in required if c not in orbits.column_names]
    if missing_cols:
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
    if "requested_provid" not in orbits.column_names or "requested_provid" not in designation_mapping.column_names:
        return orbits

    grouped: dict[str, dict[str, object]] = {}
    for req, selected, stratum in zip(
        designation_mapping["requested_provid"].to_pylist(),
        designation_mapping["selected_designation"].to_pylist(),
        designation_mapping["stratum"].to_pylist() if "stratum" in designation_mapping.column_names else [None] * designation_mapping.num_rows,
    ):
        if req is None:
            continue
        key = str(req)
        ent = grouped.setdefault(key, {"selected": set(), "strata": set()})
        if selected is not None and str(selected).strip():
            ent["selected"].add(str(selected))
        if stratum is not None and str(stratum).strip():
            ent["strata"].add(str(stratum))

    out_selected: list[str | None] = []
    out_selected_csv: list[str | None] = []
    out_n_selected: list[int] = []
    out_stratum: list[str | None] = []
    out_regime: list[str | None] = []
    out_arc: list[str | None] = []
    out_dt: list[str | None] = []
    out_u: list[str | None] = []
    out_i: list[str | None] = []

    for req in orbits["requested_provid"].to_pylist():
        if req is None:
            out_selected.append(None)
            out_selected_csv.append(None)
            out_n_selected.append(0)
            out_stratum.append(None)
            out_regime.append(None)
            out_arc.append(None)
            out_dt.append(None)
            out_u.append(None)
            out_i.append(None)
            continue
        ent = grouped.get(str(req))
        if ent is None:
            out_selected.append(None)
            out_selected_csv.append(None)
            out_n_selected.append(0)
            out_stratum.append(None)
            out_regime.append(None)
            out_arc.append(None)
            out_dt.append(None)
            out_u.append(None)
            out_i.append(None)
            continue
        selected_sorted = sorted(str(x) for x in ent["selected"])
        strata_sorted = sorted(str(x) for x in ent["strata"])
        selected_first = selected_sorted[0] if selected_sorted else None
        selected_csv = ",".join(selected_sorted) if selected_sorted else None
        stratum_first = strata_sorted[0] if strata_sorted else None
        regime, arc, dt, ub, ib = _split_stratum(stratum_first)
        out_selected.append(selected_first)
        out_selected_csv.append(selected_csv)
        out_n_selected.append(len(selected_sorted))
        out_stratum.append(stratum_first)
        out_regime.append(regime)
        out_arc.append(arc)
        out_dt.append(dt)
        out_u.append(ub)
        out_i.append(ib)

    t = orbits
    t = t.append_column("selected_designation", pa.array(out_selected, type=pa.string()))
    t = t.append_column("selected_designations_csv", pa.array(out_selected_csv, type=pa.string()))
    t = t.append_column("n_selected_designations", pa.array(out_n_selected, type=pa.int64()))
    t = t.append_column("stratum", pa.array(out_stratum, type=pa.string()))
    t = t.append_column("regime", pa.array(out_regime, type=pa.string()))
    t = t.append_column("arc_bin", pa.array(out_arc, type=pa.string()))
    t = t.append_column("dt_bin", pa.array(out_dt, type=pa.string()))
    t = t.append_column("u_bin", pa.array(out_u, type=pa.string()))
    t = t.append_column("i_bin", pa.array(out_i, type=pa.string()))
    return t


def _designation_key_columns(table: pa.Table) -> str | None:
    for key in ("requested_provid", "provid", "primary_designation", "designation"):
        if key in table.column_names:
            return key
    return None


def _per_designation_counts(table: pa.Table) -> dict[str, int]:
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


def _binning_config_for_report() -> dict[str, object]:
    cfg = BinningConfig()
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


def _write_report(
    *,
    out_path: Path,
    designations_path: Path,
    designation_resolution_path: Path,
    alias_request_mode: AliasRequestMode,
    n_designations: int,
    mapping: pa.Table,
    filtering_meta: dict[str, object] | None,
    n_orbits: int,
    n_observations: int,
    orbits_per_des: dict[str, int],
    obs_per_des: dict[str, int],
    meta: dict[str, object],
) -> None:
    lines: list[str] = []
    lines.append("# MPCQ handoff report")
    lines.append("")
    lines.append(f"Generated: {meta.get('generated_at_utc', '')}")
    lines.append("")
    lines.append("## Source")
    lines.append(f"- **Designations parquet**: `{designations_path}`")
    lines.append(f"- **Designation resolution parquet**: `{designation_resolution_path}`")
    lines.append(f"- **Designations requested**: {n_designations}")
    lines.append(f"- **Alias request mode**: `{alias_request_mode}`")
    lines.append("")

    status_counts = Counter(str(x) for x in mapping["resolution_status"].to_pylist())
    lines.append("## Resolution status")
    lines.append("")
    for k in ("resolved", "partially_resolved", "unresolved"):
        lines.append(f"- `{k}`: {int(status_counts.get(k, 0))}")
    lines.append("")

    lines.append("## Request mapping")
    lines.append("")
    lines.append(f"- **Mapping rows**: {mapping.num_rows}")
    unique_requests = sorted({str(x) for x in mapping["requested_provid"].to_pylist() if x is not None and str(x)})
    lines.append(f"- **Unique requested_provid values**: {len(unique_requests)}")
    lines.append("")

    lines.append("## Binning / stratum config")
    lines.append("")
    bc = _binning_config_for_report()
    lines.append("- **arc_bins_days** (days): " + str(bc.get("arc_bins_days")))
    lines.append("- **dt_bins_days** (days): " + str(bc.get("dt_bins_days")))
    lines.append("- **hi_i_deg**: " + str(bc.get("hi_i_deg")))
    lines.append("- **stratum format**: `" + str(bc.get("stratum_format", "")) + "`")
    lines.append("")

    if "stratum" in mapping.column_names:
        strata = [str(x) for x in mapping["stratum"].to_pylist() if x is not None]
        if strata:
            lines.append("## Stratum counts (from mapping)")
            lines.append("")
            for s, c in _stratum_counts(strata).items():
                lines.append(f"- `{s}`: {c}")
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
    lines.append("| `designation_resolution.parquet` | Canonical resolution artifact used by handoff |")
    lines.append("| `designation_mapping.parquet` | One row per selected_designation/requested_provid mapping with match counts |")
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
    designation_resolution_parquet: Path | None = None,
    alias_request_mode: AliasRequestMode = "all_linked",
) -> HandoffResult:
    if cfg is None:
        cfg = BqConfig()
    if alias_request_mode not in ("all_linked", "selected_only"):
        raise ValueError(f"Unsupported alias_request_mode={alias_request_mode!r}")

    designations_parquet = designations_parquet.resolve()
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    designations, strata = _read_designations_from_parquet(designations_parquet)
    if not designations:
        raise ValueError(f"No designations found in {designations_parquet}")

    if designation_resolution_parquet is None:
        default_resolution = designations_parquet.parent / "designation_resolution.parquet"
        if default_resolution.exists():
            resolution_path = default_resolution.resolve()
            designation_resolution = read_designation_resolution(resolution_path)
        else:
            res = materialize_designation_resolution_for_selected_designations(
                selected_designations_parquet=designations_parquet,
                out_parquet=default_resolution,
                cfg=cfg,
            )
            resolution_path = res.out_parquet
            designation_resolution = res.table
    else:
        resolution_path = Path(designation_resolution_parquet).resolve()
        designation_resolution = read_designation_resolution(resolution_path)

    mapping_tbl = _build_designation_mapping_table(
        designations=designations,
        strata=strata,
        designation_resolution=designation_resolution,
        alias_request_mode=alias_request_mode,
    )

    requests = _dedupe_keep_order(
        [str(x) for x in mapping_tbl["requested_provid"].to_pylist() if x is not None and str(x).strip()]
    )
    if not requests:
        raise ValueError("No request provids generated for handoff.")

    client = BigQueryMPCClient(dataset_id=cfg.dataset_id, views_dataset_id=cfg.views_dataset_id)
    mpc_orbits: MPCOrbits = client.query_orbits(requests)
    mpc_observations: MPCObservations = client.query_observations(requests)

    orbits_tbl = mpc_orbits.table.combine_chunks()
    obs_tbl = mpc_observations.table.combine_chunks()
    filtering_meta: dict[str, object] | None = None
    if bool(drop_null_orbits):
        orbits_tbl, obs_tbl, filtering_meta = _filter_null_orbit_rows(orbits_tbl, obs_tbl)

    orbit_counts = _count_rows_by_requested_provid(
        table=orbits_tbl,
        key_column="requested_provid",
        require_valid_column="id",
    )
    obs_counts = _count_rows_by_requested_provid(
        table=obs_tbl,
        key_column="requested_provid",
        require_valid_column="obsid",
    )
    mapping_orbit = [int(orbit_counts.get(str(x), 0)) for x in mapping_tbl["requested_provid"].to_pylist()]
    mapping_obs = [int(obs_counts.get(str(x), 0)) for x in mapping_tbl["requested_provid"].to_pylist()]
    mapping_tbl = mapping_tbl.set_column(
        mapping_tbl.schema.get_field_index("matched_orbit_rows"),
        "matched_orbit_rows",
        pa.array(mapping_orbit, type=pa.int64()),
    )
    mapping_tbl = mapping_tbl.set_column(
        mapping_tbl.schema.get_field_index("matched_observation_rows"),
        "matched_observation_rows",
        pa.array(mapping_obs, type=pa.int64()),
    )

    out_designations = out_dir / "designations.parquet"
    out_resolution = out_dir / "designation_resolution.parquet"
    out_mapping = out_dir / "designation_mapping.parquet"
    out_orbits = out_dir / "mpcq_orbits.parquet"
    out_observations = out_dir / "mpcq_observations.parquet"
    out_orbits_bins = out_dir / "mpcq_orbits_with_bins.parquet"

    shutil.copy2(designations_parquet, out_designations)
    shutil.copy2(resolution_path, out_resolution)
    pq.write_table(mapping_tbl, out_mapping)

    try:
        MPCOrbits.from_pyarrow(orbits_tbl).to_parquet(str(out_orbits))
    except Exception:
        pq.write_table(orbits_tbl, out_orbits)
    try:
        MPCObservations.from_pyarrow(obs_tbl).to_parquet(str(out_observations))
    except Exception:
        pq.write_table(obs_tbl, out_observations)

    orbits_with_bins = _enrich_orbits_with_bins(orbits=orbits_tbl, designation_mapping=mapping_tbl)
    pq.write_table(orbits_with_bins, out_orbits_bins)

    n_orbits = int(len(orbits_tbl))
    n_observations = int(len(obs_tbl))
    generated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    meta = {
        "designations_parquet": str(designations_parquet),
        "designations_copy": str(out_designations),
        "designation_resolution_parquet": str(out_resolution),
        "designation_mapping_parquet": str(out_mapping),
        "mpcq_orbits_with_bins_parquet": str(out_orbits_bins),
        "alias_request_mode": str(alias_request_mode),
        "n_designations": len(designations),
        "n_request_provids": len(requests),
        "n_orbits": n_orbits,
        "n_observations": n_observations,
        "dataset_id": cfg.dataset_id,
        "views_dataset_id": cfg.views_dataset_id,
        "generated_at_utc": generated_at,
        "binning_config": _binning_config_for_report(),
        "resolution_status_counts": dict(
            Counter(str(x) for x in mapping_tbl["resolution_status"].to_pylist())
        ),
    }
    if filtering_meta is not None:
        meta["filtering"] = filtering_meta
    if "stratum" in mapping_tbl.column_names:
        strata_nonnull = [str(x) for x in mapping_tbl["stratum"].to_pylist() if x is not None]
        if strata_nonnull:
            meta["stratum_counts"] = _stratum_counts(strata_nonnull)

    out_meta = out_dir / "meta.json"
    out_meta.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report_path = out_dir / "handoff_report.md"
    _write_report(
        out_path=report_path,
        designations_path=designations_parquet,
        designation_resolution_path=out_resolution,
        alias_request_mode=alias_request_mode,
        n_designations=len(designations),
        mapping=mapping_tbl,
        filtering_meta=filtering_meta,
        n_orbits=n_orbits,
        n_observations=n_observations,
        orbits_per_des=_per_designation_counts(orbits_tbl),
        obs_per_des=_per_designation_counts(obs_tbl),
        meta=meta,
    )

    return HandoffResult(
        out_dir=out_dir,
        designations_parquet=designations_parquet,
        designations_copy=out_designations,
        designation_resolution_parquet=out_resolution,
        designation_mapping_parquet=out_mapping,
        mpcq_orbits_parquet=out_orbits,
        mpcq_observations_parquet=out_observations,
        meta_json=out_meta,
        report_md=report_path,
        n_designations=len(designations),
        n_request_provids=len(requests),
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
        help="Path to report markdown; designations path is resolved from it.",
    )
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument(
        "--designation-resolution-parquet",
        type=str,
        default="",
        help="Optional explicit designation_resolution parquet path. Default: sibling of designations parquet.",
    )
    p.add_argument(
        "--alias-request-mode",
        type=str,
        default="all_linked",
        choices=["all_linked", "selected_only"],
        help="How to expand selected designations into mpcq requested provids.",
    )
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
        designation_resolution_parquet=(
            None if not args.designation_resolution_parquet else Path(args.designation_resolution_parquet)
        ),
        alias_request_mode=str(args.alias_request_mode),  # type: ignore[arg-type]
        drop_null_orbits=bool(args.drop_null_orbits),
    )
    print(f"out_dir={result.out_dir}")
    print(
        "n_designations="
        f"{result.n_designations} n_request_provids={result.n_request_provids} "
        f"n_orbits={result.n_orbits} n_observations={result.n_observations}"
    )
    print(f"report={result.report_md}")


if __name__ == "__main__":
    main()
