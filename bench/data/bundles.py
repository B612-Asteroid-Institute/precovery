from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from adam_core.orbits import Orbits

from precovery.search.backends.duckdb_parquet import DuckDbParquetBackend
from ..selection.designation_resolution import read_designation_resolution
from ..benchmarks.workload import month_bounds_mjd_utc


@dataclass(frozen=True)
class BundlePaths:
    out_dir: Path
    orbits_parquet: Path
    sbdb_orbit_provenance_parquet: Path
    orbit_bins_parquet: Path
    selected_orbit_mapping_parquet: Path
    truth_detections_parquet: Path
    truth_frames_parquet: Path
    truth_unresolved_designations_parquet: Path
    orbit_ids_with_truth_parquet: Path
    orbits_with_truth_parquet: Path
    orbit_ids_fast6_parquet: Path
    orbits_fast6_parquet: Path
    orbits_rejected_covariance_parquet: Path
    manifest_json: Path
    dropped_known_bad_parquet: Path


def _utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _read_selected_designations(path: Path) -> tuple[list[str], list[str | None]]:
    t = pq.read_table(path).combine_chunks()
    if "designation" not in t.column_names:
        raise ValueError(f"selected_designations parquet missing required 'designation' column: {path}")
    designations = [str(x) for x in t["designation"].to_pylist()]
    if "stratum" in t.column_names:
        strata = [None if x is None else str(x) for x in t["stratum"].to_pylist()]
    else:
        strata = [None] * len(designations)
    if len(strata) != len(designations):
        raise ValueError("selected_designations stratum length mismatch")
    return designations, strata


def _require_unique_orbit_ids(orbits: Orbits) -> list[str]:
    orbit_ids = [str(x) for x in orbits.orbit_id.to_pylist()]
    if not orbit_ids:
        return orbit_ids
    bad = [oid for oid in orbit_ids if not str(oid).strip()]
    if bad:
        raise ValueError("orbits.parquet contains empty orbit_id values")
    seen: set[str] = set()
    dup: set[str] = set()
    for oid in orbit_ids:
        if oid in seen:
            dup.add(oid)
        seen.add(oid)
    if dup:
        vals = sorted(dup)
        raise ValueError(f"orbits.parquet contains duplicate orbit_id values: {vals[:20]}")
    return orbit_ids


def _dedupe_keep_order(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        s = str(value).strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _build_selected_orbit_mapping(
    *,
    designations: list[str],
    strata: list[str | None],
    designation_resolution: pa.Table,
    orbits: Orbits,
) -> pa.Table:
    orbit_ids = [str(x) for x in orbits.orbit_id.to_pylist()]
    object_ids = (
        [None if x is None else str(x) for x in orbits.object_id.to_pylist()]
        if "object_id" in orbits.table.column_names
        else [None] * len(orbit_ids)
    )

    alias_to_orbit_ids: dict[str, set[str]] = {}
    for orbit_id, object_id in zip(orbit_ids, object_ids):
        for alias in _dedupe_keep_order([orbit_id, "" if object_id is None else object_id]):
            alias_to_orbit_ids.setdefault(alias, set()).add(str(orbit_id))

    resolution_by_selected: dict[str, tuple[str, list[str]]] = {}
    for selected, status, reqs in zip(
        designation_resolution["selected_designation"].to_pylist(),
        designation_resolution["resolution_status"].to_pylist(),
        designation_resolution["mpcq_request_provids"].to_pylist(),
    ):
        if selected is None:
            continue
        req_list = []
        if reqs is not None:
            req_list = [str(x) for x in reqs if x is not None and str(x).strip()]
        resolution_by_selected[str(selected)] = (
            "unresolved" if status is None else str(status),
            req_list,
        )

    out_selected: list[str] = []
    out_stratum: list[str | None] = []
    out_status: list[str] = []
    out_orbit_id: list[str | None] = []
    out_matched_count: list[int] = []
    out_match_reason: list[str] = []
    out_request_aliases: list[list[str]] = []

    for selected, stratum in zip(designations, strata):
        status, aliases = resolution_by_selected.get(str(selected), ("unresolved", []))
        req_aliases = _dedupe_keep_order([str(selected), *aliases])
        matched_orbit_ids: set[str] = set()
        for alias in req_aliases:
            matched_orbit_ids.update(alias_to_orbit_ids.get(alias, set()))
        matched_sorted = sorted(matched_orbit_ids)

        if len(matched_sorted) == 1:
            orbit_id: str | None = matched_sorted[0]
            reason = "matched"
        elif len(matched_sorted) == 0:
            orbit_id = None
            reason = "no_orbit_match"
        else:
            orbit_id = None
            reason = "ambiguous_orbit_match"

        out_selected.append(str(selected))
        out_stratum.append(None if stratum is None else str(stratum))
        out_status.append(str(status))
        out_orbit_id.append(orbit_id)
        out_matched_count.append(int(len(matched_sorted)))
        out_match_reason.append(reason)
        out_request_aliases.append(req_aliases)

    return pa.table(
        {
            "selected_designation": pa.array(out_selected, type=pa.large_string()),
            "stratum": pa.array(out_stratum, type=pa.large_string()),
            "resolution_status": pa.array(out_status, type=pa.large_string()),
            "orbit_id": pa.array(out_orbit_id, type=pa.large_string()),
            "matched_orbit_count": pa.array(out_matched_count, type=pa.int64()),
            "match_reason": pa.array(out_match_reason, type=pa.large_string()),
            "mpcq_request_provids": pa.array(out_request_aliases, type=pa.list_(pa.large_string())),
        }
    )


def _filter_truth_crossmatch_to_window(
    *,
    truth_crossmatch: pa.Table,
    months: Iterable[str],
    obscodes: set[str],
) -> pa.Table:
    if truth_crossmatch.num_rows == 0:
        return truth_crossmatch

    m = pc.fill_null(truth_crossmatch["matched"], False)
    m = pc.and_(m, pc.invert(pc.is_null(truth_crossmatch["match_observation_id"])))
    m = pc.and_(m, pc.invert(pc.is_null(truth_crossmatch["match_time_mjd_utc"])))
    m = pc.and_(m, pc.is_in(truth_crossmatch["obscode"], value_set=pa.array(sorted(obscodes))))

    # Month mask (OR across months).
    mt = pc.cast(truth_crossmatch["match_time_mjd_utc"], pa.float64())
    month_mask = None
    for ym in months:
        start_mjd, end_mjd = month_bounds_mjd_utc(str(ym))
        mm = pc.and_(pc.greater_equal(mt, pa.scalar(float(start_mjd))), pc.less(mt, pa.scalar(float(end_mjd))))
        month_mask = mm if month_mask is None else pc.or_(month_mask, mm)
    if month_mask is not None:
        m = pc.and_(m, month_mask)

    t = truth_crossmatch.filter(m).select(
        [
            "designation",
            "obscode",
            "truth_obsid",
            "truth_time_mjd_utc",
            "match_observation_id",
            "match_time_mjd_utc",
            "delta_t_sec",
            "distance_arcsec",
        ]
    )
    return t


def _map_truth_to_orbit_ids(
    *,
    truth_window: pa.Table,
    selected_orbit_mapping: pa.Table,
    designation_resolution: pa.Table,
) -> tuple[pa.Table, pa.Table]:
    alias_to_selected: dict[str, set[str]] = {}
    for selected, reqs in zip(
        designation_resolution["selected_designation"].to_pylist(),
        designation_resolution["mpcq_request_provids"].to_pylist(),
    ):
        if selected is None:
            continue
        selected_str = str(selected)
        aliases = [selected_str]
        if reqs is not None:
            aliases.extend([str(x) for x in reqs if x is not None and str(x).strip()])
        for alias in _dedupe_keep_order(aliases):
            alias_to_selected.setdefault(alias, set()).add(selected_str)

    selected_to_orbit: dict[str, str] = {}
    for selected, orbit_id in zip(
        selected_orbit_mapping["selected_designation"].to_pylist(),
        selected_orbit_mapping["orbit_id"].to_pylist(),
    ):
        if selected is None or orbit_id is None:
            continue
        selected_to_orbit[str(selected)] = str(orbit_id)

    mapped_orbit_ids: list[str | None] = []
    unresolved_rows: dict[tuple[str, str], dict[str, object]] = {}
    for designation in truth_window["designation"].to_pylist():
        if designation is None:
            mapped_orbit_ids.append(None)
            key = ("__null__", "designation_null")
            unresolved_rows.setdefault(
                key,
                {
                    "designation": "__null__",
                    "unresolved_reason": "designation_null",
                    "n_truth_rows": 0,
                    "selected_candidates": [],
                    "orbit_candidates": [],
                },
            )
            unresolved_rows[key]["n_truth_rows"] = int(unresolved_rows[key]["n_truth_rows"]) + 1
            continue

        des = str(designation)
        selected_candidates = set(alias_to_selected.get(des, set()))
        if not selected_candidates and des in selected_to_orbit:
            selected_candidates.add(des)
        orbit_candidates = sorted(
            {selected_to_orbit[s] for s in sorted(selected_candidates) if s in selected_to_orbit}
        )

        if len(orbit_candidates) == 1:
            mapped_orbit_ids.append(str(orbit_candidates[0]))
            continue

        mapped_orbit_ids.append(None)
        if not selected_candidates:
            reason = "designation_not_in_selected_aliases"
        elif not orbit_candidates:
            reason = "selected_designation_without_orbit_match"
        else:
            reason = "ambiguous_orbit_candidates"
        key = (des, reason)
        if key not in unresolved_rows:
            unresolved_rows[key] = {
                "designation": des,
                "unresolved_reason": reason,
                "n_truth_rows": 0,
                "selected_candidates": sorted(selected_candidates),
                "orbit_candidates": orbit_candidates,
            }
        unresolved_rows[key]["n_truth_rows"] = int(unresolved_rows[key]["n_truth_rows"]) + 1

    out = truth_window.append_column("orbit_id", pa.array(mapped_orbit_ids, type=pa.large_string()))
    mapped = out.filter(pc.is_valid(out["orbit_id"])).drop(["designation"])

    if unresolved_rows:
        rows = list(unresolved_rows.values())
        unresolved = pa.table(
            {
                "designation": pa.array([str(r["designation"]) for r in rows], type=pa.large_string()),
                "unresolved_reason": pa.array(
                    [str(r["unresolved_reason"]) for r in rows], type=pa.large_string()
                ),
                "n_truth_rows": pa.array([int(r["n_truth_rows"]) for r in rows], type=pa.int64()),
                "selected_candidates": pa.array(
                    [list(r["selected_candidates"]) for r in rows], type=pa.list_(pa.large_string())
                ),
                "orbit_candidates": pa.array(
                    [list(r["orbit_candidates"]) for r in rows], type=pa.list_(pa.large_string())
                ),
            }
        )
    else:
        unresolved = pa.table(
            {
                "designation": pa.array([], type=pa.large_string()),
                "unresolved_reason": pa.array([], type=pa.large_string()),
                "n_truth_rows": pa.array([], type=pa.int64()),
                "selected_candidates": pa.array([], type=pa.list_(pa.large_string())),
                "orbit_candidates": pa.array([], type=pa.list_(pa.large_string())),
            }
        )

    return mapped, unresolved


def _parse_stratum_parts(stratum: str | None) -> dict[str, str | None]:
    if stratum is None:
        return dict(regime=None, arc_bin=None, dt_bin=None, u_bin=None, i_bin=None)
    parts = [p.strip() for p in str(stratum).split("|") if p.strip()]
    if len(parts) != 5:
        return dict(regime=None, arc_bin=None, dt_bin=None, u_bin=None, i_bin=None)
    return dict(regime=parts[0], arc_bin=parts[1], dt_bin=parts[2], u_bin=parts[3], i_bin=parts[4])


def _bin_edges_label(v: float | None, edges: list[float], prefix: str) -> str:
    if v is None or (not np.isfinite(float(v))):
        return f"{prefix}_unknown"
    x = float(v)
    e = [float(a) for a in edges]
    if x < e[0]:
        return f"{prefix}_lt_{e[0]:g}"
    for i in range(1, len(e)):
        if x < e[i]:
            return f"{prefix}_{e[i-1]:g}_{e[i]:g}"
    return f"{prefix}_ge_{e[-1]:g}"


def _build_orbit_bins(
    *,
    orbit_ids: list[str],
    selected_orbit_mapping: pa.Table,
    cov_severity_parquet: Path,
    features_parquet: Path,
    sbdb_orbit_provenance: pa.Table,
) -> pa.Table:
    orbit_ids_arr = pa.array(orbit_ids, type=pa.large_string())
    base = pa.table({"orbit_id": orbit_ids_arr})

    # Core selection bins are sourced from selected_designations + designation_resolution mapping.
    selected_by_orbit: dict[str, list[str]] = {}
    strata_by_orbit: dict[str, list[str]] = {}
    for orbit_id, selected, stratum in zip(
        selected_orbit_mapping["orbit_id"].to_pylist(),
        selected_orbit_mapping["selected_designation"].to_pylist(),
        selected_orbit_mapping["stratum"].to_pylist(),
    ):
        if orbit_id is None:
            continue
        oid = str(orbit_id)
        if selected is not None and str(selected).strip():
            selected_by_orbit.setdefault(oid, []).append(str(selected))
        if stratum is not None and str(stratum).strip():
            strata_by_orbit.setdefault(oid, []).append(str(stratum))

    selected_first: list[str | None] = []
    stratum_first: list[str | None] = []
    regime: list[str | None] = []
    arc_bin: list[str | None] = []
    dt_bin: list[str | None] = []
    u_bin: list[str | None] = []
    i_bin: list[str | None] = []
    for oid in orbit_ids:
        selected_vals = sorted(set(selected_by_orbit.get(oid, [])))
        stratum_vals = sorted(set(strata_by_orbit.get(oid, [])))
        s = stratum_vals[0] if stratum_vals else None
        parts = _parse_stratum_parts(s)
        selected_first.append(selected_vals[0] if selected_vals else None)
        stratum_first.append(s)
        regime.append(parts["regime"])
        arc_bin.append(parts["arc_bin"])
        dt_bin.append(parts["dt_bin"])
        u_bin.append(parts["u_bin"])
        i_bin.append(parts["i_bin"])

    sel = pa.table(
        {
            "orbit_id": orbit_ids_arr,
            "selected_designation": pa.array(selected_first, type=pa.large_string()),
            "stratum": pa.array(stratum_first, type=pa.large_string()),
            "regime": pa.array(regime, type=pa.large_string()),
            "arc_bin": pa.array(arc_bin, type=pa.large_string()),
            "dt_bin": pa.array(dt_bin, type=pa.large_string()),
            "u_bin": pa.array(u_bin, type=pa.large_string()),
            "i_bin": pa.array(i_bin, type=pa.large_string()),
        }
    )

    # Covariance severity (uncertainty bins).
    cov = pq.read_table(
        str(cov_severity_parquet),
        columns=["orbit_id", "sigma_pos_rms", "anisotropy_pos", "cov_ok", "cov_invalid_reason"],
    )
    cov = cov.filter(pc.is_in(cov["orbit_id"], value_set=orbit_ids_arr))
    sigpos = cov["sigma_pos_rms"].to_pylist()
    aniso = cov["anisotropy_pos"].to_pylist()
    cov = cov.append_column(
        "sigma_pos_rms_bin",
        pa.array(
            [_bin_edges_label(v, [1e-6, 1e-4, 1e-2, 1.0, 10.0, 100.0], "sigpos") for v in sigpos],
            type=pa.large_string(),
        ),
    )
    cov = cov.append_column(
        "anisotropy_pos_bin",
        pa.array(
            [_bin_edges_label(v, [10.0, 100.0, 1000.0, 10000.0], "aniso") for v in aniso],
            type=pa.large_string(),
        ),
    )

    # Optional MPC enrichment is mapped through selected_designation -> orbit_id.
    orbit_type_by_orbit: dict[str, int | None] = {}
    u_param_by_orbit: dict[str, int | None] = {}
    if Path(features_parquet).exists():
        selected_designations = sorted(
            {
                str(x)
                for x in selected_orbit_mapping["selected_designation"].to_pylist()
                if x is not None and str(x).strip()
            }
        )
        if selected_designations:
            feat = pq.read_table(str(features_parquet), columns=["designation", "orbit_type_int", "u_param"])
            feat = feat.filter(
                pc.is_in(
                    feat["designation"],
                    value_set=pa.array(selected_designations, type=pa.large_string()),
                )
            )
            feat_by_selected: dict[str, list[tuple[int | None, int | None]]] = {}
            for des, orbit_type_int, u_param_val in zip(
                feat["designation"].to_pylist(),
                feat["orbit_type_int"].to_pylist(),
                feat["u_param"].to_pylist(),
            ):
                if des is None:
                    continue
                feat_by_selected.setdefault(str(des), []).append(
                    (
                        None if orbit_type_int is None else int(orbit_type_int),
                        None if u_param_val is None else int(u_param_val),
                    )
                )

            for oid in orbit_ids:
                orbit_type: int | None = None
                u_param: int | None = None
                for selected in sorted(set(selected_by_orbit.get(oid, []))):
                    for orbit_type_i, u_param_i in feat_by_selected.get(selected, []):
                        if orbit_type is None and orbit_type_i is not None:
                            orbit_type = int(orbit_type_i)
                        if u_param is None and u_param_i is not None:
                            u_param = int(u_param_i)
                        if orbit_type is not None and u_param is not None:
                            break
                    if orbit_type is not None and u_param is not None:
                        break
                orbit_type_by_orbit[oid] = orbit_type
                u_param_by_orbit[oid] = u_param

    feat = pa.table(
        {
            "orbit_id": orbit_ids_arr,
            "orbit_type_int": pa.array([orbit_type_by_orbit.get(oid) for oid in orbit_ids], type=pa.int64()),
            "u_param": pa.array([u_param_by_orbit.get(oid) for oid in orbit_ids], type=pa.int64()),
        }
    )

    out = base.join(sel, keys=["orbit_id"], join_type="left outer")
    out = out.join(cov, keys=["orbit_id"], join_type="left outer")
    out = out.join(feat, keys=["orbit_id"], join_type="left outer")
    out = out.join(sbdb_orbit_provenance, keys=["orbit_id"], join_type="left outer")

    orbit_type_vals = out["orbit_type_int"].to_pylist() if "orbit_type_int" in out.column_names else [None] * out.num_rows
    u_param_vals = out["u_param"].to_pylist() if "u_param" in out.column_names else [None] * out.num_rows
    out = out.append_column(
        "orbit_features_source",
        pa.array(
            [
                "mpc_optional" if (orbit_type_vals[i] is not None or u_param_vals[i] is not None) else "sbdb_core"
                for i in range(out.num_rows)
            ],
            type=pa.large_string(),
        ),
    )
    out = out.append_column(
        "orbit_features_complete",
        pa.array(
            [orbit_type_vals[i] is not None and u_param_vals[i] is not None for i in range(out.num_rows)],
            type=pa.bool_(),
        ),
    )

    if "nongrav_flag" not in out.column_names:
        out = out.append_column("nongrav_flag", pa.array([False] * out.num_rows, type=pa.bool_()))
    else:
        nongrav = out["nongrav_flag"].to_pylist()
        out = out.set_column(
            out.schema.get_field_index("nongrav_flag"),
            "nongrav_flag",
            pa.array([bool(x) if x is not None else False for x in nongrav], type=pa.bool_()),
        )
    return out


def _build_sbdb_orbit_provenance(*, orbits: Orbits) -> pa.Table:
    orbit_ids = [str(x) for x in orbits.orbit_id.to_pylist()]
    epoch_mjd_tt = [float(x) for x in orbits.coordinates.time.mjd().to_pylist()]

    a1: list[float | None] = []
    a2: list[float | None] = []
    a3: list[float | None] = []
    if "physical_parameters" in orbits.table.column_names:
        rows = orbits.table["physical_parameters"].to_pylist()
        for row in rows:
            if row is None:
                a1.append(None)
                a2.append(None)
                a3.append(None)
                continue
            v1 = row.get("a1", row.get("A1"))
            v2 = row.get("a2", row.get("A2"))
            v3 = row.get("a3", row.get("A3"))
            a1.append(None if v1 is None else float(v1))
            a2.append(None if v2 is None else float(v2))
            a3.append(None if v3 is None else float(v3))
    else:
        a1 = [None] * len(orbit_ids)
        a2 = [None] * len(orbit_ids)
        a3 = [None] * len(orbit_ids)

    has_nongrav_terms: list[bool] = []
    for v1, v2, v3 in zip(a1, a2, a3):
        has_term = False
        for v in (v1, v2, v3):
            if v is None:
                continue
            if not math.isfinite(float(v)):
                continue
            has_term = True
            break
        has_nongrav_terms.append(has_term)

    return pa.table(
        {
            "orbit_id": pa.array(orbit_ids, type=pa.large_string()),
            "orbit_source": pa.array(["sbdb"] * len(orbit_ids), type=pa.large_string()),
            "epoch_mjd_tt": pa.array(epoch_mjd_tt, type=pa.float64()),
            "has_nongrav_terms": pa.array(has_nongrav_terms, type=pa.bool_()),
            "a1": pa.array(a1, type=pa.float64()),
            "a2": pa.array(a2, type=pa.float64()),
            "a3": pa.array(a3, type=pa.float64()),
            "nongrav_flag": pa.array(has_nongrav_terms, type=pa.bool_()),
        }
    )


def _unique_orbit_ids_with_truth(truth_det: pa.Table) -> list[str]:
    if truth_det.num_rows == 0:
        return []
    ids = sorted({str(x) for x in truth_det["orbit_id"].to_pylist() if x is not None and str(x).strip()})
    return ids


def _pick_fast_orbits(
    *,
    orbit_bins: pa.Table,
    eligible_orbit_ids: list[str],
    k: int,
) -> list[str]:
    """
    Greedily select up to k orbit_ids that maximize coverage across the existing bin columns.

    Deterministic tie-break: orbit_id ascending.
    """
    if k <= 0 or not eligible_orbit_ids or orbit_bins.num_rows == 0:
        return []

    eligible_set = set(str(x) for x in eligible_orbit_ids)
    bins = orbit_bins.filter(
        pc.is_in(
            orbit_bins["orbit_id"],
            value_set=pa.array(sorted(eligible_set), type=pa.large_string()),
        )
    )
    if "cov_ok" in bins.column_names:
        # Ensure Stage-2 eligibility: sigma-point sampling requires finite covariance.
        bins = bins.filter(pc.fill_null(pc.cast(bins["cov_ok"], pa.bool_()), False))
    if bins.num_rows == 0:
        return []

    # Columns we want to represent (existing binning scheme + key severity/feature bins).
    cols = [
        "stratum",
        "regime",
        "arc_bin",
        "dt_bin",
        "u_bin",
        "i_bin",
        "orbit_type_int",
        "sigma_pos_rms_bin",
        "anisotropy_pos_bin",
    ]
    cols = [c for c in cols if c in bins.column_names]
    if not cols:
        # Fallback: just pick the first k eligible orbit_ids in sorted order.
        return sorted(eligible_orbit_ids)[: int(k)]

    orbit_ids = [str(x) for x in bins["orbit_id"].to_pylist()]

    def _row_values(i: int) -> dict[str, str]:
        out: dict[str, str] = {}
        for c in cols:
            v = bins[c][i].as_py()
            if v is None:
                continue
            s = str(v).strip()
            if not s:
                continue
            out[c] = s
        return out

    # Greedy set cover across per-column values.
    covered: dict[str, set[str]] = {c: set() for c in cols}
    selected: list[str] = []
    remaining = set(range(len(orbit_ids)))

    while remaining and len(selected) < int(k):
        best_i = None
        best_gain = -1
        best_oid = None
        for i in sorted(remaining, key=lambda j: orbit_ids[j]):
            oid = orbit_ids[i]
            if oid in selected:
                continue
            row = _row_values(i)
            gain = 0
            for c, val in row.items():
                if val not in covered[c]:
                    gain += 1
            if gain > best_gain or (gain == best_gain and (best_oid is None or oid < best_oid)):
                best_gain = gain
                best_i = i
                best_oid = oid

        if best_i is None or best_oid is None:
            break
        selected.append(best_oid)
        row = _row_values(best_i)
        for c, val in row.items():
            covered[c].add(val)
        remaining.remove(best_i)

        # If we can no longer gain anything, finish by filling with smallest remaining orbit_ids.
        if best_gain <= 0:
            rest = sorted([orbit_ids[i] for i in remaining if orbit_ids[i] not in selected])
            need = int(k) - len(selected)
            selected.extend(rest[:need])
            break

    # Ensure deterministic order in the serialized artifact.
    return sorted(selected)[: int(k)]


def bundle_paths(*, out_dir: Path) -> BundlePaths:
    out_dir = Path(out_dir)
    return BundlePaths(
        out_dir=out_dir,
        orbits_parquet=out_dir / "orbits.parquet",
        sbdb_orbit_provenance_parquet=out_dir / "orbits_sbdb_provenance.parquet",
        orbit_bins_parquet=out_dir / "orbit_bins.parquet",
        selected_orbit_mapping_parquet=out_dir / "selected_orbit_mapping.parquet",
        truth_detections_parquet=out_dir / "truth_detections.parquet",
        truth_frames_parquet=out_dir / "truth_frames.parquet",
        truth_unresolved_designations_parquet=out_dir / "truth_unresolved_designations.parquet",
        orbit_ids_with_truth_parquet=out_dir / "orbit_ids_with_truth.parquet",
        orbits_with_truth_parquet=out_dir / "orbits_with_truth.parquet",
        orbit_ids_fast6_parquet=out_dir / "orbit_ids_fast6.parquet",
        orbits_fast6_parquet=out_dir / "orbits_fast6.parquet",
        orbits_rejected_covariance_parquet=out_dir / "orbits_rejected_covariance.parquet",
        manifest_json=out_dir / "manifest.json",
        dropped_known_bad_parquet=out_dir / "dropped_orbits_known_bad.parquet",
    )


def prepare_benchmark_bundle(
    *,
    subset_dir: Path,
    out_dir: Path,
    detections_parquet: Path,
    months: Iterable[str],
    obscodes: Iterable[str],
    orbits_parquet: Path | None = None,
    truth_crossmatch_parquet: Path | None = None,
    selected_designations_parquet: Path | None = None,
    designation_resolution_parquet: Path | None = None,
    cov_severity_parquet: Path | None = None,
    features_parquet: Path | None = None,
    excluded_orbits_parquet: Path | None = None,
    keep_known_bad: bool = False,
    fast_orbits_k: int = 6,
    fail_on_unresolved_truth: bool = False,
) -> BundlePaths:
    subset_dir = Path(subset_dir)
    out_dir = Path(out_dir)
    detections_parquet = Path(detections_parquet)

    artifacts_dir = subset_dir / "artifacts"
    orbits_parquet = (
        Path(orbits_parquet)
        if orbits_parquet is not None
        else (artifacts_dir / "w_20200101_20240101__I41_T05_T08_W84" / "orbits_selected_sbdb.parquet")
    )
    orbits_inputs_dir = orbits_parquet.parent

    if truth_crossmatch_parquet is None:
        cand = orbits_inputs_dir / "truth_precovery_crossmatch.parquet"
        truth_crossmatch_parquet = cand if cand.exists() else (artifacts_dir / "truth_precovery_crossmatch.parquet")
    else:
        truth_crossmatch_parquet = Path(truth_crossmatch_parquet)

    if selected_designations_parquet is None:
        # Prefer window-specific selection when present, else fall back to the full-span selection.
        cand = orbits_inputs_dir / "selected_designations.parquet"
        selected_designations_parquet = (
            cand if cand.exists() else (artifacts_dir / "selected_designations_full_span.parquet")
        )
    else:
        selected_designations_parquet = Path(selected_designations_parquet)

    if cov_severity_parquet is None:
        cand = orbits_inputs_dir / "orbits_selected_sbdb_cov_severity.parquet"
        cov_severity_parquet = cand if cand.exists() else (artifacts_dir / "orbits_selected_sbdb_cov_severity.parquet")
    else:
        cov_severity_parquet = Path(cov_severity_parquet)

    if designation_resolution_parquet is None:
        candidates = [
            selected_designations_parquet.parent / "designation_resolution.parquet",
            orbits_inputs_dir / "designation_resolution.parquet",
            artifacts_dir / "designation_resolution.parquet",
        ]
        designation_resolution_parquet = next((p for p in candidates if p.exists()), None)
        if designation_resolution_parquet is None:
            raise FileNotFoundError(
                "designation_resolution.parquet not found. Pass --designation-resolution-parquet "
                "or generate it next to selected_designations.parquet."
            )
    else:
        designation_resolution_parquet = Path(designation_resolution_parquet)

    features_parquet = (
        Path(features_parquet)
        if features_parquet is not None
        else (artifacts_dir / "bq_designation_orbit_features_full_span.parquet")
    )
    excluded_orbits_parquet = (
        Path(excluded_orbits_parquet)
        if excluded_orbits_parquet is not None
        else (artifacts_dir / "tables" / "20260211T012705Z_trim_stage3rec0" / "excluded_orbits.parquet")
    )

    _ensure_dir(out_dir)
    paths = bundle_paths(out_dir=out_dir)

    orbits = Orbits.from_parquet(orbits_parquet)

    orbit_ids_all = _require_unique_orbit_ids(orbits)

    excluded: set[str] = set()
    if excluded_orbits_parquet.exists() and not bool(keep_known_bad):
        t_ex = pq.read_table(str(excluded_orbits_parquet), columns=["orbit_id"])
        excluded = {str(x).strip() for x in t_ex["orbit_id"].to_pylist() if x is not None and str(x).strip()}

    if excluded:
        keep_mask = np.asarray([x not in excluded for x in orbit_ids_all], dtype=bool)
        drop_ids = [orbit_ids_all[i] for i in np.nonzero(~keep_mask)[0].tolist()]
        pq.write_table(
            pa.table({"orbit_id": pa.array(drop_ids, type=pa.large_string())}),
            str(paths.dropped_known_bad_parquet),
        )
        orbits = orbits.take(pa.array(np.nonzero(keep_mask)[0], type=pa.int64()))
        orbit_ids_all = _require_unique_orbit_ids(orbits)
    else:
        pq.write_table(
            pa.table({"orbit_id": pa.array([], type=pa.large_string())}),
            str(paths.dropped_known_bad_parquet),
        )

    # Persist filtered orbits.
    orbits.to_parquet(paths.orbits_parquet)
    sbdb_provenance = _build_sbdb_orbit_provenance(orbits=orbits)
    pq.write_table(sbdb_provenance, str(paths.sbdb_orbit_provenance_parquet))

    selected_designations, selected_strata = _read_selected_designations(selected_designations_parquet)
    designation_resolution = read_designation_resolution(Path(designation_resolution_parquet))
    selected_orbit_mapping = _build_selected_orbit_mapping(
        designations=selected_designations,
        strata=selected_strata,
        designation_resolution=designation_resolution,
        orbits=orbits,
    )
    pq.write_table(selected_orbit_mapping, str(paths.selected_orbit_mapping_parquet))

    truth_cross = pq.read_table(
        str(truth_crossmatch_parquet),
        columns=[
            "matched",
            "designation",
            "obscode",
            "truth_obsid",
            "truth_time_mjd_utc",
            "match_observation_id",
            "match_time_mjd_utc",
            "delta_t_sec",
            "distance_arcsec",
        ],
    )

    # Month+station filtered truth rows (still in observation_id space).
    obscode_set = {str(x) for x in obscodes}
    truth_win = _filter_truth_crossmatch_to_window(
        truth_crossmatch=truth_cross,
        months=months,
        obscodes=obscode_set,
    )
    truth_mapped, truth_unresolved = _map_truth_to_orbit_ids(
        truth_window=truth_win,
        selected_orbit_mapping=selected_orbit_mapping,
        designation_resolution=designation_resolution,
    )
    pq.write_table(truth_unresolved, str(paths.truth_unresolved_designations_parquet))
    unresolved_truth_rows = (
        int(np.sum(np.asarray(truth_unresolved["n_truth_rows"].to_pylist(), dtype=np.int64)))
        if truth_unresolved.num_rows > 0
        else 0
    )
    if bool(fail_on_unresolved_truth) and unresolved_truth_rows > 0:
        raise RuntimeError(
            f"Truth mapping left {unresolved_truth_rows} rows unresolved "
            f"({truth_unresolved.num_rows} designation/reason groups)."
        )

    # Derive truth frame keys in the detections parquet keyspace.
    det_backend = DuckDbParquetBackend(parquet_path=detections_parquet)
    truth_base = truth_mapped.rename_columns(
        [
            "obscode",
            "truth_obsid",
            "truth_time_mjd_utc",
            "observation_id",
            "match_time_mjd_utc",
            "delta_t_sec",
            "distance_arcsec",
            "orbit_id",
        ]
    )
    truth_keys = det_backend.truth_frame_keys_from_truth_table(
        truth=truth_base.select(["orbit_id", "obscode", "observation_id"])
    )
    truth_det = truth_base.join(
        truth_keys,
        keys=["orbit_id", "obscode", "observation_id"],
        join_type="inner",
    )

    pq.write_table(truth_det, str(paths.truth_detections_parquet))

    orbit_ids_with_truth = _unique_orbit_ids_with_truth(truth_det)
    pq.write_table(
        pa.table({"orbit_id": pa.array(orbit_ids_with_truth, type=pa.large_string())}),
        str(paths.orbit_ids_with_truth_parquet),
    )
    orbit_ids_with_truth_set = set(orbit_ids_with_truth)
    if orbit_ids_with_truth:
        idx = [i for i, x in enumerate(orbit_ids_all) if x in orbit_ids_with_truth_set]
        orbits_with_truth = orbits.take(pa.array(idx, type=pa.int64()))
    else:
        orbits_with_truth = Orbits.empty()
    orbits_with_truth.to_parquet(paths.orbits_with_truth_parquet)

    if truth_det.num_rows > 0:
        truth_frames = (
            truth_det.select(["orbit_id", "obscode", "exposure_mjd_mid_key_us", "healpixel"])
            .group_by(["orbit_id", "obscode", "exposure_mjd_mid_key_us", "healpixel"])
            .aggregate([("orbit_id", "count")])
            .rename_columns(
                ["orbit_id", "obscode", "exposure_mjd_mid_key_us", "healpixel", "n_truth_detections"]
            )
        )
    else:
        truth_frames = pa.table(
            {
                "orbit_id": pa.array([], type=pa.large_string()),
                "obscode": pa.array([], type=pa.large_string()),
                "exposure_mjd_mid_key_us": pa.array([], type=pa.int64()),
                "healpixel": pa.array([], type=pa.int64()),
                "n_truth_detections": pa.array([], type=pa.int64()),
            }
        )
    pq.write_table(truth_frames, str(paths.truth_frames_parquet))

    orbit_bins = _build_orbit_bins(
        orbit_ids=orbit_ids_all,
        selected_orbit_mapping=selected_orbit_mapping,
        cov_severity_parquet=cov_severity_parquet,
        features_parquet=features_parquet,
        sbdb_orbit_provenance=sbdb_provenance,
    )
    pq.write_table(orbit_bins, str(paths.orbit_bins_parquet))

    # Persist orbit covariance rejection notes (structured artifact).
    if "cov_ok" in orbit_bins.column_names:
        bad = orbit_bins.filter(pc.equal(pc.fill_null(pc.cast(orbit_bins["cov_ok"], pa.bool_()), False), False))
        cols = [
            c
            for c in ["orbit_id", "cov_ok", "cov_invalid_reason", "sigma_pos_rms", "anisotropy_pos"]
            if c in bad.column_names
        ]
        pq.write_table(bad.select(cols), str(paths.orbits_rejected_covariance_parquet))
    else:
        pq.write_table(
            pa.table(
                {
                    "orbit_id": pa.array([], type=pa.large_string()),
                    "cov_ok": pa.array([], type=pa.bool_()),
                    "cov_invalid_reason": pa.array([], type=pa.large_string()),
                }
            ),
            str(paths.orbits_rejected_covariance_parquet),
        )

    # Deterministic small subset for fast benchmarking (truth-rich + bin-diverse).
    k_fast = int(fast_orbits_k)
    fast_ids = _pick_fast_orbits(orbit_bins=orbit_bins, eligible_orbit_ids=orbit_ids_with_truth, k=k_fast)
    pq.write_table(
        pa.table({"orbit_id": pa.array(fast_ids, type=pa.large_string())}),
        str(paths.orbit_ids_fast6_parquet),
    )
    fast_ids_set = set(fast_ids)
    if fast_ids:
        idx = [i for i, x in enumerate(orbit_ids_all) if x in fast_ids_set]
        orbits_fast = orbits.take(pa.array(idx, type=pa.int64()))
    else:
        orbits_fast = Orbits.empty()
    orbits_fast.to_parquet(paths.orbits_fast6_parquet)

    manifest = {
        "generated_at_utc": _utc_now_iso(),
        "subset_dir": str(subset_dir),
        "months": list(months),
        "obscodes": list(obscodes),
        "detections_parquet": str(detections_parquet),
        "inputs": {
            "orbits_parquet": str(orbits_parquet),
            "truth_crossmatch_parquet": str(truth_crossmatch_parquet),
            "selected_designations_parquet": str(selected_designations_parquet),
            "designation_resolution_parquet": str(designation_resolution_parquet),
            "cov_severity_parquet": str(cov_severity_parquet),
            "features_parquet": str(features_parquet),
        },
        "outputs": {
            "orbits_parquet": str(paths.orbits_parquet),
            "sbdb_orbit_provenance_parquet": str(paths.sbdb_orbit_provenance_parquet),
            "orbit_bins_parquet": str(paths.orbit_bins_parquet),
            "selected_orbit_mapping_parquet": str(paths.selected_orbit_mapping_parquet),
            "truth_detections_parquet": str(paths.truth_detections_parquet),
            "truth_frames_parquet": str(paths.truth_frames_parquet),
            "truth_unresolved_designations_parquet": str(paths.truth_unresolved_designations_parquet),
            "orbit_ids_with_truth_parquet": str(paths.orbit_ids_with_truth_parquet),
            "orbits_with_truth_parquet": str(paths.orbits_with_truth_parquet),
            "orbit_ids_fast6_parquet": str(paths.orbit_ids_fast6_parquet),
            "orbits_fast6_parquet": str(paths.orbits_fast6_parquet),
            "orbits_rejected_covariance_parquet": str(paths.orbits_rejected_covariance_parquet),
            "dropped_orbits_known_bad_parquet": str(paths.dropped_known_bad_parquet),
        },
        "counts": {
            "n_orbits": int(len(orbits)),
            "n_orbits_with_truth": int(len(orbit_ids_with_truth)),
            "n_orbits_fast": int(len(fast_ids)),
            "n_orbits_rejected_covariance": int(bad.num_rows) if "cov_ok" in orbit_bins.column_names else 0,
            "n_truth_unresolved_designations": int(truth_unresolved.num_rows),
            "n_truth_unresolved_rows": int(unresolved_truth_rows),
            "n_truth_detections": int(truth_det.num_rows),
            "n_truth_frames": int(truth_frames.num_rows),
        },
        "fast_orbits": {
            "k": int(k_fast),
            "orbit_ids": fast_ids,
        },
    }
    paths.manifest_json.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return paths


__all__ = [
    "BundlePaths",
    "bundle_paths",
    "prepare_benchmark_bundle",
]
