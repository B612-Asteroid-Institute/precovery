from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from precovery.search.gate_defaults import (
    DEFAULT_APPLY_SYSTEMATIC_IF_REPORTED_RMS_LT_ARCSEC_BY_OBSCODE,
    DEFAULT_INVALID_SIGMA_FILL_FLOOR_ARCSEC_BY_OBSCODE,
    DEFAULT_INVALID_SIGMA_FILL_FLOOR_ARCSEC_GLOBAL,
    DEFAULT_SIGMA_SYSTEMATIC_ARCSEC_BY_OBSCODE,
)


def _read_parquet(path: Path) -> pa.Table:
    return pq.read_table(str(path))


def _maybe_read_parquet(path: Path) -> pa.Table | None:
    return None if not path.exists() else _read_parquet(path)


def _fmt(x: object) -> str:
    if x is None:
        return ""
    if isinstance(x, float):
        if not np.isfinite(x):
            return ""
        if abs(x) >= 100:
            return f"{x:.1f}"
        if abs(x) >= 10:
            return f"{x:.3f}"
        return f"{x:.6f}"
    return str(x)


def _table_to_markdown(t: pa.Table, *, max_rows: int | None) -> str:
    if t.num_rows == 0 or len(t.column_names) == 0:
        return "_(empty)_\n"
    if max_rows is not None and max_rows > 0 and t.num_rows > int(max_rows):
        t = t.slice(0, int(max_rows))

    cols = list(t.column_names)
    rows: list[list[str]] = [cols]
    for i in range(int(t.num_rows)):
        rows.append([_fmt(t[c][i].as_py()) for c in cols])
    widths = [max(len(r[j]) for r in rows) for j in range(len(cols))]

    def line(vals: list[str]) -> str:
        return "| " + " | ".join(v.ljust(widths[i]) for i, v in enumerate(vals)) + " |"

    out = [line(rows[0])]
    out.append("| " + " | ".join("-" * w for w in widths) + " |")
    out.extend(line(r) for r in rows[1:])
    out.append("")
    return "\n".join(out)


def _select_existing(t: pa.Table, cols: list[str]) -> pa.Table:
    keep = [c for c in cols if c in t.column_names]
    return t.select(keep) if keep else pa.table({})


def _safe_f64(t: pa.Table, col: str) -> pa.Array | None:
    if col not in t.column_names:
        return None
    return pc.cast(t[col], pa.float64())


def _safe_i64(t: pa.Table, col: str) -> pa.Array | None:
    if col not in t.column_names:
        return None
    return pc.cast(pc.fill_null(t[col], 0), pa.int64())


def _sort(t: pa.Table, *, keys: list[tuple[str, str]]) -> pa.Table:
    present = [(c, d) for (c, d) in keys if c in t.column_names]
    return t if not present else t.sort_by(present)


@dataclass(frozen=True)
class GateSettings:
    gate_n_sigma: float
    det_sigma_floor_arcsec: float
    det_sigma_floor_arcsec_by_obscode: dict[str, float]
    det_sigma_sys_arcsec_by_obscode: dict[str, float]
    det_sigma_sys_apply_rms_lt_arcsec_by_obscode: dict[str, float]


def _parse_json_float_map(value: object) -> dict[str, float]:
    if value is None:
        return {}
    s = str(value).strip()
    if not s:
        return {}
    try:
        raw = json.loads(s)
    except Exception:
        return {}
    if not isinstance(raw, dict):
        return {}
    out: dict[str, float] = {}
    for k, v in raw.items():
        try:
            out[str(k)] = float(v)
        except Exception:
            continue
    return out


def _gate_settings_from_backend_runs(*, backend_runs: pa.Table) -> GateSettings:
    gate_n_sigma = 3.0
    det_sigma_floor_arcsec = float(DEFAULT_INVALID_SIGMA_FILL_FLOOR_ARCSEC_GLOBAL)
    det_sigma_floor_arcsec_by_obscode = dict(DEFAULT_INVALID_SIGMA_FILL_FLOOR_ARCSEC_BY_OBSCODE)
    det_sigma_sys_arcsec_by_obscode = dict(DEFAULT_SIGMA_SYSTEMATIC_ARCSEC_BY_OBSCODE)
    det_sigma_sys_apply_rms_lt_arcsec_by_obscode = dict(
        DEFAULT_APPLY_SYSTEMATIC_IF_REPORTED_RMS_LT_ARCSEC_BY_OBSCODE
    )

    if "gate_n_sigma" in backend_runs.column_names:
        v = backend_runs["gate_n_sigma"][0].as_py()
        if v is not None:
            gate_n_sigma = float(v)
    if "det_sigma_floor_arcsec" in backend_runs.column_names:
        v = backend_runs["det_sigma_floor_arcsec"][0].as_py()
        if v is not None:
            det_sigma_floor_arcsec = float(v)
    if "det_sigma_floor_arcsec_by_obscode" in backend_runs.column_names:
        m = _parse_json_float_map(backend_runs["det_sigma_floor_arcsec_by_obscode"][0].as_py())
        if m:
            det_sigma_floor_arcsec_by_obscode = m
    if "det_sigma_sys_arcsec_by_obscode" in backend_runs.column_names:
        m = _parse_json_float_map(backend_runs["det_sigma_sys_arcsec_by_obscode"][0].as_py())
        if m:
            det_sigma_sys_arcsec_by_obscode = m
    if "det_sigma_sys_apply_rms_lt_arcsec_by_obscode" in backend_runs.column_names:
        m = _parse_json_float_map(
            backend_runs["det_sigma_sys_apply_rms_lt_arcsec_by_obscode"][0].as_py()
        )
        if m:
            det_sigma_sys_apply_rms_lt_arcsec_by_obscode = m

    return GateSettings(
        gate_n_sigma=float(gate_n_sigma),
        det_sigma_floor_arcsec=float(det_sigma_floor_arcsec),
        det_sigma_floor_arcsec_by_obscode=dict(det_sigma_floor_arcsec_by_obscode),
        det_sigma_sys_arcsec_by_obscode=dict(det_sigma_sys_arcsec_by_obscode),
        det_sigma_sys_apply_rms_lt_arcsec_by_obscode=dict(
            det_sigma_sys_apply_rms_lt_arcsec_by_obscode
        ),
    )


def _rank_section(
    *,
    t: pa.Table,
    title: str,
    cols: list[str],
    sort_keys: list[tuple[str, str]],
    max_rows: int | None,
) -> str:
    t2 = _select_existing(t, cols)
    t2 = _sort(t2, keys=sort_keys)
    return f"## {title}\n\n" + _table_to_markdown(t2, max_rows=max_rows)


def _filter_mapping_audit(*, backend_runs: pa.Table) -> str:
    """
    Audit filter→canonical mapping from Stage23 artifacts.

    This reports:
    - distinct reported dataset filters used for targets
    - distinct canonical_filter_id values produced in preds
    - any (obscode, filter) pairs that required SDSS/PS1 fallback (detected by strict mapping)
    """
    if backend_runs.num_rows == 0:
        return ""
    if "bench_run_dir" not in backend_runs.column_names:
        return "## Filter mapping audit\n\n_(missing `bench_run_dir` in backend_runs; re-run benchmark with updated harness)_\n"

    # We expect one bench_run_dir per run dir; take the first.
    bench_run_dir = backend_runs["bench_run_dir"][0].as_py()
    if bench_run_dir is None or str(bench_run_dir).strip() == "":
        return "## Filter mapping audit\n\n_(missing bench_run_dir value)_\n"
    stage23 = Path(str(bench_run_dir)) / "stage23"
    targets_p = stage23 / "targets.parquet"
    preds_p = stage23 / "preds.parquet"
    if not targets_p.exists() or not preds_p.exists():
        return f"## Filter mapping audit\n\n_(missing stage23 artifacts under `{stage23}`)_\n"

    try:
        targets = _read_parquet(targets_p)
        preds = _read_parquet(preds_p)
    except Exception:
        return f"## Filter mapping audit\n\n_(failed to read stage23 artifacts under `{stage23}`)_\n"

    md: list[str] = []
    md.append("## Filter mapping audit\n")
    md.append(f"- stage23: `{stage23}`\n")

    # Distinct reported filters.
    if targets.num_rows > 0 and "obscode" in targets.column_names and "filter" in targets.column_names:
        t = targets.select(["obscode", "filter"])
        t = t.group_by(["obscode", "filter"]).aggregate([]).sort_by([("obscode", "ascending"), ("filter", "ascending")])
        md.append("### Reported dataset filters (from `targets.parquet`)\n")
        md.append(_table_to_markdown(t, max_rows=None))

    # Distinct canonical filter IDs.
    if preds.num_rows > 0 and "canonical_filter_id" in preds.column_names:
        c = preds.select(["canonical_filter_id"]).group_by(["canonical_filter_id"]).aggregate([]).sort_by(
            [("canonical_filter_id", "ascending")]
        )
        md.append("### Canonical filter IDs used (from `preds.parquet`)\n")
        md.append(_table_to_markdown(c, max_rows=None))

    # Strict mapping check: detect fallbacks.
    try:
        from adam_core.photometry.bandpasses.api import map_to_canonical_filter_bands  # type: ignore
    except Exception:
        md.append("### Strict mapping check\n\n_(adam-core photometry APIs unavailable; skipped)_\n")
        return "\n".join(md)

    if targets.num_rows == 0 or "obscode" not in targets.column_names or "filter" not in targets.column_names:
        md.append("### Strict mapping check\n\n_(targets missing obscode/filter)_\n")
        return "\n".join(md)

    obsc = pc.cast(targets["obscode"], pa.large_string())
    bands = pc.cast(targets["filter"], pa.large_string())
    # Best-effort: if strict mapping fails, parse the raised message for fallback pairs.
    fallback_pairs: list[str] = []
    try:
        _ = map_to_canonical_filter_bands(obsc, bands, allow_fallback_filters=False)
        md.append("### Strict mapping check\n\nNo SDSS/PS1 fallback pairs detected.\n")
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        # Expected format from adam-core:
        #   "No non-fallback mapping found for: CODE|band, CODE|band. Set allow_fallback_filters=True ..."
        needle = "No non-fallback mapping found for: "
        if needle in msg:
            tail = msg.split(needle, 1)[1]
            tail = tail.split(". Set allow_fallback_filters=", 1)[0]
            parts = [p.strip() for p in tail.split(",") if p.strip()]
            fallback_pairs = parts
        md.append("### Strict mapping check\n")
        if fallback_pairs:
            md.append("Some (obscode, reported band) pairs required generic-band fallbacks (SDSS/PS1):\n")
            t = pa.Table.from_pylist([{"code_band": p} for p in fallback_pairs]).sort_by([("code_band", "ascending")])
            md.append(_table_to_markdown(t, max_rows=None))
        else:
            md.append(f"_(strict mapping failed; could not parse fallbacks)_\n\n```\n{msg}\n```\n")

    return "\n".join(md).strip() + "\n"


def _truth_diagnostics_section(*, backend_runs: pa.Table, max_rows: int | None) -> str:
    """
    Truth-only diagnostics derived from stage23 artifacts + truth parquet + detections dataset.

    This is intentionally lightweight: it only touches truth rows (O(10^2-10^3)),
    not the full candidates table.
    """
    if backend_runs.num_rows == 0:
        return ""
    need_cols = {"bench_run_dir", "truth_parquet"}
    if not need_cols.issubset(set(backend_runs.column_names)):
        missing = sorted(need_cols - set(backend_runs.column_names))
        return "## Truth diagnostics\n\n_(missing columns in backend_runs: " + ", ".join(missing) + ")_\n"

    bench_run_dir = backend_runs["bench_run_dir"][0].as_py()
    truth_parquet = backend_runs["truth_parquet"][0].as_py()
    if not bench_run_dir or not truth_parquet:
        return "## Truth diagnostics\n\n_(missing bench_run_dir/truth_parquet values)_\n"

    stage23 = Path(str(bench_run_dir)) / "stage23"
    preds_p = stage23 / "preds.parquet"
    triples_p = stage23 / "triples.parquet"
    if not preds_p.exists() or not triples_p.exists():
        return f"## Truth diagnostics\n\n_(missing stage23 preds/triples under `{stage23}`)_\n"

    det_path: str | None = None
    if "detections_parquet" in backend_runs.column_names:
        vals = [backend_runs["detections_parquet"][i].as_py() for i in range(int(backend_runs.num_rows))]
        vals = [v for v in vals if v is not None and str(v).strip() != ""]
        det_path = vals[0] if vals else None
    if det_path is None or str(det_path).strip() == "":
        return "## Truth diagnostics\n\n_(missing `detections_parquet` in backend_runs; re-run benchmark with updated harness)_\n"

    truth_path = Path(str(truth_parquet))
    if not truth_path.exists():
        return f"## Truth diagnostics\n\n_(missing truth parquet `{truth_path}`)_\n"

    try:
        preds = _read_parquet(preds_p)
        triples = _read_parquet(triples_p)
        truth = _read_parquet(truth_path)
    except Exception:
        return "## Truth diagnostics\n\n_(failed to read inputs)_\n"

    # Normalize truth columns to the expected minimal set.
    # The harness already uses observation_id namespace for truth scoring.
    cols = set(truth.column_names)
    if "orbit_id" not in cols or "obscode" not in cols:
        return "## Truth diagnostics\n\n_(truth parquet missing orbit_id/obscode)_\n"
    if "observation_id" not in cols:
        return "## Truth diagnostics\n\n_(truth parquet missing observation_id)_\n"
    if "exposure_mjd_mid_key_us" not in cols or "healpixel" not in cols:
        return "## Truth diagnostics\n\n_(truth parquet missing exposure_mjd_mid_key_us/healpixel)_\n"

    # Restrict to matched truth rows if present (bench bundles typically have matched already filtered).
    if "matched" in truth.column_names:
        truth = truth.filter(pc.fill_null(pc.cast(truth["matched"], pa.bool_()), False)).drop(["matched"])

    # Join truth -> preds on (orbit_id, obscode, exposure_mjd_mid_key_us).
    want_preds = [
        "orbit_id",
        "obscode",
        "exposure_mjd_mid_key_us",
        "pred_lon_deg",
        "pred_lat_deg",
        "cov_ll_00",
        "cov_ll_01",
        "cov_ll_11",
        "pred_mag",
        "canonical_filter_id",
    ]
    preds2 = preds.select([c for c in want_preds if c in preds.column_names])
    truth = truth.join(
        preds2,
        keys=["orbit_id", "obscode", "exposure_mjd_mid_key_us"],
        join_type="left outer",
    )

    # Frame-key membership: truth in Stage3 frame-candidates if its full frame_key exists in triples.
    sep = pa.scalar("|", type=pa.large_string())

    def _frame_key(tbl: pa.Table) -> pa.Array:
        return pc.binary_join_element_wise(
            pc.cast(tbl["orbit_id"], pa.large_string()),
            pc.binary_join_element_wise(
                pc.cast(tbl["obscode"], pa.large_string()),
                pc.binary_join_element_wise(
                    pc.cast(tbl["exposure_mjd_mid_key_us"], pa.large_string()),
                    pc.cast(tbl["healpixel"], pa.large_string()),
                    sep,
                ),
                sep,
            ),
            sep,
        )

    trip_fk = _frame_key(triples)
    truth_fk = _frame_key(truth)
    in_triples = pc.is_in(truth_fk, value_set=trip_fk)
    truth = truth.append_column("_in_triples", pc.cast(in_triples, pa.bool_()))

    # Fetch observed rows for truth obsids via DuckDB (small join).
    try:
        import duckdb  # type: ignore

        def _sql_quote(s: str) -> str:
            return "'" + str(s).replace("'", "''") + "'"

        con = duckdb.connect(database=":memory:")
        con.execute(
            f"CREATE OR REPLACE VIEW det AS SELECT * FROM read_parquet({_sql_quote(str(det_path))})"
        )
        con.register("truth_ids", truth.select(["obscode", "observation_id"]))
        obs_tbl = con.execute(
            """
            SELECT
              t.obscode,
              t.observation_id,
              d.ra_deg,
              d.dec_deg,
              d.ra_sigma_deg,
              d.dec_sigma_deg,
              d.mag,
              d.mag_sigma,
              d.filter AS obs_filter
            FROM truth_ids t
            INNER JOIN det d
              ON d.obscode = t.obscode
             AND d.observation_id = t.observation_id
            """
        ).fetch_arrow_table()
        con.close()
    except Exception as e:
        return "## Truth diagnostics\n\n_(DuckDB truth join failed: " + str(e) + ")_\n"

    # Normalize join key types.
    for c in ["obscode", "observation_id"]:
        if c in obs_tbl.column_names:
            obs_tbl = obs_tbl.set_column(
                obs_tbl.schema.get_field_index(c), c, pc.cast(obs_tbl[c], pa.large_string())
            )
    truth = truth.join(obs_tbl, keys=["obscode", "observation_id"], join_type="left outer")

    gate_settings = _gate_settings_from_backend_runs(backend_runs=backend_runs)
    gate_n_sigma = gate_settings.gate_n_sigma
    det_sigma_floor_arcsec = gate_settings.det_sigma_floor_arcsec
    det_sigma_floor_arcsec_by_obscode = gate_settings.det_sigma_floor_arcsec_by_obscode
    faint_margin_mag = 0.0
    det_sigma_sys_arcsec_by_obscode = gate_settings.det_sigma_sys_arcsec_by_obscode
    det_sigma_sys_apply_rms_lt_arcsec_by_obscode = (
        gate_settings.det_sigma_sys_apply_rms_lt_arcsec_by_obscode
    )
    if "faint_frame_skip_margin_mag" in backend_runs.column_names:
        v = backend_runs["faint_frame_skip_margin_mag"][0].as_py()
        faint_margin_mag = 0.0 if v is None else float(v)

    # Sigma availability audit.
    ra_sig = pc.cast(truth["ra_sigma_deg"], pa.float64())
    dec_sig = pc.cast(truth["dec_sigma_deg"], pa.float64())
    sig_ok = pc.and_(
        pc.and_(pc.is_finite(ra_sig), pc.greater(ra_sig, pa.scalar(0.0, pa.float64()))),
        pc.and_(pc.is_finite(dec_sig), pc.greater(dec_sig, pa.scalar(0.0, pa.float64()))),
    )
    sig_missing = pc.invert(pc.fill_null(sig_ok, False))
    by_sig = (
        truth.select(["obscode"])
        .append_column("_sig_missing", pc.cast(sig_missing, pa.int64()))
        .group_by(["obscode"])
        .aggregate([("_sig_missing", "sum"), ("obscode", "count")])
        .rename_columns(["obscode", "n_sig_missing_or_invalid", "n_truth_rows"])
        .sort_by([("n_sig_missing_or_invalid", "descending"), ("obscode", "ascending")])
    )

    md: list[str] = []
    md.append("## Truth diagnostics\n")
    md.append(f"- detections_parquet: `{det_path}`\n")
    md.append(f"- gate_n_sigma: `{_fmt(gate_n_sigma)}`\n")
    md.append(f"- det_sigma_floor_arcsec: `{_fmt(det_sigma_floor_arcsec)}` (applied only when sigma missing/invalid)\n")
    md.append(
        "- det_sigma_floor_arcsec_by_obscode: "
        f"`{json.dumps(det_sigma_floor_arcsec_by_obscode, sort_keys=True)}`\n"
    )
    md.append(
        "- det_sigma_sys_arcsec_by_obscode: "
        f"`{json.dumps(det_sigma_sys_arcsec_by_obscode, sort_keys=True)}`\n"
    )
    md.append(
        "- det_sigma_sys_apply_rms_lt_arcsec_by_obscode: "
        f"`{json.dumps(det_sigma_sys_apply_rms_lt_arcsec_by_obscode, sort_keys=True)}`\n"
    )
    md.append(f"- faint_frame_skip_margin_mag: `{_fmt(faint_margin_mag)}`\n")

    md.append("### Sigma availability (truth rows)\n")
    md.append(_table_to_markdown(by_sig, max_rows=None))

    # Limiting-mag diagnostics + margin recommendation: use presets only.
    try:
        from precovery.limiting_magnitude_presets import build_default_limiting_magnitudes_table

        lm = build_default_limiting_magnitudes_table()
        codefid = pc.binary_join_element_wise(
            pc.cast(truth["obscode"], pa.large_string()),
            pc.cast(truth["canonical_filter_id"], pa.large_string()),
            sep,
        )
        lm_codefid = pc.binary_join_element_wise(lm.obscode, lm.filter_id, sep)
        idx = pc.fill_null(pc.index_in(codefid, value_set=lm_codefid), -1)
        valid = pc.greater_equal(idx, 0)
        idx_safe = pc.cast(pc.if_else(valid, idx, 0), pa.int64())
        lim = pc.take(pc.cast(lm.limiting_mag, pa.float64()), idx_safe)
        lim = pc.if_else(valid, lim, None)
        truth = truth.append_column("_limit_mag", pc.cast(lim, pa.float64()))
    except Exception:
        truth = truth.append_column("_limit_mag", pa.nulls(truth.num_rows, type=pa.float64()))

    pred_mag = pc.cast(truth["pred_mag"], pa.float64())
    limit_mag = pc.cast(truth["_limit_mag"], pa.float64())
    # Required margin per truth row (when pred_mag and limit are finite): pred_mag - limit
    req_margin = pc.subtract(pred_mag, limit_mag)
    truth = truth.append_column("_req_margin", pc.cast(req_margin, pa.float64()))

    obs_mag = pc.cast(truth["mag"], pa.float64())
    finite_req = pc.and_(pc.and_(pc.is_finite(pred_mag), pc.is_finite(limit_mag)), pc.is_finite(obs_mag))
    req = truth.filter(finite_req)
    if req.num_rows > 0:
        arr = pc.cast(req["_req_margin"], pa.float64()).to_numpy(zero_copy_only=False)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            p99 = float(np.quantile(arr, 0.99))
            # Clamp to >=0 and add a small cushion.
            reco = max(0.0, p99) + 0.05
            md.append("### Limiting-mag margin recommendation (presets-only)\n")
            md.append(f"- p99(pred_mag - limit_mag): `{_fmt(p99)}`\n")
            md.append(f"- recommended faint_frame_skip_margin_mag: `{_fmt(reco)}` (p99 + 0.05, clamped >=0)\n")
    else:
        md.append("### Limiting-mag margin recommendation\n\n_(no finite pred_mag/limit_mag rows)_\n")

    # Limiting-mag skip approximation at current margin (truth rows with finite pred_mag/limit).
    finite_skip = pc.and_(pc.is_finite(pred_mag), pc.is_finite(limit_mag))
    too_faint = pc.greater(pred_mag, pc.add(limit_mag, pa.scalar(float(faint_margin_mag), pa.float64())))
    too_faint = pc.and_(finite_skip, pc.fill_null(too_faint, False))
    skip_by_code = (
        truth.select(["obscode"])
        .append_column("_too_faint", pc.cast(too_faint, pa.int64()))
        .group_by(["obscode"])
        .aggregate([("_too_faint", "sum"), ("obscode", "count")])
        .rename_columns(["obscode", "n_truth_pred_mag_gt_limit_plus_margin", "n_truth_rows"])
        .sort_by([("n_truth_pred_mag_gt_limit_plus_margin", "descending"), ("obscode", "ascending")])
    )
    md.append("### Limiting-mag skip approximation (truth rows)\n")
    md.append(_table_to_markdown(skip_by_code, max_rows=None))

    # innov_ellipse truth-only diagnostics (compute chi2 on truth rows that are in triples).
    in_triples_tbl = truth.filter(pc.cast(truth["_in_triples"], pa.bool_()))
    need = {"pred_lon_deg", "pred_lat_deg", "cov_ll_00", "cov_ll_01", "cov_ll_11", "ra_deg", "dec_deg"}
    if in_triples_tbl.num_rows > 0 and need.issubset(set(in_triples_tbl.column_names)):
        q = _compute_residual_quantiles(
            in_triples_tbl,
            gate_n_sigma=gate_n_sigma,
            det_sigma_floor_arcsec=det_sigma_floor_arcsec,
            det_sigma_floor_arcsec_by_obscode=det_sigma_floor_arcsec_by_obscode,
            det_sigma_sys_arcsec_by_obscode=det_sigma_sys_arcsec_by_obscode,
            det_sigma_sys_apply_rms_lt_arcsec_by_obscode=det_sigma_sys_apply_rms_lt_arcsec_by_obscode,
        )
        if q is not None:
            md.append("### innov_ellipse truth-only diagnostics (truth rows in Stage3 frame-candidates)\n")
            md.append(
                f"- n_truth_in_triples: `{q['n_truth_in_triples']}`\n"
                f"- n_keep: `{q['n_keep']}`\n"
                f"- n_reject: `{q['n_reject']}`\n"
            )
            if "chi2_p50" in q:
                md.append(
                    f"- chi2 quantiles (all): p50 `{_fmt(q['chi2_p50'])}`, p90 `{_fmt(q['chi2_p90'])}`, p99 `{_fmt(q['chi2_p99'])}`\n"
                )
            if "offset_arcsec_p50" in q:
                md.append(
                    f"- offset_arcsec quantiles (all): p50 `{_fmt(q['offset_arcsec_p50'])}`, p90 `{_fmt(q['offset_arcsec_p90'])}`, p99 `{_fmt(q['offset_arcsec_p99'])}`\n"
                )

    # Catastrophic miss detector: truth rows not in triples (Stage3 geometry miss bucket).
    miss_geom = truth.filter(pc.invert(pc.cast(truth["_in_triples"], pa.bool_())))
    if miss_geom.num_rows > 0 and {"pred_lon_deg", "pred_lat_deg", "ra_deg", "dec_deg"}.issubset(
        set(miss_geom.column_names)
    ):
        lon_o = pc.cast(miss_geom["ra_deg"], pa.float64()).to_numpy(zero_copy_only=False)
        lat_o = pc.cast(miss_geom["dec_deg"], pa.float64()).to_numpy(zero_copy_only=False)
        lon_p = pc.cast(miss_geom["pred_lon_deg"], pa.float64()).to_numpy(zero_copy_only=False)
        lat_p = pc.cast(miss_geom["pred_lat_deg"], pa.float64()).to_numpy(zero_copy_only=False)
        cos_lat = np.cos(np.deg2rad(lat_p))
        cos_lat = np.where(np.isfinite(cos_lat) & (np.abs(cos_lat) > 1e-12), cos_lat, 1e-12)
        # Wrap-safe dlon in degrees.
        dlon = (lon_o - lon_p + 180.0) % 360.0 - 180.0
        dx = dlon * cos_lat
        dy = lat_o - lat_p
        off_arcsec = np.sqrt(dx * dx + dy * dy) * 3600.0
        miss_geom = miss_geom.append_column("_offset_arcsec", pa.array(off_arcsec, type=pa.float64()))

        # Summarize by orbit_id.
        m = miss_geom.select(["orbit_id", "_offset_arcsec"]).group_by(["orbit_id"]).aggregate(
            [("_offset_arcsec", "count"), ("_offset_arcsec", "max")]
        ).rename_columns(["orbit_id", "n_truth_geom_missed", "max_offset_arcsec"])
        m = m.sort_by([("max_offset_arcsec", "descending"), ("n_truth_geom_missed", "descending")])
        md.append("### Catastrophic miss detector (truth not in Stage3 frame-candidates)\n")
        md.append(_table_to_markdown(m, max_rows=max_rows))

    return "\n".join(md).strip() + "\n"


def _compute_residual_quantiles(
    in_triples_tbl: pa.Table,
    *,
    gate_n_sigma: float,
    det_sigma_floor_arcsec: float,
    det_sigma_floor_arcsec_by_obscode: dict[str, float],
    det_sigma_sys_arcsec_by_obscode: dict[str, float],
    det_sigma_sys_apply_rms_lt_arcsec_by_obscode: dict[str, float],
) -> dict[str, float | int] | None:
    """
    Compute p50/p90/p99 for offset_arcsec and chi2 on truth rows in triples.
    Same formula as the innov_ellipse truth-only diagnostics block.
    Returns None if table empty or missing required columns.
    """
    need = {"pred_lon_deg", "pred_lat_deg", "cov_ll_00", "cov_ll_01", "cov_ll_11", "ra_deg", "dec_deg"}
    if in_triples_tbl.num_rows == 0 or not need.issubset(set(in_triples_tbl.column_names)):
        return None
    obsc = pc.cast(in_triples_tbl["obscode"], pa.large_string()).to_numpy(zero_copy_only=False)
    lon_o = pc.cast(in_triples_tbl["ra_deg"], pa.float64()).to_numpy(zero_copy_only=False)
    lat_o = pc.cast(in_triples_tbl["dec_deg"], pa.float64()).to_numpy(zero_copy_only=False)
    lon_p = pc.cast(in_triples_tbl["pred_lon_deg"], pa.float64()).to_numpy(zero_copy_only=False)
    lat_p = pc.cast(in_triples_tbl["pred_lat_deg"], pa.float64()).to_numpy(zero_copy_only=False)
    sig_lon = pc.cast(in_triples_tbl["ra_sigma_deg"], pa.float64()).to_numpy(zero_copy_only=False)
    sig_lat = pc.cast(in_triples_tbl["dec_sigma_deg"], pa.float64()).to_numpy(zero_copy_only=False)
    c00 = pc.cast(in_triples_tbl["cov_ll_00"], pa.float64()).to_numpy(zero_copy_only=False)
    c01 = pc.cast(in_triples_tbl["cov_ll_01"], pa.float64()).to_numpy(zero_copy_only=False)
    c11 = pc.cast(in_triples_tbl["cov_ll_11"], pa.float64()).to_numpy(zero_copy_only=False)

    cos_lat = np.cos(np.deg2rad(lat_p))
    cos_lat = np.where(np.isfinite(cos_lat) & (np.abs(cos_lat) > 1e-12), cos_lat, 1e-12)
    dlon = (lon_o - lon_p + 180.0) % 360.0 - 180.0
    x = dlon * cos_lat
    y = lat_o - lat_p
    off_arcsec = np.sqrt(x * x + y * y) * 3600.0

    lon_ok = np.isfinite(sig_lon) & (sig_lon > 0.0)
    lat_ok = np.isfinite(sig_lat) & (sig_lat > 0.0)
    if det_sigma_floor_arcsec_by_obscode:
        floor_arcsec = np.full(sig_lon.shape, float(det_sigma_floor_arcsec), dtype=np.float64)
        for code, floor_arc in det_sigma_floor_arcsec_by_obscode.items():
            m = obsc == str(code)
            if np.any(m):
                floor_arcsec[m] = float(floor_arc)
        fill = np.where(
            np.isfinite(floor_arcsec) & (floor_arcsec > 0.0),
            floor_arcsec / 3600.0,
            0.0,
        )
    else:
        f = float(det_sigma_floor_arcsec)
        fill = (f / 3600.0) if f > 0.0 else 0.0
    sig_lon = np.where(lon_ok, sig_lon, fill)
    sig_lat = np.where(lat_ok, sig_lat, fill)

    if det_sigma_sys_arcsec_by_obscode and det_sigma_sys_apply_rms_lt_arcsec_by_obscode:
        sys_arc = np.zeros_like(sig_lon, dtype=np.float64)
        trig_arc = np.full_like(sig_lon, -np.inf, dtype=np.float64)
        for code, sys in det_sigma_sys_arcsec_by_obscode.items():
            if code not in det_sigma_sys_apply_rms_lt_arcsec_by_obscode:
                continue
            m = obsc == code
            if not np.any(m):
                continue
            sys_arc[m] = float(sys)
            trig_arc[m] = float(det_sigma_sys_apply_rms_lt_arcsec_by_obscode[code])
        if np.any(sys_arc > 0.0):
            rms_arc = np.sqrt(sig_lon * sig_lon + sig_lat * sig_lat) * 3600.0
            apply = (sys_arc > 0.0) & np.isfinite(trig_arc) & (rms_arc < trig_arc)
            if np.any(apply):
                sys_deg = sys_arc / 3600.0
                sig_lon = np.where(apply, np.sqrt(sig_lon * sig_lon + sys_deg * sys_deg), sig_lon)
                sig_lat = np.where(apply, np.sqrt(sig_lat * sig_lat + sys_deg * sys_deg), sig_lat)

    a_p = (cos_lat * cos_lat) * c00
    b_p = cos_lat * c01
    d_p = c11
    var_x = (sig_lon * cos_lat) ** 2
    var_y = sig_lat**2
    a = a_p + var_x
    d = d_p + var_y
    b = b_p
    det = a * d - b * b
    det = np.where(np.isfinite(det) & (det > 0.0), det, np.inf)
    chi2 = (x * x * d + y * y * a - 2.0 * x * y * b) / det
    chi2 = np.where(np.isfinite(chi2), chi2, np.inf)

    n_keep = int(np.sum(chi2 <= float(gate_n_sigma) ** 2))
    n_reject = int(in_triples_tbl.num_rows) - n_keep

    out: dict[str, float | int] = {
        "n_truth_in_triples": int(in_triples_tbl.num_rows),
        "n_keep": n_keep,
        "n_reject": n_reject,
    }
    if off_arcsec.size and np.isfinite(off_arcsec).any():
        q = np.quantile(off_arcsec[np.isfinite(off_arcsec)], [0.5, 0.9, 0.99])
        out["offset_arcsec_p50"] = float(q[0])
        out["offset_arcsec_p90"] = float(q[1])
        out["offset_arcsec_p99"] = float(q[2])
    if chi2.size and np.isfinite(chi2).any():
        q = np.quantile(chi2[np.isfinite(chi2)], [0.5, 0.9, 0.99])
        out["chi2_p50"] = float(q[0])
        out["chi2_p90"] = float(q[1])
        out["chi2_p99"] = float(q[2])
    return out


def _residual_quantiles_per_run_section(*, backend_runs: pa.Table) -> str:
    """
    Per-run p50/p90/p99 for offset_arcsec and chi2 on truth detections in Stage3 frame-candidates.
    Reuses the same residual computation as the innov_ellipse truth-only diagnostics.
    """
    if backend_runs.num_rows == 0:
        return ""
    need_cols = {"bench_run_dir", "truth_parquet", "detections_parquet"}
    if not need_cols.issubset(set(backend_runs.column_names)):
        return ""
    truth_parquet = backend_runs["truth_parquet"][0].as_py()
    if not truth_parquet:
        return ""
    det_vals = [backend_runs["detections_parquet"][i].as_py() for i in range(int(backend_runs.num_rows))]
    det_path = next((v for v in det_vals if v is not None and str(v).strip() != ""), None)
    if not det_path:
        return ""

    truth_path = Path(str(truth_parquet))
    if not truth_path.exists():
        return ""

    try:
        truth = _read_parquet(truth_path)
    except Exception:
        return ""

    cols = set(truth.column_names)
    if "orbit_id" not in cols or "obscode" not in cols or "observation_id" not in cols:
        return ""
    if "exposure_mjd_mid_key_us" not in cols or "healpixel" not in cols:
        return ""
    if "matched" in truth.column_names:
        truth = truth.filter(pc.fill_null(pc.cast(truth["matched"], pa.bool_()), False)).drop(["matched"])

    try:
        import duckdb  # type: ignore

        def _sql_quote(s: str) -> str:
            return "'" + str(s).replace("'", "''") + "'"

        con = duckdb.connect(database=":memory:")
        con.execute(
            f"CREATE OR REPLACE VIEW det AS SELECT * FROM read_parquet({_sql_quote(str(det_path))})"
        )
        con.register("truth_ids", truth.select(["obscode", "observation_id"]))
        obs_tbl = con.execute(
            """
            SELECT t.obscode, t.observation_id, d.ra_deg, d.dec_deg, d.ra_sigma_deg, d.dec_sigma_deg, d.mag
            FROM truth_ids t
            INNER JOIN det d ON d.obscode = t.obscode AND d.observation_id = t.observation_id
            """
        ).fetch_arrow_table()
        con.close()
    except Exception:
        return ""

    for c in ["obscode", "observation_id"]:
        if c in obs_tbl.column_names:
            obs_tbl = obs_tbl.set_column(
                obs_tbl.schema.get_field_index(c), c, pc.cast(obs_tbl[c], pa.large_string())
            )
    truth_base = truth.join(obs_tbl, keys=["obscode", "observation_id"], join_type="left outer")

    gate_settings = _gate_settings_from_backend_runs(backend_runs=backend_runs)
    gate_n_sigma = gate_settings.gate_n_sigma
    det_sigma_floor_arcsec = gate_settings.det_sigma_floor_arcsec
    det_sigma_floor_arcsec_by_obscode = gate_settings.det_sigma_floor_arcsec_by_obscode
    det_sigma_sys_arcsec_by_obscode = gate_settings.det_sigma_sys_arcsec_by_obscode
    det_sigma_sys_apply_rms_lt_arcsec_by_obscode = (
        gate_settings.det_sigma_sys_apply_rms_lt_arcsec_by_obscode
    )

    want_preds = [
        "orbit_id", "obscode", "exposure_mjd_mid_key_us",
        "pred_lon_deg", "pred_lat_deg", "cov_ll_00", "cov_ll_01", "cov_ll_11",
    ]
    sep = pa.scalar("|", type=pa.large_string())

    def _frame_key(tbl: pa.Table) -> pa.Array:
        return pc.binary_join_element_wise(
            pc.cast(tbl["orbit_id"], pa.large_string()),
            pc.binary_join_element_wise(
                pc.cast(tbl["obscode"], pa.large_string()),
                pc.binary_join_element_wise(
                    pc.cast(tbl["exposure_mjd_mid_key_us"], pa.large_string()),
                    pc.cast(tbl["healpixel"], pa.large_string()),
                    sep,
                ),
                sep,
            ),
            sep,
        )

    rows: list[dict[str, object]] = []
    for i in range(int(backend_runs.num_rows)):
        bench_run_dir = backend_runs["bench_run_dir"][i].as_py()
        if not bench_run_dir:
            continue
        stage23 = Path(str(bench_run_dir)) / "stage23"
        preds_p = stage23 / "preds.parquet"
        triples_p = stage23 / "triples.parquet"
        if not preds_p.exists() or not triples_p.exists():
            continue
        try:
            preds = _read_parquet(preds_p)
            triples = _read_parquet(triples_p)
        except Exception:
            continue
        preds2 = preds.select([c for c in want_preds if c in preds.column_names])
        if not preds2.column_names:
            continue
        truth_run = truth_base.join(
            preds2,
            keys=["orbit_id", "obscode", "exposure_mjd_mid_key_us"],
            join_type="left outer",
        )
        trip_fk = _frame_key(triples)
        truth_fk = _frame_key(truth_run)
        in_triples = pc.is_in(truth_fk, value_set=trip_fk)
        truth_run = truth_run.append_column("_in_triples", pc.cast(in_triples, pa.bool_()))
        in_triples_tbl = truth_run.filter(pc.cast(truth_run["_in_triples"], pa.bool_()))

        q = _compute_residual_quantiles(
            in_triples_tbl,
            gate_n_sigma=gate_n_sigma,
            det_sigma_floor_arcsec=det_sigma_floor_arcsec,
            det_sigma_floor_arcsec_by_obscode=det_sigma_floor_arcsec_by_obscode,
            det_sigma_sys_arcsec_by_obscode=det_sigma_sys_arcsec_by_obscode,
            det_sigma_sys_apply_rms_lt_arcsec_by_obscode=det_sigma_sys_apply_rms_lt_arcsec_by_obscode,
        )
        row: dict[str, object] = {
            "run": i + 1,
            "stage2_strategy": backend_runs["stage2_strategy"][i].as_py() if "stage2_strategy" in backend_runs.column_names else "",
            "build_elapsed_s": backend_runs["build_elapsed_s"][i].as_py() if "build_elapsed_s" in backend_runs.column_names else None,
        }
        if q:
            row["n_truth_in_triples"] = q["n_truth_in_triples"]
            row["offset_arcsec_p50"] = q.get("offset_arcsec_p50")
            row["offset_arcsec_p90"] = q.get("offset_arcsec_p90")
            row["offset_arcsec_p99"] = q.get("offset_arcsec_p99")
            row["chi2_p50"] = q.get("chi2_p50")
            row["chi2_p90"] = q.get("chi2_p90")
            row["chi2_p99"] = q.get("chi2_p99")
        else:
            row["n_truth_in_triples"] = 0
            row["offset_arcsec_p50"] = row["offset_arcsec_p90"] = row["offset_arcsec_p99"] = None
            row["chi2_p50"] = row["chi2_p90"] = row["chi2_p99"] = None
        rows.append(row)

    if not rows:
        return ""

    cols_order = [
        "run", "stage2_strategy", "build_elapsed_s", "n_truth_in_triples",
        "offset_arcsec_p50", "offset_arcsec_p90", "offset_arcsec_p99",
        "chi2_p50", "chi2_p90", "chi2_p99",
    ]
    arrays: dict[str, pa.Array] = {}
    for c in cols_order:
        vals = [r[c] for r in rows]
        if c == "run":
            arrays[c] = pa.array(vals, type=pa.int64())
        elif c == "stage2_strategy":
            arrays[c] = pa.array([str(v) for v in vals], type=pa.large_string())
        elif c == "build_elapsed_s":
            arrays[c] = pa.array([float(v) if v is not None else None for v in vals], type=pa.float64())
        elif c == "n_truth_in_triples":
            arrays[c] = pa.array([int(v) for v in vals], type=pa.int64())
        else:
            arrays[c] = pa.array([float(v) if v is not None else None for v in vals], type=pa.float64())
    tbl = pa.table(arrays)
    return "## Residual quantiles vs truth (per run)\n\nTruth detections in Stage3 frame-candidates: offset_arcsec (arcsec) and chi2 (innovation ellipse).\n\n" + _table_to_markdown(tbl, max_rows=None)


def _summarize_microtimings_row(row: dict[str, object]) -> list[str]:
    # Keep this stable and non-exhaustive: highlight the biggest subcomponents we can see.
    build = float(row.get("build_elapsed_s") or 0.0)
    if build <= 0:
        return []

    candidates = [
        ("propagation_elapsed_s", "propagation"),
        ("build_pred_mag_elapsed_s", "pred_mag"),
        ("build_footprint_elapsed_s", "footprint_total"),
        ("build_triples_elapsed_s", "triples"),
        ("footprint.vertices_elapsed_s", "fp_vertices"),
        ("footprint.rasterize_elapsed_s", "fp_rasterize"),
        ("footprint.pad_neighbors_elapsed_s", "fp_pad_neighbors"),
        ("stage2.window_centers_elapsed_s", "stage2_window_centers"),
        ("stage2.assist_propagate_centers_elapsed_s", "stage2_assist_centers"),
        ("stage2.sigma_point_variants_elapsed_s", "stage2_sigma_points"),
        ("stage2.propagate2body_nominal_elapsed_s", "stage2_2body_nominal"),
        ("stage2.ephemeris_nominal_elapsed_s", "stage2_ephem_nominal"),
        ("stage2.propagate2body_variants_elapsed_s", "stage2_2body_variants"),
        ("stage2.ephemeris_variants_elapsed_s", "stage2_ephem_variants"),
        ("stage2.variant_collapse_elapsed_s", "stage2_variant_collapse"),
        ("stage2.variant_lonlat_elapsed_s", "stage2_variant_lonlat"),
        ("stage2.reconstruct_cov_elapsed_s", "stage2_reconstruct_cov"),
        ("gate.mag_residual_elapsed_s", "gate_mag_residual"),
        ("gate.innov_ellipse_elapsed_s", "gate_innov_ellipse"),
    ]
    seen: list[tuple[str, float]] = []
    for k, label in candidates:
        v = row.get(k)
        if v is None:
            continue
        try:
            fv = float(v)
        except Exception:
            continue
        if not np.isfinite(fv) or fv <= 0:
            continue
        seen.append((label, fv))
    if not seen:
        return []

    seen.sort(key=lambda x: x[1], reverse=True)
    top = seen[: min(8, len(seen))]
    lines = []
    for label, sec in top:
        frac = sec / build
        lines.append(f"- {label}: {sec:.3f}s ({frac:.1%} of build_elapsed_s)")
    return lines


def _per_orbit_outliers(
    per_orbit: pa.Table,
    *,
    workload: str,
    backend: str,
    max_rows: int,
) -> str:
    t = per_orbit
    for col in ("workload", "backend"):
        if col not in t.column_names:
            return ""
    t = t.filter(pc.equal(t["workload"], pa.scalar(str(workload), type=pa.large_string())))
    t = t.filter(pc.equal(t["backend"], pa.scalar(str(backend), type=pa.large_string())))
    if t.num_rows == 0:
        return ""

    # Ensure missed-truth columns exist.
    def i64(col: str) -> pa.Array:
        if col not in t.column_names:
            return pa.array([0] * int(t.num_rows), type=pa.int64())
        return pc.cast(pc.fill_null(t[col], 0), pa.int64())

    missed_det = pc.subtract(i64("n_detections_truth_total"), i64("n_detections_truth_final"))
    missed_frames = pc.subtract(i64("n_frames_truth_available"), i64("n_frames_truth_final"))
    out = t
    out = out.append_column("missed_truth_detections", missed_det)
    out = out.append_column("missed_truth_frames", missed_frames)

    out = _sort(
        out,
        keys=[
            ("missed_truth_detections", "descending"),
            ("missed_truth_frames", "descending"),
            ("n_detections_false_positive_final", "descending"),
            ("n_detections_unknown_final", "descending"),
            ("n_detections_candidates", "descending"),
            ("orbit_id", "ascending"),
        ],
    )

    cols = [
        "orbit_id",
        "missed_truth_detections",
        "missed_truth_frames",
        "miss_stage3_geom_frames",
        "miss_stage3_lim_mag_frames",
        "miss_stage4_innov",
        "n_detections_truth_total",
        "n_detections_truth_final",
        "n_frames_truth_available",
        "n_frames_truth_final",
        "n_detections_candidates",
        "n_detections_gate_matched",
        "n_detections_magnitude_rejected",
        "n_detections_false_positive_final",
        "n_detections_unknown_final",
        "miss_truth_total",
    ]
    out2 = _select_existing(out, cols)
    return (
        f"### Per-orbit outliers (workload={workload} backend={backend})\n\n"
        + _table_to_markdown(out2, max_rows=int(max_rows))
    )


@dataclass(frozen=True)
class BinWorst:
    filename: str
    workload: str | None
    backend: str | None
    group_col: str
    group_val: str
    truth_total: int
    truth_final: int
    truth_recall: float | None
    det_candidates: int | None
    frames_geom: int | None


def _worst_bins(
    run_dir: Path,
    *,
    min_truth_detections: int,
    max_rows: int,
) -> str:
    by_bin = Path(run_dir) / "by_bin"
    if not by_bin.exists():
        return "## Worst bins (missing `by_bin/`)\n\n"

    worst: list[BinWorst] = []
    for p in sorted(by_bin.glob("*.parquet")):
        t = _read_parquet(p)
        if t.num_rows == 0:
            continue

        # Guess the group column (written as the first non-workload/backend column in our generator).
        group_col = None
        for c in t.column_names:
            if c in ("workload", "backend", "truth_recall", "truth_frame_coverage"):
                continue
            if c.startswith("n_"):
                continue
            group_col = c
            break
        if group_col is None:
            continue

        truth_total_arr = _safe_i64(t, "n_detections_truth_total")
        truth_final_arr = _safe_i64(t, "n_detections_truth_final")
        if truth_total_arr is None or truth_final_arr is None:
            continue

        # Filter to bins with meaningful truth.
        t2 = t.filter(pc.greater_equal(truth_total_arr, pa.scalar(int(min_truth_detections), type=pa.int64())))
        if t2.num_rows == 0:
            continue

        # Prefer existing recall, else compute.
        recall = _safe_f64(t2, "truth_recall")
        if recall is None:
            denom = pc.cast(_safe_i64(t2, "n_detections_truth_total"), pa.float64())
            numer = pc.cast(_safe_i64(t2, "n_detections_truth_final"), pa.float64())
            recall = pc.if_else(pc.greater(denom, 0), pc.divide(numer, denom), pa.scalar(None, type=pa.float64()))
            t2 = t2.append_column("truth_recall", recall)

        t2 = _sort(t2, keys=[("truth_recall", "ascending"), ("n_detections_truth_total", "descending")])

        # Take top-k worst per file.
        for i in range(min(int(max_rows), int(t2.num_rows))):
            rec = t2.slice(i, 1).to_pylist()[0]
            group_raw = rec.get(group_col)
            group_val = "unknown" if group_raw is None or str(group_raw).strip() == "" else str(group_raw)
            worst.append(
                BinWorst(
                    filename=p.name,
                    workload=(None if rec.get("workload") is None else str(rec["workload"])),
                    backend=(None if rec.get("backend") is None else str(rec["backend"])),
                    group_col=str(group_col),
                    group_val=group_val,
                    truth_total=int(rec.get("n_detections_truth_total") or 0),
                    truth_final=int(rec.get("n_detections_truth_final") or 0),
                    truth_recall=None if rec.get("truth_recall") is None else float(rec["truth_recall"]),
                    det_candidates=(None if rec.get("n_detections_candidates") is None else int(rec["n_detections_candidates"])),
                    frames_geom=(None if rec.get("n_frames_geometry_matched") is None else int(rec["n_frames_geometry_matched"])),
                )
            )

    if not worst:
        return "## Worst bins\n\n_(no bins with truth found)_\n"

    worst.sort(key=lambda w: (float("inf") if w.truth_recall is None else float(w.truth_recall), -w.truth_total))
    rows = [
        {
            "file": w.filename,
            "workload": w.workload,
            "backend": w.backend,
            "group": w.group_col,
            "value": w.group_val,
            "n_truth_total": w.truth_total,
            "n_truth_final": w.truth_final,
            "truth_recall": w.truth_recall,
            "n_det_candidates": w.det_candidates,
            "n_frames_geom": w.frames_geom,
        }
        for w in worst[: int(max_rows)]
    ]
    t_out = pa.Table.from_pylist(rows)
    return "## Worst bins (lowest recall, min truth threshold)\n\n" + _table_to_markdown(t_out, max_rows=max_rows)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Rank and highlight benchmark results under a run directory."
    )
    p.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Directory containing backend_runs.parquet, per_orbit.parquet, and optionally by_bin/.",
    )
    p.add_argument(
        "--out-md",
        type=str,
        default="",
        help="Optional markdown output path (defaults to stdout only).",
    )
    p.add_argument("--max-rows", type=int, default=20, help="Max rows per table section.")
    p.add_argument(
        "--min-truth-detections",
        type=int,
        default=10,
        help="Minimum truth detections for by-bin worst-bin ranking.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    run_dir = Path(args.run_dir)
    max_rows = int(args.max_rows)
    min_truth = int(args.min_truth_detections)

    backend_runs_path = run_dir / "backend_runs.parquet"
    per_orbit_path = run_dir / "per_orbit.parquet"
    backend_runs = _maybe_read_parquet(backend_runs_path)
    per_orbit = _maybe_read_parquet(per_orbit_path)

    if backend_runs is None:
        raise SystemExit(f"Missing {backend_runs_path}")

    md: list[str] = []
    md.append(f"# Benchmark ranking report: `{run_dir}`\n")
    md.append(f"- backend_runs: `{backend_runs_path}`\n")
    md.append(f"- per_orbit: `{per_orbit_path}` ({'present' if per_orbit is not None else 'missing'})\n")

    # Stage23 filter mapping audit (helps interpret limiting-mag and photometry behavior).
    md.append(_filter_mapping_audit(backend_runs=backend_runs))
    md.append(_truth_diagnostics_section(backend_runs=backend_runs, max_rows=max_rows))
    residual_section = _residual_quantiles_per_run_section(backend_runs=backend_runs)
    if residual_section:
        md.append(residual_section)

    # Workload/backends present.
    if "workload" in backend_runs.column_names and "backend" in backend_runs.column_names:
        md.append("## Runs present\n")
        md.append(
            _table_to_markdown(
                backend_runs.select(["workload", "backend"]).group_by(["workload", "backend"]).aggregate([]),
                max_rows=None,
            )
        )

    # Core ranking tables.
    md.append(
        _rank_section(
            t=backend_runs,
            title="Absolute timing rank (build_elapsed_s)",
            cols=[
                "workload",
                "backend",
                "n_orbits",
                "n_targets",
                "build_elapsed_s",
                "propagation_elapsed_s",
                "build_footprint_elapsed_s",
                "backend_elapsed_s",
            ],
            sort_keys=[("build_elapsed_s", "ascending"), ("backend", "ascending")],
            max_rows=max_rows,
        )
    )

    md.append(
        _rank_section(
            t=backend_runs,
            title="Truth recovery rank (truth_recall)",
            cols=[
                "workload",
                "backend",
                "n_orbits",
                "n_detections_truth_total",
                "n_detections_truth_final",
                "truth_recall",
                "truth_frame_coverage",
                "n_detections_false_positive_final",
                "n_detections_unknown_final",
            ],
            sort_keys=[
                ("truth_recall", "descending"),
                ("truth_frame_coverage", "descending"),
                ("build_elapsed_s", "ascending"),
            ],
            max_rows=max_rows,
        )
    )

    # Microtiming highlights: show the biggest contributors for top-rows by build time.
    if backend_runs.num_rows > 0:
        t_sort = _sort(backend_runs, keys=[("build_elapsed_s", "descending")])
        n_show = min(int(max_rows), int(t_sort.num_rows), 5)
        md.append("## Microtiming highlights (largest build_elapsed_s rows)\n")
        for rec in t_sort.slice(0, n_show).to_pylist():
            wl = rec.get("workload")
            be = rec.get("backend")
            build = rec.get("build_elapsed_s")
            md.append(f"### workload={wl} backend={be} build_elapsed_s={_fmt(build)}\n")
            lines = _summarize_microtimings_row(rec)
            md.append("\n".join(lines) + ("\n" if lines else "_(no microtimings columns present)_\n"))

    # Per-orbit outliers.
    if per_orbit is not None and per_orbit.num_rows > 0:
        md.append("## Per-orbit outliers\n")
        wl_vals = (
            sorted({str(x) for x in backend_runs["workload"].to_pylist()})
            if "workload" in backend_runs.column_names
            else []
        )
        be_vals = (
            sorted({str(x) for x in backend_runs["backend"].to_pylist()})
            if "backend" in backend_runs.column_names
            else []
        )
        for wl in wl_vals[:5]:
            for be in be_vals[:5]:
                sec = _per_orbit_outliers(per_orbit, workload=wl, backend=be, max_rows=max_rows)
                if sec:
                    md.append(sec)

    # Worst bins from by_bin outputs.
    md.append(_worst_bins(run_dir, min_truth_detections=min_truth, max_rows=max_rows))

    out = "\n".join(md).strip() + "\n"
    print(out)
    if args.out_md:
        Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_md).write_text(out, encoding="utf-8")


if __name__ == "__main__":
    main()
