from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from adam_core.orbits import Orbits

from precovery.search.backends.bigquery_virtual import (
    BigQueryVirtualBackend,
    BqDetectionsTableConfig,
)
from precovery.search.backends.clickhouse_local import ClickHouseLocalBackend
from precovery.search.backends.duckdb_parquet import DuckDbParquetBackend
from precovery.search.pipeline_types import MonthWindow
from .runner import run_benchmark
from .workload import WorkloadSpec


def _parse_float_map_json(raw: str, *, arg_name: str) -> dict[str, float] | None:
    s = str(raw).strip()
    if not s:
        return None
    try:
        obj = json.loads(s)
    except Exception as e:
        raise SystemExit(f"{arg_name} must be valid JSON object; got parse error: {e}") from e
    if not isinstance(obj, dict):
        raise SystemExit(f"{arg_name} must be a JSON object mapping obscode -> arcsec float")
    out: dict[str, float] = {}
    for k, v in obj.items():
        ks = str(k).strip()
        if not ks:
            raise SystemExit(f"{arg_name} contains an empty obscode key")
        try:
            out[ks] = float(v)
        except Exception as e:
            raise SystemExit(f"{arg_name} value for key {ks!r} is not a float: {v!r}") from e
    return out


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backend benchmark harness (join + innov gate).")
    p.add_argument("--subset-dir", type=str, required=True, help="Local subset dir containing config.json (+ backend data).")
    p.add_argument("--orbits-parquet", type=str, required=True, help="Orbits parquet (adam-core Orbits).")
    p.add_argument("--truth-parquet", type=str, default="", help="Optional truth crossmatch parquet.")

    p.add_argument("--month", action="append", required=True, help="Benchmark month window YYYY-MM (repeatable).")
    p.add_argument("--obscode", action="append", required=True, help="Observatory code (repeatable).")

    p.add_argument(
        "--backend",
        action="append",
        choices=["duckdb", "clickhouse", "bigquery"],
        default=[],
        help="Backend to run (repeatable).",
    )
    p.add_argument("--duckdb-parquet", type=str, default="", help="Parquet dataset path for DuckDB backend.")
    p.add_argument("--clickhouse-table", type=str, default="", help="ClickHouse table name.")
    p.add_argument("--clickhouse-host", type=str, default="localhost", help="ClickHouse host.")
    p.add_argument("--clickhouse-port", type=int, default=8123, help="ClickHouse HTTP port.")
    p.add_argument("--clickhouse-user", type=str, default="default", help="ClickHouse username.")
    p.add_argument("--clickhouse-password", type=str, default="", help="ClickHouse password.")
    p.add_argument("--clickhouse-database", type=str, default="default", help="ClickHouse database.")
    p.add_argument("--bigquery-table", type=str, default="", help="BigQuery table/view for detections.")
    p.add_argument("--bigquery-exec", action="store_true", help="Allow tiny BigQuery executions (counts only).")
    p.add_argument("--bigquery-max-bytes", type=int, default=0, help="Set bq maximum bytes billed (0=unset).")

    p.add_argument("--max-orbits", type=int, default=25)
    p.add_argument("--max-targets", type=int, default=2000)
    p.add_argument(
        "--stage2-strategy",
        type=str,
        default="",
        help="Stage 2 propagation strategy override (default is WorkloadSpec default).",
    )
    p.add_argument(
        "--footprint",
        type=str,
        default="",
        choices=[
            "",
            "cov_polygon_reconstructed_moc",
            "cov_polygon_reconstructed_moc_pad_neighbors",
        ],
        help="Footprint strategy override (default is WorkloadSpec default).",
    )
    p.add_argument(
        "--max-processes",
        type=int,
        default=0,
        help="Max worker processes for propagation (0=default/auto; 1=single-process).",
    )
    p.add_argument(
        "--execution-mode",
        type=str,
        default="memory",
        choices=["memory", "disk"],
        help="Pipeline execution mode (memory default, disk for file-first chunked execution).",
    )
    p.add_argument(
        "--chunk-rows-stage23",
        type=int,
        default=0,
        help="Approximate row budget per Stage23 chunk in disk mode (<=0 uses RAM-budget auto sizing).",
    )
    p.add_argument(
        "--chunk-rows-stage4",
        type=int,
        default=0,
        help="Approximate row budget per Stage4 chunk in disk mode (<=0 uses RAM-budget auto sizing).",
    )
    p.add_argument(
        "--max-inflight-chunks",
        type=int,
        default=2,
        help="Reserved disk-mode backpressure control for writer inflight chunks.",
    )
    p.add_argument(
        "--runtime-tmp-dir",
        type=str,
        default="",
        help="Optional temp directory override used by disk mode (TMPDIR/RAY_TMPDIR).",
    )
    p.add_argument(
        "--min-free-disk-gb",
        type=float,
        default=10.0,
        help="Disk-mode preflight minimum free disk in GB.",
    )
    p.add_argument(
        "--window-size-days",
        type=int,
        default=0,
        help="Stage-2 window size in days (default: 7 when 0 or omitted; uses WorkloadSpec default).",
    )
    p.add_argument(
        "--det-sigma-floor-arcsec",
        type=float,
        default=None,
        help=(
            "Global fill floor (arcsec) for missing/invalid detection sigmas. "
            "Default: production default from gate_defaults.py."
        ),
    )
    p.add_argument(
        "--det-sigma-floor-arcsec-by-obscode-json",
        type=str,
        default="",
        help=(
            "Optional JSON object override for per-obscode invalid-sigma fill floors (arcsec), "
            "e.g. '{\"I41\": 0.1, \"T05\": 0.3, \"T08\": 0.3}'. "
            "Default: production default map from gate_defaults.py."
        ),
    )
    p.add_argument(
        "--det-sigma-sys-arcsec-by-obscode-json",
        type=str,
        default="",
        help=(
            "Optional JSON object override for per-obscode sigma systematic terms (arcsec), "
            "e.g. '{\"T05\": 0.3, \"T08\": 0.3}'. "
            "Default: production default map from gate_defaults.py."
        ),
    )
    p.add_argument(
        "--det-sigma-sys-apply-rms-lt-arcsec-by-obscode-json",
        type=str,
        default="",
        help=(
            "Optional JSON object override for per-obscode trigger thresholds (arcsec) that gate "
            "application of sigma systematic terms, e.g. '{\"T05\": 0.4, \"T08\": 0.45}'. "
            "Default: production default map from gate_defaults.py."
        ),
    )
    p.add_argument(
        "--gate-n-sigma",
        type=float,
        default=0.0,
        help="Override innovation-ellipse gate n_sigma (<=0 uses workload default). "
        "This does not change the Stage-3 footprint n_sigma.",
    )
    p.add_argument(
        "--max-on-sky-sigma-major-arcsec",
        type=float,
        default=None,
        help="Optional Stage-3 uncertainty budget (major-axis 1-sigma, arcsec).",
    )
    p.add_argument(
        "--targets-from-truth",
        action="store_true",
        help="When --truth-parquet is provided, derive target exposure times from truth rows "
        "(instead of enumerating all exposures in the month).",
    )
    p.add_argument(
        "--gate-totals",
        action="store_true",
        help="Also compute and report overall gate rejection totals per backend "
        "(may re-fetch candidates; avoid on very large runs unless needed).",
    )
    p.add_argument(
        "--detailed-timings",
        action="store_true",
        help="Record micro-timings for selected sub-steps (e.g. footprint vertices/rasterize/padding, "
        "mag-residual vs innovation-ellipse gating).",
    )
    p.add_argument(
        "--out-parquet",
        type=str,
        default="",
        help="Optional parquet path to append one row per backend run for later analysis.",
    )
    p.add_argument(
        "--out-per-orbit-parquet",
        type=str,
        default="",
        help="Optional parquet path to append per-orbit metrics (requires --truth-parquet).",
    )
    p.add_argument(
        "--orbit-bins-parquet",
        type=str,
        default="",
        help="Optional orbit bins parquet (orbit_id -> bin columns) for by-bin aggregations.",
    )
    p.add_argument(
        "--out-by-bin-dir",
        type=str,
        default="",
        help="Optional directory to write by-bin parquet aggregations (requires --out-per-orbit-parquet output).",
    )
    return p.parse_args()


def _fill_null_i64(t: pa.Table, cols: list[str]) -> pa.Table:
    out = t
    for c in cols:
        if c not in out.column_names:
            continue
        out = out.set_column(
            out.schema.get_field_index(c),
            c,
            pc.cast(pc.fill_null(out[c], 0), pa.int64()),
        )
    return out


def _aggregate_by_bin(*, per_orbit: pa.Table, group_cols: list[str]) -> pa.Table:
    t = per_orbit
    t = _fill_null_i64(
        t,
        [
            "n_frames_candidates",
            "n_frames_geometry_matched",
            "n_frames_stage3_rejected_any",
            "n_frames_uncertainty_rejected",
            "n_frames_truth_available",
            "n_frames_truth_final",
            "n_detections_candidates",
            "n_detections_gate_matched",
            "n_detections_innov_ellipse_rejected",
            "n_detections_magnitude_rejected",
            "n_detections_truth_total",
            "n_detections_truth_final",
            "n_detections_false_positive_final",
            "n_detections_unknown_final",
        ],
    )
    g = t.group_by(group_cols).aggregate(
        [
            ("orbit_id", "count"),
            ("n_frames_geometry_matched", "sum"),
            ("n_frames_stage3_rejected_any", "sum"),
            ("n_frames_uncertainty_rejected", "sum"),
            ("n_frames_truth_available", "sum"),
            ("n_frames_truth_final", "sum"),
            ("n_detections_candidates", "sum"),
            ("n_detections_gate_matched", "sum"),
            ("n_detections_innov_ellipse_rejected", "sum"),
            ("n_detections_magnitude_rejected", "sum"),
            ("n_detections_truth_total", "sum"),
            ("n_detections_truth_final", "sum"),
            ("n_detections_false_positive_final", "sum"),
            ("n_detections_unknown_final", "sum"),
        ]
    )
    # Rename aggregate columns to stable names.
    ren = []
    for name in g.column_names:
        if name.endswith("_count"):
            ren.append("n_orbits")
        elif name.endswith("_sum"):
            ren.append(name[: -len("_sum")])
        else:
            ren.append(name)
    g = g.rename_columns(ren)

    # Derived ratios (total-weighted).
    truth = pc.cast(g["n_detections_truth_total"], pa.float64())
    hit = pc.cast(g["n_detections_truth_final"], pa.float64())
    frames = pc.cast(g["n_frames_truth_available"], pa.float64())
    frames_hit = pc.cast(g["n_frames_truth_final"], pa.float64())
    recall = pc.if_else(pc.greater(truth, 0), pc.divide(hit, truth), pa.scalar(None, type=pa.float64()))
    coverage = pc.if_else(
        pc.greater(frames, 0), pc.divide(frames_hit, frames), pa.scalar(None, type=pa.float64())
    )
    g = g.append_column("truth_recall", recall)
    g = g.append_column("truth_frame_coverage", coverage)
    return g


def _append_results_parquet(*, out_path: Path, rows: list[dict[str, object]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t_new = pa.Table.from_pylist(rows) if rows else pa.table({})
    if not out_path.exists():
        pq.write_table(t_new, out_path)
        return
    t_old = pq.read_table(out_path)
    pq.write_table(pa.concat_tables([t_old, t_new], promote_options="default"), out_path)


def _append_table_parquet(*, out_path: Path, table: pa.Table) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not out_path.exists():
        pq.write_table(table, out_path)
        return
    t_old = pq.read_table(out_path)
    pq.write_table(pa.concat_tables([t_old, table], promote_options="default"), out_path)


def main() -> None:
    args = _parse_args()
    subset_dir = Path(args.subset_dir)
    orbits = Orbits.from_parquet(args.orbits_parquet)

    if not args.backend:
        raise SystemExit(
            "No backends specified. Pass one or more --backend values."
        )

    window = MonthWindow(year_months=tuple(args.month), obscodes=tuple(args.obscode))
    wl_kwargs: dict[str, object] = {}
    if args.stage2_strategy:
        wl_kwargs["stage2_strategy"] = str(args.stage2_strategy)
    if args.footprint:
        wl_kwargs["footprint"] = str(args.footprint)
    if int(args.window_size_days) > 0:
        wl_kwargs["window_size_days"] = int(args.window_size_days)
    workload = WorkloadSpec(subset_dir=subset_dir, window=window, **wl_kwargs)

    backends = []
    for b in args.backend:
        if b == "duckdb":
            if not args.duckdb_parquet:
                raise SystemExit("--duckdb-parquet is required for duckdb backend")
            backends.append(DuckDbParquetBackend(parquet_path=Path(args.duckdb_parquet)))
        elif b == "clickhouse":
            if not args.clickhouse_table:
                raise SystemExit("--clickhouse-table is required for clickhouse backend")
            backends.append(
                ClickHouseLocalBackend(
                    table=str(args.clickhouse_table),
                    host=str(args.clickhouse_host),
                    port=int(args.clickhouse_port),
                    user=str(args.clickhouse_user),
                    password=str(args.clickhouse_password),
                    database=str(args.clickhouse_database),
                )
            )
        elif b == "bigquery":
            table = args.bigquery_table or "moeyens-thor-dev.production_aims.aims"
            max_bytes = None if int(args.bigquery_max_bytes) <= 0 else int(args.bigquery_max_bytes)
            backends.append(
                BigQueryVirtualBackend(
                    cfg=BqDetectionsTableConfig(table=str(table)),
                    allow_execute=bool(args.bigquery_exec),
                    maximum_bytes_billed=max_bytes,
                )
            )

    # Enumeration should come from one of the selected backends. This avoids keeping a
    # separate sqlite implementation around "just for enumeration".
    enum_backend = next(
        (
            b
            for b in backends
            if bool(getattr(b, "capabilities", None))
            and bool(getattr(b.capabilities, "supports_enumerate_targets", False))
        ),
        None,
    )
    if enum_backend is None:
        raise SystemExit(
            "No selected backend supports target enumeration. Include --backend duckdb."
        )

    floor_by_obscode = _parse_float_map_json(
        args.det_sigma_floor_arcsec_by_obscode_json,
        arg_name="--det-sigma-floor-arcsec-by-obscode-json",
    )
    sys_by_obscode = _parse_float_map_json(
        args.det_sigma_sys_arcsec_by_obscode_json,
        arg_name="--det-sigma-sys-arcsec-by-obscode-json",
    )
    trig_by_obscode = _parse_float_map_json(
        args.det_sigma_sys_apply_rms_lt_arcsec_by_obscode_json,
        arg_name="--det-sigma-sys-apply-rms-lt-arcsec-by-obscode-json",
    )

    truth_path = Path(args.truth_parquet) if args.truth_parquet else None
    backend_rows, per_orbit = run_benchmark(
        workload=workload,
        orbits=orbits,
        orbits_parquet=str(args.orbits_parquet),
        enumerate_backend=enum_backend,
        backends=backends,
        max_orbits=int(args.max_orbits) if args.max_orbits > 0 else None,
        max_targets=int(args.max_targets) if args.max_targets > 0 else None,
        truth_path=truth_path,
        targets_from_truth=bool(args.targets_from_truth),
        det_sigma_floor_arcsec=(
            None if args.det_sigma_floor_arcsec is None else float(args.det_sigma_floor_arcsec)
        ),
        det_sigma_floor_arcsec_by_obscode=floor_by_obscode,
        det_sigma_sys_arcsec_by_obscode=sys_by_obscode,
        det_sigma_sys_apply_rms_lt_arcsec_by_obscode=trig_by_obscode,
        gate_n_sigma=None if float(args.gate_n_sigma) <= 0.0 else float(args.gate_n_sigma),
        max_processes=None if int(args.max_processes) <= 0 else int(args.max_processes),
        compute_gate_totals=bool(args.gate_totals),
        detailed_timings=bool(args.detailed_timings),
        execution_mode=str(args.execution_mode),
        chunk_rows_stage23=int(args.chunk_rows_stage23),
        chunk_rows_stage4=int(args.chunk_rows_stage4),
        max_inflight_chunks=int(args.max_inflight_chunks),
        runtime_tmp_dir=(None if str(args.runtime_tmp_dir).strip() == "" else str(args.runtime_tmp_dir)),
        min_free_disk_gb=float(args.min_free_disk_gb),
        max_on_sky_sigma_major_arcsec=(
            None
            if args.max_on_sky_sigma_major_arcsec is None
            else float(args.max_on_sky_sigma_major_arcsec)
        ),
    )

    # Print flat rows (one per backend) for easy diffing.
    print(json.dumps(backend_rows.table.to_pylist(), indent=2, sort_keys=True))

    if args.out_parquet:
        _append_table_parquet(out_path=Path(args.out_parquet), table=backend_rows.table)

    if args.out_per_orbit_parquet:
        if per_orbit is None or per_orbit.table.num_rows == 0:
            return
        _append_table_parquet(out_path=Path(args.out_per_orbit_parquet), table=per_orbit.table)

    if args.out_by_bin_dir and args.orbit_bins_parquet:
        if per_orbit is None or per_orbit.table.num_rows == 0:
            return
        bins = pq.read_table(args.orbit_bins_parquet)
        # Normalize join key type.
        if "orbit_id" in bins.column_names:
            bins = bins.set_column(
                bins.schema.get_field_index("orbit_id"),
                "orbit_id",
                pc.cast(bins["orbit_id"], pa.large_string()),
            )
        per = per_orbit.table
        per = per.set_column(
            per.schema.get_field_index("orbit_id"),
            "orbit_id",
            pc.cast(per["orbit_id"], pa.large_string()),
        )
        per_binned = per.join(bins, keys=["orbit_id"], join_type="left outer")

        out_dir = Path(args.out_by_bin_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        groupings: list[tuple[str, list[str]]] = [
            ("by_stratum.parquet", ["workload", "backend", "stratum"]),
            ("by_regime.parquet", ["workload", "backend", "regime"]),
            ("by_arc_bin.parquet", ["workload", "backend", "arc_bin"]),
            ("by_dt_bin.parquet", ["workload", "backend", "dt_bin"]),
            ("by_u_bin.parquet", ["workload", "backend", "u_bin"]),
            ("by_i_bin.parquet", ["workload", "backend", "i_bin"]),
            ("by_orbit_type_int.parquet", ["workload", "backend", "orbit_type_int"]),
            ("by_sigma_pos_rms_bin.parquet", ["workload", "backend", "sigma_pos_rms_bin"]),
            ("by_anisotropy_pos_bin.parquet", ["workload", "backend", "anisotropy_pos_bin"]),
        ]
        for fname, cols in groupings:
            cols_ok = [c for c in cols if c in per_binned.column_names]
            if len(cols_ok) < 3:
                continue
            agg = _aggregate_by_bin(per_orbit=per_binned, group_cols=cols_ok)
            _append_table_parquet(out_path=out_dir / fname, table=agg)


if __name__ == "__main__":
    main()
