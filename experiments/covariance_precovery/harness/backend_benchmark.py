from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from adam_core.orbits import Orbits

from ..bench.backends.baseline_sqlite_blobs import BaselineSqliteBlobBackend
from ..bench.backends.bigquery_virtual import BigQueryVirtualBackend, BqDetectionsTableConfig
from ..bench.backends.clickhouse_local import ClickHouseLocalBackend
from ..bench.backends.duckdb_parquet import DuckDbParquetBackend
from ..bench.runner import run_benchmark
from ..bench.types import MonthWindow
from ..bench.workload import WorkloadSpec


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backend benchmark harness (join + innov gate).")
    p.add_argument("--subset-dir", type=str, required=True, help="Local subset dir containing index.db (+ blobs).")
    p.add_argument("--orbits-parquet", type=str, required=True, help="Orbits parquet (adam-core Orbits).")
    p.add_argument("--truth-parquet", type=str, default="", help="Optional truth crossmatch parquet.")

    p.add_argument("--month", action="append", required=True, help="Benchmark month window YYYY-MM (repeatable).")
    p.add_argument("--obscode", action="append", required=True, help="Observatory code (repeatable).")

    p.add_argument(
        "--backend",
        action="append",
        default=[],
        help="Backend to run: baseline|duckdb|clickhouse|bigquery (repeatable).",
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
        "--max-processes",
        type=int,
        default=0,
        help="Max worker processes for propagation (0=default/auto; 1=single-process).",
    )
    p.add_argument("--det-sigma-floor-arcsec", type=float, default=0.10)
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
        "--out-parquet",
        type=str,
        default="",
        help="Optional parquet path to append one row per backend run for later analysis.",
    )
    return p.parse_args()


def _append_results_parquet(*, out_path: Path, rows: list[dict[str, object]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t_new = pa.Table.from_pylist(rows) if rows else pa.table({})
    if not out_path.exists():
        pq.write_table(t_new, out_path)
        return
    t_old = pq.read_table(out_path)
    pq.write_table(pa.concat_tables([t_old, t_new], promote_options="default"), out_path)


def main() -> None:
    args = _parse_args()
    subset_dir = Path(args.subset_dir)
    orbits = Orbits.from_parquet(args.orbits_parquet)

    if not args.backend:
        args.backend = ["baseline"]

    window = MonthWindow(year_months=tuple(args.month), obscodes=tuple(args.obscode))
    wl_kwargs: dict[str, object] = {}
    if args.stage2_strategy:
        wl_kwargs["stage2_strategy"] = str(args.stage2_strategy)
    workload = WorkloadSpec(subset_dir=subset_dir, window=window, **wl_kwargs)

    enum_backend = BaselineSqliteBlobBackend()

    backends = []
    for b in args.backend:
        if b == "baseline":
            backends.append(BaselineSqliteBlobBackend())
        elif b == "duckdb":
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
        else:
            raise SystemExit(f"Unknown backend: {b}")

    truth_path = Path(args.truth_parquet) if args.truth_parquet else None
    res = run_benchmark(
        workload=workload,
        orbits=orbits,
        enumerate_backend=enum_backend,
        backends=backends,
        max_orbits=int(args.max_orbits) if args.max_orbits > 0 else None,
        max_targets=int(args.max_targets) if args.max_targets > 0 else None,
        truth_path=truth_path,
        targets_from_truth=bool(args.targets_from_truth),
        det_sigma_floor_arcsec=float(args.det_sigma_floor_arcsec),
        max_processes=None if int(args.max_processes) <= 0 else int(args.max_processes),
        compute_gate_totals=bool(args.gate_totals),
    )

    # Print a compact JSON summary for easy diffing across runs.
    out = {
        "workload": res.workload_label,
        "n_orbits": res.n_orbits,
        "n_targets": res.n_targets,
        "n_triples": res.n_triples,
        "build": {
            "elapsed_s": res.build_elapsed_s,
            "predict_elapsed_s": res.build_predict_elapsed_s,
            "pred_mag_elapsed_s": res.build_pred_mag_elapsed_s,
            "footprint_elapsed_s": res.build_footprint_elapsed_s,
            "triples_elapsed_s": res.build_triples_elapsed_s,
            "n_pairs_skipped_faint": res.n_pairs_skipped_faint,
        },
        "backends": [
            {
                "backend": r.backend,
                "elapsed_s": r.elapsed_s,
                "bytes_estimate": r.bytes_estimate,
                "counts_rows": len(r.counts),
                "gate_totals": None
                if r.gate_totals is None
                else {
                    "n_candidates": r.gate_totals.n_candidates,
                    "n_accepted": r.gate_totals.n_accepted,
                    "n_rejected_innov_ellipse": r.gate_totals.n_rejected_innov_ellipse,
                    "n_rejected_mag_residual": r.gate_totals.n_rejected_mag_residual,
                },
                "truth": None
                if r.truth is None
                else {
                    "n_truth": r.truth.n_truth,
                    "n_hit": r.truth.n_hit,
                    "recall": r.truth.recall,
                },
            }
            for r in res.backend_results
        ],
    }
    print(json.dumps(out, indent=2, sort_keys=True))

    if args.out_parquet:
        rows: list[dict[str, object]] = []
        for r in res.backend_results:
            row: dict[str, object] = {
                "workload": res.workload_label,
                "backend": r.backend,
                "subset_dir": str(subset_dir),
                "months": list(args.month),
                "obscodes": list(args.obscode),
                "stage2_strategy": str(workload.stage2_strategy),
                "orbits_parquet": str(args.orbits_parquet),
                "truth_parquet": str(args.truth_parquet) if args.truth_parquet else None,
                "targets_from_truth": bool(args.targets_from_truth),
                "max_orbits": int(args.max_orbits),
                "max_targets": int(args.max_targets),
                "max_processes": int(args.max_processes),
                "n_orbits": int(res.n_orbits),
                "n_targets": int(res.n_targets),
                "n_triples": int(res.n_triples),
                "build_elapsed_s": float(res.build_elapsed_s),
                "build_predict_elapsed_s": float(res.build_predict_elapsed_s),
                "build_pred_mag_elapsed_s": float(res.build_pred_mag_elapsed_s),
                "build_footprint_elapsed_s": float(res.build_footprint_elapsed_s),
                "build_triples_elapsed_s": float(res.build_triples_elapsed_s),
                "build_n_pairs_skipped_faint": int(res.n_pairs_skipped_faint),
                "backend_elapsed_s": float(r.elapsed_s),
                "counts_rows": int(len(r.counts)),
                "bytes_estimate": None if r.bytes_estimate is None else int(r.bytes_estimate),
            }
            if r.gate_totals is not None:
                row.update(
                    {
                        "gate_n_candidates": int(r.gate_totals.n_candidates),
                        "gate_n_accepted": int(r.gate_totals.n_accepted),
                        "gate_n_rejected_innov_ellipse": int(r.gate_totals.n_rejected_innov_ellipse),
                        "gate_n_rejected_mag_residual": int(r.gate_totals.n_rejected_mag_residual),
                    }
                )
            else:
                row.update(
                    {
                        "gate_n_candidates": None,
                        "gate_n_accepted": None,
                        "gate_n_rejected_innov_ellipse": None,
                        "gate_n_rejected_mag_residual": None,
                    }
                )
            if r.truth is not None:
                row.update(
                    {
                        "truth_n_truth": int(r.truth.n_truth),
                        "truth_n_hit": int(r.truth.n_hit),
                        "truth_recall": float(r.truth.recall),
                    }
                )
            else:
                row.update(
                    {
                        "truth_n_truth": None,
                        "truth_n_hit": None,
                        "truth_recall": None,
                    }
                )
            rows.append(row)
        _append_results_parquet(out_path=Path(args.out_parquet), rows=rows)


if __name__ == "__main__":
    main()

