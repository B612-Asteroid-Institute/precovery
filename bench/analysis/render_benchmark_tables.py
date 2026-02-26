from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render benchmark results as markdown tables.")
    p.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Directory containing backend_runs.parquet, per_orbit.parquet, and by_bin/.",
    )
    p.add_argument(
        "--out-md",
        type=str,
        default="",
        help="Optional markdown path to write output (defaults to stdout only).",
    )
    p.add_argument(
        "--max-rows",
        type=int,
        default=50,
        help="Max rows per table (0=unlimited).",
    )
    return p.parse_args()


def _fmt_cell(x: object) -> str:
    if x is None:
        return ""
    if isinstance(x, float):
        if np.isnan(x) or np.isinf(x):
            return ""
        # Reasonable default precision for this benchmark context.
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
    rows = [cols]
    for i in range(t.num_rows):
        rows.append([_fmt_cell(t[c][i].as_py()) for c in cols])

    widths = [max(len(r[j]) for r in rows) for j in range(len(cols))]

    def line(vals: list[str]) -> str:
        return "| " + " | ".join(v.ljust(widths[i]) for i, v in enumerate(vals)) + " |"

    out = [line(rows[0])]
    out.append("| " + " | ".join("-" * w for w in widths) + " |")
    for r in rows[1:]:
        out.append(line(r))
    out.append("")
    return "\n".join(out)


def _read_parquet(path: Path) -> pa.Table:
    return pq.read_table(str(path))


def _select_existing(t: pa.Table, cols: list[str]) -> pa.Table:
    keep = [c for c in cols if c in t.column_names]
    return t.select(keep) if keep else pa.table({})


def _render_totals(run_dir: Path, *, max_rows: int | None) -> str:
    t = _read_parquet(run_dir / "backend_runs.parquet")
    cols = [
        "workload",
        "backend",
        "footprint",
        "n_orbits",
        "n_targets",
        "n_frames_candidates",
        "n_frames_geometry_matched",
        "n_frames_lim_mag_rejected",
        "n_frames_final",
        "n_frames_truth_available",
        "n_frames_truth_final",
        "n_matched_frames",
        "n_detections_candidates",
        "n_detections_gate_matched",
        "n_detections_magnitude_rejected",
        "n_detections_truth_total",
        "n_detections_truth_frame_candidates",
        "n_detections_truth_final",
        "n_detections_false_positive_final",
        "n_detections_unknown_final",
        "truth_recall",
        "truth_frame_coverage",
        "n_unique_frames_selected",
        "build_elapsed_s",
        "propagation_elapsed_s",
        "build_footprint_elapsed_s",
        "backend_elapsed_s",
        "gate_n_rejected_mag_residual",
        "gate_n_rejected_innov_ellipse",
    ]
    t2 = _select_existing(t, cols)
    return "## Totals (`backend_runs.parquet`)\n\n" + _table_to_markdown(t2, max_rows=max_rows)


def _render_top_orbits(run_dir: Path, *, max_rows: int | None) -> str:
    p = run_dir / "per_orbit.parquet"
    if not p.exists():
        return "## Per-orbit (missing `per_orbit.parquet`)\n\n"

    t = _read_parquet(p)
    if t.num_rows == 0:
        return "## Per-orbit (`per_orbit.parquet`)\n\n_(empty)_\n"

    # For display, treat missing numeric metrics as 0.
    for c in [
        "n_frames_candidates",
        "n_frames_geometry_matched",
        "n_frames_lim_mag_rejected",
        "n_frames_final",
        "n_frames_truth_available",
        "n_frames_truth_final",
        "n_detections_candidates",
        "n_detections_gate_matched",
        "n_detections_magnitude_rejected",
        "n_detections_truth_total",
        "n_detections_truth_final",
        "n_detections_false_positive_final",
        "n_detections_unknown_final",
    ]:
        if c in t.column_names:
            t = t.set_column(
                t.schema.get_field_index(c),
                c,
                pc.cast(pc.fill_null(t[c], 0), pa.int64() if pa.types.is_integer(t[c].type) else t[c].type),
            )

    # Compute missed truth columns (fill null -> 0).
    def i64(col: str) -> pa.Array:
        if col not in t.column_names:
            return pa.array([0] * t.num_rows, type=pa.int64())
        return pc.cast(pc.fill_null(t[col], 0), pa.int64())

    missed_det = pc.subtract(i64("n_detections_truth_total"), i64("n_detections_truth_final"))
    missed_frames = pc.subtract(i64("n_frames_truth_available"), i64("n_frames_truth_final"))

    out = t
    out = out.append_column("missed_truth_detections", missed_det)
    out = out.append_column("missed_truth_frames", missed_frames)

    cols = [
        "orbit_id",
        "backend",
        "n_frames_geometry_matched",
        "n_frames_lim_mag_rejected",
        "n_frames_final",
        "miss_stage3_geom_frames",
        "miss_stage3_lim_mag_frames",
        "n_detections_candidates",
        "n_detections_gate_matched",
        "n_detections_magnitude_rejected",
        "n_detections_false_positive_final",
        "n_detections_unknown_final",
        "n_frames_truth_available",
        "n_frames_truth_final",
        "missed_truth_frames",
        "n_detections_truth_total",
        "n_detections_truth_final",
        "missed_truth_detections",
        "miss_stage4_innov",
        "miss_truth_total",
    ]
    out = _select_existing(out, cols)
    if out.num_rows > 0 and "missed_truth_detections" in out.column_names:
        out = out.sort_by(
            [
                ("missed_truth_detections", "descending"),
                ("missed_truth_frames", "descending"),
                ("n_detections_false_positive_final", "descending"),
                ("orbit_id", "ascending"),
            ]
        )
    return "## Per-orbit (sorted by missed truth)\n\n" + _table_to_markdown(out, max_rows=max_rows)


def _render_by_bin(run_dir: Path, *, max_rows: int | None) -> str:
    by_bin = run_dir / "by_bin"
    if not by_bin.exists():
        return "## By-bin (missing `by_bin/`)\n\n"

    parts: list[str] = ["## By-bin breakdowns (`by_bin/*.parquet`)\n"]
    for p in sorted(by_bin.glob("*.parquet")):
        t = _read_parquet(p)
        group_col = p.name.replace("by_", "").replace(".parquet", "")
        cols = [
            group_col,
            "workload",
            "backend",
            "n_orbits",
            "n_frames_geometry_matched",
            "n_frames_truth_available",
            "n_frames_truth_final",
            "n_detections_candidates",
            "n_detections_gate_matched",
            "n_detections_magnitude_rejected",
            "n_detections_truth_total",
            "n_detections_truth_final",
            "n_detections_false_positive_final",
            "n_detections_unknown_final",
            "truth_recall",
            "truth_frame_coverage",
        ]
        t2 = _select_existing(t, cols)
        parts.append(f"### `{p.name}`\n")
        parts.append(_table_to_markdown(t2, max_rows=max_rows))
    return "\n".join(parts)


def main() -> None:
    args = _parse_args()
    run_dir = Path(args.run_dir)
    max_rows = None if int(args.max_rows) <= 0 else int(args.max_rows)

    md = []
    md.append(f"# Benchmark summary: `{run_dir}`\n")
    md.append(_render_totals(run_dir, max_rows=max_rows))
    md.append(_render_top_orbits(run_dir, max_rows=max_rows))
    md.append(_render_by_bin(run_dir, max_rows=max_rows))
    out = "\n".join(md).strip() + "\n"

    print(out)
    if args.out_md:
        Path(args.out_md).write_text(out, encoding="utf-8")


if __name__ == "__main__":
    main()

