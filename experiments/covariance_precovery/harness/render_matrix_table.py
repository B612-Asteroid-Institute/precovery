from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


def _fmt_cell(x: object) -> str:
    if x is None:
        return ""
    # pandas uses NaN for missing
    try:
        if pd.isna(x):  # type: ignore[arg-type]
            return ""
    except Exception:  # noqa: BLE001
        pass
    if isinstance(x, float):
        if abs(x - round(x)) < 1e-9 and abs(x) < 1e15:
            return str(int(round(x)))
        return f"{x:.6g}"
    return str(x)


def _print_markdown_table(df: pd.DataFrame, cols: list[str]) -> None:
    headers = cols
    rows = [[_fmt_cell(df.iloc[i][c]) for c in headers] for i in range(len(df))]
    widths = [len(h) for h in headers]
    for r in rows:
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], len(cell))

    def md_row(cells: list[str]) -> str:
        return "| " + " | ".join(cells[i].ljust(widths[i]) for i in range(len(cells))) + " |"

    def md_sep() -> str:
        return "| " + " | ".join("-" * w for w in widths) + " |"

    print(md_row(headers))
    print(md_sep())
    for r in rows:
        print(md_row(r))


def build_matrix_table(*, stage2_run_dir: Path, stage3_run_dir: Path) -> pd.DataFrame:
    m2_path = stage2_run_dir / "metrics.parquet"
    m3_path = stage3_run_dir / "metrics.parquet"
    c3_path = stage3_run_dir / "coverage.parquet"

    if not m3_path.exists():
        raise FileNotFoundError(f"Missing Stage 3 metrics: {m3_path}")
    if not c3_path.exists():
        raise FileNotFoundError(f"Missing Stage 3 coverage: {c3_path}")

    m3 = pq.read_table(m3_path).to_pandas()
    c3 = pq.read_table(c3_path).to_pandas()

    df = m3.merge(
        c3,
        on=[
            "stage2_run_dir",
            "subset_dir",
            "strategy",
            "variant_kind",
            "footprint",
            "healpix_nside",
        ],
        how="left",
    )

    # Attach Stage 2 timing + shapes per (strategy, variant_kind).
    if m2_path.exists():
        m2 = pq.read_table(m2_path).to_pandas()
        m2_key = m2[
            [
                "strategy",
                "variant_kind",
                "runtime_total_sec",
                "runtime_sec",
                "io_sec",
                "n_orbits",
                "n_orbits_covok",
                "n_time_targets",
                "n_ephem_rows_mean",
                "n_variant_orbits",
                "n_ephem_rows_variants",
                "error",
            ]
        ].copy()
        m2_key = m2_key.rename(
            columns={
                "runtime_total_sec": "stage2_runtime_total_sec",
                "runtime_sec": "stage2_runtime_sec",
                "io_sec": "stage2_io_sec",
                "n_orbits": "stage2_n_orbits",
                "n_orbits_covok": "stage2_n_orbits_covok",
                "n_time_targets": "stage2_n_time_targets",
                "n_ephem_rows_mean": "stage2_n_ephem_rows_mean",
                "n_variant_orbits": "stage2_n_variant_orbits",
                "n_ephem_rows_variants": "stage2_n_ephem_rows_variants",
                "error": "stage2_error",
            }
        )
    else:
        # Incremental mode: Stage 2 may still be running, so the combined metrics parquet
        # isn't written yet. Fall back to the per-strategy meta.json files.
        rows: list[dict[str, object]] = []
        strategies_dir = stage2_run_dir / "strategies"
        for meta_path in strategies_dir.glob("**/meta.json") if strategies_dir.exists() else []:
            try:
                meta = json.loads(meta_path.read_text())
            except Exception:  # noqa: BLE001
                continue
            rows.append(
                dict(
                    strategy=meta.get("strategy"),
                    variant_kind=meta.get("variant_kind"),
                    stage2_runtime_total_sec=meta.get("runtime_total_sec"),
                    stage2_runtime_sec=meta.get("runtime_sec"),
                    stage2_io_sec=meta.get("io_sec"),
                    stage2_n_orbits=meta.get("n_orbits"),
                    stage2_n_orbits_covok=meta.get("n_orbits_covok"),
                    stage2_n_time_targets=meta.get("n_time_targets"),
                    stage2_n_ephem_rows_mean=meta.get("n_rows") if meta.get("kind") == "mean_ephemeris" else None,
                    stage2_n_variant_orbits=meta.get("n_variant_orbits"),
                    stage2_n_ephem_rows_variants=meta.get("n_rows")
                    if meta.get("kind") == "variants_ephemeris"
                    else None,
                    stage2_error=meta.get("error"),
                )
            )
        m2_key = pd.DataFrame.from_records(rows)

    df = df.merge(m2_key, on=["strategy", "variant_kind"], how="left")

    # Derived metrics
    if "n_truth" in df.columns and "n_covered" in df.columns:
        df["n_missed"] = df["n_truth"] - df["n_covered"]
    else:
        df["n_missed"] = None

    df["total_runtime_sec"] = df["stage2_runtime_total_sec"] + df["runtime_sec"]
    df["n_errors"] = df["n_errors"].fillna(0).astype(int)

    return df


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Render a Stage2×Stage3 matrix table from Stage 2/3 metrics + coverage parquets.\n\n"
            "Tip: If stage3_run_dir is omitted, it defaults to sibling stage3/<stage2_run_dir.name>."
        )
    )
    p.add_argument("--stage2-run-dir", type=str, required=True)
    p.add_argument("--stage3-run-dir", type=str, default=None)
    p.add_argument(
        "--format",
        type=str,
        default="markdown",
        choices=["markdown", "csv"],
        help="Output format (default: markdown).",
    )
    p.add_argument(
        "--hide-errors",
        action="store_true",
        help="Hide rows with errors (n_errors>0) in the output.",
    )
    p.add_argument(
        "--only-missed",
        action="store_true",
        help="Only include rows with missed truth keys (n_missed>0).",
    )
    args = p.parse_args()

    stage2_run_dir = Path(args.stage2_run_dir)
    stage3_run_dir = (
        Path(args.stage3_run_dir)
        if args.stage3_run_dir is not None
        else stage2_run_dir.parent.parent / "stage3" / stage2_run_dir.name
    )

    df = build_matrix_table(stage2_run_dir=stage2_run_dir, stage3_run_dir=stage3_run_dir)

    # Column subset aimed at “what matters”: misses + runtime + (optional) extra frames.
    cols = [
        "strategy",
        "variant_kind",
        "footprint",
        "healpix_nside",
        "coverage",
        "n_truth",
        "n_covered",
        "n_missed",
        "n_selected",
        "n_extra_frames",
        "runtime_sec",
        "stage2_runtime_total_sec",
        "total_runtime_sec",
        "n_errors",
        "error",
        "stage2_error",
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = None

    out = df[cols].sort_values(["strategy", "variant_kind", "footprint"]).reset_index(drop=True)
    if bool(args.hide_errors):
        out = out[out["n_errors"] == 0].reset_index(drop=True)
    if bool(args.only_missed):
        out = out[out["n_missed"].fillna(0).astype(int) > 0].reset_index(drop=True)

    if str(args.format) == "csv":
        print(out.to_csv(index=False))
        return

    _print_markdown_table(out, cols=cols)


if __name__ == "__main__":
    main()

