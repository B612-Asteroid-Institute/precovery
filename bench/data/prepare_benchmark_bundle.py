from __future__ import annotations

import argparse
from pathlib import Path
from .bundles import prepare_benchmark_bundle


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prepare a standalone monthly benchmark bundle (orbits + truth + bins)."
    )
    p.add_argument("--subset-dir", type=str, required=True)
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--detections-parquet", type=str, required=True, help="DuckDB parquet dataset path.")

    p.add_argument("--month", action="append", required=True, help="Month window YYYY-MM (repeatable).")
    p.add_argument("--obscode", action="append", required=True, help="Observatory code (repeatable).")

    p.add_argument("--orbits-parquet", type=str, default="", help="Input Orbits parquet.")
    p.add_argument("--truth-crossmatch-parquet", type=str, default="", help="Truth precovery crossmatch parquet.")
    p.add_argument("--selected-designations-parquet", type=str, default="", help="Selected designations parquet.")
    p.add_argument(
        "--designation-resolution-parquet",
        type=str,
        default="",
        help="Canonical designation resolution parquet (auto-discovered if omitted).",
    )
    p.add_argument("--cov-severity-parquet", type=str, default="", help="Orbit covariance severity parquet.")
    p.add_argument("--features-parquet", type=str, default="", help="Orbit features parquet (orbit_type_int/u_param).")
    p.add_argument(
        "--excluded-orbits-parquet",
        type=str,
        default="",
        help="Optional parquet listing known-bad orbit_ids to exclude (Stage-3-zero-hit outliers).",
    )

    p.add_argument(
        "--keep-known-bad",
        action="store_true",
        help="Do not drop known-bad orbit_ids from --excluded-orbits-parquet.",
    )
    p.add_argument(
        "--fast-orbits-k",
        type=int,
        default=6,
        help="Number of orbits to include in the small fast benchmark subset (default: 6).",
    )
    p.add_argument(
        "--fail-on-unresolved-truth",
        action="store_true",
        help="Fail bundle creation if any truth designation cannot be resolved to an orbit_id.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    prepare_benchmark_bundle(
        subset_dir=Path(args.subset_dir),
        out_dir=Path(args.out_dir),
        detections_parquet=Path(args.detections_parquet),
        months=[str(x) for x in args.month],
        obscodes=[str(x) for x in args.obscode],
        orbits_parquet=(Path(args.orbits_parquet) if args.orbits_parquet else None),
        truth_crossmatch_parquet=(
            Path(args.truth_crossmatch_parquet) if args.truth_crossmatch_parquet else None
        ),
        selected_designations_parquet=(
            Path(args.selected_designations_parquet) if args.selected_designations_parquet else None
        ),
        designation_resolution_parquet=(
            Path(args.designation_resolution_parquet) if args.designation_resolution_parquet else None
        ),
        cov_severity_parquet=(Path(args.cov_severity_parquet) if args.cov_severity_parquet else None),
        features_parquet=(Path(args.features_parquet) if args.features_parquet else None),
        excluded_orbits_parquet=(
            Path(args.excluded_orbits_parquet) if args.excluded_orbits_parquet else None
        ),
        keep_known_bad=bool(args.keep_known_bad),
        fast_orbits_k=int(args.fast_orbits_k),
        fail_on_unresolved_truth=bool(args.fail_on_unresolved_truth),
    )


if __name__ == "__main__":
    main()
