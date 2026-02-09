from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import quivr as qv

# NOTE: Use absolute imports so this file can be run as a script path.
from experiments.covariance_precovery.data.subset_db import (
    GCS_ROOT,
    download_full_index_only,
    write_index_only_manifest,
)
from experiments.covariance_precovery.selection.bq_select import BqConfig
from experiments.covariance_precovery.selection.covariance_severity import (
    compute_covariance_severity,
)
from experiments.covariance_precovery.selection.fetch_selected_orbits_sbdb import (
    fetch_selected_orbits_via_sbdb,
)
from experiments.covariance_precovery.selection.long_span_sampling import (
    TimeBlock,
    generate_even_time_blocks,
    select_designations_across_blocks,
)
from experiments.covariance_precovery.selection.subset_designations import (
    read_subset_window,
)
from experiments.covariance_precovery.selection.subset_sampling import SamplingConfig
from experiments.covariance_precovery.truth.precovery_truth_crossmatch import (
    crossmatch_truth_to_precovery_subset,
)
from experiments.covariance_precovery.truth.subset_truth_observations import (
    fetch_and_persist_truth_observations_for_subset_selection,
)


@dataclass(frozen=True)
class PopulateConfig:
    db_dir: Path
    gcs_root: str = GCS_ROOT
    obscodes_csv: str | None = None
    n_blocks: int = 12
    block_days: int = 90
    n_total_designations: int = 600
    seed: int = 0


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_csv(s: str | None) -> list[str] | None:
    if s is None:
        return None
    items = [p.strip() for p in str(s).split(",")]
    items = [p for p in items if p]
    return None if not items else items


def populate_long_span_dataset(
    *,
    cfg: PopulateConfig,
    ensure_index_only: bool,
    select_designations: bool,
    fetch_orbits: bool,
    compute_cov_severity: bool,
    fetch_truth: bool,
    crossmatch_truth: bool,
    lazy_download_blobs: bool,
) -> dict[str, Path]:
    """
    Populate all artifacts needed to begin Stage2–Stage4 evaluations later.

    This intentionally does NOT run any evaluation harnesses.
    """
    db_dir = Path(cfg.db_dir).expanduser().resolve()
    db_dir.mkdir(parents=True, exist_ok=True)
    artifacts = db_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)

    if bool(ensure_index_only):
        download_full_index_only(dest_db_dir=db_dir, gcs_root=str(cfg.gcs_root))
        write_index_only_manifest(dest_db_dir=db_dir, gcs_root=str(cfg.gcs_root))

    # Discover time span from the local index.
    win = read_subset_window(db_dir)
    blocks: list[TimeBlock] = generate_even_time_blocks(
        start_utc=win.start_utc,
        end_utc_exclusive=win.end_utc_exclusive,
        n_blocks=int(cfg.n_blocks),
        block_days=int(cfg.block_days),
    )
    stns = _parse_csv(cfg.obscodes_csv) or list(win.obscodes)
    if len(stns) > 50:
        raise ValueError(
            f"Refusing to run BQ queries with {len(stns)} obscodes. "
            "Pass a smaller list via --obscodes."
        )

    out: dict[str, Path] = {}
    # Persist the block plan.
    blocks_meta = {
        "subset_dir": str(db_dir),
        "index_db": str(db_dir / "index.db"),
        "obscodes": stns,
        "blocks": [
            {
                "label": b.label,
                "start_utc": b.start_utc.isoformat().replace("+00:00", "Z"),
                "end_utc_exclusive": b.end_utc_exclusive.isoformat().replace("+00:00", "Z"),
            }
            for b in blocks
        ],
        "generated_at_utc": _now_utc(),
    }
    blocks_meta_path = artifacts / "long_span_blocks_plan.json"
    blocks_meta_path.write_text(json.dumps(blocks_meta, indent=2, sort_keys=True) + "\n")
    out["blocks_plan_json"] = blocks_meta_path

    selected_path = artifacts / "selected_designations_long_span.parquet"
    if bool(select_designations):
        selected_path = select_designations_across_blocks(
            subset_dir=db_dir,
            blocks=blocks,
            sampling_cfg=SamplingConfig(n_total=int(cfg.n_total_designations), seed=int(cfg.seed)),
            cfg=BqConfig(),
            obscodes=stns,
        )
    out["selected_designations_parquet"] = selected_path

    orbits_path = artifacts / "orbits_selected_sbdb.parquet"
    if bool(fetch_orbits):
        r = fetch_selected_orbits_via_sbdb(
            subset_dir=db_dir,
            selected_designations_parquet=selected_path,
        )
        orbits_path = r.orbits_parquet
        out["orbits_selected_sbdb_parquet"] = orbits_path
        out["orbits_selected_sbdb_failures_parquet"] = r.failures_parquet
        out["orbits_selected_sbdb_meta_json"] = r.meta_json

    if bool(compute_cov_severity):
        # Uses the SBDB orbit covariance at epoch for stratification.
        from adam_core.orbits import Orbits

        orbits = Orbits.from_parquet(str(orbits_path))
        sev = compute_covariance_severity(orbits=orbits)
        sev_path = artifacts / "orbits_selected_sbdb_cov_severity.parquet"
        sev.to_parquet(str(sev_path))
        out["cov_severity_parquet"] = sev_path

    if bool(fetch_truth):
        truth = fetch_and_persist_truth_observations_for_subset_selection(
            subset_dir=db_dir,
            cfg=BqConfig(),
            selected_designations_parquet=selected_path,
        )
        out["truth_observations_parquet"] = truth.truth_parquet
        out["truth_observations_meta_json"] = truth.meta_json

    if bool(crossmatch_truth):
        truth_parquet = out.get("truth_observations_parquet")
        if truth_parquet is None:
            truth_parquet = artifacts / "truth_observations_selected.parquet"
        cm = crossmatch_truth_to_precovery_subset(
            subset_dir=db_dir,
            truth_parquet=Path(truth_parquet),
            lazy_download_blobs=bool(lazy_download_blobs),
            gcs_root=str(cfg.gcs_root),
        )
        out["truth_precovery_crossmatch_parquet"] = cm.out_parquet
        out["truth_precovery_crossmatch_meta_json"] = cm.out_meta_json

    # Persist a pipeline manifest for reproducibility.
    manifest = {
        "subset_dir": str(db_dir),
        "gcs_root": str(cfg.gcs_root),
        "obscodes": stns,
        "n_blocks": int(cfg.n_blocks),
        "block_days": int(cfg.block_days),
        "n_total_designations": int(cfg.n_total_designations),
        "seed": int(cfg.seed),
        "steps": {
            "ensure_index_only": bool(ensure_index_only),
            "select_designations": bool(select_designations),
            "fetch_orbits": bool(fetch_orbits),
            "compute_cov_severity": bool(compute_cov_severity),
            "fetch_truth": bool(fetch_truth),
            "crossmatch_truth": bool(crossmatch_truth),
            "lazy_download_blobs": bool(lazy_download_blobs),
        },
        "outputs": {k: str(v) for k, v in out.items()},
        "generated_at_utc": _now_utc(),
    }
    manifest_path = artifacts / "populate_long_span_dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    out["manifest_json"] = manifest_path

    return out


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(
        description=(
            "Populate long-span experiment inputs (index-only DB + selection + SBDB orbits + truth + crossmatch) "
            "without running Stage2–Stage4 evaluations."
        )
    )
    p.add_argument("--db-dir", type=str, required=True, help="Local DB dir (will contain config.json + index.db).")
    p.add_argument("--gcs-root", type=str, default=GCS_ROOT)
    p.add_argument(
        "--ensure-index-only",
        action="store_true",
        help="If set, download config.json + full index.db into --db-dir.",
    )
    p.add_argument(
        "--obscodes",
        type=str,
        default=None,
        help="Comma-separated obscodes to use for BQ selection (default: all obscodes in index.db).",
    )
    p.add_argument("--n-blocks", type=int, default=12)
    p.add_argument("--block-days", type=int, default=90)
    p.add_argument("--n-total-designations", type=int, default=600)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--select-designations", action="store_true")
    p.add_argument("--fetch-orbits", action="store_true")
    p.add_argument("--compute-cov-severity", action="store_true")
    p.add_argument("--fetch-truth", action="store_true")
    p.add_argument("--crossmatch-truth", action="store_true")
    p.add_argument(
        "--lazy-download-blobs",
        action="store_true",
        help="When crossmatching truth, download missing blobs from GCS on demand.",
    )

    args = p.parse_args()
    cfg = PopulateConfig(
        db_dir=Path(args.db_dir),
        gcs_root=str(args.gcs_root),
        obscodes_csv=args.obscodes,
        n_blocks=int(args.n_blocks),
        block_days=int(args.block_days),
        n_total_designations=int(args.n_total_designations),
        seed=int(args.seed),
    )

    out = populate_long_span_dataset(
        cfg=cfg,
        ensure_index_only=bool(args.ensure_index_only),
        select_designations=bool(args.select_designations),
        fetch_orbits=bool(args.fetch_orbits),
        compute_cov_severity=bool(args.compute_cov_severity),
        fetch_truth=bool(args.fetch_truth),
        crossmatch_truth=bool(args.crossmatch_truth),
        lazy_download_blobs=bool(args.lazy_download_blobs),
    )
    # Print key outputs for shell scripting convenience.
    for k, v in out.items():
        print(f"{k}={v}")


if __name__ == "__main__":
    main()

