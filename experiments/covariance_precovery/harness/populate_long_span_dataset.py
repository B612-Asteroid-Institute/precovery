from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import quivr as qv
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

# NOTE: Use absolute imports so this file can be run as a script path.
from experiments.covariance_precovery.data.gcs_copy import default_copy_tool
from experiments.covariance_precovery.data.subset_db import (
    GCS_ROOT,
    download_full_index_only,
    write_index_only_manifest,
)
from experiments.covariance_precovery.selection.bq_select import (
    BqConfig,
    bq_table_ref,
    export_bq_table_to_gcs_parquet,
    materialize_designation_orbit_features_for_window_to_table,
)
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
from experiments.covariance_precovery.selection.subset_sampling import (
    SamplingConfig,
    select_designations_from_features,
)
from experiments.covariance_precovery.truth.precovery_truth_crossmatch import (
    count_distinct_data_uris_for_truth,
    crossmatch_truth_to_precovery_subset,
)
from experiments.covariance_precovery.truth.subset_truth_observations import (
    TruthObservationByDesignation,
    fetch_truth_observations_for_designations,
)


@dataclass(frozen=True)
class PopulateConfig:
    db_dir: Path
    gcs_root: str = GCS_ROOT
    obscodes_csv: str | None = None
    analysis_start_utc: str | None = None
    analysis_end_utc_exclusive: str | None = None
    run_tag: str | None = None
    n_blocks: int = 12
    block_days: int = 90
    n_total_designations: int = 600
    seed: int = 0
    bq_project_id: str = "moeyens-thor-dev"
    bq_dataset_id: str = "ai_aleck_scratch"
    gcs_export_prefix: str = "gs://ak-scratch/precovery/covariance_precovery/bq_exports"


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_utc_arg(s: str | None) -> datetime | None:
    """
    Parse a UTC timestamp argument.

    Accepts:
    - YYYY-MM-DD
    - ISO-8601 with optional 'Z' suffix (e.g. 2020-01-01T00:00:00Z)
    """
    if s is None:
        return None
    ss = str(s).strip()
    if not ss:
        return None
    if "T" not in ss and len(ss) == 10:
        ss = ss + "T00:00:00Z"
    if ss.endswith("Z"):
        ss = ss[:-1] + "+00:00"
    dt = datetime.fromisoformat(ss)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _mjd_from_datetime_utc(dt: datetime) -> float:
    # 1970-01-01 00:00:00 UTC is MJD 40587.0
    dtu = dt.astimezone(timezone.utc)
    return float(dtu.timestamp()) / 86400.0 + 40587.0


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
    materialize_full_span_features: bool,
    select_designations: bool,
    fetch_orbits: bool,
    compute_cov_severity: bool,
    fetch_truth: bool,
    crossmatch_truth: bool,
    lazy_download_blobs: bool,
    count_crossmatch_blobs: bool,
) -> dict[str, Path]:
    """
    Populate all artifacts needed to begin Stage2–Stage4 evaluations later.

    This intentionally does NOT run any evaluation harnesses.
    """
    db_dir = Path(cfg.db_dir).expanduser().resolve()
    db_dir.mkdir(parents=True, exist_ok=True)
    artifacts_root = db_dir / "artifacts"
    artifacts_root.mkdir(parents=True, exist_ok=True)

    if bool(ensure_index_only):
        download_full_index_only(dest_db_dir=db_dir, gcs_root=str(cfg.gcs_root))
        write_index_only_manifest(dest_db_dir=db_dir, gcs_root=str(cfg.gcs_root))

    # Discover time span from the local index.
    win = read_subset_window(db_dir)
    analysis_start = _parse_utc_arg(cfg.analysis_start_utc) or win.start_utc
    analysis_end_exclusive = _parse_utc_arg(cfg.analysis_end_utc_exclusive) or win.end_utc_exclusive
    if analysis_end_exclusive <= analysis_start:
        raise ValueError("analysis_end_utc_exclusive must be > analysis_start_utc")
    if analysis_start < win.start_utc or analysis_end_exclusive > win.end_utc_exclusive:
        raise ValueError(
            "Analysis window must be within the subset index window. "
            f"Got [{analysis_start.isoformat()}, {analysis_end_exclusive.isoformat()}), "
            f"index window is [{win.start_utc.isoformat()}, {win.end_utc_exclusive.isoformat()})."
        )

    blocks: list[TimeBlock] = generate_even_time_blocks(
        start_utc=analysis_start,
        end_utc_exclusive=analysis_end_exclusive,
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

    stn_label = "_".join(sorted(str(s) for s in stns))
    stn_label = "".join(c if (c.isalnum() or c == "_") else "_" for c in stn_label)
    start_tag = analysis_start.date().isoformat().replace("-", "")
    end_tag = analysis_end_exclusive.date().isoformat().replace("-", "")
    auto_tag = f"w_{start_tag}_{end_tag}__{stn_label}"
    run_tag = str(cfg.run_tag).strip() if cfg.run_tag is not None else None
    if not run_tag and (analysis_start != win.start_utc or analysis_end_exclusive != win.end_utc_exclusive):
        run_tag = auto_tag
    artifacts = artifacts_root if not run_tag else (artifacts_root / run_tag)
    artifacts.mkdir(parents=True, exist_ok=True)

    # Optional: materialize a single window features table in BQ and export to local Parquet.
    full_span_features_parquet: Path | None = None
    if bool(materialize_full_span_features):
        merged = artifacts / "bq_designation_orbit_features.parquet"
        meta_path = artifacts / "bq_designation_orbit_features_meta.json"
        if merged.exists() and meta_path.exists():
            full_span_features_parquet = merged
            out["bq_full_span_features_parquet"] = merged
            out["bq_full_span_features_meta_json"] = meta_path
        else:
            table_id = f"precovery_designation_orbit_features__{stn_label}__{start_tag}_{end_tag}"

            dest_table = bq_table_ref(
                project_id=str(cfg.bq_project_id),
                dataset_id=str(cfg.bq_dataset_id),
                table_id=str(table_id),
            )
            materialize_designation_orbit_features_for_window_to_table(
                cfg=BqConfig(),
                start_utc=analysis_start,
                end_utc=analysis_end_exclusive,
                obscodes=stns,
                destination_table=str(dest_table),
                replace=True,
            )

            gcs_dir = f"{str(cfg.gcs_export_prefix).rstrip('/')}/{table_id}"
            export_uri = f"{gcs_dir}/part-*.parquet"
            export_bq_table_to_gcs_parquet(source_table=str(dest_table), destination_uri=str(export_uri))

            local_shards_dir = artifacts / "bq_full_span_features_shards"
            local_shards_dir.mkdir(parents=True, exist_ok=True)
            copy_tool = default_copy_tool()
            copy_tool.cp(str(gcs_dir), local_shards_dir, recursive=True)

            parquet_files = sorted(p for p in local_shards_dir.rglob("*.parquet") if p.is_file())
            if not parquet_files:
                raise RuntimeError(f"No Parquet files downloaded under {local_shards_dir}")

            dataset = ds.dataset([str(p) for p in parquet_files], format="parquet")
            schema = dataset.schema
            writer = pq.ParquetWriter(str(merged), schema=schema, compression="snappy")
            try:
                for batch in dataset.to_batches(batch_size=65536):
                    writer.write_table(pa.Table.from_batches([batch], schema=schema))
            finally:
                writer.close()

            meta = {
                "subset_dir": str(db_dir),
                "obscodes": stns,
                "analysis_window": {
                    "start_utc": analysis_start.isoformat().replace("+00:00", "Z"),
                    "end_utc_exclusive": analysis_end_exclusive.isoformat().replace("+00:00", "Z"),
                },
                "bq_project_id": str(cfg.bq_project_id),
                "bq_dataset_id": str(cfg.bq_dataset_id),
                "bq_destination_table": str(dest_table),
                "gcs_export_dir": str(gcs_dir),
                "local_shards_dir": str(local_shards_dir),
                "merged_parquet": str(merged),
                "generated_at_utc": _now_utc(),
            }
            meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")

            full_span_features_parquet = merged
            out["bq_full_span_features_parquet"] = merged
            out["bq_full_span_features_meta_json"] = meta_path
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
        if full_span_features_parquet is not None:
            from experiments.covariance_precovery.selection.bq_select import (
                DesignationOrbitWindowFeatures,
            )

            selected_path = artifacts / "selected_designations.parquet"
            if not selected_path.exists():
                feats = DesignationOrbitWindowFeatures.from_parquet(str(full_span_features_parquet))
                window_mid_mjd = 0.5 * (
                    _mjd_from_datetime_utc(analysis_start) + _mjd_from_datetime_utc(analysis_end_exclusive)
                )
                selected = select_designations_from_features(
                    features=feats,
                    window_mid_mjd=float(window_mid_mjd),
                    cfg=SamplingConfig(n_total=int(cfg.n_total_designations), seed=int(cfg.seed)),
                )
                selected.to_parquet(str(selected_path))
        else:
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
        # Reuse if already present.
        failures_path = artifacts / "orbits_selected_sbdb_failures.parquet"
        meta_path = artifacts / "orbits_selected_sbdb_meta.json"
        if orbits_path.exists() and failures_path.exists() and meta_path.exists():
            out["orbits_selected_sbdb_parquet"] = orbits_path
            out["orbits_selected_sbdb_failures_parquet"] = failures_path
            out["orbits_selected_sbdb_meta_json"] = meta_path
        else:
            r = fetch_selected_orbits_via_sbdb(
                subset_dir=db_dir,
                selected_designations_parquet=selected_path,
                artifacts_dir=artifacts,
            )
            orbits_path = r.orbits_parquet
            out["orbits_selected_sbdb_parquet"] = orbits_path
            out["orbits_selected_sbdb_failures_parquet"] = r.failures_parquet
            out["orbits_selected_sbdb_meta_json"] = r.meta_json

    if bool(compute_cov_severity):
        # Uses the SBDB orbit covariance at epoch for stratification.
        from adam_core.orbits import Orbits

        sev_path = artifacts / "orbits_selected_sbdb_cov_severity.parquet"
        if not sev_path.exists():
            orbits = Orbits.from_parquet(str(orbits_path))
            sev = compute_covariance_severity(orbits=orbits)
            sev.to_parquet(str(sev_path))
        out["cov_severity_parquet"] = sev_path

    if bool(fetch_truth):
        truth_parquet = artifacts / "truth_observations_selected.parquet"
        truth_meta_path = artifacts / "truth_observations_selected_meta.json"
        if truth_parquet.exists() and truth_meta_path.exists():
            out["truth_observations_parquet"] = truth_parquet
            out["truth_observations_meta_json"] = truth_meta_path
        else:
            # Fetch truth observations for the selected designations.
            from experiments.covariance_precovery.selection.subset_sampling import SelectedDesignations

            sel = SelectedDesignations.from_parquet(str(selected_path))
            designations = [str(x) for x in sel.designation.to_pylist()]
            if not designations:
                raise ValueError("No selected designations; cannot fetch truth.")

            if full_span_features_parquet is not None:
                truth_all = fetch_truth_observations_for_designations(
                    cfg=BqConfig(),
                    designations=designations,
                    obscodes=stns,
                    start_utc=analysis_start,
                    end_utc=analysis_end_exclusive,
                )
                truth_windows_meta: dict[str, object] = {
                    "mode": "analysis_window",
                    "start_utc": analysis_start.isoformat().replace("+00:00", "Z"),
                    "end_utc_exclusive": analysis_end_exclusive.isoformat().replace("+00:00", "Z"),
                }
            else:
                truth_all = TruthObservationByDesignation.empty()
                for b in blocks:
                    t = fetch_truth_observations_for_designations(
                        cfg=BqConfig(),
                        designations=designations,
                        obscodes=stns,
                        start_utc=b.start_utc,
                        end_utc=b.end_utc_exclusive,
                    )
                    truth_all = qv.concatenate([truth_all, t])
                truth_windows_meta = {
                    "mode": "blocks",
                    "blocks": [
                        {
                            "label": b.label,
                            "start_utc": b.start_utc.isoformat().replace("+00:00", "Z"),
                            "end_utc_exclusive": b.end_utc_exclusive.isoformat().replace("+00:00", "Z"),
                        }
                        for b in blocks
                    ],
                }

            truth_all.to_parquet(str(truth_parquet))
            truth_meta = {
                "subset_dir": str(db_dir),
                "selected_designations_parquet": str(selected_path),
                "n_selected_designations": int(len(designations)),
                "n_truth_observations": int(len(truth_all)),
                "obscodes": stns,
                "truth_windows": truth_windows_meta,
                "analysis_window": {
                    "start_utc": analysis_start.isoformat().replace("+00:00", "Z"),
                    "end_utc_exclusive": analysis_end_exclusive.isoformat().replace("+00:00", "Z"),
                },
                "run_tag": run_tag,
                "generated_at_utc": _now_utc(),
            }
            truth_meta_path.write_text(json.dumps(truth_meta, indent=2, sort_keys=True) + "\n")

            out["truth_observations_parquet"] = truth_parquet
            out["truth_observations_meta_json"] = truth_meta_path

    truth_parquet_for_cm = out.get("truth_observations_parquet") or (artifacts / "truth_observations_selected.parquet")

    if bool(count_crossmatch_blobs):
        if not truth_parquet_for_cm.exists():
            raise FileNotFoundError(
                f"Cannot count crossmatch blobs: truth parquet not found at {truth_parquet_for_cm}"
            )
        blob_json = artifacts / "truth_precovery_blob_count.json"
        uris_txt = artifacts / "truth_precovery_blob_uris.txt"
        if not blob_json.exists():
            blob_res = count_distinct_data_uris_for_truth(
                subset_dir=db_dir,
                truth_parquet=Path(truth_parquet_for_cm),
                artifacts_dir=artifacts,
            )
            out["truth_precovery_blob_count_json"] = blob_res.out_json
            out["truth_precovery_blob_uris_txt"] = blob_res.uris_txt
        else:
            out["truth_precovery_blob_count_json"] = blob_json
            if uris_txt.exists():
                out["truth_precovery_blob_uris_txt"] = uris_txt

    if bool(crossmatch_truth):
        out_parquet = artifacts / "truth_precovery_crossmatch.parquet"
        out_meta = artifacts / "truth_precovery_crossmatch_meta.json"
        if out_parquet.exists() and out_meta.exists():
            out["truth_precovery_crossmatch_parquet"] = out_parquet
            out["truth_precovery_crossmatch_meta_json"] = out_meta
        else:
            truth_parquet = truth_parquet_for_cm
            cm = crossmatch_truth_to_precovery_subset(
                subset_dir=db_dir,
                truth_parquet=Path(truth_parquet),
                lazy_download_blobs=bool(lazy_download_blobs),
                gcs_root=str(cfg.gcs_root),
                artifacts_dir=artifacts,
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
        "bq_project_id": str(cfg.bq_project_id),
        "bq_dataset_id": str(cfg.bq_dataset_id),
        "gcs_export_prefix": str(cfg.gcs_export_prefix),
        "analysis_window": {
            "start_utc": analysis_start.isoformat().replace("+00:00", "Z"),
            "end_utc_exclusive": analysis_end_exclusive.isoformat().replace("+00:00", "Z"),
        },
        "run_tag": run_tag,
        "steps": {
            "ensure_index_only": bool(ensure_index_only),
            "materialize_full_span_features": bool(materialize_full_span_features),
            "select_designations": bool(select_designations),
            "fetch_orbits": bool(fetch_orbits),
            "compute_cov_severity": bool(compute_cov_severity),
            "fetch_truth": bool(fetch_truth),
            "crossmatch_truth": bool(crossmatch_truth),
            "lazy_download_blobs": bool(lazy_download_blobs),
            "count_crossmatch_blobs": bool(count_crossmatch_blobs),
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
    p.add_argument(
        "--analysis-start-utc",
        type=str,
        default=None,
        help="Override analysis start (UTC). Example: 2020-01-01 or 2020-01-01T00:00:00Z",
    )
    p.add_argument(
        "--analysis-end-utc-exclusive",
        type=str,
        default=None,
        help="Override analysis end exclusive (UTC). Example: 2024-01-01 or 2024-01-01T00:00:00Z",
    )
    p.add_argument(
        "--run-tag",
        type=str,
        default=None,
        help="Optional tag to write artifacts under <db-dir>/artifacts/<run-tag>/ (avoids clobbering).",
    )

    p.add_argument(
        "--materialize-full-span-features",
        action="store_true",
        help=(
            "If set, run a single analysis-window BQ group-by query, write results to a BQ table, "
            "export to Parquet in GCS, download locally, and merge into one Parquet file."
        ),
    )
    p.add_argument("--bq-project-id", type=str, default="moeyens-thor-dev")
    p.add_argument("--bq-dataset-id", type=str, default="ai_aleck_scratch")
    p.add_argument(
        "--gcs-export-prefix",
        type=str,
        default="gs://ak-scratch/precovery/covariance_precovery/bq_exports",
        help="GCS prefix for exported Parquet shards.",
    )

    p.add_argument("--select-designations", action="store_true")
    p.add_argument("--fetch-orbits", action="store_true")
    p.add_argument("--compute-cov-severity", action="store_true")
    p.add_argument("--fetch-truth", action="store_true")
    p.add_argument("--crossmatch-truth", action="store_true")
    p.add_argument(
        "--count-crossmatch-blobs",
        action="store_true",
        help="Pre-count distinct frame data URIs needed for crossmatch (index + truth only, no download). Writes truth_precovery_blob_count.json.",
    )
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
        analysis_start_utc=args.analysis_start_utc,
        analysis_end_utc_exclusive=args.analysis_end_utc_exclusive,
        run_tag=args.run_tag,
        n_blocks=int(args.n_blocks),
        block_days=int(args.block_days),
        n_total_designations=int(args.n_total_designations),
        seed=int(args.seed),
        bq_project_id=str(args.bq_project_id),
        bq_dataset_id=str(args.bq_dataset_id),
        gcs_export_prefix=str(args.gcs_export_prefix),
    )

    out = populate_long_span_dataset(
        cfg=cfg,
        ensure_index_only=bool(args.ensure_index_only),
        materialize_full_span_features=bool(args.materialize_full_span_features),
        select_designations=bool(args.select_designations),
        fetch_orbits=bool(args.fetch_orbits),
        compute_cov_severity=bool(args.compute_cov_severity),
        fetch_truth=bool(args.fetch_truth),
        crossmatch_truth=bool(args.crossmatch_truth),
        lazy_download_blobs=bool(args.lazy_download_blobs),
        count_crossmatch_blobs=bool(args.count_crossmatch_blobs),
    )
    # Print key outputs for shell scripting convenience.
    for k, v in out.items():
        print(f"{k}={v}")


if __name__ == "__main__":
    main()

