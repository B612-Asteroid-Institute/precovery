from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
from typing import Literal, Mapping

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import time

from adam_core.orbits import Orbits

from .artifacts import (
    read_stage23_artifacts,
    stage23_artifacts_exist,
    write_stage23_artifacts,
)
from .chunking import resolve_time_chunk_size_from_ram_budget
from .pipeline_build import build_predictions_and_triples
from .pipeline_types import (
    AcceptedCounts,
    BenchTargets,
    PredictedTargets,
    PredictedTriples,
    Stage3OrbitMetrics,
    Stage4OrbitMetrics,
    SubsetPaths,
)
from .backends.protocols import GateParams, SearchBackend
from .gate_counts import (
    GateTotals,
    gate_keep_and_reject_masks_python,
)
from .results import AcceptedDetections

STAGE23_EFFECTIVE_BYTES_PER_ROW = 512.0
# Stage4 chunk sizing must account for candidate amplification from triples->detections join.
# This is intentionally conservative so default auto-chunking stays spill-safe on dense frames.
STAGE4_EFFECTIVE_BYTES_PER_TRIPLE_ROW = 50_000.0


@dataclass(frozen=True)
class BuildStage23:
    preds: PredictedTargets
    triples: PredictedTriples
    stage3_orbit_metrics: Stage3OrbitMetrics

    build_observers_elapsed_s: float
    build_predict_elapsed_s: float
    build_pred_mag_elapsed_s: float
    build_footprint_elapsed_s: float
    build_triples_elapsed_s: float

    n_frames_skipped_limiting_mag: int
    n_frames_skipped_uncertainty: int
    n_targets_preprop_viability_rejected: int
    n_targets_preprop_time_limited: int
    n_targets_failfast_dynamics_error: int
    micro_timings: Mapping[str, float]


@dataclass(frozen=True)
class BackendGateResult:
    accepted_counts: AcceptedCounts
    totals: GateTotals
    elapsed_s: float


@dataclass(frozen=True)
class BackendGateRowsResult:
    accepted_detections: AcceptedDetections
    accepted_counts: AcceptedCounts
    totals: GateTotals
    orbit_metrics: Stage4OrbitMetrics
    elapsed_s: float


@dataclass(frozen=True)
class Stage14RunPython:
    targets: BenchTargets
    build: BuildStage23
    gate: BackendGateResult


@dataclass(frozen=True)
class Stage14RunRowsPython:
    targets: BenchTargets
    build: BuildStage23
    gate: BackendGateRowsResult


@dataclass(frozen=True)
class DiskRowsRunSummary:
    stage23_run_dir: Path
    stage4_run_dir: Path
    stage3_orbit_metrics: Stage3OrbitMetrics
    stage4_orbit_metrics: Stage4OrbitMetrics
    accepted_counts: AcceptedCounts
    gate_totals: GateTotals
    build_micro_timings: dict[str, float]
    gate_micro_timings: dict[str, float]
    build_observers_elapsed_s: float
    build_predict_elapsed_s: float
    build_pred_mag_elapsed_s: float
    build_footprint_elapsed_s: float
    build_triples_elapsed_s: float
    n_frames_skipped_limiting_mag: int
    n_frames_skipped_uncertainty: int
    n_targets_preprop_viability_rejected: int
    n_targets_preprop_time_limited: int
    n_targets_failfast_dynamics_error: int
    backend_elapsed_s: float
    n_unique_frames_selected: int
    n_detections_selected_exposure_keys: int | None = None
    n_detections_selected_frame_keys: int | None = None
    n_detections_healpixel_nonmatch: int | None = None


def _free_disk_gb(path: Path) -> float:
    usage = shutil.disk_usage(str(path))
    return float(usage.free) / float(1024**3)


def _ensure_min_free_disk(*, path: Path, min_free_disk_gb: float) -> None:
    if float(min_free_disk_gb) <= 0.0:
        return
    free_gb = _free_disk_gb(path)
    if free_gb < float(min_free_disk_gb):
        raise RuntimeError(
            f"Insufficient free disk for disk execution mode at {path}: "
            f"free_gb={free_gb:.3f} required_gb={float(min_free_disk_gb):.3f}"
        )


def _sum_timings(dst: dict[str, float], src: Mapping[str, float] | None) -> None:
    if src is None:
        return
    for k, v in src.items():
        dst[str(k)] = float(dst.get(str(k), 0.0) + float(v))


def _ensure_empty_parquet(*, path: Path, schema: pa.Schema) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_arrays([pa.array([], type=f.type) for f in schema], names=schema.names), path)


def _resolve_disk_chunk_rows(
    *,
    requested_chunk_rows: int,
    total_rows_hint: int,
    effective_bytes_per_row: float,
    ram_budget_frac: float = 0.10,
) -> tuple[int, dict[str, object]]:
    req = int(requested_chunk_rows)
    total = int(max(0, int(total_rows_hint)))
    if req > 0:
        resolved = int(max(1, req))
        return resolved, {
            "mode": "manual",
            "requested_chunk_rows": int(req),
            "resolved_chunk_rows": int(resolved),
            "total_rows_hint": int(total),
        }

    n_targets = int(max(1, total))
    resolved_rows, meta = resolve_time_chunk_size_from_ram_budget(
        n_time_targets=int(n_targets),
        rows_per_time_target=1,
        ram_budget_frac=float(ram_budget_frac),
        effective_bytes_per_row=float(effective_bytes_per_row),
        min_chunk=1,
    )
    resolved = int(max(1, int(resolved_rows)))
    if total > 0:
        resolved = int(min(int(resolved), int(total)))
    return resolved, {
        "mode": "auto_ram_budget",
        "requested_chunk_rows": int(req),
        "resolved_chunk_rows": int(resolved),
        "total_rows_hint": int(total),
        "ram_budget_frac": float(ram_budget_frac),
        "effective_bytes_per_row": float(effective_bytes_per_row),
        "ram_chunk_meta": {
            "system_ram_bytes": None if meta.system_ram_bytes is None else int(meta.system_ram_bytes),
            "max_usage_bytes": None if meta.max_usage_bytes is None else int(meta.max_usage_bytes),
            "max_rows_fit": int(meta.max_rows_fit),
        },
    }


def _unique_frames_count_from_triples_file(triples_path: Path) -> int:
    if not triples_path.exists():
        return 0
    try:
        import duckdb  # type: ignore

        con = duckdb.connect(database=":memory:")
        try:
            p = str(triples_path).replace("'", "''")
            row = con.execute(
                f"""
                SELECT COUNT(DISTINCT (obscode, exposure_mjd_mid_key_us, healpixel)) AS n
                FROM read_parquet('{p}')
                """
            ).fetchone()
            if row is None or row[0] is None:
                return 0
            return int(row[0])
        finally:
            con.close()
    except Exception:
        t = pq.read_table(triples_path, columns=["obscode", "exposure_mjd_mid_key_us", "healpixel"])
        if t.num_rows == 0:
            return 0
        return int(t.group_by(["obscode", "exposure_mjd_mid_key_us", "healpixel"]).aggregate([]).num_rows)


def _preds_subset_for_triples(*, preds_table: pa.Table, triples_table: pa.Table) -> pa.Table:
    """
    Return the subset of prediction rows needed by `triples_table` keys.
    """
    if triples_table.num_rows == 0 or preds_table.num_rows == 0:
        return preds_table.slice(0, 0)
    key_pairs = triples_table.select(["orbit_id", "target_idx"]).group_by(["orbit_id", "target_idx"]).aggregate([])
    if key_pairs.num_rows == 0:
        return preds_table.slice(0, 0)
    return preds_table.join(key_pairs, keys=["orbit_id", "target_idx"], join_type="inner")


def _aggregate_accepted_counts_tables(parts: list[pa.Table]) -> pa.Table:
    if not parts:
        return AcceptedCounts.empty().table
    t = pa.concat_tables(parts, promote_options="default")
    if t.num_rows == 0:
        return AcceptedCounts.empty().table
    gb = t.group_by(["orbit_id", "target_idx"]).aggregate(
        [
            ("n_candidates", "sum"),
            ("n_accepted", "sum"),
        ]
    )
    gb = gb.rename_columns(["orbit_id", "target_idx", "n_candidates", "n_accepted"])
    return AcceptedCounts.from_pyarrow(gb).table


def _aggregate_stage4_orbit_metrics_tables(parts: list[pa.Table]) -> pa.Table:
    if not parts:
        return Stage4OrbitMetrics.empty().table
    t = pa.concat_tables(parts, promote_options="default")
    if t.num_rows == 0:
        return Stage4OrbitMetrics.empty().table
    gb = t.group_by(["orbit_id"]).aggregate(
        [
            ("n_detections_candidates", "sum"),
            ("n_detections_gate_matched", "sum"),
            ("n_detections_innov_ellipse_rejected", "sum"),
            ("n_detections_magnitude_rejected", "sum"),
            ("n_detections_truth_gate_matched", "sum"),
            ("n_detections_truth_magnitude_rejected", "sum"),
            ("n_detections_truth_final", "sum"),
            ("n_detections_false_positive_final", "sum"),
            ("n_detections_unknown_final", "sum"),
        ]
    )
    gb = gb.rename_columns(
        [
            "orbit_id",
            "n_detections_candidates",
            "n_detections_gate_matched",
            "n_detections_innov_ellipse_rejected",
            "n_detections_magnitude_rejected",
            "n_detections_truth_gate_matched",
            "n_detections_truth_magnitude_rejected",
            "n_detections_truth_final",
            "n_detections_false_positive_final",
            "n_detections_unknown_final",
        ]
    )
    return Stage4OrbitMetrics.from_pyarrow(gb).table


def _detection_key_match_totals_from_backend(
    *,
    backend: SearchBackend,
    subset: SubsetPaths,
    triples_parquet_path: Path,
) -> tuple[int | None, int | None, int | None]:
    caps = getattr(backend, "capabilities", None)
    if caps is not None and not bool(
        getattr(caps, "supports_detection_key_match_totals_from_triples_parquet", False)
    ):
        return None, None, None
    fn = getattr(backend, "detection_key_match_totals_from_triples_parquet", None)
    if fn is None or not callable(fn):
        return None, None, None
    try:
        out = fn(subset=subset, triples_parquet=str(triples_parquet_path))
    except Exception:
        return None, None, None
    if not isinstance(out, tuple) or len(out) != 3:
        return None, None, None
    n_exp_raw, n_frame_raw, n_nonmatch_raw = out
    n_exp = None if n_exp_raw is None else int(n_exp_raw)
    n_frame = None if n_frame_raw is None else int(n_frame_raw)
    n_nonmatch = None if n_nonmatch_raw is None else int(n_nonmatch_raw)
    return n_exp, n_frame, n_nonmatch


def run_stage1_to_stage4_rows_disk_summary(
    *,
    backend: SearchBackend,
    subset: SubsetPaths,
    orbits: Orbits,
    start_mjd_utc: float,
    end_mjd_utc: float,
    obscodes: tuple[str, ...],
    window_size_days: int,
    stage2_strategy: str,
    healpix_nside: int,
    footprint,
    max_processes: int | None,
    limit_codefid_keys: pa.Array,
    limit_codefid_vals: pa.Array,
    faint_margin_mag: float,
    gate: GateParams,
    targets: BenchTargets,
    dataset_pixels_by_target: dict[tuple[str, int], np.ndarray] | None = None,
    truth_frames: pa.Table | None = None,
    truth_ids_key: pa.Array | None = None,
    run_dir: Path,
    reuse_stage23_artifacts: bool = True,
    write_stage23_artifacts_flag: bool = True,
    write_stage4_artifacts_flag: bool = True,
    detailed_timings: bool = False,
    chunk_rows_stage23: int = 0,
    chunk_rows_stage4: int = 0,
    max_inflight_chunks: int = 2,
    runtime_tmp_dir: Path | None = None,
    min_free_disk_gb: float = 10.0,
    max_on_sky_sigma_major_arcsec: float | None = None,
    preprop_viability_policy: str = "off",
    preprop_short_arc_days_threshold: float = 14.0,
    preprop_time_limit_days_short_arc: float = 30.0,
    preprop_time_limit_days_default: float = 90.0,
    preprop_max_sigma_r_over_r: float | None = None,
    preprop_max_covariance_condition: float | None = None,
    preprop_viability_min_score: float = 0.25,
    preprop_fail_open_on_scoring_error: bool = False,
    compute_detection_key_match_totals: bool = False,
) -> DiskRowsRunSummary:
    _ = int(max_inflight_chunks)  # Reserved for future async writer scheduling.
    run_dir = Path(run_dir)
    stage23_dir = run_dir / "stage23"
    stage4_dir = run_dir / "stage4" / str(backend.name)
    stage23_meta_path = stage23_dir / "meta.json"
    stage23_manifest_path = stage23_dir / "manifest.json"
    stage23_targets_path = stage23_dir / "targets.parquet"
    stage23_preds_path = stage23_dir / "preds.parquet"
    stage23_triples_path = stage23_dir / "triples.parquet"
    stage23_metrics_path = stage23_dir / "stage3_orbit_metrics.parquet"
    preds_dataset_dir = stage23_dir / "preds_dataset"
    triples_dataset_dir = stage23_dir / "triples_dataset"

    stage4_meta_path = stage4_dir / "meta.json"
    stage4_accepted_path = stage4_dir / "accepted_detections.parquet"
    stage4_counts_path = stage4_dir / "accepted_counts.parquet"
    stage4_metrics_path = stage4_dir / "stage4_orbit_metrics.parquet"
    stage4_accepted_ds_dir = stage4_dir / "accepted_detections_dataset"

    run_dir.mkdir(parents=True, exist_ok=True)
    _ensure_min_free_disk(path=run_dir, min_free_disk_gb=float(min_free_disk_gb))

    old_tmpdir = os.environ.get("TMPDIR")
    old_ray_tmpdir = os.environ.get("RAY_TMPDIR")
    if runtime_tmp_dir is not None:
        runtime_tmp = Path(runtime_tmp_dir)
        runtime_tmp.mkdir(parents=True, exist_ok=True)
        os.environ["TMPDIR"] = str(runtime_tmp)
        os.environ["RAY_TMPDIR"] = str(runtime_tmp / "ray")

    try:
        stage23_ready = bool(stage23_artifacts_exist(run_dir=run_dir))
        manifest: dict[str, object] = {}
        chunks: list[dict[str, object]] = []
        build_micro_timings: dict[str, float] = {}
        build_observers_elapsed_s = 0.0
        build_predict_elapsed_s = 0.0
        build_pred_mag_elapsed_s = 0.0
        build_footprint_elapsed_s = 0.0
        build_triples_elapsed_s = 0.0
        n_frames_skipped_limiting_mag = 0
        n_frames_skipped_uncertainty = 0
        n_targets_preprop_viability_rejected = 0
        n_targets_preprop_time_limited = 0
        n_targets_failfast_dynamics_error = 0
        n_unique_frames_selected = 0

        if (
            bool(reuse_stage23_artifacts)
            and stage23_ready
            and stage23_manifest_path.exists()
        ):
            manifest = json.loads(stage23_manifest_path.read_text())
            raw_chunks = manifest.get("chunks", [])
            if isinstance(raw_chunks, list):
                chunks = [dict(x) for x in raw_chunks if isinstance(x, dict)]
            n_unique_frames_selected = int(manifest.get("n_unique_frames_selected", 0) or 0)
            if stage23_meta_path.exists():
                try:
                    m = json.loads(stage23_meta_path.read_text())
                    mt = m.get("micro_timings", {})
                    if isinstance(mt, dict):
                        build_micro_timings = {str(k): float(v) for k, v in mt.items()}
                    build_observers_elapsed_s = float(m.get("build_observers_elapsed_s", 0.0) or 0.0)
                    build_predict_elapsed_s = float(m.get("build_predict_elapsed_s", 0.0) or 0.0)
                    build_pred_mag_elapsed_s = float(m.get("build_pred_mag_elapsed_s", 0.0) or 0.0)
                    build_footprint_elapsed_s = float(m.get("build_footprint_elapsed_s", 0.0) or 0.0)
                    build_triples_elapsed_s = float(m.get("build_triples_elapsed_s", 0.0) or 0.0)
                    n_frames_skipped_limiting_mag = int(m.get("n_frames_skipped_limiting_mag", 0) or 0)
                    n_frames_skipped_uncertainty = int(m.get("n_frames_skipped_uncertainty", 0) or 0)
                    n_targets_preprop_viability_rejected = int(
                        m.get("n_targets_preprop_viability_rejected", 0) or 0
                    )
                    n_targets_preprop_time_limited = int(
                        m.get("n_targets_preprop_time_limited", 0) or 0
                    )
                    n_targets_failfast_dynamics_error = int(
                        m.get("n_targets_failfast_dynamics_error", 0) or 0
                    )
                except Exception:
                    pass
        elif bool(reuse_stage23_artifacts) and stage23_ready and not stage23_manifest_path.exists():
            # Backward-compatible reuse: single-file artifacts from memory mode.
            chunks = [
                {
                    "chunk_index": 0,
                    "preds_path": str(stage23_preds_path),
                    "triples_path": str(stage23_triples_path),
                }
            ]
            n_unique_frames_selected = _unique_frames_count_from_triples_file(stage23_triples_path)
            if stage23_meta_path.exists():
                try:
                    m = json.loads(stage23_meta_path.read_text())
                    mt = m.get("micro_timings", {})
                    if isinstance(mt, dict):
                        build_micro_timings = {str(k): float(v) for k, v in mt.items()}
                    build_observers_elapsed_s = float(m.get("build_observers_elapsed_s", 0.0) or 0.0)
                    build_predict_elapsed_s = float(m.get("build_predict_elapsed_s", 0.0) or 0.0)
                    build_pred_mag_elapsed_s = float(m.get("build_pred_mag_elapsed_s", 0.0) or 0.0)
                    build_footprint_elapsed_s = float(m.get("build_footprint_elapsed_s", 0.0) or 0.0)
                    build_triples_elapsed_s = float(m.get("build_triples_elapsed_s", 0.0) or 0.0)
                    n_frames_skipped_limiting_mag = int(m.get("n_frames_skipped_limiting_mag", 0) or 0)
                    n_frames_skipped_uncertainty = int(m.get("n_frames_skipped_uncertainty", 0) or 0)
                    n_targets_preprop_viability_rejected = int(
                        m.get("n_targets_preprop_viability_rejected", 0) or 0
                    )
                    n_targets_preprop_time_limited = int(
                        m.get("n_targets_preprop_time_limited", 0) or 0
                    )
                    n_targets_failfast_dynamics_error = int(
                        m.get("n_targets_failfast_dynamics_error", 0) or 0
                    )
                except Exception:
                    pass
        else:
            if stage23_dir.exists():
                shutil.rmtree(stage23_dir)
            stage23_dir.mkdir(parents=True, exist_ok=True)
            preds_dataset_dir.mkdir(parents=True, exist_ok=True)
            triples_dataset_dir.mkdir(parents=True, exist_ok=True)
            targets.to_parquet(str(stage23_targets_path))

            n_targets = max(1, int(len(targets)))
            total_rows_hint = int(max(0, int(len(orbits)) * int(n_targets)))
            resolved_stage23_rows, stage23_chunk_cfg = _resolve_disk_chunk_rows(
                requested_chunk_rows=int(chunk_rows_stage23),
                total_rows_hint=int(total_rows_hint),
                effective_bytes_per_row=float(STAGE23_EFFECTIVE_BYTES_PER_ROW),
                ram_budget_frac=0.10,
            )
            resolved_stage4_rows, stage4_chunk_cfg = _resolve_disk_chunk_rows(
                requested_chunk_rows=int(chunk_rows_stage4),
                total_rows_hint=int(total_rows_hint),
                effective_bytes_per_row=float(STAGE4_EFFECTIVE_BYTES_PER_TRIPLE_ROW),
                ram_budget_frac=0.10,
            )
            row_budget = int(max(1, int(resolved_stage23_rows)))
            orbit_chunk_size = int(max(1, row_budget // n_targets))
            unique_frames: set[tuple[str, int, int]] = set()

            preds_writer: pq.ParquetWriter | None = None
            triples_writer: pq.ParquetWriter | None = None
            metrics_writer: pq.ParquetWriter | None = None
            try:
                chunk_index = 0
                for o0 in range(0, int(len(orbits)), int(orbit_chunk_size)):
                    o1 = int(min(int(len(orbits)), int(o0) + int(orbit_chunk_size)))
                    orbits_chunk = orbits.take(list(range(o0, o1)))
                    build = build_stage23(
                        orbits=orbits_chunk,
                        targets=targets,
                        start_mjd_utc=float(start_mjd_utc),
                        window_size_days=int(window_size_days),
                        stage2_strategy=str(stage2_strategy),
                        healpix_nside=int(healpix_nside),
                        footprint=footprint,
                        max_processes=max_processes,
                        limit_codefid_keys=limit_codefid_keys,
                        limit_codefid_vals=limit_codefid_vals,
                        faint_margin_mag=float(faint_margin_mag),
                        max_on_sky_sigma_major_arcsec=(
                            None
                            if max_on_sky_sigma_major_arcsec is None
                            else float(max_on_sky_sigma_major_arcsec)
                        ),
                        preprop_viability_policy=str(preprop_viability_policy),
                        preprop_short_arc_days_threshold=float(
                            preprop_short_arc_days_threshold
                        ),
                        preprop_time_limit_days_short_arc=float(
                            preprop_time_limit_days_short_arc
                        ),
                        preprop_time_limit_days_default=float(
                            preprop_time_limit_days_default
                        ),
                        preprop_max_sigma_r_over_r=(
                            None
                            if preprop_max_sigma_r_over_r is None
                            else float(preprop_max_sigma_r_over_r)
                        ),
                        preprop_max_covariance_condition=(
                            None
                            if preprop_max_covariance_condition is None
                            else float(preprop_max_covariance_condition)
                        ),
                        preprop_viability_min_score=float(preprop_viability_min_score),
                        preprop_fail_open_on_scoring_error=bool(
                            preprop_fail_open_on_scoring_error
                        ),
                        max_mag_residual_fainter_mag=gate.max_mag_residual_fainter_mag,
                        max_mag_residual_brighter_mag=gate.max_mag_residual_brighter_mag,
                        dataset_pixels_by_target=dataset_pixels_by_target,
                        truth_frames=truth_frames,
                        detailed_timings=bool(detailed_timings),
                    )

                    preds_tbl = build.preds.table
                    triples_tbl = build.triples.table
                    metrics_tbl = build.stage3_orbit_metrics.table

                    preds_chunk_path = preds_dataset_dir / f"part-{chunk_index:06d}.parquet"
                    triples_chunk_path = triples_dataset_dir / f"part-{chunk_index:06d}.parquet"
                    pq.write_table(preds_tbl, preds_chunk_path)
                    pq.write_table(triples_tbl, triples_chunk_path)

                    if preds_writer is None:
                        preds_writer = pq.ParquetWriter(str(stage23_preds_path), preds_tbl.schema)
                    if triples_writer is None:
                        triples_writer = pq.ParquetWriter(str(stage23_triples_path), triples_tbl.schema)
                    if metrics_writer is None:
                        metrics_writer = pq.ParquetWriter(str(stage23_metrics_path), metrics_tbl.schema)
                    preds_writer.write_table(preds_tbl)
                    triples_writer.write_table(triples_tbl)
                    metrics_writer.write_table(metrics_tbl)

                    if triples_tbl.num_rows > 0:
                        tuniq = triples_tbl.select(["obscode", "exposure_mjd_mid_key_us", "healpixel"]).group_by(
                            ["obscode", "exposure_mjd_mid_key_us", "healpixel"]
                        ).aggregate([])
                        u_obscode = pc.cast(tuniq["obscode"], pa.large_string()).to_pylist()
                        u_key = pc.cast(tuniq["exposure_mjd_mid_key_us"], pa.int64()).to_pylist()
                        u_pix = pc.cast(tuniq["healpixel"], pa.int64()).to_pylist()
                        for oc, key, pix in zip(u_obscode, u_key, u_pix):
                            if oc is None or key is None or pix is None:
                                continue
                            unique_frames.add((str(oc), int(key), int(pix)))

                    chunks.append(
                        {
                            "chunk_index": int(chunk_index),
                            "orbit_start": int(o0),
                            "orbit_stop": int(o1),
                            "preds_path": str(preds_chunk_path),
                            "triples_path": str(triples_chunk_path),
                            "n_preds": int(preds_tbl.num_rows),
                            "n_triples": int(triples_tbl.num_rows),
                            "n_stage3_rows": int(metrics_tbl.num_rows),
                        }
                    )
                    _sum_timings(build_micro_timings, build.micro_timings)
                    build_observers_elapsed_s += float(build.build_observers_elapsed_s)
                    build_predict_elapsed_s += float(build.build_predict_elapsed_s)
                    build_pred_mag_elapsed_s += float(build.build_pred_mag_elapsed_s)
                    build_footprint_elapsed_s += float(build.build_footprint_elapsed_s)
                    build_triples_elapsed_s += float(build.build_triples_elapsed_s)
                    n_frames_skipped_limiting_mag += int(build.n_frames_skipped_limiting_mag)
                    n_frames_skipped_uncertainty += int(build.n_frames_skipped_uncertainty)
                    n_targets_preprop_viability_rejected += int(
                        build.n_targets_preprop_viability_rejected
                    )
                    n_targets_preprop_time_limited += int(build.n_targets_preprop_time_limited)
                    n_targets_failfast_dynamics_error += int(
                        build.n_targets_failfast_dynamics_error
                    )
                    chunk_index += 1
            finally:
                if preds_writer is not None:
                    preds_writer.close()
                if triples_writer is not None:
                    triples_writer.close()
                if metrics_writer is not None:
                    metrics_writer.close()

            _ensure_empty_parquet(path=stage23_preds_path, schema=PredictedTargets.empty().table.schema)
            _ensure_empty_parquet(path=stage23_triples_path, schema=PredictedTriples.empty().table.schema)
            _ensure_empty_parquet(path=stage23_metrics_path, schema=Stage3OrbitMetrics.empty().table.schema)

            n_unique_frames_selected = int(len(unique_frames))
            stage23_meta = {
                "schema_version": 1,
                "kind": "stage23_artifacts",
                "n_targets": int(len(targets)),
                "n_preds": int(pq.read_metadata(stage23_preds_path).num_rows),
                "n_triples": int(pq.read_metadata(stage23_triples_path).num_rows),
                "n_orbits_metrics": int(pq.read_metadata(stage23_metrics_path).num_rows),
                "micro_timings": {str(k): float(v) for k, v in build_micro_timings.items()},
                "build_observers_elapsed_s": float(build_observers_elapsed_s),
                "build_predict_elapsed_s": float(build_predict_elapsed_s),
                "build_pred_mag_elapsed_s": float(build_pred_mag_elapsed_s),
                "build_footprint_elapsed_s": float(build_footprint_elapsed_s),
                "build_triples_elapsed_s": float(build_triples_elapsed_s),
                "n_frames_skipped_limiting_mag": int(n_frames_skipped_limiting_mag),
                "n_frames_skipped_uncertainty": int(n_frames_skipped_uncertainty),
                "n_targets_preprop_viability_rejected": int(
                    n_targets_preprop_viability_rejected
                ),
                "n_targets_preprop_time_limited": int(n_targets_preprop_time_limited),
                "n_targets_failfast_dynamics_error": int(
                    n_targets_failfast_dynamics_error
                ),
                "start_mjd_utc": float(start_mjd_utc),
                "end_mjd_utc": float(end_mjd_utc),
                "window_size_days": int(window_size_days),
                "stage2_strategy": str(stage2_strategy),
                "healpix_nside": int(healpix_nside),
                "max_on_sky_sigma_major_arcsec": (
                    None
                    if max_on_sky_sigma_major_arcsec is None
                    else float(max_on_sky_sigma_major_arcsec)
                ),
                "preprop_viability_policy": str(preprop_viability_policy),
                "preprop_short_arc_days_threshold": float(
                    preprop_short_arc_days_threshold
                ),
                "preprop_time_limit_days_short_arc": float(
                    preprop_time_limit_days_short_arc
                ),
                "preprop_time_limit_days_default": float(
                    preprop_time_limit_days_default
                ),
                "preprop_max_sigma_r_over_r": (
                    None
                    if preprop_max_sigma_r_over_r is None
                    else float(preprop_max_sigma_r_over_r)
                ),
                "preprop_max_covariance_condition": (
                    None
                    if preprop_max_covariance_condition is None
                    else float(preprop_max_covariance_condition)
                ),
                "preprop_viability_min_score": float(preprop_viability_min_score),
                "preprop_fail_open_on_scoring_error": bool(
                    preprop_fail_open_on_scoring_error
                ),
                "chunking": {
                    "chunk_rows_stage23_input": int(chunk_rows_stage23),
                    "chunk_rows_stage4_input": int(chunk_rows_stage4),
                    "chunk_rows_stage23_resolved": int(resolved_stage23_rows),
                    "chunk_rows_stage4_resolved": int(resolved_stage4_rows),
                    "row_budget_resolved": int(row_budget),
                    "orbit_chunk_size": int(orbit_chunk_size),
                    "n_targets": int(n_targets),
                    "total_rows_hint": int(total_rows_hint),
                    "stage23": dict(stage23_chunk_cfg),
                    "stage4": dict(stage4_chunk_cfg),
                },
            }
            if bool(write_stage23_artifacts_flag):
                stage23_meta_path.write_text(json.dumps(stage23_meta, indent=2, sort_keys=True) + "\n")

            manifest = {
                "schema_version": 1,
                "kind": "stage23_chunk_manifest",
                "execution_mode": "disk",
                "n_chunks": int(len(chunks)),
                "n_unique_frames_selected": int(n_unique_frames_selected),
                "chunks": chunks,
            }
            if bool(write_stage23_artifacts_flag):
                stage23_manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

        if not chunks:
            chunks = [
                {
                    "chunk_index": 0,
                    "preds_path": str(stage23_preds_path),
                    "triples_path": str(stage23_triples_path),
                }
            ]

        if n_unique_frames_selected <= 0:
            n_unique_frames_selected = _unique_frames_count_from_triples_file(stage23_triples_path)

        if stage4_dir.exists():
            shutil.rmtree(stage4_dir)
        stage4_dir.mkdir(parents=True, exist_ok=True)
        stage4_accepted_ds_dir.mkdir(parents=True, exist_ok=True)

        gate_micro_timings: dict[str, float] = {}
        gate_n_candidates = 0
        gate_n_accepted = 0
        gate_n_rej_innov = 0
        gate_n_rej_mag = 0
        backend_elapsed_s = 0.0
        n_accepted_detections = 0

        accepted_writer: pq.ParquetWriter | None = None
        counts_writer: pq.ParquetWriter | None = None
        stage4_metrics_writer: pq.ParquetWriter | None = None
        stage4_subchunk_rows_resolved = 0
        stage4_subchunk_cfg: dict[str, object] = {}
        try:
            for chunk in chunks:
                preds_path = Path(str(chunk["preds_path"]))
                triples_path = Path(str(chunk["triples_path"]))
                preds_chunk = PredictedTargets.from_parquet(str(preds_path))
                triples_chunk = PredictedTriples.from_parquet(str(triples_path))
                n_triples_chunk = int(len(triples_chunk))
                if n_triples_chunk <= 0:
                    continue

                stage4_subchunk_rows, stage4_subchunk_cfg_local = _resolve_disk_chunk_rows(
                    requested_chunk_rows=int(chunk_rows_stage4),
                    total_rows_hint=int(n_triples_chunk),
                    effective_bytes_per_row=float(STAGE4_EFFECTIVE_BYTES_PER_TRIPLE_ROW),
                    ram_budget_frac=0.10,
                )
                stage4_subchunk_rows = int(max(1, min(int(stage4_subchunk_rows), int(n_triples_chunk))))
                stage4_subchunk_rows_resolved = int(stage4_subchunk_rows)
                stage4_subchunk_cfg = dict(stage4_subchunk_cfg_local)

                counts_parts: list[pa.Table] = []
                stage4_metric_parts: list[pa.Table] = []
                subchunk_index = 0
                for r0 in range(0, int(n_triples_chunk), int(stage4_subchunk_rows)):
                    _ensure_min_free_disk(path=run_dir, min_free_disk_gb=float(min_free_disk_gb))
                    r1 = int(min(int(n_triples_chunk), int(r0) + int(stage4_subchunk_rows)))
                    triples_sub_tbl = triples_chunk.table.slice(int(r0), int(r1 - r0))
                    if triples_sub_tbl.num_rows == 0:
                        continue
                    preds_sub_tbl = _preds_subset_for_triples(
                        preds_table=preds_chunk.table,
                        triples_table=triples_sub_tbl,
                    )
                    if preds_sub_tbl.num_rows == 0:
                        continue

                    triples_sub = PredictedTriples.from_pyarrow(triples_sub_tbl)
                    preds_sub = PredictedTargets.from_pyarrow(preds_sub_tbl)
                    orbit_ids_sub = (
                        preds_sub.table.select(["orbit_id"]).group_by(["orbit_id"]).aggregate([])["orbit_id"]
                        if len(preds_sub) > 0
                        else pa.array([], type=pa.large_string())
                    )
                    t_local: dict[str, float] | None = {} if bool(detailed_timings) else None
                    gate_res = stage4_fetch_and_gate_rows_python(
                        backend=backend,
                        subset=subset,
                        triples=triples_sub,
                        preds=preds_sub,
                        gate=gate,
                        orbit_ids=pc.cast(orbit_ids_sub, pa.large_string()),
                        truth_ids_key=truth_ids_key,
                        timings=t_local,
                    )

                    accepted_tbl = gate_res.accepted_detections.table
                    counts_parts.append(gate_res.accepted_counts.table)
                    stage4_metric_parts.append(gate_res.orbit_metrics.table)

                    accepted_chunk_path = stage4_accepted_ds_dir / (
                        f"part-{int(chunk.get('chunk_index', 0)):06d}-{int(subchunk_index):06d}.parquet"
                    )
                    pq.write_table(accepted_tbl, accepted_chunk_path)
                    if accepted_writer is None:
                        accepted_writer = pq.ParquetWriter(str(stage4_accepted_path), accepted_tbl.schema)
                    accepted_writer.write_table(accepted_tbl)
                    n_accepted_detections += int(accepted_tbl.num_rows)

                    gate_n_candidates += int(gate_res.totals.n_candidates)
                    gate_n_accepted += int(gate_res.totals.n_accepted)
                    gate_n_rej_innov += int(gate_res.totals.n_rejected_innov_ellipse)
                    gate_n_rej_mag += int(gate_res.totals.n_rejected_mag_residual)
                    backend_elapsed_s += float(gate_res.elapsed_s)
                    _sum_timings(gate_micro_timings, t_local)
                    subchunk_index += 1

                counts_tbl = _aggregate_accepted_counts_tables(counts_parts)
                stage4_tbl = _aggregate_stage4_orbit_metrics_tables(stage4_metric_parts)

                if counts_writer is None:
                    counts_writer = pq.ParquetWriter(str(stage4_counts_path), counts_tbl.schema)
                if stage4_metrics_writer is None:
                    stage4_metrics_writer = pq.ParquetWriter(str(stage4_metrics_path), stage4_tbl.schema)
                counts_writer.write_table(counts_tbl)
                stage4_metrics_writer.write_table(stage4_tbl)
        finally:
            if accepted_writer is not None:
                accepted_writer.close()
            if counts_writer is not None:
                counts_writer.close()
            if stage4_metrics_writer is not None:
                stage4_metrics_writer.close()

        _ensure_empty_parquet(path=stage4_accepted_path, schema=AcceptedDetections.empty().table.schema)
        _ensure_empty_parquet(path=stage4_counts_path, schema=AcceptedCounts.empty().table.schema)
        _ensure_empty_parquet(path=stage4_metrics_path, schema=Stage4OrbitMetrics.empty().table.schema)
        n_det_exp_keys: int | None = None
        n_det_frame_keys: int | None = None
        n_det_healpixel_nonmatch: int | None = None
        if bool(compute_detection_key_match_totals):
            n_det_exp_keys, n_det_frame_keys, n_det_healpixel_nonmatch = _detection_key_match_totals_from_backend(
                backend=backend,
                subset=subset,
                triples_parquet_path=stage23_triples_path,
            )

        if bool(write_stage4_artifacts_flag):
            stage4_meta = {
                "schema_version": 1,
                "kind": "stage4_artifacts",
                "backend": str(backend.name),
                "n_accepted_detections": int(n_accepted_detections),
                "n_counts_rows": int(pq.read_metadata(stage4_counts_path).num_rows),
                "n_orbit_metrics_rows": int(pq.read_metadata(stage4_metrics_path).num_rows),
                "gate_totals": {
                    "n_candidates": int(gate_n_candidates),
                    "n_accepted": int(gate_n_accepted),
                    "n_rejected_innov_ellipse": int(gate_n_rej_innov),
                    "n_rejected_mag_residual": int(gate_n_rej_mag),
                },
                "n_detections_selected_exposure_keys": (
                    None if n_det_exp_keys is None else int(n_det_exp_keys)
                ),
                "n_detections_selected_frame_keys": (
                    None if n_det_frame_keys is None else int(n_det_frame_keys)
                ),
                "n_detections_healpixel_nonmatch": (
                    None if n_det_healpixel_nonmatch is None else int(n_det_healpixel_nonmatch)
                ),
                "micro_timings": {str(k): float(v) for k, v in gate_micro_timings.items()},
                "start_mjd_utc": float(start_mjd_utc),
                "end_mjd_utc": float(end_mjd_utc),
                "window_size_days": int(window_size_days),
                "stage2_strategy": str(stage2_strategy),
                "healpix_nside": int(healpix_nside),
                "stage4_subchunk_rows_resolved": int(stage4_subchunk_rows_resolved),
                "stage4_subchunk_cfg": dict(stage4_subchunk_cfg),
            }
            stage4_meta_path.write_text(json.dumps(stage4_meta, indent=2, sort_keys=True) + "\n")

        return DiskRowsRunSummary(
            stage23_run_dir=run_dir,
            stage4_run_dir=stage4_dir,
            stage3_orbit_metrics=Stage3OrbitMetrics.from_parquet(str(stage23_metrics_path)),
            stage4_orbit_metrics=Stage4OrbitMetrics.from_parquet(str(stage4_metrics_path)),
            accepted_counts=AcceptedCounts.from_parquet(str(stage4_counts_path)),
            gate_totals=GateTotals(
                int(gate_n_candidates),
                int(gate_n_accepted),
                int(gate_n_rej_innov),
                int(gate_n_rej_mag),
            ),
            build_micro_timings=dict(build_micro_timings),
            gate_micro_timings=dict(gate_micro_timings),
            build_observers_elapsed_s=float(build_observers_elapsed_s),
            build_predict_elapsed_s=float(build_predict_elapsed_s),
            build_pred_mag_elapsed_s=float(build_pred_mag_elapsed_s),
            build_footprint_elapsed_s=float(build_footprint_elapsed_s),
            build_triples_elapsed_s=float(build_triples_elapsed_s),
            n_frames_skipped_limiting_mag=int(n_frames_skipped_limiting_mag),
            n_frames_skipped_uncertainty=int(n_frames_skipped_uncertainty),
            n_targets_preprop_viability_rejected=int(
                n_targets_preprop_viability_rejected
            ),
            n_targets_preprop_time_limited=int(n_targets_preprop_time_limited),
            n_targets_failfast_dynamics_error=int(n_targets_failfast_dynamics_error),
            backend_elapsed_s=float(backend_elapsed_s),
            n_unique_frames_selected=int(n_unique_frames_selected),
            n_detections_selected_exposure_keys=(
                None if n_det_exp_keys is None else int(n_det_exp_keys)
            ),
            n_detections_selected_frame_keys=(
                None if n_det_frame_keys is None else int(n_det_frame_keys)
            ),
            n_detections_healpixel_nonmatch=(
                None if n_det_healpixel_nonmatch is None else int(n_det_healpixel_nonmatch)
            ),
        )
    finally:
        if runtime_tmp_dir is not None:
            if old_tmpdir is None:
                os.environ.pop("TMPDIR", None)
            else:
                os.environ["TMPDIR"] = old_tmpdir
            if old_ray_tmpdir is None:
                os.environ.pop("RAY_TMPDIR", None)
            else:
                os.environ["RAY_TMPDIR"] = old_ray_tmpdir


def build_stage23(
    *,
    orbits: Orbits,
    targets: BenchTargets,
    start_mjd_utc: float,
    window_size_days: int,
    stage2_strategy: str,
    healpix_nside: int,
    footprint,
    max_processes: int | None,
    limit_codefid_keys: pa.Array,
    limit_codefid_vals: pa.Array,
    faint_margin_mag: float,
    max_on_sky_sigma_major_arcsec: float | None,
    preprop_viability_policy: str = "off",
    preprop_short_arc_days_threshold: float = 14.0,
    preprop_time_limit_days_short_arc: float = 30.0,
    preprop_time_limit_days_default: float = 90.0,
    preprop_max_sigma_r_over_r: float | None = None,
    preprop_max_covariance_condition: float | None = None,
    preprop_viability_min_score: float = 0.25,
    preprop_fail_open_on_scoring_error: bool = False,
    max_mag_residual_fainter_mag: float | None,
    max_mag_residual_brighter_mag: float | None,
    dataset_pixels_by_target: dict[tuple[str, int], np.ndarray] | None = None,
    truth_frames: pa.Table | None = None,
    detailed_timings: bool,
) -> BuildStage23:
    (
        preds,
        triples,
        frame_metrics,
        build_observers_s,
        build_predict_s,
        build_pred_mag_s,
        build_footprint_s,
        build_triples_s,
        n_pairs_skipped_faint,
        n_pairs_skipped_uncertainty,
        micro_timings,
    ) = build_predictions_and_triples(
        orbits=orbits,
        targets=targets,
        start_mjd_utc=float(start_mjd_utc),
        window_size_days=int(window_size_days),
        stage2_strategy=str(stage2_strategy),
        healpix_nside=int(healpix_nside),
        footprint=footprint,
        max_processes=max_processes,
        limit_codefid_keys=limit_codefid_keys,
        limit_codefid_vals=limit_codefid_vals,
        faint_margin_mag=float(faint_margin_mag),
        max_on_sky_sigma_major_arcsec=(
            None
            if max_on_sky_sigma_major_arcsec is None
            else float(max_on_sky_sigma_major_arcsec)
        ),
        preprop_viability_policy=str(preprop_viability_policy),
        preprop_short_arc_days_threshold=float(preprop_short_arc_days_threshold),
        preprop_time_limit_days_short_arc=float(preprop_time_limit_days_short_arc),
        preprop_time_limit_days_default=float(preprop_time_limit_days_default),
        preprop_max_sigma_r_over_r=(
            None
            if preprop_max_sigma_r_over_r is None
            else float(preprop_max_sigma_r_over_r)
        ),
        preprop_max_covariance_condition=(
            None
            if preprop_max_covariance_condition is None
            else float(preprop_max_covariance_condition)
        ),
        preprop_viability_min_score=float(preprop_viability_min_score),
        preprop_fail_open_on_scoring_error=bool(preprop_fail_open_on_scoring_error),
        max_mag_residual_fainter_mag=max_mag_residual_fainter_mag,
        max_mag_residual_brighter_mag=max_mag_residual_brighter_mag,
        dataset_pixels_by_target=dataset_pixels_by_target,
        truth_frames=truth_frames,
        detailed_timings=bool(detailed_timings),
    )

    return BuildStage23(
        preds=preds,
        triples=triples,
        stage3_orbit_metrics=frame_metrics,
        build_observers_elapsed_s=float(build_observers_s),
        build_predict_elapsed_s=float(build_predict_s),
        build_pred_mag_elapsed_s=float(build_pred_mag_s),
        build_footprint_elapsed_s=float(build_footprint_s),
        build_triples_elapsed_s=float(build_triples_s),
        n_frames_skipped_limiting_mag=int(n_pairs_skipped_faint),
        n_frames_skipped_uncertainty=int(n_pairs_skipped_uncertainty),
        n_targets_preprop_viability_rejected=int(
            pc.sum(
                pc.cast(
                    pc.fill_null(frame_metrics.n_targets_preprop_viability_rejected, 0), pa.int64()
                )
            ).as_py()
            if len(frame_metrics) > 0
            else 0
        ),
        n_targets_preprop_time_limited=int(
            pc.sum(
                pc.cast(
                    pc.fill_null(frame_metrics.n_targets_preprop_time_limited, 0), pa.int64()
                )
            ).as_py()
            if len(frame_metrics) > 0
            else 0
        ),
        n_targets_failfast_dynamics_error=int(
            pc.sum(
                pc.cast(
                    pc.fill_null(frame_metrics.n_targets_failfast_dynamics_error, 0), pa.int64()
                )
            ).as_py()
            if len(frame_metrics) > 0
            else 0
        ),
        micro_timings=dict(micro_timings),
    )


def stage4_fetch_and_gate_python(
    *,
    backend: SearchBackend,
    subset: SubsetPaths,
    triples: PredictedTriples,
    preds: PredictedTargets,
    gate: GateParams,
    orbit_ids: pa.Array | None = None,
    timings: dict[str, float] | None = None,
) -> BackendGateResult:
    if len(triples) == 0:
        return BackendGateResult(
            accepted_counts=AcceptedCounts.empty(),
            totals=GateTotals(0, 0, 0, 0),
            elapsed_s=0.0,
        )
    t0 = time.perf_counter()
    if bool(getattr(backend.capabilities, "supports_filter_triples_to_existing_frames", False)):
        triples = backend.filter_triples_to_existing_frames(subset=subset, triples=triples)
        if len(triples) == 0:
            return BackendGateResult(
                accepted_counts=AcceptedCounts.empty(),
                totals=GateTotals(0, 0, 0, 0),
                elapsed_s=float(time.perf_counter() - t0),
            )
    candidates = backend.fetch_candidates(subset=subset, triples=triples, limit=None)
    keep_innov, rej_innov, rej_mag, keep_final = gate_keep_and_reject_masks_python(
        candidates=candidates, preds=preds, gate=gate, timings=timings
    )
    totals = GateTotals(
        n_candidates=int(len(candidates)),
        n_accepted=int(np.count_nonzero(np.asarray(keep_final, dtype=bool))),
        n_rejected_innov_ellipse=int(np.count_nonzero(np.asarray(rej_innov, dtype=bool))),
        n_rejected_mag_residual=int(np.count_nonzero(np.asarray(rej_mag, dtype=bool))),
    )

    t = candidates.table.select(["orbit_id", "target_idx", "observation_id"])
    t = t.append_column("_keep", pa.array(np.asarray(keep_final, dtype=bool)))
    gb = t.group_by(["orbit_id", "target_idx"]).aggregate(
        [
            ("observation_id", "count"),
            ("_keep", "sum"),
        ]
    )
    gb = gb.rename_columns(["orbit_id", "target_idx", "n_candidates", "n_accepted"])
    return BackendGateResult(
        accepted_counts=AcceptedCounts.from_pyarrow(gb),
        totals=totals,
        elapsed_s=float(time.perf_counter() - t0),
    )


def stage4_fetch_and_gate_rows_python(
    *,
    backend: SearchBackend,
    subset: SubsetPaths,
    triples: PredictedTriples,
    preds: PredictedTargets,
    gate: GateParams,
    orbit_ids: pa.Array | None = None,
    truth_ids_key: pa.Array | None = None,
    timings: dict[str, float] | None = None,
) -> BackendGateRowsResult:
    if len(triples) == 0:
        return BackendGateRowsResult(
            accepted_detections=AcceptedDetections.empty(),
            accepted_counts=AcceptedCounts.empty(),
            totals=GateTotals(0, 0, 0, 0),
            orbit_metrics=Stage4OrbitMetrics.empty(),
            elapsed_s=0.0,
        )
    t0 = time.perf_counter()
    if bool(getattr(backend.capabilities, "supports_filter_triples_to_existing_frames", False)):
        triples = backend.filter_triples_to_existing_frames(subset=subset, triples=triples)
        if len(triples) == 0:
            return BackendGateRowsResult(
                accepted_detections=AcceptedDetections.empty(),
                accepted_counts=AcceptedCounts.empty(),
                totals=GateTotals(0, 0, 0, 0),
                orbit_metrics=Stage4OrbitMetrics.empty(),
                elapsed_s=float(time.perf_counter() - t0),
            )
    candidates = backend.fetch_candidates(subset=subset, triples=triples, limit=None)
    keep_innov, rej_innov, rej_mag, keep_final = gate_keep_and_reject_masks_python(
        candidates=candidates, preds=preds, gate=gate, timings=timings
    )
    keep_innov = np.asarray(keep_innov, dtype=bool)
    rej_innov = np.asarray(rej_innov, dtype=bool)
    rej_mag = np.asarray(rej_mag, dtype=bool)
    keep_final = np.asarray(keep_final, dtype=bool)

    totals = GateTotals(
        n_candidates=int(len(candidates)),
        n_accepted=int(np.count_nonzero(keep_final)),
        n_rejected_innov_ellipse=int(np.count_nonzero(rej_innov)),
        n_rejected_mag_residual=int(np.count_nonzero(rej_mag)),
    )

    import pyarrow.compute as pc

    # Per-(orbit,target) accepted counts.
    t_counts = candidates.table.select(["orbit_id", "target_idx", "observation_id"])
    t_counts = t_counts.append_column("_keep", pa.array(keep_final))
    gb = t_counts.group_by(["orbit_id", "target_idx"]).aggregate(
        [
            ("observation_id", "count"),
            ("_keep", "sum"),
        ]
    )
    gb = gb.rename_columns(["orbit_id", "target_idx", "n_candidates", "n_accepted"])
    accepted_counts = AcceptedCounts.from_pyarrow(gb)

    # Minimal per-candidate table for orbit-level aggregations and accepted row extraction.
    t = candidates.table.select(
        ["orbit_id", "target_idx", "obscode", "exposure_mjd_mid_key_us", "observation_id"]
    )
    t = t.append_column("_keep_innov", pa.array(keep_innov))
    t = t.append_column("_rej_mag", pa.array(rej_mag))
    t = t.append_column("_keep_final", pa.array(keep_final))

    # Truth membership (composite key = orbit_id|observation_id).
    if truth_ids_key is not None and len(truth_ids_key) > 0 and t.num_rows > 0:
        sep = pa.scalar("|", type=pa.large_string())
        cand_key = pc.binary_join_element_wise(
            pc.cast(t["orbit_id"], pa.large_string()),
            pc.cast(t["observation_id"], pa.large_string()),
            sep,
        )
        is_truth = pc.is_in(cand_key, value_set=truth_ids_key)
    else:
        is_truth = pa.array([False] * t.num_rows, type=pa.bool_())
    t = t.append_column("_is_truth", pc.cast(is_truth, pa.bool_()))

    # Orbit-level detection breakdown (always emit stable schema; truth metrics are 0 without truth).
    det_gb = t.group_by(["orbit_id"]).aggregate(
        [
            ("observation_id", "count"),
            ("_keep_innov", "sum"),
            ("_keep_innov", "count"),
            ("_rej_mag", "sum"),
        ]
    ).rename_columns(
        [
            "orbit_id",
            "n_detections_candidates",
            "n_detections_gate_matched",
            "_n_detections_innov_tested",
            "n_detections_magnitude_rejected",
        ]
    )
    det_gb = det_gb.append_column(
        "n_detections_innov_ellipse_rejected",
        pc.subtract(
            pc.cast(det_gb["_n_detections_innov_tested"], pa.int64()),
            pc.cast(det_gb["n_detections_gate_matched"], pa.int64()),
        ),
    )
    det_gb = det_gb.select(
        [
            "orbit_id",
            "n_detections_candidates",
            "n_detections_gate_matched",
            "n_detections_innov_ellipse_rejected",
            "n_detections_magnitude_rejected",
        ]
    )

    truth_gate = pc.and_(t["_is_truth"], t["_keep_innov"])
    truth_mag_rej = pc.and_(t["_is_truth"], t["_rej_mag"])
    truth_final = pc.and_(t["_is_truth"], t["_keep_final"])
    t2 = t.append_column("_truth_gate", truth_gate)
    t2 = t2.append_column("_truth_mag_rej", truth_mag_rej)
    t2 = t2.append_column("_truth_final", truth_final)
    truth_gb = t2.group_by(["orbit_id"]).aggregate(
        [
            ("_truth_gate", "sum"),
            ("_truth_mag_rej", "sum"),
            ("_truth_final", "sum"),
        ]
    ).rename_columns(
        [
            "orbit_id",
            "n_detections_truth_gate_matched",
            "n_detections_truth_magnitude_rejected",
            "n_detections_truth_final",
        ]
    )

    orbit_breakdown = det_gb.join(truth_gb, keys=["orbit_id"], join_type="left outer")
    for c in [
        "n_detections_candidates",
        "n_detections_gate_matched",
        "n_detections_innov_ellipse_rejected",
        "n_detections_magnitude_rejected",
        "n_detections_truth_gate_matched",
        "n_detections_truth_magnitude_rejected",
        "n_detections_truth_final",
    ]:
        if c in orbit_breakdown.column_names:
            orbit_breakdown = orbit_breakdown.set_column(
                orbit_breakdown.schema.get_field_index(c),
                c,
                pc.cast(pc.fill_null(orbit_breakdown[c], 0), pa.int64()),
            )
        else:
            orbit_breakdown = orbit_breakdown.append_column(c, pa.array([0] * orbit_breakdown.num_rows, type=pa.int64()))

    # Exposure-key final false-positive/unknown accounting (matches benchmark harness semantics).
    acc = t2.filter(pc.field("_keep_final")).select(
        ["orbit_id", "obscode", "exposure_mjd_mid_key_us", "observation_id", "_is_truth"]
    )
    if acc.num_rows > 0:
        exp = acc.group_by(["orbit_id", "obscode", "exposure_mjd_mid_key_us"]).aggregate(
            [
                ("observation_id", "count"),
                ("_is_truth", "sum"),
            ]
        ).rename_columns(
            [
                "orbit_id",
                "obscode",
                "exposure_mjd_mid_key_us",
                "_n_in_exposure",
                "_n_truth_in_exposure",
            ]
        )
        exp = exp.append_column(
            "_fp",
            pc.max_element_wise(
                pc.subtract(pc.cast(exp["_n_in_exposure"], pa.int64()), pa.scalar(1, type=pa.int64())),
                pa.scalar(0, type=pa.int64()),
            ),
        )
        fp_by_orbit = exp.group_by(["orbit_id"]).aggregate([("_fp", "sum")]).rename_columns(
            ["orbit_id", "n_detections_false_positive_final"]
        )

        exp_any = exp.group_by(["orbit_id"]).aggregate([("exposure_mjd_mid_key_us", "count")]).rename_columns(
            ["orbit_id", "_n_exposures_with_any"]
        )
        exp_truth = exp.filter(pc.greater(pc.cast(exp["_n_truth_in_exposure"], pa.int64()), pa.scalar(0, type=pa.int64())))
        if exp_truth.num_rows > 0:
            exp_truth = exp_truth.group_by(["orbit_id"]).aggregate([("exposure_mjd_mid_key_us", "count")]).rename_columns(
                ["orbit_id", "_n_exposures_with_truth"]
            )
        else:
            exp_truth = pa.table(
                {
                    "orbit_id": pa.array([], type=pa.large_string()),
                    "_n_exposures_with_truth": pa.array([], type=pa.int64()),
                }
            )
        unk = exp_any.join(exp_truth, keys=["orbit_id"], join_type="left outer")
        unk = unk.append_column(
            "n_detections_unknown_final",
            pc.subtract(
                pc.cast(unk["_n_exposures_with_any"], pa.int64()),
                pc.cast(pc.fill_null(unk["_n_exposures_with_truth"], 0), pa.int64()),
            ),
        ).select(["orbit_id", "n_detections_unknown_final"])
    else:
        fp_by_orbit = pa.table(
            {
                "orbit_id": pa.array([], type=pa.large_string()),
                "n_detections_false_positive_final": pa.array([], type=pa.int64()),
            }
        )
        unk = pa.table(
            {
                "orbit_id": pa.array([], type=pa.large_string()),
                "n_detections_unknown_final": pa.array([], type=pa.int64()),
            }
        )

    orbit_breakdown = orbit_breakdown.join(fp_by_orbit, keys=["orbit_id"], join_type="left outer")
    orbit_breakdown = orbit_breakdown.join(unk, keys=["orbit_id"], join_type="left outer")
    for c in ["n_detections_false_positive_final", "n_detections_unknown_final"]:
        if c in orbit_breakdown.column_names:
            orbit_breakdown = orbit_breakdown.set_column(
                orbit_breakdown.schema.get_field_index(c),
                c,
                pc.cast(pc.fill_null(orbit_breakdown[c], 0), pa.int64()),
            )
        else:
            orbit_breakdown = orbit_breakdown.append_column(c, pa.array([0] * orbit_breakdown.num_rows, type=pa.int64()))

    if orbit_ids is not None and len(orbit_ids) > 0:
        base = pa.table({"orbit_id": pc.cast(orbit_ids, pa.large_string())})
        orbit_breakdown = base.join(orbit_breakdown, keys=["orbit_id"], join_type="left outer")
        for c in orbit_breakdown.column_names:
            if c == "orbit_id":
                continue
            orbit_breakdown = orbit_breakdown.set_column(
                orbit_breakdown.schema.get_field_index(c),
                c,
                pc.cast(pc.fill_null(orbit_breakdown[c], 0), pa.int64()),
            )
    orbit_metrics = Stage4OrbitMetrics.from_pyarrow(
        orbit_breakdown.select(
            [
                "orbit_id",
                "n_detections_candidates",
                "n_detections_gate_matched",
                "n_detections_innov_ellipse_rejected",
                "n_detections_magnitude_rejected",
                "n_detections_truth_gate_matched",
                "n_detections_truth_magnitude_rejected",
                "n_detections_truth_final",
                "n_detections_false_positive_final",
                "n_detections_unknown_final",
            ]
        )
    )

    n_keep_final = int(np.count_nonzero(keep_final))
    if n_keep_final == 0:
        return BackendGateRowsResult(
            accepted_detections=AcceptedDetections.empty(),
            accepted_counts=accepted_counts,
            totals=totals,
            orbit_metrics=orbit_metrics,
            elapsed_s=float(time.perf_counter() - t0),
        )

    # Avoid large hash-join explosions when key multiplicities are high:
    # accepted detections are exactly the keep_final-filtered candidate rows.
    acc_full = candidates.table.filter(pa.array(keep_final, type=pa.bool_()))
    acc_full = acc_full.select(
        [
            "orbit_id",
            "target_idx",
            "obscode",
            "exposure_mjd_mid_key_us",
            "healpixel",
            "filter",
            "observation_id",
            "obstime_mjd_utc",
            "ra_deg",
            "dec_deg",
            "ra_sigma_deg",
            "dec_sigma_deg",
            "mag",
            "mag_sigma",
        ]
    )
    return BackendGateRowsResult(
        accepted_detections=AcceptedDetections.from_pyarrow(acc_full),
        accepted_counts=accepted_counts,
        totals=totals,
        orbit_metrics=orbit_metrics,
        elapsed_s=float(time.perf_counter() - t0),
    )


def run_stage1_to_stage4_python(
    *,
    backend: SearchBackend,
    subset: SubsetPaths,
    orbits: Orbits,
    start_mjd_utc: float,
    end_mjd_utc: float,
    obscodes: tuple[str, ...],
    window_size_days: int,
    stage2_strategy: str,
    healpix_nside: int,
    footprint,
    max_processes: int | None,
    limit_codefid_keys: pa.Array,
    limit_codefid_vals: pa.Array,
    faint_margin_mag: float,
    gate: GateParams,
    max_on_sky_sigma_major_arcsec: float | None = None,
    preprop_viability_policy: str = "off",
    preprop_short_arc_days_threshold: float = 14.0,
    preprop_time_limit_days_short_arc: float = 30.0,
    preprop_time_limit_days_default: float = 90.0,
    preprop_max_sigma_r_over_r: float | None = None,
    preprop_max_covariance_condition: float | None = None,
    preprop_viability_min_score: float = 0.25,
    preprop_fail_open_on_scoring_error: bool = False,
    detailed_timings: bool,
    timings: dict[str, float] | None = None,
    stage23_run_dir: Path | None = None,
    reuse_stage23_artifacts: bool = True,
    write_stage23_artifacts_flag: bool = True,
    targets: BenchTargets | None = None,
    dataset_pixels_by_target: dict[tuple[str, int], np.ndarray] | None = None,
    truth_frames: pa.Table | None = None,
    execution_mode: Literal["memory", "disk"] = "memory",
    chunk_rows_stage23: int = 0,
    chunk_rows_stage4: int = 0,
    max_inflight_chunks: int = 2,
    runtime_tmp_dir: Path | None = None,
    min_free_disk_gb: float = 10.0,
) -> Stage14RunPython:
    if str(execution_mode) == "disk":
        rows_out = run_stage1_to_stage4_rows_python(
            backend=backend,
            subset=subset,
            orbits=orbits,
            start_mjd_utc=float(start_mjd_utc),
            end_mjd_utc=float(end_mjd_utc),
            obscodes=tuple(map(str, obscodes)),
            window_size_days=int(window_size_days),
            stage2_strategy=str(stage2_strategy),
            healpix_nside=int(healpix_nside),
            footprint=footprint,
            max_processes=max_processes,
            limit_codefid_keys=limit_codefid_keys,
            limit_codefid_vals=limit_codefid_vals,
            faint_margin_mag=float(faint_margin_mag),
            max_on_sky_sigma_major_arcsec=(
                None
                if max_on_sky_sigma_major_arcsec is None
                else float(max_on_sky_sigma_major_arcsec)
            ),
            gate=gate,
            detailed_timings=bool(detailed_timings),
            timings=timings,
            stage23_run_dir=stage23_run_dir,
            reuse_stage23_artifacts=bool(reuse_stage23_artifacts),
            write_stage23_artifacts_flag=bool(write_stage23_artifacts_flag),
            targets=targets,
            dataset_pixels_by_target=dataset_pixels_by_target,
            truth_frames=truth_frames,
            truth_ids_key=None,
            execution_mode="disk",
            chunk_rows_stage23=int(chunk_rows_stage23),
            chunk_rows_stage4=int(chunk_rows_stage4),
            max_inflight_chunks=int(max_inflight_chunks),
            runtime_tmp_dir=runtime_tmp_dir,
            min_free_disk_gb=float(min_free_disk_gb),
            preprop_viability_policy=str(preprop_viability_policy),
            preprop_short_arc_days_threshold=float(preprop_short_arc_days_threshold),
            preprop_time_limit_days_short_arc=float(preprop_time_limit_days_short_arc),
            preprop_time_limit_days_default=float(preprop_time_limit_days_default),
            preprop_max_sigma_r_over_r=(
                None
                if preprop_max_sigma_r_over_r is None
                else float(preprop_max_sigma_r_over_r)
            ),
            preprop_max_covariance_condition=(
                None
                if preprop_max_covariance_condition is None
                else float(preprop_max_covariance_condition)
            ),
            preprop_viability_min_score=float(preprop_viability_min_score),
            preprop_fail_open_on_scoring_error=bool(preprop_fail_open_on_scoring_error),
            write_stage4_artifacts_flag=False,
        )
        gate_res = BackendGateResult(
            accepted_counts=rows_out.gate.accepted_counts,
            totals=rows_out.gate.totals,
            elapsed_s=float(rows_out.gate.elapsed_s),
        )
        return Stage14RunPython(targets=rows_out.targets, build=rows_out.build, gate=gate_res)

    if (
        stage23_run_dir is not None
        and bool(reuse_stage23_artifacts)
        and stage23_artifacts_exist(run_dir=stage23_run_dir)
    ):
        art = read_stage23_artifacts(run_dir=stage23_run_dir)
        targets = art.targets
        build = BuildStage23(
            preds=art.preds,
            triples=art.triples,
            stage3_orbit_metrics=art.stage3_orbit_metrics,
            build_observers_elapsed_s=float(art.build_observers_elapsed_s),
            build_predict_elapsed_s=float(art.build_predict_elapsed_s),
            build_pred_mag_elapsed_s=float(art.build_pred_mag_elapsed_s),
            build_footprint_elapsed_s=float(art.build_footprint_elapsed_s),
            build_triples_elapsed_s=float(art.build_triples_elapsed_s),
            n_frames_skipped_limiting_mag=int(art.n_frames_skipped_limiting_mag),
            n_frames_skipped_uncertainty=int(art.n_frames_skipped_uncertainty),
            n_targets_preprop_viability_rejected=int(
                art.n_targets_preprop_viability_rejected
            ),
            n_targets_preprop_time_limited=int(art.n_targets_preprop_time_limited),
            n_targets_failfast_dynamics_error=int(
                art.n_targets_failfast_dynamics_error
            ),
            micro_timings=dict(art.micro_timings),
        )
    else:
        if targets is None:
            targets = backend.enumerate_targets(
                subset=subset,
                start_mjd_utc=float(start_mjd_utc),
                end_mjd_utc=float(end_mjd_utc),
                obscodes=tuple(map(str, obscodes)),
            )
        if dataset_pixels_by_target is None and bool(
            getattr(getattr(backend, "capabilities", None), "supports_frame_pixels_by_target", False)
        ):
            try:
                dataset_pixels_by_target = backend.frame_pixels_by_target(subset=subset, targets=targets)
            except Exception:
                dataset_pixels_by_target = None
        build = build_stage23(
            orbits=orbits,
            targets=targets,
            start_mjd_utc=float(start_mjd_utc),
            window_size_days=int(window_size_days),
            stage2_strategy=str(stage2_strategy),
            healpix_nside=int(healpix_nside),
            footprint=footprint,
            max_processes=max_processes,
            limit_codefid_keys=limit_codefid_keys,
            limit_codefid_vals=limit_codefid_vals,
            faint_margin_mag=float(faint_margin_mag),
            max_on_sky_sigma_major_arcsec=(
                None
                if max_on_sky_sigma_major_arcsec is None
                else float(max_on_sky_sigma_major_arcsec)
            ),
            preprop_viability_policy=str(preprop_viability_policy),
            preprop_short_arc_days_threshold=float(preprop_short_arc_days_threshold),
            preprop_time_limit_days_short_arc=float(preprop_time_limit_days_short_arc),
            preprop_time_limit_days_default=float(preprop_time_limit_days_default),
            preprop_max_sigma_r_over_r=(
                None
                if preprop_max_sigma_r_over_r is None
                else float(preprop_max_sigma_r_over_r)
            ),
            preprop_max_covariance_condition=(
                None
                if preprop_max_covariance_condition is None
                else float(preprop_max_covariance_condition)
            ),
            preprop_viability_min_score=float(preprop_viability_min_score),
            preprop_fail_open_on_scoring_error=bool(preprop_fail_open_on_scoring_error),
            max_mag_residual_fainter_mag=gate.max_mag_residual_fainter_mag,
            max_mag_residual_brighter_mag=gate.max_mag_residual_brighter_mag,
            dataset_pixels_by_target=dataset_pixels_by_target,
            truth_frames=truth_frames,
            detailed_timings=bool(detailed_timings),
        )
        if stage23_run_dir is not None and bool(write_stage23_artifacts_flag):
            write_stage23_artifacts(
                run_dir=stage23_run_dir,
                targets=targets,
                preds=build.preds,
                triples=build.triples,
                stage3_orbit_metrics=build.stage3_orbit_metrics,
                micro_timings=dict(build.micro_timings),
                build_observers_elapsed_s=float(build.build_observers_elapsed_s),
                build_predict_elapsed_s=float(build.build_predict_elapsed_s),
                build_pred_mag_elapsed_s=float(build.build_pred_mag_elapsed_s),
                build_footprint_elapsed_s=float(build.build_footprint_elapsed_s),
                build_triples_elapsed_s=float(build.build_triples_elapsed_s),
                n_frames_skipped_limiting_mag=int(build.n_frames_skipped_limiting_mag),
                n_frames_skipped_uncertainty=int(build.n_frames_skipped_uncertainty),
                n_targets_preprop_viability_rejected=int(
                    build.n_targets_preprop_viability_rejected
                ),
                n_targets_preprop_time_limited=int(build.n_targets_preprop_time_limited),
                n_targets_failfast_dynamics_error=int(
                    build.n_targets_failfast_dynamics_error
                ),
                extra_meta={
                    "start_mjd_utc": float(start_mjd_utc),
                    "end_mjd_utc": float(end_mjd_utc),
                    "window_size_days": int(window_size_days),
                    "stage2_strategy": str(stage2_strategy),
                    "healpix_nside": int(healpix_nside),
                    "max_on_sky_sigma_major_arcsec": (
                        None
                        if max_on_sky_sigma_major_arcsec is None
                        else float(max_on_sky_sigma_major_arcsec)
                    ),
                },
            )
    gate_res = stage4_fetch_and_gate_python(
        backend=backend,
        subset=subset,
        triples=build.triples,
        preds=build.preds,
        gate=gate,
        orbit_ids=pa.array(orbits.orbit_id.to_pylist(), type=pa.large_string()),
        timings=timings,
    )
    return Stage14RunPython(targets=targets, build=build, gate=gate_res)


def run_stage1_to_stage4_rows_python(
    *,
    backend: SearchBackend,
    subset: SubsetPaths,
    orbits: Orbits,
    start_mjd_utc: float,
    end_mjd_utc: float,
    obscodes: tuple[str, ...],
    window_size_days: int,
    stage2_strategy: str,
    healpix_nside: int,
    footprint,
    max_processes: int | None,
    limit_codefid_keys: pa.Array,
    limit_codefid_vals: pa.Array,
    faint_margin_mag: float,
    gate: GateParams,
    max_on_sky_sigma_major_arcsec: float | None = None,
    preprop_viability_policy: str = "off",
    preprop_short_arc_days_threshold: float = 14.0,
    preprop_time_limit_days_short_arc: float = 30.0,
    preprop_time_limit_days_default: float = 90.0,
    preprop_max_sigma_r_over_r: float | None = None,
    preprop_max_covariance_condition: float | None = None,
    preprop_viability_min_score: float = 0.25,
    preprop_fail_open_on_scoring_error: bool = False,
    detailed_timings: bool,
    timings: dict[str, float] | None = None,
    stage23_run_dir: Path | None = None,
    reuse_stage23_artifacts: bool = True,
    write_stage23_artifacts_flag: bool = True,
    targets: BenchTargets | None = None,
    dataset_pixels_by_target: dict[tuple[str, int], np.ndarray] | None = None,
    truth_frames: pa.Table | None = None,
    truth_ids_key: pa.Array | None = None,
    execution_mode: Literal["memory", "disk"] = "memory",
    chunk_rows_stage23: int = 0,
    chunk_rows_stage4: int = 0,
    max_inflight_chunks: int = 2,
    runtime_tmp_dir: Path | None = None,
    min_free_disk_gb: float = 10.0,
    write_stage4_artifacts_flag: bool = True,
) -> Stage14RunRowsPython:
    if str(execution_mode) == "disk":
        if targets is None:
            targets = backend.enumerate_targets(
                subset=subset,
                start_mjd_utc=float(start_mjd_utc),
                end_mjd_utc=float(end_mjd_utc),
                obscodes=tuple(map(str, obscodes)),
            )
        if dataset_pixels_by_target is None and bool(
            getattr(getattr(backend, "capabilities", None), "supports_frame_pixels_by_target", False)
        ):
            try:
                dataset_pixels_by_target = backend.frame_pixels_by_target(subset=subset, targets=targets)
            except Exception:
                dataset_pixels_by_target = None
        run_dir = stage23_run_dir
        if run_dir is None:
            run_dir = subset.artifacts_dir / "disk_runs" / "default"
        summary = run_stage1_to_stage4_rows_disk_summary(
            backend=backend,
            subset=subset,
            orbits=orbits,
            start_mjd_utc=float(start_mjd_utc),
            end_mjd_utc=float(end_mjd_utc),
            obscodes=tuple(map(str, obscodes)),
            window_size_days=int(window_size_days),
            stage2_strategy=str(stage2_strategy),
            healpix_nside=int(healpix_nside),
            footprint=footprint,
            max_processes=max_processes,
            limit_codefid_keys=limit_codefid_keys,
            limit_codefid_vals=limit_codefid_vals,
            faint_margin_mag=float(faint_margin_mag),
            max_on_sky_sigma_major_arcsec=(
                None
                if max_on_sky_sigma_major_arcsec is None
                else float(max_on_sky_sigma_major_arcsec)
            ),
            gate=gate,
            targets=targets,
            dataset_pixels_by_target=dataset_pixels_by_target,
            truth_frames=truth_frames,
            truth_ids_key=truth_ids_key,
            run_dir=Path(run_dir),
            reuse_stage23_artifacts=bool(reuse_stage23_artifacts),
            write_stage23_artifacts_flag=bool(write_stage23_artifacts_flag),
            write_stage4_artifacts_flag=bool(write_stage4_artifacts_flag),
            detailed_timings=bool(detailed_timings),
            chunk_rows_stage23=int(chunk_rows_stage23),
            chunk_rows_stage4=int(chunk_rows_stage4),
            max_inflight_chunks=int(max_inflight_chunks),
            runtime_tmp_dir=runtime_tmp_dir,
            min_free_disk_gb=float(min_free_disk_gb),
            preprop_viability_policy=str(preprop_viability_policy),
            preprop_short_arc_days_threshold=float(preprop_short_arc_days_threshold),
            preprop_time_limit_days_short_arc=float(preprop_time_limit_days_short_arc),
            preprop_time_limit_days_default=float(preprop_time_limit_days_default),
            preprop_max_sigma_r_over_r=(
                None
                if preprop_max_sigma_r_over_r is None
                else float(preprop_max_sigma_r_over_r)
            ),
            preprop_max_covariance_condition=(
                None
                if preprop_max_covariance_condition is None
                else float(preprop_max_covariance_condition)
            ),
            preprop_viability_min_score=float(preprop_viability_min_score),
            preprop_fail_open_on_scoring_error=bool(preprop_fail_open_on_scoring_error),
        )
        if timings is not None:
            _sum_timings(timings, summary.gate_micro_timings)

        art = read_stage23_artifacts(run_dir=Path(run_dir))
        build = BuildStage23(
            preds=art.preds,
            triples=art.triples,
            stage3_orbit_metrics=art.stage3_orbit_metrics,
            build_observers_elapsed_s=float(summary.build_observers_elapsed_s),
            build_predict_elapsed_s=float(summary.build_predict_elapsed_s),
            build_pred_mag_elapsed_s=float(summary.build_pred_mag_elapsed_s),
            build_footprint_elapsed_s=float(summary.build_footprint_elapsed_s),
            build_triples_elapsed_s=float(summary.build_triples_elapsed_s),
            n_frames_skipped_limiting_mag=int(summary.n_frames_skipped_limiting_mag),
            n_frames_skipped_uncertainty=int(summary.n_frames_skipped_uncertainty),
            n_targets_preprop_viability_rejected=int(
                summary.n_targets_preprop_viability_rejected
            ),
            n_targets_preprop_time_limited=int(summary.n_targets_preprop_time_limited),
            n_targets_failfast_dynamics_error=int(
                summary.n_targets_failfast_dynamics_error
            ),
            micro_timings=dict(summary.build_micro_timings),
        )
        accepted_path = Path(run_dir) / "stage4" / str(backend.name) / "accepted_detections.parquet"
        accepted = AcceptedDetections.from_parquet(str(accepted_path))
        gate_res = BackendGateRowsResult(
            accepted_detections=accepted,
            accepted_counts=summary.accepted_counts,
            totals=summary.gate_totals,
            orbit_metrics=summary.stage4_orbit_metrics,
            elapsed_s=float(summary.backend_elapsed_s),
        )
        return Stage14RunRowsPython(targets=art.targets, build=build, gate=gate_res)

    if (
        stage23_run_dir is not None
        and bool(reuse_stage23_artifacts)
        and stage23_artifacts_exist(run_dir=stage23_run_dir)
    ):
        art = read_stage23_artifacts(run_dir=stage23_run_dir)
        targets = art.targets
        build = BuildStage23(
            preds=art.preds,
            triples=art.triples,
            stage3_orbit_metrics=art.stage3_orbit_metrics,
            build_observers_elapsed_s=float(art.build_observers_elapsed_s),
            build_predict_elapsed_s=float(art.build_predict_elapsed_s),
            build_pred_mag_elapsed_s=float(art.build_pred_mag_elapsed_s),
            build_footprint_elapsed_s=float(art.build_footprint_elapsed_s),
            build_triples_elapsed_s=float(art.build_triples_elapsed_s),
            n_frames_skipped_limiting_mag=int(art.n_frames_skipped_limiting_mag),
            n_frames_skipped_uncertainty=int(art.n_frames_skipped_uncertainty),
            n_targets_preprop_viability_rejected=int(
                art.n_targets_preprop_viability_rejected
            ),
            n_targets_preprop_time_limited=int(art.n_targets_preprop_time_limited),
            n_targets_failfast_dynamics_error=int(
                art.n_targets_failfast_dynamics_error
            ),
            micro_timings=dict(art.micro_timings),
        )
    else:
        if targets is None:
            targets = backend.enumerate_targets(
                subset=subset,
                start_mjd_utc=float(start_mjd_utc),
                end_mjd_utc=float(end_mjd_utc),
                obscodes=tuple(map(str, obscodes)),
            )
        if dataset_pixels_by_target is None and bool(
            getattr(getattr(backend, "capabilities", None), "supports_frame_pixels_by_target", False)
        ):
            try:
                dataset_pixels_by_target = backend.frame_pixels_by_target(subset=subset, targets=targets)
            except Exception:
                dataset_pixels_by_target = None
        build = build_stage23(
            orbits=orbits,
            targets=targets,
            start_mjd_utc=float(start_mjd_utc),
            window_size_days=int(window_size_days),
            stage2_strategy=str(stage2_strategy),
            healpix_nside=int(healpix_nside),
            footprint=footprint,
            max_processes=max_processes,
            limit_codefid_keys=limit_codefid_keys,
            limit_codefid_vals=limit_codefid_vals,
            faint_margin_mag=float(faint_margin_mag),
            max_on_sky_sigma_major_arcsec=(
                None
                if max_on_sky_sigma_major_arcsec is None
                else float(max_on_sky_sigma_major_arcsec)
            ),
            preprop_viability_policy=str(preprop_viability_policy),
            preprop_short_arc_days_threshold=float(preprop_short_arc_days_threshold),
            preprop_time_limit_days_short_arc=float(preprop_time_limit_days_short_arc),
            preprop_time_limit_days_default=float(preprop_time_limit_days_default),
            preprop_max_sigma_r_over_r=(
                None
                if preprop_max_sigma_r_over_r is None
                else float(preprop_max_sigma_r_over_r)
            ),
            preprop_max_covariance_condition=(
                None
                if preprop_max_covariance_condition is None
                else float(preprop_max_covariance_condition)
            ),
            preprop_viability_min_score=float(preprop_viability_min_score),
            preprop_fail_open_on_scoring_error=bool(preprop_fail_open_on_scoring_error),
            max_mag_residual_fainter_mag=gate.max_mag_residual_fainter_mag,
            max_mag_residual_brighter_mag=gate.max_mag_residual_brighter_mag,
            dataset_pixels_by_target=dataset_pixels_by_target,
            truth_frames=truth_frames,
            detailed_timings=bool(detailed_timings),
        )
        if stage23_run_dir is not None and bool(write_stage23_artifacts_flag):
            write_stage23_artifacts(
                run_dir=stage23_run_dir,
                targets=targets,
                preds=build.preds,
                triples=build.triples,
                stage3_orbit_metrics=build.stage3_orbit_metrics,
                micro_timings=dict(build.micro_timings),
                build_observers_elapsed_s=float(build.build_observers_elapsed_s),
                build_predict_elapsed_s=float(build.build_predict_elapsed_s),
                build_pred_mag_elapsed_s=float(build.build_pred_mag_elapsed_s),
                build_footprint_elapsed_s=float(build.build_footprint_elapsed_s),
                build_triples_elapsed_s=float(build.build_triples_elapsed_s),
                n_frames_skipped_limiting_mag=int(build.n_frames_skipped_limiting_mag),
                n_frames_skipped_uncertainty=int(build.n_frames_skipped_uncertainty),
                n_targets_preprop_viability_rejected=int(
                    build.n_targets_preprop_viability_rejected
                ),
                n_targets_preprop_time_limited=int(build.n_targets_preprop_time_limited),
                n_targets_failfast_dynamics_error=int(
                    build.n_targets_failfast_dynamics_error
                ),
                extra_meta={
                    "start_mjd_utc": float(start_mjd_utc),
                    "end_mjd_utc": float(end_mjd_utc),
                    "window_size_days": int(window_size_days),
                    "stage2_strategy": str(stage2_strategy),
                    "healpix_nside": int(healpix_nside),
                    "max_on_sky_sigma_major_arcsec": (
                        None
                        if max_on_sky_sigma_major_arcsec is None
                        else float(max_on_sky_sigma_major_arcsec)
                    ),
                },
            )
    gate_res = stage4_fetch_and_gate_rows_python(
        backend=backend,
        subset=subset,
        triples=build.triples,
        preds=build.preds,
        gate=gate,
        orbit_ids=pa.array(orbits.orbit_id.to_pylist(), type=pa.large_string()),
        truth_ids_key=truth_ids_key,
        timings=timings,
    )
    return Stage14RunRowsPython(targets=targets, build=build, gate=gate_res)
