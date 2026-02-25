from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa

from .pipeline_types import (
    AcceptedCounts,
    BenchTargets,
    PredictedTargets,
    PredictedTriples,
    Stage3OrbitMetrics,
    Stage4OrbitMetrics,
)
from .gate_counts import GateTotals
from .results import AcceptedDetections


@dataclass(frozen=True)
class Stage23Artifacts:
    targets: BenchTargets
    preds: PredictedTargets
    triples: PredictedTriples
    stage3_orbit_metrics: Stage3OrbitMetrics
    micro_timings: dict[str, float]
    build_observers_elapsed_s: float = 0.0
    build_predict_elapsed_s: float = 0.0
    build_pred_mag_elapsed_s: float = 0.0
    build_footprint_elapsed_s: float = 0.0
    build_triples_elapsed_s: float = 0.0
    n_frames_skipped_limiting_mag: int = 0


@dataclass(frozen=True)
class Stage4Artifacts:
    accepted_detections: AcceptedDetections
    accepted_counts: AcceptedCounts
    stage4_orbit_metrics: Stage4OrbitMetrics
    totals: GateTotals
    micro_timings: dict[str, float]


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def stage23_artifacts_paths(*, run_dir: Path) -> dict[str, Path]:
    d = Path(run_dir) / "stage23"
    return {
        "dir": d,
        "meta": d / "meta.json",
        "targets": d / "targets.parquet",
        "preds": d / "preds.parquet",
        "triples": d / "triples.parquet",
        "stage3_orbit_metrics": d / "stage3_orbit_metrics.parquet",
    }


def stage23_artifacts_exist(*, run_dir: Path) -> bool:
    p = stage23_artifacts_paths(run_dir=run_dir)
    return (
        p["targets"].exists()
        and p["preds"].exists()
        and p["triples"].exists()
        and p["stage3_orbit_metrics"].exists()
    )


def _sanitize_backend_name(name: str) -> str:
    s = str(name).strip()
    if not s:
        return "backend"
    return s.replace("/", "_").replace("\\", "_")


def stage4_artifacts_paths(*, run_dir: Path, backend_name: str) -> dict[str, Path]:
    b = _sanitize_backend_name(backend_name)
    d = Path(run_dir) / "stage4" / b
    return {
        "dir": d,
        "meta": d / "meta.json",
        "accepted_detections": d / "accepted_detections.parquet",
        "accepted_counts": d / "accepted_counts.parquet",
        "stage4_orbit_metrics": d / "stage4_orbit_metrics.parquet",
    }


def stage4_artifacts_exist(*, run_dir: Path, backend_name: str) -> bool:
    p = stage4_artifacts_paths(run_dir=run_dir, backend_name=backend_name)
    return (
        p["accepted_detections"].exists()
        and p["accepted_counts"].exists()
        and p["stage4_orbit_metrics"].exists()
    )


def write_stage4_artifacts(
    *,
    run_dir: Path,
    backend_name: str,
    accepted_detections: AcceptedDetections,
    accepted_counts: AcceptedCounts,
    stage4_orbit_metrics: Stage4OrbitMetrics,
    totals: GateTotals,
    micro_timings: dict[str, float],
    extra_meta: dict[str, object] | None = None,
) -> None:
    p = stage4_artifacts_paths(run_dir=run_dir, backend_name=backend_name)
    _ensure_dir(p["dir"])

    accepted_detections.to_parquet(str(p["accepted_detections"]))
    accepted_counts.to_parquet(str(p["accepted_counts"]))
    stage4_orbit_metrics.to_parquet(str(p["stage4_orbit_metrics"]))

    meta: dict[str, object] = {
        "schema_version": 1,
        "kind": "stage4_artifacts",
        "backend": str(backend_name),
        "n_accepted_detections": int(len(accepted_detections)),
        "n_counts_rows": int(len(accepted_counts)),
        "n_orbit_metrics_rows": int(len(stage4_orbit_metrics)),
        "gate_totals": {
            "n_candidates": int(totals.n_candidates),
            "n_accepted": int(totals.n_accepted),
            "n_rejected_innov_ellipse": int(totals.n_rejected_innov_ellipse),
            "n_rejected_mag_residual": int(totals.n_rejected_mag_residual),
        },
        "micro_timings": {str(k): float(v) for k, v in dict(micro_timings).items()},
    }
    if extra_meta:
        meta.update({str(k): v for k, v in dict(extra_meta).items()})
    p["meta"].write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")


def read_stage4_artifacts(*, run_dir: Path, backend_name: str) -> Stage4Artifacts:
    p = stage4_artifacts_paths(run_dir=run_dir, backend_name=backend_name)
    if not stage4_artifacts_exist(run_dir=run_dir, backend_name=backend_name):
        raise FileNotFoundError(f"Missing stage4 artifacts under {p['dir']}")

    accepted_detections = AcceptedDetections.from_parquet(str(p["accepted_detections"]))
    accepted_counts = AcceptedCounts.from_parquet(str(p["accepted_counts"]))
    m = Stage4OrbitMetrics.from_parquet(str(p["stage4_orbit_metrics"]))

    totals = GateTotals(0, 0, 0, 0)
    micro_timings: dict[str, float] = {}
    if p["meta"].exists():
        try:
            meta = json.loads(p["meta"].read_text())
            gt = meta.get("gate_totals", {})
            if isinstance(gt, dict):
                totals = GateTotals(
                    int(gt.get("n_candidates", 0) or 0),
                    int(gt.get("n_accepted", 0) or 0),
                    int(gt.get("n_rejected_innov_ellipse", 0) or 0),
                    int(gt.get("n_rejected_mag_residual", 0) or 0),
                )
            mt = meta.get("micro_timings", {})
            if isinstance(mt, dict):
                micro_timings = {str(k): float(v) for k, v in mt.items()}
        except Exception:
            totals = GateTotals(0, 0, 0, 0)
            micro_timings = {}

    return Stage4Artifacts(
        accepted_detections=accepted_detections,
        accepted_counts=accepted_counts,
        stage4_orbit_metrics=m,
        totals=totals,
        micro_timings=micro_timings,
    )


def write_stage23_artifacts(
    *,
    run_dir: Path,
    targets: BenchTargets,
    preds: PredictedTargets,
    triples: PredictedTriples,
    stage3_orbit_metrics: Stage3OrbitMetrics,
    micro_timings: dict[str, float],
    build_observers_elapsed_s: float = 0.0,
    build_predict_elapsed_s: float = 0.0,
    build_pred_mag_elapsed_s: float = 0.0,
    build_footprint_elapsed_s: float = 0.0,
    build_triples_elapsed_s: float = 0.0,
    n_frames_skipped_limiting_mag: int = 0,
    extra_meta: dict[str, object] | None = None,
) -> None:
    p = stage23_artifacts_paths(run_dir=run_dir)
    _ensure_dir(p["dir"])

    targets.to_parquet(str(p["targets"]))
    preds.to_parquet(str(p["preds"]))
    triples.to_parquet(str(p["triples"]))
    stage3_orbit_metrics.to_parquet(str(p["stage3_orbit_metrics"]))

    meta: dict[str, object] = {
        "schema_version": 1,
        "kind": "stage23_artifacts",
        "n_targets": int(len(targets)),
        "n_preds": int(len(preds)),
        "n_triples": int(len(triples)),
        "n_orbits_metrics": int(len(stage3_orbit_metrics)),
        "micro_timings": {str(k): float(v) for k, v in dict(micro_timings).items()},
        "build_observers_elapsed_s": float(build_observers_elapsed_s),
        "build_predict_elapsed_s": float(build_predict_elapsed_s),
        "build_pred_mag_elapsed_s": float(build_pred_mag_elapsed_s),
        "build_footprint_elapsed_s": float(build_footprint_elapsed_s),
        "build_triples_elapsed_s": float(build_triples_elapsed_s),
        "n_frames_skipped_limiting_mag": int(n_frames_skipped_limiting_mag),
    }
    if extra_meta:
        meta.update({str(k): v for k, v in dict(extra_meta).items()})

    p["meta"].write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")


def read_stage23_artifacts(*, run_dir: Path) -> Stage23Artifacts:
    p = stage23_artifacts_paths(run_dir=run_dir)
    if not stage23_artifacts_exist(run_dir=run_dir):
        raise FileNotFoundError(f"Missing stage23 artifacts under {p['dir']}")

    targets = BenchTargets.from_parquet(str(p["targets"]))
    preds = PredictedTargets.from_parquet(str(p["preds"]))
    triples = PredictedTriples.from_parquet(str(p["triples"]))
    m = Stage3OrbitMetrics.from_parquet(str(p["stage3_orbit_metrics"]))

    micro_timings: dict[str, float] = {}
    build_observers_elapsed_s = 0.0
    build_predict_elapsed_s = 0.0
    build_pred_mag_elapsed_s = 0.0
    build_footprint_elapsed_s = 0.0
    build_triples_elapsed_s = 0.0
    n_frames_skipped_limiting_mag = 0
    if p["meta"].exists():
        try:
            meta = json.loads(p["meta"].read_text())
            mt = meta.get("micro_timings", {})
            if isinstance(mt, dict):
                micro_timings = {str(k): float(v) for k, v in mt.items()}
            build_observers_elapsed_s = float(meta.get("build_observers_elapsed_s", 0.0) or 0.0)
            build_predict_elapsed_s = float(meta.get("build_predict_elapsed_s", 0.0) or 0.0)
            build_pred_mag_elapsed_s = float(meta.get("build_pred_mag_elapsed_s", 0.0) or 0.0)
            build_footprint_elapsed_s = float(meta.get("build_footprint_elapsed_s", 0.0) or 0.0)
            build_triples_elapsed_s = float(meta.get("build_triples_elapsed_s", 0.0) or 0.0)
            n_frames_skipped_limiting_mag = int(meta.get("n_frames_skipped_limiting_mag", 0) or 0)
        except Exception:
            micro_timings = {}

    return Stage23Artifacts(
        targets=targets,
        preds=preds,
        triples=triples,
        stage3_orbit_metrics=m,
        micro_timings=micro_timings,
        build_observers_elapsed_s=float(build_observers_elapsed_s),
        build_predict_elapsed_s=float(build_predict_elapsed_s),
        build_pred_mag_elapsed_s=float(build_pred_mag_elapsed_s),
        build_footprint_elapsed_s=float(build_footprint_elapsed_s),
        build_triples_elapsed_s=float(build_triples_elapsed_s),
        n_frames_skipped_limiting_mag=int(n_frames_skipped_limiting_mag),
    )


def _pyarrow_table_nbytes(t: pa.Table) -> int:
    try:
        return int(t.nbytes)
    except Exception:
        return 0

