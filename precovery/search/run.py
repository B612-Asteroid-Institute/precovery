from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa

from adam_core.orbits import Orbits

from precovery.config import Config

from .artifacts import write_stage4_artifacts
from .backend_pipeline import Stage14RunRowsPython, run_stage1_to_stage4_rows_python
from .pipeline_types import BenchTargets, Stage3OrbitMetrics, Stage4OrbitMetrics, AcceptedCounts, SubsetPaths
from .backends.factory import backend_from_config
from .backends.protocols import GateParams, SearchBackend
from .footprints import CovPolygonReconstructedMoc
from .gate_counts import GateTotals
from .gate_params_factory import build_gate_params
from .results import AcceptedDetections
from .runtime_config import load_mag_gate_config


@dataclass(frozen=True)
class BackendSearchRun:
    accepted: AcceptedDetections
    accepted_counts: AcceptedCounts
    build_micro_timings: dict[str, float]
    n_frames_skipped_limiting_mag: int


@dataclass(frozen=True)
class BackendBenchmarkRun:
    """
    File-first benchmark-oriented run summary.

    This is intentionally small: large intermediate tables live on disk under `run_dir`.
    """

    stage23_run_dir: Path | None
    stage4_run_dir: Path | None

    stage3_orbit_metrics: Stage3OrbitMetrics
    stage4_orbit_metrics: Stage4OrbitMetrics

    build_observers_elapsed_s: float
    build_predict_elapsed_s: float
    build_pred_mag_elapsed_s: float
    build_footprint_elapsed_s: float
    build_triples_elapsed_s: float
    build_elapsed_s: float
    n_frames_skipped_limiting_mag: int
    n_unique_frames_selected: int

    build_micro_timings: dict[str, float]
    gate_micro_timings: dict[str, float]
    gate_totals: GateTotals
    backend_elapsed_s: float
    bytes_estimate: int | None
    n_matched_frames: int | None


def precover_orbits_backend(
    *,
    orbits: Orbits,
    subset_dir: Path,
    start_mjd: float,
    end_mjd: float,
    obscodes: tuple[str, ...] = (),
    window_size_days: int = 7,
    stage2_strategy: str = "assist_window_then_2body_variants:sigma_points",
    max_processes: int | None = None,
    n_sigma: float = 3.0,
    polygon_vertices: int = 32,
    detailed_timings: bool = False,
    stage23_run_dir: Path | None = None,
    reuse_stage23_artifacts: bool = True,
    write_stage23_artifacts: bool = True,
) -> BackendSearchRun:
    """
    Run backend-adapter precovery (DuckDB/ClickHouse/BigQuery) for a set of orbits.

    This path is intentionally independent of the legacy SQLite blob store; it treats the
    detections dataset as the source of truth and uses backend adapters for Stage 1 and Stage 4.
    """
    subset_dir = Path(subset_dir)
    cfg_path = subset_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config.json: {cfg_path}")

    cfg = Config.from_json(str(cfg_path))
    backend = backend_from_config(config=cfg, subset_dir=subset_dir)

    # Magnitude + limiting-magnitude configuration.
    limit_keys, limit_vals, faint_margin, max_faint, max_bright = load_mag_gate_config(
        subset_dir=subset_dir,
        obscodes=set(obscodes) if obscodes else None,
        config=cfg,
    )
    gate = build_gate_params(
        innovation_gate_n_sigma=float(n_sigma),
        max_mag_residual_fainter_mag=max_faint,
        max_mag_residual_brighter_mag=max_bright,
    )

    fp = CovPolygonReconstructedMoc(
        n_sigma=float(n_sigma),
        polygon_vertices=int(polygon_vertices),
    )
    timings = {} if bool(detailed_timings) else None
    out: Stage14RunRowsPython = run_stage1_to_stage4_rows_python(
        backend=backend,
        subset=SubsetPaths(subset_dir=subset_dir),
        orbits=orbits,
        start_mjd_utc=float(start_mjd),
        end_mjd_utc=float(end_mjd),
        obscodes=tuple(map(str, obscodes)),
        window_size_days=int(window_size_days),
        stage2_strategy=str(stage2_strategy),
        healpix_nside=int(getattr(cfg, "nside", 32)),
        footprint=fp,
        max_processes=max_processes,
        limit_codefid_keys=limit_keys,
        limit_codefid_vals=limit_vals,
        faint_margin_mag=float(faint_margin),
        gate=gate,
        detailed_timings=bool(detailed_timings),
        timings=timings,
        stage23_run_dir=stage23_run_dir,
        reuse_stage23_artifacts=bool(reuse_stage23_artifacts),
        write_stage23_artifacts_flag=bool(write_stage23_artifacts),
    )
    return BackendSearchRun(
        accepted=out.gate.accepted_detections,
        accepted_counts=out.gate.accepted_counts,
        build_micro_timings=dict(out.build.micro_timings),
        n_frames_skipped_limiting_mag=int(out.build.n_frames_skipped_limiting_mag),
    )


def precover_orbits_with_backend_for_benchmark(
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
    run_dir: Path | None = None,
    reuse_stage23_artifacts: bool = True,
    write_stage23_artifacts: bool = True,
    write_stage4_artifacts_flag: bool = True,
    detailed_timings: bool = False,
) -> BackendBenchmarkRun:
    """
    Benchmark-friendly precover wrapper.

    - The caller provides Stage-1 inputs (`targets`, optionally `dataset_pixels_by_target`) so Stage 2/3 is shared
      across backends for fair comparison.
    - When `run_dir` is provided, Stage23 and Stage4 outputs are written to disk and the return value stays small.
    """
    stage23_run_dir = run_dir
    stage4_run_dir = None if run_dir is None else (Path(run_dir) / "stage4" / str(backend.name))

    timings = {} if bool(detailed_timings) else None
    out: Stage14RunRowsPython = run_stage1_to_stage4_rows_python(
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
        gate=gate,
        detailed_timings=bool(detailed_timings),
        timings=timings,
        stage23_run_dir=stage23_run_dir,
        reuse_stage23_artifacts=bool(reuse_stage23_artifacts),
        write_stage23_artifacts_flag=bool(write_stage23_artifacts),
        targets=targets,
        dataset_pixels_by_target=dataset_pixels_by_target,
        truth_frames=truth_frames,
        truth_ids_key=truth_ids_key,
    )

    gate_micro = {} if timings is None else dict(timings)

    build_elapsed_s = float(
        float(out.build.build_observers_elapsed_s)
        + float(out.build.build_predict_elapsed_s)
        + float(out.build.build_pred_mag_elapsed_s)
        + float(out.build.build_footprint_elapsed_s)
        + float(out.build.build_triples_elapsed_s)
    )

    triples_tbl = out.build.triples.table
    if triples_tbl.num_rows > 0:
        n_unique_frames_selected = int(
            triples_tbl.select(["obscode", "exposure_mjd_mid_key_us", "healpixel"])
            .group_by(["obscode", "exposure_mjd_mid_key_us", "healpixel"])
            .aggregate([])
            .num_rows
        )
    else:
        n_unique_frames_selected = 0

    if run_dir is not None and bool(write_stage4_artifacts_flag):
        write_stage4_artifacts(
            run_dir=Path(run_dir),
            backend_name=str(backend.name),
            accepted_detections=out.gate.accepted_detections,
            accepted_counts=out.gate.accepted_counts,
            stage4_orbit_metrics=out.gate.orbit_metrics,
            totals=out.gate.totals,
            micro_timings=gate_micro,
            extra_meta={
                "start_mjd_utc": float(start_mjd_utc),
                "end_mjd_utc": float(end_mjd_utc),
                "window_size_days": int(window_size_days),
                "stage2_strategy": str(stage2_strategy),
                "healpix_nside": int(healpix_nside),
            },
        )

    bytes_est = getattr(backend, "last_bytes_estimate", None)
    bytes_estimate = None if bytes_est is None else int(bytes_est)
    try:
        n_matched_frames = (
            None
            if not hasattr(backend, "count_matched_frame_keys")
            else int(backend.count_matched_frame_keys(triples=out.build.triples))  # type: ignore[attr-defined]
        )
    except Exception:
        n_matched_frames = None

    return BackendBenchmarkRun(
        stage23_run_dir=stage23_run_dir,
        stage4_run_dir=stage4_run_dir,
        stage3_orbit_metrics=out.build.stage3_orbit_metrics,
        stage4_orbit_metrics=out.gate.orbit_metrics,
        build_observers_elapsed_s=float(out.build.build_observers_elapsed_s),
        build_predict_elapsed_s=float(out.build.build_predict_elapsed_s),
        build_pred_mag_elapsed_s=float(out.build.build_pred_mag_elapsed_s),
        build_footprint_elapsed_s=float(out.build.build_footprint_elapsed_s),
        build_triples_elapsed_s=float(out.build.build_triples_elapsed_s),
        build_elapsed_s=build_elapsed_s,
        n_frames_skipped_limiting_mag=int(out.build.n_frames_skipped_limiting_mag),
        n_unique_frames_selected=int(n_unique_frames_selected),
        build_micro_timings=dict(out.build.micro_timings),
        gate_micro_timings=gate_micro,
        gate_totals=out.gate.totals,
        backend_elapsed_s=float(out.gate.elapsed_s),
        bytes_estimate=bytes_estimate,
        n_matched_frames=n_matched_frames,
    )
