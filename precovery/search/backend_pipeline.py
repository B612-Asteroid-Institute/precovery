from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pyarrow as pa
import time

from adam_core.orbits import Orbits

from .artifacts import (
    read_stage23_artifacts,
    stage23_artifacts_exist,
    write_stage23_artifacts,
)
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
            ("_rej_mag", "sum"),
        ]
    ).rename_columns(
        [
            "orbit_id",
            "n_detections_candidates",
            "n_detections_gate_matched",
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
    detailed_timings: bool,
    timings: dict[str, float] | None = None,
    stage23_run_dir: Path | None = None,
    reuse_stage23_artifacts: bool = True,
    write_stage23_artifacts_flag: bool = True,
    targets: BenchTargets | None = None,
    dataset_pixels_by_target: dict[tuple[str, int], np.ndarray] | None = None,
    truth_frames: pa.Table | None = None,
) -> Stage14RunPython:
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
                extra_meta={
                    "start_mjd_utc": float(start_mjd_utc),
                    "end_mjd_utc": float(end_mjd_utc),
                    "window_size_days": int(window_size_days),
                    "stage2_strategy": str(stage2_strategy),
                    "healpix_nside": int(healpix_nside),
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
    detailed_timings: bool,
    timings: dict[str, float] | None = None,
    stage23_run_dir: Path | None = None,
    reuse_stage23_artifacts: bool = True,
    write_stage23_artifacts_flag: bool = True,
    targets: BenchTargets | None = None,
    dataset_pixels_by_target: dict[tuple[str, int], np.ndarray] | None = None,
    truth_frames: pa.Table | None = None,
    truth_ids_key: pa.Array | None = None,
) -> Stage14RunRowsPython:
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
                extra_meta={
                    "start_mjd_utc": float(start_mjd_utc),
                    "end_mjd_utc": float(end_mjd_utc),
                    "window_size_days": int(window_size_days),
                    "stage2_strategy": str(stage2_strategy),
                    "healpix_nside": int(healpix_nside),
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
