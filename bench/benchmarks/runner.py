from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from adam_core.orbits import Orbits

from precovery.search.footprints import CovPolygonReconstructedMoc
from precovery.config import Config
from precovery.search.covariance_psd import repair_orbits_covariance_psd_for_sampling
from precovery.search.gate_params_factory import build_gate_params
from precovery.search.pipeline_types import (
    BenchTargets,
)
from precovery.search.backends.protocols import SearchBackend as BenchBackend
from precovery.search.runtime_config import (
    load_mag_gate_config,
    load_preprop_viability_policy_config,
    load_stage3_uncertainty_budget_config,
)
from precovery.search.time_key import mjd_to_time_key_us
from precovery.search.assist_perturbers import perturber_warnings_for_orbit_ids

from .results import BackendRunResults, PerOrbitRunResults
from .workload import WorkloadSpec, utc_now_iso


def _discover_stage2_bad_orbit_ids(*, subset_dir: Path) -> set[str]:
    """
    Discover bad orbit IDs recorded by Stage-2 runs under:
      <subset_dir>/artifacts/stage2/**/bad_orbits.json
    """
    root = Path(subset_dir) / "artifacts" / "stage2"
    if not root.exists():
        return set()
    out: set[str] = set()
    for p in root.glob("**/bad_orbits.json"):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        ids = d.get("bad_orbit_ids") or []
        for x in ids:
            if x is None:
                continue
            s = str(x).strip()
            if s:
                out.add(s)
    return out


def _drop_orbits_by_id(orbits: Orbits, *, drop_ids: set[str]) -> Orbits:
    if len(orbits) == 0 or not drop_ids:
        return orbits
    value_set = pa.array(sorted(drop_ids), type=pa.large_string())
    mask = pc.invert(pc.is_in(orbits.orbit_id, value_set=value_set))
    idx = np.nonzero(mask.to_numpy(zero_copy_only=False).astype(bool))[0]
    return orbits.take(idx.tolist())


def _repair_orbits_covariance_psd_for_sampling(
    orbits: Orbits,
    *,
    abs_tol: float = 1e-15,
    rel_tol: float = 1e-10,
) -> tuple[Orbits, int, int]:
    # Backwards-compatible wrapper: benchmark code delegates policy to core.
    return repair_orbits_covariance_psd_for_sampling(orbits, abs_tol=abs_tol, rel_tol=rel_tol)


def _sample_orbits(orbits: Orbits, *, max_orbits: int | None) -> Orbits:
    if max_orbits is None or len(orbits) <= int(max_orbits):
        return orbits
    return orbits.take(list(range(int(max_orbits))))


def _sample_targets(targets: BenchTargets, *, max_targets: int | None) -> BenchTargets:
    if max_targets is None or len(targets) <= int(max_targets):
        return targets
    return targets.take(list(range(int(max_targets))))


def _load_truth_table(truth_path: Path) -> pa.Table:
    t = pq.read_table(str(truth_path))
    cols = set(t.column_names)

    orbit_col = "orbit_id" if "orbit_id" in cols else ("designation" if "designation" in cols else None)
    if orbit_col is None:
        raise ValueError(f"Truth table missing expected orbit id column. Found: {sorted(cols)}")

    # Prefer crossmatch-to-precovery IDs when present: these are the observation IDs our
    # candidate backends will emit/recover.
    if "match_observation_id" in cols:
        obs_col = "match_observation_id"
    elif "observation_id" in cols:
        obs_col = "observation_id"
    elif "obsid" in cols:
        obs_col = "obsid"
    elif "truth_obsid" in cols:
        # Fallback: truth observation IDs (may be a different namespace than precovery obs IDs).
        obs_col = "truth_obsid"
    else:
        raise ValueError(f"Truth table missing expected observation id column. Found: {sorted(cols)}")

    # Time column for window filtering (prefer matched time when present).
    time_col = (
        "match_time_mjd_utc"
        if "match_time_mjd_utc" in cols
        else ("truth_time_mjd_utc" if "truth_time_mjd_utc" in cols else None)
    )
    if time_col is None:
        raise ValueError(f"Truth table missing expected time column. Found: {sorted(cols)}")

    select_cols = [orbit_col, obs_col, "obscode", time_col]
    # Canonical truth_detections bundles include the dataset exposure join key.
    if "exposure_mjd_mid_key_us" in cols:
        select_cols.append("exposure_mjd_mid_key_us")
    if "matched" in cols:
        select_cols.append("matched")
    out_cols = ["orbit_id", "observation_id", "obscode", "time_mjd_utc"]
    if "exposure_mjd_mid_key_us" in cols:
        out_cols.append("exposure_mjd_mid_key_us")
    if "matched" in cols:
        out_cols.append("matched")
    t2 = t.select(select_cols).rename_columns(out_cols)

    if "matched" in t2.column_names:
        m = pc.fill_null(t2["matched"], False)
        t2 = t2.filter(m).drop(["matched"])

    t2 = t2.filter(pc.invert(pc.is_null(t2["observation_id"])))
    t2 = t2.filter(pc.invert(pc.is_null(t2["time_mjd_utc"])))
    return t2


def _truth_targets_for_workload(*, truth: pa.Table, start_mjd_utc: float, end_mjd_utc: float) -> BenchTargets:
    """
    Build target exposure midpoints from truth crossmatch rows within the workload time bounds.

    This avoids enumerating all exposures in the month when we specifically want truth scoring.
    """
    if truth.num_rows == 0:
        return BenchTargets.empty()
    m = pc.greater_equal(truth["time_mjd_utc"], pa.scalar(float(start_mjd_utc)))
    m = pc.and_(m, pc.less(truth["time_mjd_utc"], pa.scalar(float(end_mjd_utc))))
    t2 = truth.filter(m).select(["obscode", "time_mjd_utc"])
    if t2.num_rows == 0:
        return BenchTargets.empty()
    # Unique (obscode,time) targets.
    t2 = t2.group_by(["obscode", "time_mjd_utc"]).aggregate([]).sort_by([("time_mjd_utc", "ascending")])
    mids = np.asarray(t2["time_mjd_utc"].to_numpy(zero_copy_only=False), dtype=np.float64)
    return BenchTargets.from_kwargs(
        obscode=[str(x) for x in t2["obscode"].to_pylist()],
        exposure_mjd_mid_utc=mids,
        # Truth tables do not reliably carry the dataset filter band; use a placeholder.
        filter=["V"] * int(mids.size),
        exposure_mjd_mid_key_us=mjd_to_time_key_us(mids),
    )


def _with_truth_exposure_key(*, truth: pa.Table) -> pa.Table:
    """
    Ensure truth rows include a canonical exposure key column.
    """
    if truth.num_rows == 0:
        return truth

    if "exposure_mjd_mid_key_us" in truth.column_names:
        return truth.set_column(
            truth.schema.get_field_index("exposure_mjd_mid_key_us"),
            "exposure_mjd_mid_key_us",
            pc.cast(truth["exposure_mjd_mid_key_us"], pa.int64()),
        )

    key_us = mjd_to_time_key_us(
        np.asarray(truth["time_mjd_utc"].to_numpy(zero_copy_only=False), dtype=np.float64)
    )
    return truth.append_column("exposure_mjd_mid_key_us", pa.array(key_us, type=pa.int64()))


def _filter_truth_to_targets(*, truth: pa.Table, targets: BenchTargets) -> pa.Table:
    """
    Restrict truth rows to the sampled target exposure set for denominator-consistent scoring.
    """
    if truth.num_rows == 0 or len(targets) == 0:
        return truth.slice(0, 0)

    truth_k = _with_truth_exposure_key(truth=truth)
    truth_k = truth_k.set_column(
        truth_k.schema.get_field_index("obscode"),
        "obscode",
        pc.cast(truth_k["obscode"], pa.large_string()),
    )
    target_keys = (
        targets.table.select(["obscode", "exposure_mjd_mid_key_us"])
        .group_by(["obscode", "exposure_mjd_mid_key_us"])
        .aggregate([])
    )
    out = truth_k.join(
        target_keys,
        keys=["obscode", "exposure_mjd_mid_key_us"],
        join_type="inner",
    )
    return out.select(truth_k.column_names)


def run_benchmark(
    *,
    workload: WorkloadSpec,
    orbits: Orbits,
    orbits_parquet: str,
    enumerate_backend: BenchBackend,
    backends: Iterable[BenchBackend],
    max_orbits: int | None = 25,
    max_targets: int | None = 2000,
    truth_path: Path | None = None,
    targets_from_truth: bool = False,
    det_sigma_floor_arcsec: float | None = None,
    det_sigma_floor_arcsec_by_obscode: dict[str, float] | None = None,
    det_sigma_sys_arcsec_by_obscode: dict[str, float] | None = None,
    det_sigma_sys_apply_rms_lt_arcsec_by_obscode: dict[str, float] | None = None,
    gate_n_sigma: float | None = None,
    max_processes: int | None = None,
    compute_gate_totals: bool = False,
    detailed_timings: bool = False,
    execution_mode: str = "memory",
    chunk_rows_stage23: int = 0,
    chunk_rows_stage4: int = 0,
    max_inflight_chunks: int = 2,
    runtime_tmp_dir: str | None = None,
    min_free_disk_gb: float = 10.0,
    max_on_sky_sigma_major_arcsec: float | None = None,
) -> tuple[BackendRunResults, PerOrbitRunResults | None]:
    """
    Run the canonical workload across backends and collect comparable metrics.

    Metric definitions and terminology are centralized in `bench/benchmarks/metrics_spec.py`.
    """
    subset = workload.subset_paths()
    start_mjd, end_mjd = workload.bounds_mjd_utc()
    backends = list(backends)

    cfg_path = Path(workload.subset_dir) / "config.json"
    cfg = Config.from_json(str(cfg_path)) if cfg_path.exists() else Config()
    limit_keys, limit_vals, faint_margin, max_faint, max_bright = load_mag_gate_config(
        subset_dir=Path(workload.subset_dir),
        obscodes=set(workload.window.obscodes) if workload.window.obscodes else None,
        config=cfg,
    )
    cfg_max_on_sky = load_stage3_uncertainty_budget_config(config=cfg)
    preprop_cfg = load_preprop_viability_policy_config(config=cfg)
    max_on_sky_budget = (
        float(max_on_sky_sigma_major_arcsec)
        if max_on_sky_sigma_major_arcsec is not None
        else cfg_max_on_sky
    )

    # Orbit sample + covariance QC (reject non-finite / non-PSD covariances up front so we don't
    # include guaranteed failures in Stage 3/4 metrics).
    orbits_use0 = _sample_orbits(orbits, max_orbits=max_orbits)
    bad_ids = _discover_stage2_bad_orbit_ids(subset_dir=Path(workload.subset_dir))
    orbits_use0 = _drop_orbits_by_id(orbits_use0, drop_ids=bad_ids)
    orbits_use, n_cov_repaired, n_cov_rejected = _repair_orbits_covariance_psd_for_sampling(orbits_use0)
    orbit_ids_use = [str(x) for x in orbits_use.orbit_id.to_pylist()]
    truth_window_tbl = _load_truth_table(truth_path) if truth_path is not None and truth_path.exists() else None
    if truth_window_tbl is not None:
        # Restrict truth rows to the current window obscodes, orbit sample, and time range.
        m = pc.is_in(
            truth_window_tbl["obscode"],
            value_set=pa.array(list(workload.window.obscodes), type=pa.large_string()),
        )
        m = pc.and_(
            m,
            pc.is_in(
                truth_window_tbl["orbit_id"],
                value_set=pa.array(
                    [str(x) for x in orbits_use.orbit_id.to_pylist()], type=pa.large_string()
                ),
            ),
        )
        m = pc.and_(m, pc.greater_equal(truth_window_tbl["time_mjd_utc"], pa.scalar(float(start_mjd))))
        m = pc.and_(m, pc.less(truth_window_tbl["time_mjd_utc"], pa.scalar(float(end_mjd))))
        truth_window_tbl = truth_window_tbl.filter(m)
        if truth_window_tbl.num_rows == 0:
            truth_window_tbl = None

    if truth_window_tbl is not None and bool(targets_from_truth):
        # When scoring on truth-only targets, ensure we still use the dataset's exposure midpoints
        # and filter band so limiting-magnitude filtering matches the full-month run.
        if "exposure_mjd_mid_key_us" in truth_window_tbl.column_names:
            truth_keys = (
                truth_window_tbl.select(["obscode", "exposure_mjd_mid_key_us"])
                .group_by(["obscode", "exposure_mjd_mid_key_us"])
                .aggregate([])
            )
            targets_all = enumerate_backend.enumerate_targets(
                subset=subset,
                start_mjd_utc=float(start_mjd),
                end_mjd_utc=float(end_mjd),
                obscodes=workload.window.obscodes,
            )
            t_join = targets_all.table.join(
                truth_keys,
                keys=["obscode", "exposure_mjd_mid_key_us"],
                join_type="inner",
            ).sort_by([("exposure_mjd_mid_utc", "ascending"), ("obscode", "ascending")])
            targets = BenchTargets.from_pyarrow(t_join)
        else:
            targets = _truth_targets_for_workload(
                truth=truth_window_tbl, start_mjd_utc=float(start_mjd), end_mjd_utc=float(end_mjd)
            )
            if len(targets) == 0:
                targets = enumerate_backend.enumerate_targets(
                    subset=subset,
                    start_mjd_utc=float(start_mjd),
                    end_mjd_utc=float(end_mjd),
                    obscodes=workload.window.obscodes,
                )
    else:
        targets = enumerate_backend.enumerate_targets(
            subset=subset,
            start_mjd_utc=float(start_mjd),
            end_mjd_utc=float(end_mjd),
            obscodes=workload.window.obscodes,
        )
    targets = _sample_targets(targets, max_targets=max_targets)

    # Truth scoring denominators are target-scoped: with capped targets, only score truth rows
    # that are reachable by the sampled target set.
    truth_tbl: pa.Table | None = None
    if truth_window_tbl is not None and truth_window_tbl.num_rows > 0:
        truth_tbl = _filter_truth_to_targets(truth=truth_window_tbl, targets=targets)
        if truth_tbl.num_rows == 0:
            truth_tbl = None

    truth_ids_key: pa.Array | None = None
    truth_frames_tbl: pa.Table | None = None
    truth_frames_by_orbit: pa.Table | None = None
    if truth_tbl is not None and truth_tbl.num_rows > 0:
        # Distinct truth detections in the backend observation_id namespace.
        truth_ids = (
            truth_tbl.select(["orbit_id", "observation_id"])
            .group_by(["orbit_id", "observation_id"])
            .aggregate([])
        )
        sep = pa.scalar("|", type=pa.large_string())
        truth_ids_key = pc.binary_join_element_wise(
            pc.cast(truth_ids["orbit_id"], pa.large_string()),
            pc.cast(truth_ids["observation_id"], pa.large_string()),
            sep,
        )

        # Build truth frame keys by mapping truth observation_ids into the detections keyspace.
        truth_frame_keys: pa.Table | None = None
        for b in (enumerate_backend, *backends):
            if hasattr(b, "truth_frame_keys_from_truth_table"):
                try:
                    truth_frame_keys = b.truth_frame_keys_from_truth_table(truth=truth_tbl)  # type: ignore[attr-defined]
                    break
                except Exception:
                    truth_frame_keys = None

        if truth_frame_keys is not None and truth_frame_keys.num_rows > 0:
            truth_frame_keys_norm = pa.table(
                {
                    "orbit_id": pc.cast(truth_frame_keys["orbit_id"], pa.large_string()),
                    "obscode": pc.cast(truth_frame_keys["obscode"], pa.large_string()),
                    "exposure_mjd_mid_key_us": pc.cast(
                        truth_frame_keys["exposure_mjd_mid_key_us"], pa.int64()
                    ),
                    "healpixel": pc.cast(truth_frame_keys["healpixel"], pa.int64()),
                    "observation_id": pc.cast(
                        truth_frame_keys["observation_id"], pa.large_string()
                    ),
                }
            )
            truth_frames_tbl = (
                truth_frame_keys_norm.group_by(
                    ["orbit_id", "obscode", "exposure_mjd_mid_key_us", "healpixel"]
                )
                .aggregate([("observation_id", "count")])
                .rename_columns(
                    [
                        "orbit_id",
                        "obscode",
                        "exposure_mjd_mid_key_us",
                        "healpixel",
                        "n_truth_detections",
                    ]
                )
            )
            truth_frames_by_orbit = (
                truth_frames_tbl.group_by(["orbit_id"])
                .aggregate(
                    [
                        ("healpixel", "count"),
                        ("n_truth_detections", "sum"),
                    ]
                )
                .rename_columns(
                    [
                        "orbit_id",
                        "n_frames_truth_available",
                        "n_detections_truth_total",
                    ]
                )
            )

    # Dataset-wide candidate frame keys for this run's *target set*:
    #   (obscode, exposure_mjd_mid_key_us, healpixel)
    #
    # This is constant across orbits for the run, and is the reference set for
    # n_frames_geometry_matched / n_frames_geometry_rejected accounting.
    n_frames_candidates: int | None = None
    count_frames_candidates_elapsed_s: float | None = None
    dataset_pixels_by_target: dict[tuple[str, int], np.ndarray] | None = None

    # Production primitive (backend-owned): (obscode,key_us) -> array[healpixel]
    # Used to restrict footprint pixels to frames that actually exist in the dataset.
    if bool(getattr(getattr(enumerate_backend, "capabilities", None), "supports_frame_pixels_by_target", False)):
        try:
            t0_frames = time.perf_counter()
            dataset_pixels_by_target = enumerate_backend.frame_pixels_by_target(  # type: ignore[attr-defined]
                subset=subset,
                targets=targets,
            )
            count_frames_candidates_elapsed_s = float(time.perf_counter() - t0_frames)
            n_frames_candidates = int(
                sum(int(v.size) for v in (dataset_pixels_by_target or {}).values())
            )
        except Exception:
            n_frames_candidates = None
            count_frames_candidates_elapsed_s = None
            dataset_pixels_by_target = None

    # Fallback: count distinct frame keys in the full window (may include targets we didn't search).
    if n_frames_candidates is None and hasattr(enumerate_backend, "count_total_window_frames"):
        try:
            t_frames0 = time.perf_counter()
            n_frames_candidates = int(
                enumerate_backend.count_total_window_frames(  # type: ignore[attr-defined]
                    start_mjd_utc=float(start_mjd),
                    end_mjd_utc=float(end_mjd),
                    obscodes=tuple(workload.window.obscodes),
                )
            )
            count_frames_candidates_elapsed_s = float(time.perf_counter() - t_frames0)
        except Exception:
            n_frames_candidates = None
            count_frames_candidates_elapsed_s = None

    gate = build_gate_params(
        innovation_gate_n_sigma=(
            float(workload.n_sigma) if gate_n_sigma is None else float(gate_n_sigma)
        ),
        invalid_sigma_fill_floor_arcsec_global=det_sigma_floor_arcsec,
        invalid_sigma_fill_floor_arcsec_by_obscode=det_sigma_floor_arcsec_by_obscode,
        sigma_systematic_arcsec_by_obscode=det_sigma_sys_arcsec_by_obscode,
        apply_systematic_if_reported_rms_lt_arcsec_by_obscode=(
            det_sigma_sys_apply_rms_lt_arcsec_by_obscode
        ),
        max_mag_residual_fainter_mag=max_faint,
        max_mag_residual_brighter_mag=max_bright,
    )

    fp_name = str(workload.footprint)
    if fp_name == "cov_polygon_reconstructed_moc":
        footprint = CovPolygonReconstructedMoc(n_sigma=float(workload.n_sigma), pad_neighbors=False)
    elif fp_name == "cov_polygon_reconstructed_moc_pad_neighbors":
        footprint = CovPolygonReconstructedMoc(n_sigma=float(workload.n_sigma), pad_neighbors=True)
    else:
        raise ValueError(f"Unknown footprint: {workload.footprint!r}")

    workload_label = str(workload.window.label())
    months_csv = ",".join(workload.window.year_months)
    obscodes_csv = ",".join(workload.window.obscodes)

    # Stage23 + Stage4 outputs are written under a run directory so we can reuse Stage23
    # artifacts across backends and avoid holding large intermediates in the runner.
    from precovery.search.run import precover_orbits_with_backend_for_benchmark

    run_stamp = utc_now_iso().replace(":", "").replace(".", "")
    bench_run_dir = Path(workload.subset_dir) / "artifacts" / "bench_runs" / workload_label / run_stamp

    base: pa.Table | None = None

    # Prepare per-backend outputs.
    backend_rows: list[dict[str, object]] = []
    per_orbit_tables: list[pa.Table] = []

    for b in backends:
        bench = precover_orbits_with_backend_for_benchmark(
            backend=b,
            subset=subset,
            orbits=orbits_use,
            start_mjd_utc=float(start_mjd),
            end_mjd_utc=float(end_mjd),
            obscodes=tuple(map(str, workload.window.obscodes)),
            window_size_days=int(workload.window_size_days),
            stage2_strategy=str(workload.stage2_strategy),
            healpix_nside=int(workload.healpix_nside),
            footprint=footprint,
            max_processes=max_processes,
            limit_codefid_keys=limit_keys,
            limit_codefid_vals=limit_vals,
            faint_margin_mag=float(faint_margin),
            gate=gate,
            targets=targets,
            dataset_pixels_by_target=dataset_pixels_by_target,
            truth_frames=truth_frames_tbl,
            truth_ids_key=truth_ids_key,
            run_dir=bench_run_dir,
            reuse_stage23_artifacts=True,
            write_stage23_artifacts=True,
            write_stage4_artifacts_flag=True,
            detailed_timings=bool(detailed_timings),
            execution_mode=str(execution_mode),
            chunk_rows_stage23=int(chunk_rows_stage23),
            chunk_rows_stage4=int(chunk_rows_stage4),
            max_inflight_chunks=int(max_inflight_chunks),
            runtime_tmp_dir=(None if runtime_tmp_dir is None or str(runtime_tmp_dir).strip() == "" else Path(str(runtime_tmp_dir))),
            min_free_disk_gb=float(min_free_disk_gb),
            max_on_sky_sigma_major_arcsec=(
                None if max_on_sky_budget is None else float(max_on_sky_budget)
            ),
            preprop_viability_policy=str(preprop_cfg.policy),
            preprop_short_arc_days_threshold=float(preprop_cfg.short_arc_days_threshold),
            preprop_time_limit_days_short_arc=float(
                preprop_cfg.time_limit_days_short_arc
            ),
            preprop_time_limit_days_default=float(preprop_cfg.time_limit_days_default),
            preprop_max_sigma_r_over_r=(
                None
                if preprop_cfg.max_sigma_r_over_r is None
                else float(preprop_cfg.max_sigma_r_over_r)
            ),
            preprop_max_covariance_condition=(
                None
                if preprop_cfg.max_covariance_condition is None
                else float(preprop_cfg.max_covariance_condition)
            ),
            preprop_fail_open_on_scoring_error=bool(
                preprop_cfg.fail_open_on_scoring_error
            ),
        )

        if base is None:
            base = pa.table(
                {
                    "orbit_id": pa.array(
                        sorted({str(x) for x in orbit_ids_use}), type=pa.large_string()
                    )
                }
            )
            fm_tbl = bench.stage3_orbit_metrics.table
            if fm_tbl.num_rows > 0 and "orbit_id" in fm_tbl.column_names:
                base = base.join(fm_tbl, keys=["orbit_id"], join_type="left outer")
            for c in [
                "n_frames_geometry_matched",
                "n_frames_stage3_rejected_any",
                "n_frames_lim_mag_rejected",
                "n_frames_uncertainty_rejected",
                "n_frames_truth_geometry_matched",
                "n_frames_truth_stage3_rejected_any",
                "n_frames_lim_mag_truth_rejected",
                "n_frames_truth_uncertainty_rejected",
                "n_detections_truth_frame_candidates",
                "n_targets_preprop_viability_rejected",
                "n_targets_preprop_time_limited",
                "n_targets_failfast_dynamics_error",
                "n_targets_eval_total",
                "n_targets_eval_after_policy",
            ]:
                if c not in base.column_names:
                    base = base.append_column(c, pa.array([0] * base.num_rows, type=pa.int64()))
                else:
                    base = base.set_column(
                        base.schema.get_field_index(c),
                        c,
                        pc.cast(pc.fill_null(base[c], 0), pa.int64()),
                    )
            for c in ["completed_full_time_period_check"]:
                if c not in base.column_names:
                    base = base.append_column(c, pa.array([True] * base.num_rows, type=pa.bool_()))
                else:
                    base = base.set_column(
                        base.schema.get_field_index(c),
                        c,
                        pc.cast(pc.fill_null(base[c], True), pa.bool_()),
                    )
            for c in [
                "preprop_decision",
                "preprop_reason",
                "preprop_trigger_metric",
                "failfast_stage",
                "failfast_reason",
            ]:
                if c not in base.column_names:
                    base = base.append_column(c, pa.nulls(base.num_rows, type=pa.large_string()))
            for c in [
                "preprop_trigger_value",
                "preprop_trigger_threshold",
                "preprop_time_limit_days_applied",
                "preprop_first_excluded_target_mjd_utc",
                "failfast_time_mjd_tdb",
                "failfast_t0_mjd_tdb",
                "failfast_t1_mjd_tdb",
                "failfast_dt_days",
            ]:
                if c not in base.column_names:
                    base = base.append_column(c, pa.nulls(base.num_rows, type=pa.float64()))

            if truth_frames_by_orbit is not None and truth_frames_by_orbit.num_rows > 0:
                base = base.join(truth_frames_by_orbit, keys=["orbit_id"], join_type="left outer")
            for c in ["n_frames_truth_available", "n_detections_truth_total"]:
                if c not in base.column_names:
                    base = base.append_column(c, pa.array([0] * base.num_rows, type=pa.int64()))
                else:
                    base = base.set_column(
                        base.schema.get_field_index(c),
                        c,
                        pc.cast(pc.fill_null(base[c], 0), pa.int64()),
                    )

            if n_frames_candidates is None:
                frames_candidates_arr = pa.nulls(base.num_rows, type=pa.int64())
            else:
                frames_candidates_arr = pa.array([int(n_frames_candidates)] * base.num_rows, type=pa.int64())
            base = base.append_column("n_frames_candidates", frames_candidates_arr)

            n_geom = pc.cast(base["n_frames_geometry_matched"], pa.int64())
            n_stage3_rej = pc.cast(base["n_frames_stage3_rejected_any"], pa.int64())
            n_truth_geom = pc.cast(base["n_frames_truth_geometry_matched"], pa.int64())
            n_truth_stage3_rej = pc.cast(base["n_frames_truth_stage3_rejected_any"], pa.int64())
            n_final = pc.subtract(n_geom, n_stage3_rej)
            n_truth_final = pc.subtract(n_truth_geom, n_truth_stage3_rej)
            base = base.append_column("n_frames_final", pc.cast(n_final, pa.int64()))
            base = base.append_column("n_frames_truth_final", pc.cast(n_truth_final, pa.int64()))

            if n_frames_candidates is None:
                base = base.append_column("n_frames_geometry_rejected", pa.nulls(base.num_rows, type=pa.int64()))
            else:
                base = base.append_column(
                    "n_frames_geometry_rejected",
                    pc.subtract(pc.cast(frames_candidates_arr, pa.int64()), n_geom),
                )

        det = bench.stage4_orbit_metrics.table

        # Join Stage-4 detection metrics onto the Stage-3 per-orbit base.
        out = base.join(det, keys=["orbit_id"], join_type="left outer")  # type: ignore[union-attr]
        for c in [
            "n_detections_candidates",
            "n_detections_gate_matched",
            "n_detections_innov_ellipse_rejected",
            "n_detections_magnitude_rejected",
            "n_detections_truth_gate_matched",
            "n_detections_truth_magnitude_rejected",
            "n_detections_truth_final",
            "n_detections_false_positive_final",
            "n_detections_unknown_final",
        ]:
            if c in out.column_names:
                out = out.set_column(
                    out.schema.get_field_index(c),
                    c,
                    pc.cast(pc.fill_null(out[c], 0), pa.int64()),
                )
            else:
                out = out.append_column(c, pa.array([0] * out.num_rows, type=pa.int64()))

        # Populate remaining truth columns.
        if "n_detections_truth_frame_candidates" not in out.column_names:
            out = out.append_column(
                "n_detections_truth_frame_candidates", pa.array([0] * out.num_rows, type=pa.int64())
            )
        else:
            out = out.set_column(
                out.schema.get_field_index("n_detections_truth_frame_candidates"),
                "n_detections_truth_frame_candidates",
                pc.cast(pc.fill_null(out["n_detections_truth_frame_candidates"], 0), pa.int64()),
            )

        # Add workload/backend labels.
        keep_cols = [
            "orbit_id",
            # Stage-3 frames.
            "n_frames_candidates",
            "n_frames_geometry_matched",
            "n_frames_geometry_rejected",
            "n_frames_truth_available",
            "n_frames_truth_geometry_matched",
            "n_frames_stage3_rejected_any",
            "n_frames_truth_stage3_rejected_any",
            "n_frames_lim_mag_rejected",
            "n_frames_lim_mag_truth_rejected",
            "n_frames_uncertainty_rejected",
            "n_frames_truth_uncertainty_rejected",
            "n_frames_final",
            "n_frames_truth_final",
            "n_targets_preprop_viability_rejected",
            "n_targets_preprop_time_limited",
            "n_targets_failfast_dynamics_error",
            "n_targets_eval_total",
            "n_targets_eval_after_policy",
            "completed_full_time_period_check",
            "preprop_decision",
            "preprop_reason",
            "preprop_trigger_metric",
            "preprop_trigger_value",
            "preprop_trigger_threshold",
            "preprop_time_limit_days_applied",
            "preprop_first_excluded_target_mjd_utc",
            "failfast_stage",
            "failfast_reason",
            "failfast_time_mjd_tdb",
            "failfast_t0_mjd_tdb",
            "failfast_t1_mjd_tdb",
            "failfast_dt_days",
            # Stage-4 detections.
            "n_detections_candidates",
            "n_detections_gate_matched",
            "n_detections_innov_ellipse_rejected",
            "n_detections_magnitude_rejected",
            "n_detections_truth_total",
            "n_detections_truth_frame_candidates",
            "n_detections_truth_gate_matched",
            "n_detections_truth_magnitude_rejected",
            "n_detections_truth_final",
            "n_detections_false_positive_final",
            "n_detections_unknown_final",
        ]
        out = out.select([c for c in keep_cols if c in out.column_names])

        # Per-orbit truth loss attribution (stable derived columns).
        # These are purely derived from existing per-orbit truth counts.
        def _i64(col: str) -> pa.Array:
            if col not in out.column_names:
                return pa.array([0] * out.num_rows, type=pa.int64())
            return pc.cast(pc.fill_null(out[col], 0), pa.int64())

        miss_stage3_geom_frames = pc.subtract(
            _i64("n_frames_truth_available"), _i64("n_frames_truth_geometry_matched")
        )
        miss_stage3_lim_mag_frames = _i64("n_frames_lim_mag_truth_rejected")
        miss_stage4_innov = pc.subtract(
            _i64("n_detections_truth_frame_candidates"),
            _i64("n_detections_truth_gate_matched"),
        )
        miss_truth_total = pc.subtract(_i64("n_detections_truth_total"), _i64("n_detections_truth_final"))

        out = out.append_column("miss_stage3_geom_frames", pc.cast(miss_stage3_geom_frames, pa.int64()))
        out = out.append_column("miss_stage3_lim_mag_frames", pc.cast(miss_stage3_lim_mag_frames, pa.int64()))
        out = out.append_column("miss_stage4_innov", pc.cast(miss_stage4_innov, pa.int64()))
        out = out.append_column("miss_truth_total", pc.cast(miss_truth_total, pa.int64()))

        # When using ASSIST: attach perturber warning for orbits that are ASSIST perturbers (e.g. Pluto).
        if "assist" in str(workload.stage2_strategy).lower():
            orbit_ids = out["orbit_id"].to_pylist()
            perturber_warnings = perturber_warnings_for_orbit_ids(orbit_ids)
            out = out.append_column(
                "perturber_warning",
                pa.array(perturber_warnings, type=pa.large_string()),
            )
        else:
            out = out.append_column(
                "perturber_warning",
                pa.nulls(out.num_rows, type=pa.large_string()),
            )

        out = out.append_column(
            "workload", pa.array([workload_label] * out.num_rows, type=pa.large_string())
        )
        out = out.append_column(
            "backend", pa.array([str(b.name)] * out.num_rows, type=pa.large_string())
        )
        per_orbit_tables.append(out)

        # Backend-level totals (aggregate from per-orbit).
        def _sum_i64(col: str) -> int | None:
            if col not in out.column_names:
                return None
            s = pc.sum(pc.cast(pc.fill_null(out[col], 0), pa.int64()))
            return None if s is None or s.as_py() is None else int(s.as_py())

        n_truth_total = _sum_i64("n_detections_truth_total") or 0
        n_truth_final = _sum_i64("n_detections_truth_final") or 0
        recall = None if n_truth_total <= 0 else float(n_truth_final) / float(n_truth_total)

        n_tf_total = _sum_i64("n_frames_truth_available") or 0
        n_tf_final = _sum_i64("n_frames_truth_final") or 0
        coverage = None if n_tf_total <= 0 else float(n_tf_final) / float(n_tf_total)

        gate_mag_elapsed_s = None
        gate_innov_elapsed_s = None
        if bool(detailed_timings):
            gate_mag_elapsed_s = float(bench.gate_micro_timings.get("gate.mag_residual_elapsed_s", 0.0))
            gate_innov_elapsed_s = float(bench.gate_micro_timings.get("gate.innov_ellipse_elapsed_s", 0.0))

        if bool(detailed_timings):
            mt = dict(bench.build_micro_timings)
            stage2_window_centers_elapsed_s: float | None = float(mt.get("stage2.window_centers_elapsed_s", 0.0))
            stage2_assist_propagate_centers_elapsed_s: float | None = float(
                mt.get("stage2.assist_propagate_centers_elapsed_s", 0.0)
            )
            stage2_sigma_point_variants_elapsed_s: float | None = float(
                mt.get("stage2.sigma_point_variants_elapsed_s", 0.0)
            )
            stage2_propagate2body_nominal_elapsed_s: float | None = float(
                mt.get("stage2.propagate2body_nominal_elapsed_s", 0.0)
            )
            stage2_ephemeris_nominal_elapsed_s: float | None = float(
                mt.get("stage2.ephemeris_nominal_elapsed_s", 0.0)
            )
            stage2_propagate2body_variants_elapsed_s: float | None = float(
                mt.get("stage2.propagate2body_variants_elapsed_s", 0.0)
            )
            stage2_ephemeris_variants_elapsed_s: float | None = float(
                mt.get("stage2.ephemeris_variants_elapsed_s", 0.0)
            )
            stage2_variant_collapse_elapsed_s: float | None = float(
                mt.get("stage2.variant_collapse_elapsed_s", 0.0)
            )
            stage2_variant_lonlat_elapsed_s: float | None = float(
                mt.get("stage2.variant_lonlat_elapsed_s", 0.0)
            )
            stage2_reconstruct_cov_elapsed_s: float | None = float(
                mt.get("stage2.reconstruct_cov_elapsed_s", 0.0)
            )
            stage2_variant_lonlat_fallback_count: int | None = int(
                float(mt.get("stage2.variant_lonlat_fallback_count", 0.0))
            )

            fp_vertices_elapsed_s: float | None = float(mt.get("footprint.vertices_elapsed_s", 0.0))
            fp_rasterize_elapsed_s: float | None = float(mt.get("footprint.rasterize_elapsed_s", 0.0))
            fp_pad_neighbors_elapsed_s: float | None = float(
                mt.get("footprint.pad_neighbors_elapsed_s", 0.0)
            )
        else:
            stage2_window_centers_elapsed_s = None
            stage2_assist_propagate_centers_elapsed_s = None
            stage2_sigma_point_variants_elapsed_s = None
            stage2_propagate2body_nominal_elapsed_s = None
            stage2_ephemeris_nominal_elapsed_s = None
            stage2_propagate2body_variants_elapsed_s = None
            stage2_ephemeris_variants_elapsed_s = None
            stage2_variant_collapse_elapsed_s = None
            stage2_variant_lonlat_elapsed_s = None
            stage2_reconstruct_cov_elapsed_s = None
            stage2_variant_lonlat_fallback_count = None
            fp_vertices_elapsed_s = None
            fp_rasterize_elapsed_s = None
            fp_pad_neighbors_elapsed_s = None

        backend_rows.append(
            dict(
                workload=workload_label,
                footprint=str(workload.footprint),
                backend=str(b.name),
                bench_run_stamp=str(run_stamp),
                bench_run_dir=str(bench_run_dir),
                subset_dir=str(workload.subset_dir),
                detections_parquet=(
                    None
                    if getattr(b, "parquet_path", None) is None
                    else str(getattr(b, "parquet_path"))
                ),
                months_csv=months_csv,
                obscodes_csv=obscodes_csv,
                stage2_strategy=str(workload.stage2_strategy),
                orbits_parquet=str(orbits_parquet),
                truth_parquet=None if truth_path is None else str(truth_path),
                targets_from_truth=bool(targets_from_truth),
                gate_n_sigma=float(gate.innovation_gate_n_sigma),
                det_sigma_floor_arcsec=float(gate.invalid_sigma_fill_floor_arcsec_global),
                det_sigma_floor_arcsec_by_obscode=(
                    None
                    if gate.invalid_sigma_fill_floor_arcsec_by_obscode is None
                    else json.dumps(dict(gate.invalid_sigma_fill_floor_arcsec_by_obscode), sort_keys=True)
                ),
                det_sigma_sys_arcsec_by_obscode=(
                    None
                    if gate.sigma_systematic_arcsec_by_obscode is None
                    else json.dumps(dict(gate.sigma_systematic_arcsec_by_obscode), sort_keys=True)
                ),
                det_sigma_sys_apply_rms_lt_arcsec_by_obscode=(
                    None
                    if gate.apply_systematic_if_reported_rms_lt_arcsec_by_obscode is None
                    else json.dumps(
                        dict(gate.apply_systematic_if_reported_rms_lt_arcsec_by_obscode), sort_keys=True
                    )
                ),
                faint_frame_skip_margin_mag=float(faint_margin),
                max_mag_residual_fainter_mag=None if max_faint is None else float(max_faint),
                max_mag_residual_brighter_mag=None if max_bright is None else float(max_bright),
                max_on_sky_sigma_major_arcsec=(
                    None if max_on_sky_budget is None else float(max_on_sky_budget)
                ),
                max_orbits=None if max_orbits is None else int(max_orbits),
                max_targets=None if max_targets is None else int(max_targets),
                max_processes=None if max_processes is None else int(max_processes),
                n_orbits=int(len(orbits_use)),
                n_targets=int(len(targets)),
                n_frames_candidates=None if n_frames_candidates is None else int(n_frames_candidates),
                n_total_window_frames=None if n_frames_candidates is None else int(n_frames_candidates),
                count_total_window_frames_elapsed_s=(
                    None
                    if count_frames_candidates_elapsed_s is None
                    else float(count_frames_candidates_elapsed_s)
                ),
                n_orbits_cov_repaired=int(n_cov_repaired),
                n_orbits_cov_rejected=int(n_cov_rejected),
                n_frames_geometry_matched=_sum_i64("n_frames_geometry_matched"),
                n_frames_geometry_rejected=_sum_i64("n_frames_geometry_rejected"),
                n_frames_truth_available=_sum_i64("n_frames_truth_available"),
                n_frames_truth_geometry_matched=_sum_i64("n_frames_truth_geometry_matched"),
                n_frames_stage3_rejected_any=_sum_i64("n_frames_stage3_rejected_any"),
                n_frames_truth_stage3_rejected_any=_sum_i64("n_frames_truth_stage3_rejected_any"),
                n_frames_lim_mag_rejected=_sum_i64("n_frames_lim_mag_rejected"),
                n_frames_lim_mag_truth_rejected=_sum_i64("n_frames_lim_mag_truth_rejected"),
                n_frames_uncertainty_rejected=_sum_i64("n_frames_uncertainty_rejected"),
                n_frames_truth_uncertainty_rejected=_sum_i64("n_frames_truth_uncertainty_rejected"),
                n_frames_final=_sum_i64("n_frames_final"),
                n_frames_truth_final=_sum_i64("n_frames_truth_final"),
                n_unique_frames_selected=int(bench.n_unique_frames_selected),
                select_frames_elapsed_s=float(bench.build_footprint_elapsed_s + bench.build_triples_elapsed_s),
                n_frames_skipped_limiting_mag=int(bench.n_frames_skipped_limiting_mag),
                n_frames_skipped_uncertainty=int(bench.n_frames_skipped_uncertainty),
                n_targets_preprop_viability_rejected=_sum_i64(
                    "n_targets_preprop_viability_rejected"
                ),
                n_targets_preprop_time_limited=_sum_i64(
                    "n_targets_preprop_time_limited"
                ),
                n_targets_failfast_dynamics_error=_sum_i64(
                    "n_targets_failfast_dynamics_error"
                ),
                build_elapsed_s=float(bench.build_elapsed_s),
                observer_creation_elapsed_s=float(bench.build_observers_elapsed_s),
                propagation_elapsed_s=float(bench.build_predict_elapsed_s),
                build_pred_mag_elapsed_s=float(bench.build_pred_mag_elapsed_s),
                build_footprint_elapsed_s=float(bench.build_footprint_elapsed_s),
                build_triples_elapsed_s=float(bench.build_triples_elapsed_s),
                stage2_window_centers_elapsed_s=stage2_window_centers_elapsed_s,
                stage2_assist_propagate_centers_elapsed_s=stage2_assist_propagate_centers_elapsed_s,
                stage2_sigma_point_variants_elapsed_s=stage2_sigma_point_variants_elapsed_s,
                stage2_propagate2body_nominal_elapsed_s=stage2_propagate2body_nominal_elapsed_s,
                stage2_ephemeris_nominal_elapsed_s=stage2_ephemeris_nominal_elapsed_s,
                stage2_propagate2body_variants_elapsed_s=stage2_propagate2body_variants_elapsed_s,
                stage2_ephemeris_variants_elapsed_s=stage2_ephemeris_variants_elapsed_s,
                stage2_variant_collapse_elapsed_s=stage2_variant_collapse_elapsed_s,
                stage2_variant_lonlat_elapsed_s=stage2_variant_lonlat_elapsed_s,
                stage2_reconstruct_cov_elapsed_s=stage2_reconstruct_cov_elapsed_s,
                stage2_variant_lonlat_fallback_count=stage2_variant_lonlat_fallback_count,
                footprint_vertices_elapsed_s=fp_vertices_elapsed_s,
                footprint_rasterize_elapsed_s=fp_rasterize_elapsed_s,
                footprint_pad_neighbors_elapsed_s=fp_pad_neighbors_elapsed_s,
                gate_mag_residual_elapsed_s=gate_mag_elapsed_s,
                gate_innov_ellipse_elapsed_s=gate_innov_elapsed_s,
                backend_elapsed_s=float(bench.backend_elapsed_s),
                bytes_estimate=None if bench.bytes_estimate is None else int(bench.bytes_estimate),
                n_matched_frames=bench.n_matched_frames,
                n_detections_candidates=_sum_i64("n_detections_candidates"),
                n_detections_gate_matched=_sum_i64("n_detections_gate_matched"),
                n_detections_innov_ellipse_rejected=_sum_i64(
                    "n_detections_innov_ellipse_rejected"
                ),
                n_detections_magnitude_rejected=_sum_i64("n_detections_magnitude_rejected"),
                n_detections_selected_exposure_keys=(
                    None
                    if bench.n_detections_selected_exposure_keys is None
                    else int(bench.n_detections_selected_exposure_keys)
                ),
                n_detections_selected_frame_keys=(
                    None
                    if bench.n_detections_selected_frame_keys is None
                    else int(bench.n_detections_selected_frame_keys)
                ),
                n_detections_healpixel_nonmatch=(
                    None
                    if bench.n_detections_healpixel_nonmatch is None
                    else int(bench.n_detections_healpixel_nonmatch)
                ),
                n_detections_truth_total=_sum_i64("n_detections_truth_total"),
                n_detections_truth_frame_candidates=_sum_i64("n_detections_truth_frame_candidates"),
                n_detections_truth_gate_matched=_sum_i64("n_detections_truth_gate_matched"),
                n_detections_truth_magnitude_rejected=_sum_i64("n_detections_truth_magnitude_rejected"),
                n_detections_truth_final=_sum_i64("n_detections_truth_final"),
                n_detections_false_positive_final=_sum_i64("n_detections_false_positive_final"),
                n_detections_unknown_final=_sum_i64("n_detections_unknown_final"),
                gate_n_candidates=(
                    int(bench.gate_totals.n_candidates)
                    if bool(compute_gate_totals) or truth_tbl is not None
                    else None
                ),
                gate_n_accepted=(
                    int(bench.gate_totals.n_accepted) if bool(compute_gate_totals) or truth_tbl is not None else None
                ),
                gate_n_rejected_innov_ellipse=(
                    int(bench.gate_totals.n_rejected_innov_ellipse)
                    if bool(compute_gate_totals) or truth_tbl is not None
                    else None
                ),
                gate_n_rejected_mag_residual=(
                    int(bench.gate_totals.n_rejected_mag_residual)
                    if bool(compute_gate_totals) or truth_tbl is not None
                    else None
                ),
                truth_recall=recall,
                truth_frame_coverage=coverage,
                orbit_ids_assist_perturber_warning=(
                    json.dumps([
                        out["orbit_id"][i].as_py()
                        for i in range(out.num_rows)
                        if out["perturber_warning"][i].as_py() is not None
                    ])
                    if "perturber_warning" in out.column_names
                    and any(out["perturber_warning"][i].as_py() is not None for i in range(out.num_rows))
                    else None
                ),
            )
        )

    backend_rows_tbl = BackendRunResults.from_pyarrow(pa.Table.from_pylist(backend_rows))
    per_orbit = (
        None
        if not per_orbit_tables
        else PerOrbitRunResults.from_pyarrow(
            pa.concat_tables(per_orbit_tables, promote_options="default")
        )
    )
    return backend_rows_tbl, per_orbit
