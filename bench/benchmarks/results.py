from __future__ import annotations

import quivr as qv


class BackendRunResults(qv.Table):
    """
    One row per backend run, with run-level fields repeated for easy filtering/aggregation.

    Column names are intended to match `BENCHMARK_SPECS.md` canonical metrics wherever applicable.
    """

    workload = qv.LargeStringColumn()
    footprint = qv.LargeStringColumn()
    backend = qv.LargeStringColumn()
    bench_run_stamp = qv.LargeStringColumn()
    bench_run_dir = qv.LargeStringColumn()
    subset_dir = qv.LargeStringColumn()
    detections_parquet = qv.LargeStringColumn(nullable=True)
    months_csv = qv.LargeStringColumn()
    obscodes_csv = qv.LargeStringColumn()
    stage2_strategy = qv.LargeStringColumn()
    orbits_parquet = qv.LargeStringColumn()
    truth_parquet = qv.LargeStringColumn(nullable=True)
    targets_from_truth = qv.BooleanColumn()
    gate_n_sigma = qv.Float64Column()
    det_sigma_floor_arcsec = qv.Float64Column()
    det_sigma_floor_arcsec_by_obscode = qv.LargeStringColumn(nullable=True)
    det_sigma_sys_arcsec_by_obscode = qv.LargeStringColumn(nullable=True)
    det_sigma_sys_apply_rms_lt_arcsec_by_obscode = qv.LargeStringColumn(nullable=True)
    faint_frame_skip_margin_mag = qv.Float64Column()
    max_mag_residual_fainter_mag = qv.Float64Column(nullable=True)
    max_mag_residual_brighter_mag = qv.Float64Column(nullable=True)
    max_orbits = qv.Int64Column(nullable=True)
    max_targets = qv.Int64Column(nullable=True)
    max_processes = qv.Int64Column(nullable=True)

    n_orbits = qv.Int64Column()
    n_targets = qv.Int64Column()

    # Dataset-wide frame keys in the window (constant per orbit).
    n_frames_candidates = qv.Int64Column(nullable=True)

    # Stage-3 frame accounting aggregated across orbits (sums of per-orbit counts).
    n_frames_geometry_matched = qv.Int64Column(nullable=True)
    n_frames_geometry_rejected = qv.Int64Column(nullable=True)
    n_frames_truth_available = qv.Int64Column(nullable=True)
    n_frames_truth_geometry_matched = qv.Int64Column(nullable=True)
    n_frames_lim_mag_rejected = qv.Int64Column(nullable=True)
    n_frames_lim_mag_truth_rejected = qv.Int64Column(nullable=True)
    n_frames_final = qv.Int64Column(nullable=True)
    n_frames_truth_final = qv.Int64Column(nullable=True)

    # Total unique frame keys available in the dataset for this window.
    # Frame key = (obscode, exposure_mjd_mid_key_us, healpixel).
    n_total_window_frames = qv.Int64Column(nullable=True)  # legacy synonym for n_frames_candidates
    count_total_window_frames_elapsed_s = qv.Float64Column(nullable=True)

    # Orbit covariance QC (Stage 2 eligibility).
    n_orbits_cov_repaired = qv.Int64Column(nullable=True)
    n_orbits_cov_rejected = qv.Int64Column(nullable=True)

    # Legacy/diagnostic: unique selected frame keys across all orbits (drops orbit multiplicity).
    n_unique_frames_selected = qv.Int64Column(nullable=True)
    select_frames_elapsed_s = qv.Float64Column()
    n_frames_skipped_limiting_mag = qv.Int64Column()

    # Build timing breakdown.
    build_elapsed_s = qv.Float64Column()
    observer_creation_elapsed_s = qv.Float64Column()
    propagation_elapsed_s = qv.Float64Column()
    build_pred_mag_elapsed_s = qv.Float64Column()
    build_footprint_elapsed_s = qv.Float64Column()
    build_triples_elapsed_s = qv.Float64Column()

    # Optional micro-timings (when enabled in the benchmark runner).
    stage2_window_centers_elapsed_s = qv.Float64Column(nullable=True)
    stage2_assist_propagate_centers_elapsed_s = qv.Float64Column(nullable=True)
    stage2_sigma_point_variants_elapsed_s = qv.Float64Column(nullable=True)
    stage2_propagate2body_nominal_elapsed_s = qv.Float64Column(nullable=True)
    stage2_ephemeris_nominal_elapsed_s = qv.Float64Column(nullable=True)
    stage2_propagate2body_variants_elapsed_s = qv.Float64Column(nullable=True)
    stage2_ephemeris_variants_elapsed_s = qv.Float64Column(nullable=True)
    stage2_variant_collapse_elapsed_s = qv.Float64Column(nullable=True)
    stage2_variant_lonlat_elapsed_s = qv.Float64Column(nullable=True)
    stage2_reconstruct_cov_elapsed_s = qv.Float64Column(nullable=True)
    stage2_variant_lonlat_fallback_count = qv.Int64Column(nullable=True)
    footprint_vertices_elapsed_s = qv.Float64Column(nullable=True)
    footprint_rasterize_elapsed_s = qv.Float64Column(nullable=True)
    footprint_pad_neighbors_elapsed_s = qv.Float64Column(nullable=True)
    gate_mag_residual_elapsed_s = qv.Float64Column(nullable=True)
    gate_innov_ellipse_elapsed_s = qv.Float64Column(nullable=True)

    # Backend join + gate time.
    backend_elapsed_s = qv.Float64Column()
    bytes_estimate = qv.Int64Column(nullable=True)

    # Frame join coverage proxy.
    n_matched_frames = qv.Int64Column(nullable=True)

    # Detection-level counts (Stage 4).
    n_detections_candidates = qv.Int64Column(nullable=True)
    n_detections_gate_matched = qv.Int64Column(nullable=True)
    n_detections_magnitude_rejected = qv.Int64Column(nullable=True)

    n_detections_truth_total = qv.Int64Column(nullable=True)
    n_detections_truth_frame_candidates = qv.Int64Column(nullable=True)
    n_detections_truth_gate_matched = qv.Int64Column(nullable=True)
    n_detections_truth_magnitude_rejected = qv.Int64Column(nullable=True)
    n_detections_truth_final = qv.Int64Column(nullable=True)

    n_detections_false_positive_final = qv.Int64Column(nullable=True)
    n_detections_unknown_final = qv.Int64Column(nullable=True)

    # Optional detailed gate totals (debugging / analysis).
    gate_n_candidates = qv.Int64Column(nullable=True)
    gate_n_accepted = qv.Int64Column(nullable=True)
    gate_n_rejected_innov_ellipse = qv.Int64Column(nullable=True)
    gate_n_rejected_mag_residual = qv.Int64Column(nullable=True)

    # Truth scoring.
    # Retained derived ratios for convenience in summaries.
    truth_recall = qv.Float64Column(nullable=True)
    truth_frame_coverage = qv.Float64Column(nullable=True)

    # When using ASSIST: orbit_ids that matched ASSIST perturbers (e.g. Pluto); results invalid for those.
    orbit_ids_assist_perturber_warning = qv.LargeStringColumn(nullable=True)


class PerOrbitRunResults(qv.Table):
    """
    One row per (workload, backend, orbit_id), used for later by-bin aggregation.
    """

    orbit_id = qv.LargeStringColumn()

    # Stage-3 frame metrics (per-orbit).
    n_frames_candidates = qv.Int64Column(nullable=True)
    n_frames_geometry_matched = qv.Int64Column(nullable=True)
    n_frames_geometry_rejected = qv.Int64Column(nullable=True)
    n_frames_truth_available = qv.Int64Column(nullable=True)
    n_frames_truth_geometry_matched = qv.Int64Column(nullable=True)
    n_frames_lim_mag_rejected = qv.Int64Column(nullable=True)
    n_frames_lim_mag_truth_rejected = qv.Int64Column(nullable=True)
    n_frames_final = qv.Int64Column(nullable=True)
    n_frames_truth_final = qv.Int64Column(nullable=True)

    # Stage-4 detection metrics (per-orbit).
    n_detections_candidates = qv.Int64Column(nullable=True)
    n_detections_gate_matched = qv.Int64Column(nullable=True)
    n_detections_magnitude_rejected = qv.Int64Column(nullable=True)

    # Truth detection metrics (per-orbit).
    n_detections_truth_total = qv.Int64Column(nullable=True)
    n_detections_truth_frame_candidates = qv.Int64Column(nullable=True)
    n_detections_truth_gate_matched = qv.Int64Column(nullable=True)
    n_detections_truth_magnitude_rejected = qv.Int64Column(nullable=True)
    n_detections_truth_final = qv.Int64Column(nullable=True)

    # Final false-positive accounting (per-orbit).
    n_detections_false_positive_final = qv.Int64Column(nullable=True)
    n_detections_unknown_final = qv.Int64Column(nullable=True)

    # Derived truth loss attribution (per-orbit).
    miss_stage3_geom_frames = qv.Int64Column(nullable=True)
    miss_stage3_lim_mag_frames = qv.Int64Column(nullable=True)
    miss_stage4_innov = qv.Int64Column(nullable=True)
    miss_truth_total = qv.Int64Column(nullable=True)

    workload = qv.LargeStringColumn()
    backend = qv.LargeStringColumn()

    # When using ASSIST propagation: warning if this orbit is an ASSIST perturber (e.g. Pluto).
    perturber_warning = qv.LargeStringColumn(nullable=True)
