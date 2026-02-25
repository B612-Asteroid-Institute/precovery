from __future__ import annotations

from dataclasses import dataclass

import pyarrow as pa
import quivr as qv


@dataclass
class SearchAgg:
    """
    Lightweight, opt-in performance counters and timings for `precovery.search`.

    Design goals
    ------------
    - Extremely low overhead when disabled (callers pass `None`).
    - When enabled, only cheap integer increments + a handful of timers.
    - Fields are intentionally aligned with Stage 4 harness semantics:
      frames_loaded, observations_loaded, accepted, and IO/prep/filter timings.
    """

    # counts (volume)
    n_targets: int = 0
    n_predicted_rows: int = 0
    n_predicted_pixels: int = 0
    # Distinct (obscode, exposure_mjd_mid, healpixel) keys requested by Stage 3.
    # (Production note: this is about "selected" sky positions, not necessarily observed frames.)
    n_selected_frames_unique: int = 0
    n_frames_joined: int = 0
    # Distinct observed frame keys returned by the DB join (drops dataset_id multiplicity).
    n_matched_frames_unique: int = 0
    n_frames_loaded: int = 0
    n_frames_faint_skipped: int = 0
    n_observations_loaded: int = 0
    n_accepted: int = 0
    n_candidates: int = 0
    # Gate rejection totals (detection-level).
    n_rejected_innov_ellipse: int = 0
    n_rejected_tolerance: int = 0
    n_rejected_mag_residual: int = 0

    # timings (seconds)
    enum_sec: float = 0.0
    propagate_sec: float = 0.0
    footprint_sec: float = 0.0
    join_frames_sec: float = 0.0
    io_sec: float = 0.0
    filter_sec: float = 0.0
    photometry_sec: float = 0.0
    total_sec: float = 0.0

    # Optional per-target debug rows (OFF by default; can be large).
    per_target_rows: list[dict[str, object]] | None = None
    # Optional micro-timing accumulator (OFF by default; aligns with benchmark timing keys).
    micro_timings: dict[str, float] | None = None

    def enable_per_target(self) -> None:
        """
        Enable per-target metrics collection.
        """
        if self.per_target_rows is None:
            self.per_target_rows = []

    def enable_micro_timings(self) -> None:
        """
        Enable micro-timing accumulation (Stage 2/3/4 sub-steps).
        """
        if self.micro_timings is None:
            self.micro_timings = {}


class SearchMetrics(qv.Table):
    """
    One-row, run-level metrics table for production precovery search.

    This mirrors the Stage 4 harness idea: aggregated counts + coarse timings.
    """

    orbit_id = qv.LargeStringColumn(nullable=True)
    propagation_strategy = qv.LargeStringColumn()
    footprint = qv.LargeStringColumn()
    healpix_nside = qv.Int64Column()
    n_sigma = qv.Float64Column()
    window_size_days = qv.Int64Column()
    target_chunk_size = qv.Int64Column()
    max_processes = qv.Int64Column(nullable=True)

    n_targets = qv.Int64Column()
    n_predicted_rows = qv.Int64Column()
    n_predicted_pixels = qv.Int64Column()
    n_selected_frames_unique = qv.Int64Column()
    n_frames_joined = qv.Int64Column()
    n_matched_frames_unique = qv.Int64Column()
    n_frames_loaded = qv.Int64Column()
    n_frames_faint_skipped = qv.Int64Column()
    n_observations_loaded = qv.Int64Column()
    n_accepted = qv.Int64Column()
    n_candidates = qv.Int64Column()
    n_rejected_innov_ellipse = qv.Int64Column()
    n_rejected_tolerance = qv.Int64Column()
    n_rejected_mag_residual = qv.Int64Column()

    enum_sec = qv.Float64Column()
    propagate_sec = qv.Float64Column()
    footprint_sec = qv.Float64Column()
    join_frames_sec = qv.Float64Column()
    io_sec = qv.Float64Column()
    filter_sec = qv.Float64Column()
    photometry_sec = qv.Float64Column()
    total_sec = qv.Float64Column()

    # Optional micro-timings (nullable).
    stage2_window_centers_elapsed_s = qv.Float64Column(nullable=True)
    stage2_assist_propagate_centers_elapsed_s = qv.Float64Column(nullable=True)
    stage2_sigma_point_variants_elapsed_s = qv.Float64Column(nullable=True)
    stage2_propagate2body_nominal_elapsed_s = qv.Float64Column(nullable=True)
    stage2_ephemeris_nominal_elapsed_s = qv.Float64Column(nullable=True)
    stage2_propagate2body_variants_elapsed_s = qv.Float64Column(nullable=True)
    stage2_variant_lonlat_elapsed_s = qv.Float64Column(nullable=True)
    stage2_reconstruct_cov_elapsed_s = qv.Float64Column(nullable=True)
    stage2_variant_lonlat_fallback_count = qv.Int64Column(nullable=True)
    footprint_vertices_elapsed_s = qv.Float64Column(nullable=True)
    footprint_rasterize_elapsed_s = qv.Float64Column(nullable=True)
    footprint_pad_neighbors_elapsed_s = qv.Float64Column(nullable=True)
    gate_mag_residual_elapsed_s = qv.Float64Column(nullable=True)
    gate_innov_ellipse_elapsed_s = qv.Float64Column(nullable=True)


class SearchPerTarget(qv.Table):
    """
    Optional per-target rollup (OFF by default).

    This is the production analogue of Stage 4's `per_target.parquet`: it lets us
    identify scan-volume outliers (e.g. QU127-style behavior) without running the
    entire experiments harness.
    """

    orbit_id = qv.LargeStringColumn(nullable=True)
    obscode = qv.LargeStringColumn()
    exposure_mjd_mid = qv.Float64Column()
    target_row = qv.Int64Column()

    n_predicted_pixels = qv.Int64Column()
    n_frames_joined = qv.Int64Column()
    n_frames_loaded = qv.Int64Column()
    n_observations_loaded = qv.Int64Column()
    n_accepted = qv.Int64Column()

    footprint_sec = qv.Float64Column(nullable=True)
    io_sec = qv.Float64Column(nullable=True)
    filter_sec = qv.Float64Column(nullable=True)


def metrics_row(
    *,
    orbit_id: str | None,
    propagation_strategy: str,
    footprint: str,
    healpix_nside: int,
    n_sigma: float,
    window_size_days: int,
    target_chunk_size: int,
    max_processes: int | None,
    agg: SearchAgg,
) -> SearchMetrics:
    """
    Convert a `SearchAgg` into a single-row `SearchMetrics` table.
    """
    t = agg.micro_timings or {}
    return SearchMetrics.from_kwargs(
        orbit_id=[None if orbit_id is None else str(orbit_id)],
        propagation_strategy=[str(propagation_strategy)],
        footprint=[str(footprint)],
        healpix_nside=[int(healpix_nside)],
        n_sigma=[float(n_sigma)],
        window_size_days=[int(window_size_days)],
        target_chunk_size=[int(target_chunk_size)],
        max_processes=[None if max_processes is None else int(max_processes)],
        n_targets=[int(agg.n_targets)],
        n_predicted_rows=[int(agg.n_predicted_rows)],
        n_predicted_pixels=[int(agg.n_predicted_pixels)],
        n_selected_frames_unique=[int(agg.n_selected_frames_unique)],
        n_frames_joined=[int(agg.n_frames_joined)],
        n_matched_frames_unique=[int(agg.n_matched_frames_unique)],
        n_frames_loaded=[int(agg.n_frames_loaded)],
        n_frames_faint_skipped=[int(agg.n_frames_faint_skipped)],
        n_observations_loaded=[int(agg.n_observations_loaded)],
        n_accepted=[int(agg.n_accepted)],
        n_candidates=[int(agg.n_candidates)],
        n_rejected_innov_ellipse=[int(agg.n_rejected_innov_ellipse)],
        n_rejected_tolerance=[int(agg.n_rejected_tolerance)],
        n_rejected_mag_residual=[int(agg.n_rejected_mag_residual)],
        enum_sec=[float(agg.enum_sec)],
        propagate_sec=[float(agg.propagate_sec)],
        footprint_sec=[float(agg.footprint_sec)],
        join_frames_sec=[float(agg.join_frames_sec)],
        io_sec=[float(agg.io_sec)],
        filter_sec=[float(agg.filter_sec)],
        photometry_sec=[float(agg.photometry_sec)],
        total_sec=[float(agg.total_sec)],
        stage2_window_centers_elapsed_s=[t.get("stage2.window_centers_elapsed_s")],
        stage2_assist_propagate_centers_elapsed_s=[t.get("stage2.assist_propagate_centers_elapsed_s")],
        stage2_sigma_point_variants_elapsed_s=[t.get("stage2.sigma_point_variants_elapsed_s")],
        stage2_propagate2body_nominal_elapsed_s=[t.get("stage2.propagate2body_nominal_elapsed_s")],
        stage2_ephemeris_nominal_elapsed_s=[t.get("stage2.ephemeris_nominal_elapsed_s")],
        stage2_propagate2body_variants_elapsed_s=[t.get("stage2.propagate2body_variants_elapsed_s")],
        stage2_variant_lonlat_elapsed_s=[t.get("stage2.variant_lonlat_elapsed_s")],
        stage2_reconstruct_cov_elapsed_s=[t.get("stage2.reconstruct_cov_elapsed_s")],
        stage2_variant_lonlat_fallback_count=[
            None
            if t.get("stage2.variant_lonlat_fallback_count") is None
            else int(float(t.get("stage2.variant_lonlat_fallback_count", 0.0)))
        ],
        footprint_vertices_elapsed_s=[t.get("footprint.vertices_elapsed_s")],
        footprint_rasterize_elapsed_s=[t.get("footprint.rasterize_elapsed_s")],
        footprint_pad_neighbors_elapsed_s=[t.get("footprint.pad_neighbors_elapsed_s")],
        gate_mag_residual_elapsed_s=[t.get("gate.mag_residual_elapsed_s")],
        gate_innov_ellipse_elapsed_s=[t.get("gate.innov_ellipse_elapsed_s")],
    )

