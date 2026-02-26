# Covariance Precovery Benchmark Specs (Authoritative)

This file is the **single source of truth** for how we run and interpret the covariance precovery benchmarking harness.
It exists to prevent “benchmark drift” and to ensure all future work stays aligned with the intended optimization goals,
accuracy constraints, and required statistics.

## Goals

- **Primary goal**: **Fastest possible recovery of complete truth-level detections** while **minimizing false positives**.
- **How we get faster**:
  - **A)** optimize the overall algorithm/relationships between components, optimize/replace individual components
  - **B)** reduce the amount of computation performed (while maintaining required accuracy)

## Canonical stages (conceptual)

- **Stage 1 (enumerate targets)**: define the target set, typically one row per distinct \((\mathrm{obscode}, \mathrm{exposure\_mjd\_mid})\).
- **Stage 2 (predict)**: propagate orbit(s) and generate ephemeris at target times (high accuracy required).
- **Stage 3 (footprint → healpixels)**: compute on-sky covariance footprint and rasterize to **intersecting healpixels**.
- **Stage 4 (detection filtering / “gate”)**: join detections by \((\mathrm{obscode}, \mathrm{time\_key}, \mathrm{healpixel})\), then apply:
  - magnitude-based gates (cheap scalar operations)
  - geometry gate (innovation ellipse / Mahalanobis)
- **Stage 5 (optional)**: density/index-only analysis (not the primary focus right now).

## Definitions and required outputs

### IDs and normalization (critical)

- **All object/orbit IDs must be normalized and joined early** so there is no ambiguity across:
  - orbits parquet(s)
  - truth detections
  - truth crossmatch artifacts
  - benchmark outputs (per-orbit + by-bin)
- The normalized ID is referred to as **`orbit_id`** everywhere once normalized.
- If multiple ID namespaces exist upstream (packed/unpacked, SBDB display strings, MPC designations), normalize to a single stable key.

### “Gate” terminology

- **Frame gating**: Stage 3 selecting **frame keys** (via footprint → pixels).
- **Detection gating**: Stage 4 filtering **candidate detections** (mag residual and innovation ellipse).
- When reporting, metrics must make clear whether “gating” refers to **frames** or **detections**.

### Core count metrics (must be present in benchmark outputs)

All canonical metrics are defined **per-orbit** first, then aggregated across orbits for run-level summaries and by-bin
reporting. (Some dataset-wide quantities are constant across orbits for a given run; we still record them per-orbit for
simplicity and consistent aggregation.)

#### Per-orbit metrics (exact definitions and column names)

All of these are per-orbit:

- **`n_frames_candidates`**: (this is just the total unique frames in the window, it will be identical for every orbit in a given run but we can record it per-orbit if that's easier).
- **`n_frames_geometry_matched`**: (this is the total frames for the orbit as a result of the footprint crossmatching. this is before filtering by limiting magnitude so it represents just the on sky geometry matching)
- **`n_frames_geometry_rejected`**: (this is simply n_frames_candidates - n_frames_geometry_matched)
- **`n_frames_truth_available`**: (renamed form n_total_truth_frames) which represents the number of frames for that orbit that has at least one truth detection in it
- **`n_frames_truth_geometry_matched`**: (the number of frames for that orbit that matched in footprint crossmatching that was also a truth frame)
- **`n_frames_lim_mag_rejected`**: (per orbit, the number of frames that are rejected by limiting magnitude)
- **`n_frames_lim_mag_truth_rejected`**: (the number of frames rejected by limiting magnitude filter that had truth detections in them)
- **`n_frames_final`**: (the number of total frames that survived both geometry and limiting magnitude filtering)
- **`n_frames_truth_final`**: (the number of truth frames remaining after both geometry filtering and limiting magnitude filtering)

- **`n_detections_candidates`**: (for an orbit, the number of total detections that were fetched related to the n_matched_frames), e.g. before our filtering
- **`n_detections_gate_matched`**: (for an orbit, the number of detections which passed the innov_ellipse filtering, but before magnitude outlier rejection)
- **`n_detections_magnitude_rejected`**: (the total number of detections that survived gate innov_ellipse filtering but were then rejected by magnitude outlier rejection)

- **`n_detections_truth_total`**: (all the truth detections that existed for this orbit in the search window, should equal n_frames_truth_available)
- **`n_detections_truth_frame_candidates`**: (all truth detections available in the frames that made it into n_frames_final. e.g. the maximum truth detections the detection filtering could find based on what survived the frame selection step)
- **`n_detections_truth_gate_matched`**: (for the orbit, all truth detections that survived the innov_ellipse gate filtering)
- **`n_detections_truth_magnitude_rejected`**: (the number of truth detections which were rejected during magnitude outlier rejection)
- **`n_detections_truth_final`**: (the number of truth detections we recovered, surviving all filtering)
- **`n_detections_false_positive_final`**: (the number of detections that survived all filtering and we know to be a false positive, which is all detections matched above the first detection in a single unique obscode/time combination. note: this is not per frame, as you can have multiple frames in an exposure, and what defines a definite false positive is any match above the first in a unique exposure)
- **`n_detections_unknown_final`**: (the number of detections that made it past all filtering that are not known truth detections but could itself be a real unattributed precovery or it coudl be a false positive)

## Truth scoring (required artifacts and stats)

### Tolerances

- **Time tolerance**: **±60 s**
- **Spatial tolerance**: **5 arcsec**

### Required truth artifacts

- **Truth detections**: truth rows expressed in the same **`observation_id`** namespace the backends emit.
- **Truth frames**: truth detections must also be aggregated into **truth frame keys**:
  - \((\mathrm{orbit\_id}, \mathrm{obscode}, \mathrm{exposure\_mjd\_mid\_key\_us}, \mathrm{healpixel})\)
  - This is required so we can score footprint selection quality independently of Stage-4 detection gating.

### Required truth stats (per run, per backend)

- **`n_frames_truth_available`**: number of truth frames we hope to find in the window (sum across orbits).
- **`n_matched_frames`**: number of selected frames that join to detections in the dataset (backend-dependent; join coverage proxy).
- **`n_frames_truth_final`**: number of truth frames that survive Stage 3 (geometry + limiting-magnitude filtering).
- **`n_detections_truth_total`**: number of truth detections in the window (sum across orbits).
- **`n_detections_truth_final`**: number of truth detections recovered (survive all filtering).
- **`truth_recall`**: `n_detections_truth_final / n_detections_truth_total`
- **`truth_frame_coverage`**: `n_frames_truth_final / n_frames_truth_available`

## Photometry requirements

- Ephemeris generation must compute **predicted apparent magnitude in V** once (when H is available).
- All gating must use **V → canonical filter** conversion via **`convert_magnitude`** with **`composition="C"`**.
- Magnitude outlier rejection must be recorded distinctly from geometry gating (innovation ellipse).
- Limiting magnitude filtering must be recorded (skipped reasons / counts).

## Propagation / ephemeris requirements

- Benchmark workloads must use the **high-accuracy propagation strategy**:
  - **ASSIST window + 2-body** (`assist_window_then_2body_variants:sigma_points`)
  - **7 day window** (canonical default): Stage-2 window size in days; 2-body propagation is valid within ±window/2 of each ASSIST window center.
- **Default behavior (code and harness)**:
  - `precovery/search/run.run_precovery()` defaults: `window_size_days=7`, `stage2_strategy="assist_window_then_2body_variants:sigma_points"`.
  - `bench/benchmarks/workload.WorkloadSpec` defaults: same. The backend_benchmark CLI uses these when `--window-size-days` is not set (or ≤0).
- Optional comparison runs (e.g. 7d vs 30d vs full N-body) use `--window-size-days N` or a different strategy flag; these are not the default.
- Optimizations in `adam-core`/`adam-assist` must maintain strict numerical equivalence where required.
- The STM-based covariance propagation prototype is **explicitly out of scope / rejected**.

## Footprint requirements

- We are optimizing for the **fastest and most accurate way to get intersecting healpixels**.
- **Do not use `cov_disc`** for production-like benchmarks (it pulls in too many frames).
- The benchmark harness must preserve the accuracy characteristics of **`cov_polygon_reconstructed_moc`**.
- **Neighbor dilation/padding is not assumed**:
  - It must be explicitly controlled and benchmarked.
  - It is not “default-on just because legacy did it.”

## Backends and storage

- **DuckDB is the canonical backend** for local benchmarking runs.
- The SQLite “baseline blobs” implementation is **disabled** in the benchmark harness.
- External backends (ClickHouse/BigQuery virtual) may exist for comparison, but the harness must remain DuckDB-first.

## Required benchmark reporting dimensions

All of the following must be recordable:

- **Per object** (per normalized `orbit_id`)
- **Per bin** (see below)

### Binning scheme (must reuse the existing one)

The population was selected for the existing binning scheme. Reuse it (do not invent a new one):

- **Selection stratum** bins (as stored in selection artifacts): `stratum`, plus parsed components:
  - `regime`, `arc_bin`, `dt_bin`, `u_bin`, `i_bin`
- **Orbit-type bin**: `orbit_type_int`
- **Covariance severity bins** (from stored covariance severity artifacts):
  - `sigma_pos_rms_bin`
  - `anisotropy_pos_bin`

Orbit-to-bin mapping must be persisted as a reusable artifact (e.g. `orbit_bins.parquet`).

## Canonical month and inputs

- We use a canonical “test month” and associated dataset for repeatable comparisons.
- The month must be chosen based on **high truth coverage for the selected orbit population**.
- The benchmarking bundle must be standalone and include:
  - normalized orbits parquet
  - truth detections parquet for the month
  - truth frames parquet for the month
  - orbit bins parquet
  - **orbit IDs with truth detections** for the month (and an `Orbits` parquet filtered to those IDs)
  - a small **bin-representative “fast” orbit subset** (6 orbits) for quick iteration
  - a manifest describing inputs/outputs and window parameters

### Canonical month choices (current)

- **Standard-candles smoke month**: `2024-09` / `T05`
  - Used for fast iteration on algorithmic changes.
- **Full-population truth-rich month (581-orbit population)**: **`2021-09` / `I41,T05,T08`**
  - Chosen by maximizing **distinct orbits with truth detections** and maintaining broad **bin coverage** (strata/orbit types/uncertainty bins).
  - Canonical prepared bundle folder (local): `bench/local_db/full_precovery_n32/artifacts/bench_bundle_2021-09_I41_T05_T08`
  - Canonical **fast6** orbit IDs (bin-representative, truth-rich): `101604`, `104P`, `2017 SQ220`, `402P`, `C/2020 K1`, `P/2021 Q5`
    - Note: `134340` (Pluto) is intentionally excluded because its covariance is undefined/non-finite in the SBDB inputs and it is not a meaningful sigma-point benchmark target for our pipeline.

### Bundle outputs (canonical filenames)

When creating a standalone bundle via `bench/data/prepare_benchmark_bundle.py`,
the output directory must include the following additional reusable artifacts:

- **All orbits with truth detections in the month window**:
  - `orbit_ids_with_truth.parquet` (single column `orbit_id`)
  - `orbits_with_truth.parquet` (adam-core `Orbits` parquet filtered to those IDs)
- **Fast bin-representative subset (default: 6 orbits)**:
  - `orbit_ids_fast6.parquet` (single column `orbit_id`)
  - `orbits_fast6.parquet` (adam-core `Orbits` parquet filtered to those IDs)

## Implementation notes (where this is enforced)

 - Benchmark metric definitions and terminology are centralized in `bench/benchmarks/metrics_spec.py`.
- Production detection sigma floors are applied in `precovery/search/detection_filter.py` and are reused by benchmark gating.

