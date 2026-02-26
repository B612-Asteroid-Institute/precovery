# Covariance precovery experiments

This directory contains the benchmark harness, curated datasets/bundles, and run notes for the
covariance precovery pipeline.

## Where the benchmark assets live (orbits / detections / truth / bins)

These assets are **not tracked in git** (see `.gitignore`), and are expected to exist on disk under:

- `bench/local_db/` (gitignored)

In practice, we keep one or more “subset dirs” there. A **subset dir** is the root passed as
`--subset-dir` to `bench/benchmarks/backend_benchmark.py` and must contain `config.json` plus an `artifacts/`
directory (Stage 2/3/4 outputs, bundles, etc.).

### Canonical “44-orbit” benchmark assets (truth-rich month bundle)

This is the run we’ve been iterating on most recently:

- **Subset dir**:
  - `bench/local_db/full_precovery_n32/`
- **Prepared monthly bundle** (contains the orbit sets, truth inputs, bins, manifest):
  - `bench/local_db/full_precovery_n32/artifacts/bench_bundle_2021-09_I41_T05_T08/`
- **The “44 orbits” set**:
  - `.../bench_bundle_2021-09_I41_T05_T08/orbits_with_truth.parquet`
  - This is “all orbits with truth detections in the month window” for the bundle; historically it was **44**.
- **Fast iteration set**:
  - `.../bench_bundle_2021-09_I41_T05_T08/orbits_fast6.parquet`
- **Truth inputs used by the benchmark harness**:
  - `.../bench_bundle_2021-09_I41_T05_T08/truth_detections.parquet`
  - `.../bench_bundle_2021-09_I41_T05_T08/truth_frames.parquet`
- **Orbit-to-bin mapping (for by-bin aggregations)**:
  - `.../bench_bundle_2021-09_I41_T05_T08/orbit_bins.parquet`
- **Detections parquet dataset (DuckDB backend)**:
  - `bench/local_db/full_precovery_n32/artifacts/throughput_2021-09_I41_T05_T08/detections_parquet_fullmonth_with_filter/`
    - `part-*.parquet`

### “Standard” command we used for the 44-orbit run (DuckDB + truth scoring)

Default propagation strategy: **ASSIST window + 2-body**, **7 day window** (see `BENCHMARK_SPECS.md`). The command below uses these defaults (no `--window-size-days`).

**Full run (all targets)**:

```bash
pdm run python -m bench.benchmarks.backend_benchmark \
  --subset-dir "bench/local_db/full_precovery_n32" \
  --orbits-parquet "bench/local_db/full_precovery_n32/artifacts/bench_bundle_2021-09_I41_T05_T08/orbits_with_truth.parquet" \
  --truth-parquet "bench/local_db/full_precovery_n32/artifacts/bench_bundle_2021-09_I41_T05_T08/truth_detections.parquet" \
  --month 2021-09 \
  --obscode I41 --obscode T05 --obscode T08 \
  --backend duckdb \
  --duckdb-parquet "bench/local_db/full_precovery_n32/artifacts/throughput_2021-09_I41_T05_T08/detections_parquet_fullmonth_with_filter" \
  --targets-from-truth \
  --max-orbits 0 --max-targets 0
```

## Benchmark specification (authoritative)

- `BENCHMARK_SPECS.md`

## Fixture hardening notes (2026-02-25)

The benchmark fixture pipeline was hardened to make identity resolution deterministic and
alias-aware, and to remove join-critical heuristic designation normalization.

### New required/expected artifacts for hardened bundles

- `designation_resolution.parquet`
  - canonical selected designation -> linked alias expansion (`mpcq_request_provids`) and resolution status
- `selected_orbit_mapping.parquet`
  - selected designation -> resolved `orbit_id` mapping (with explicit match reason/count)
- `truth_unresolved_designations.parquet`
  - explicit unresolved truth designations and counts (instead of silent fallback behavior)
- `orbits_sbdb_provenance.parquet`
  - SBDB orbit provenance metadata used by bundle/bin outputs

### SBDB provenance + non-grav metadata

Bundle prep now persists SBDB-derived provenance/enrichment fields:

- `orbit_source` (constant: `sbdb`)
- `epoch_mjd_tt`
- `has_nongrav_terms` (derived from finite SBDB A-terms)
- `a1`, `a2`, `a3` (nullable)
- `nongrav_flag` (boolean, included in `orbit_bins.parquet`)

This phase is metadata-only for stratified benchmarking/readiness; propagation dynamics are unchanged.

### CLI additions

- `bench/data/prepare_benchmark_bundle.py`
  - `--designation-resolution-parquet`
  - `--fail-on-unresolved-truth`
- `bench/selection/fetch_mpcq_handoff.py`
  - `--designation-resolution-parquet`
  - `--alias-request-mode` (`all_linked` default, `selected_only` optional)

## Run notes / reports

- `RUN_NOTES_20260206_covariance_precovery.md`
- `report_20260211_population.md`
- `performance_notes.md`

## Curated datasets / bundles (paths are the source of truth)

The recommended way to discover “what datasets exist” is via each bundle’s `manifest.json`.

Example canonical bundle:

- `bench/local_db/full_precovery_n32/artifacts/bench_bundle_2021-09_I41_T05_T08/manifest.json`

Selected bundle outputs used by the harness:

- Orbits:
  - `.../bench_bundle_2021-09_I41_T05_T08/orbits.parquet`
  - `.../bench_bundle_2021-09_I41_T05_T08/orbits_fast6_canonical.parquet`
  - `.../bench_bundle_2021-09_I41_T05_T08/orbits_with_truth.parquet`
- Truth crossmatch:
  - `.../w_20200101_20240101__I41_T05_T08_W84/truth_precovery_crossmatch.parquet`
- Detection parquet dataset used by DuckDB backend:
  - `.../throughput_2021-09_I41_T05_T08/detections_parquet_fullmonth_with_filter/part-*.parquet`

## “How do I regenerate X?”

- **Prepare a benchmark bundle**: `bench/data/prepare_benchmark_bundle.py` (writes `manifest.json`)
- **Export detections**:
  - from local DB subset: `bench/data/export_precovery_month_parquet.py`
  - from BigQuery AIMS: `bench/data/export_aims_month_parquet.py`
- **Orbit selection / curation**: `bench/selection/` (notably `bench/selection/bq_select.py`, `bench/selection/covariance_severity.py`)
