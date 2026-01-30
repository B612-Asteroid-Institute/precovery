# Stage 4: detection filtering benchmark (handoff summary)

Stage 4 is implemented in [`experiments/covariance_precovery/harness/stage4_detection_filter_bench.py`](harness/stage4_detection_filter_bench.py).

## Purpose

Stage 4 benchmarks **detection-level filtering** after candidate frames have been selected (via Stage 2+3 artifacts). It measures:

- **runtime** split into I/O vs filtering vs χ² gating
- **detection-level recall** vs the truth crossmatch (did we recover the specific truth detection IDs?)

It is intentionally an “atomic harness”: it **replays** the expensive part of precovery (loading detections and matching), but in a controlled, logged, repeatable way.

## Inputs

Stage 4 consumes:

- **Subset DB** (same as other stages)
  - `subset_dir/index.db` (sqlite `frames` table, contains `data_uri`, `data_offset`, `data_length`)
  - `subset_dir/data/.../*.data` (binary observation blobs)
- **Truth crossmatch**
  - `subset_dir/artifacts/truth_precovery_crossmatch.parquet`
  - plus `subset_dir/artifacts/orbits_selected_sbdb.parquet` to map `designation -> orbit_id` consistently
- **Stage 2 artifacts** (required)
  - `stage2_run_dir/inputs/frame_time_targets.parquet`
  - `stage2_run_dir/strategies/**/(mean_ephemeris|variants_ephemeris)/part-*.parquet`

## High-level algorithm

For each Stage 2 strategy output (and each footprint + detection filter variant):

1. **Align truth to Stage 2 targets**
   - `truth_precovery_crossmatch.parquet` rows with `matched==True` are mapped to:
     - `orbit_id` (via designation normalization + `orbits_selected_sbdb.parquet`)
     - `target_idx` (nearest `frame_time_targets.parquet` entry for that obscode within 60s)
   - This yields the truth set of expected `(orbit_id, match_observation_id)` pairs.

2. **For each ephemeris row (mean strategies) or each grouped variant cloud (variant strategies)**
   - compute predicted footprint pixels at `healpix_nside` (same footprint vocabulary as Stage 3)
   - query `index.db` to get concrete frames at `(obscode, exposure_mjd_mid, healpixel IN predicted_pixels)`

3. **Load observations from the binary `.data` files (legacy style)**
   - Stage 4 reads detections via **byte offsets** exactly like the production DB:
     - `FrameIndex.frames` provides `data_uri`, `data_offset`, `data_length`
     - `FrameDB.get_observations(...)` opens the `.data` file, `seek(data_offset)`, and unpacks records until `data_length` is consumed

4. **Apply detection filters**
   - `chi2_only`: run `find_observation_matches_covariance(observations, ephem, n_sigma)`
   - `footprint_then_chi2`: prefilter detections using `footprint.contains(ra, dec)` then run χ²

5. **Score recall**
   - Each accepted observation is identified by `(orbit_id, observation_id)`
   - Stage 4 counts a truth detection as recovered if that pair matches truth’s `(orbit_id, match_observation_id)`

## Matching-time assumption (important)

`find_observation_matches_covariance(...)` asserts ephemeris times match observation times.

Stage 4 defaults to building a “paired” ephemeris by **repeating the Stage 2 ephemeris row** to length `N_obs`.

- This is fast, but assumes observation timestamps are effectively at the Stage 2 target midpoint.
- If a dataset’s detections have distinct per-detection timestamps, use `--fallback-per-obs-ephem` to generate a correct per-observation ephemeris using `generate_ephem_for_per_obs_timestamps(...)` (more expensive).

## Outputs

Stage 4 writes to:

`subset_dir/artifacts/stage4/<stage2_run_dir.name>/`

- `metrics.parquet`: one row per `(strategy, variant_kind, footprint, detection_filter)`
  - includes `io_sec`, `filter_sec`, `chi2_sec`, `runtime_total_sec`, plus size counters
- `coverage.parquet`: one row per `(strategy, variant_kind, footprint, detection_filter)`
  - includes `n_truth_matched`, `n_recovered`, `recall`
- `meta.json`: configuration summary for the run

## Why Stage 4 can be slow (and how to run it sanely)

Stage 4 is often **I/O bound** because loading observations requires reading and unpacking binary blobs from disk.

Detection-level filtering does **not** reduce this I/O, because it happens *after* observations are loaded.

Recommended controls for practical runs:

- `--only-truth`: evaluate only orbit/target pairs that appear in truth (large speedup)
- `--max-orbits`, `--max-targets`: cap workload for iteration/debugging
- `--strategies ...`: run a single strategy first
- `--filters chi2_only` (or `footprint_then_chi2`) to avoid evaluating multiple filters
- `--obs-cache-max-frames N`: caches `ObservationsTable` by `(data_uri, offset, length)` to avoid re-reading the same frames across footprint variants

## CLI entrypoint

Stage 4 is runnable directly:

```bash
pdm run python experiments/covariance_precovery/harness/stage4_detection_filter_bench.py \
  --subset-dir <subset_dir> \
  --stage2-run-dir <stage2_run_dir> \
  --healpix-nside 32 \
  --only-truth \
  --obs-cache-max-frames 1024
```

