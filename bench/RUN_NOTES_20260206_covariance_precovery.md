# Covariance precovery run notes (2026-02-06)
Scope: 27 orbits over a dense 1-day window, with Stage 4 `innov_ellipse` gating. Results look strong overall; one object (`2019 QU127`) has unusually large uncertainty and dominates scan volume and accept counts.

## Dataset / subset
- **Subset DB (nside=32)**: `bench/local_db/subset_2019-08_atlas-ztf-nsc`
- **Time window**: MJD \([58721.0, 58722.0]\) = UTC \([2019-08-26, 2019-08-27]\)

## Stage 2 (propagation) — 27-orbit run
- **Run dir**: `.../artifacts/stage2/20260205T164238Z/`
- **n_orbits**: 27
- **strategies**:
  - `2body_with_covariance`
  - `assist_variants:sigma_points`
  - `assist_window_then_2body_variants:sigma_points`
- **meta**: `bench/local_db/subset_2019-08_atlas-ztf-nsc/artifacts/stage2/20260205T164238Z/meta.json`

## Stage 3 (healpixel selection / coverage)
### “fast” Stage 3 feeding Stage 4
- **Run dir**: `.../artifacts/stage3_fast/20260205T164238Z/`
- **nside**: 32
- **footprints**: `cov_disc`, `cov_disc_reconstructed`
- **meta**: `bench/local_db/subset_2019-08_atlas-ztf-nsc/artifacts/stage3_fast/20260205T164238Z/meta.json`

### Footprint comparison sweeps (same stage2 run)
Goal: compare candidate footprints’ frame volume at different nsides, keeping truth coverage high.

- **nside=32 run dir**: `.../artifacts/stage3_fp_compare_n32/20260205T210258Z/20260205T164238Z/`
- **nside=64 run dir**: `.../artifacts/stage3_fp_compare_n64/20260205T212501Z/20260205T164238Z/`
- **nside=128 run dir**: `.../artifacts/stage3_fp_compare_n128/20260205T212843Z/20260205T164238Z/`

Footprints included in fp-compare runs:
- `sample_direct`
- `sample_polygon_moc:convex_hull`
- `cov_disc_reconstructed`
- `cov_polygon_reconstructed_moc`
- `cov_mc_reconstructed`

## Stage 4 (detection filter bench) — completed runs
Stage 4 outputs live under:
`bench/local_db/subset_2019-08_atlas-ztf-nsc/artifacts/`

### Stage 4 “fast” baseline
- **Run dir**: `.../artifacts/stage4_fast/20260205T164238Z/`
- **Inputs**:
  - stage2: `.../stage2/20260205T164238Z`
  - stage3: `.../stage3_fast/20260205T164238Z`
  - strategies: `2body_with_covariance`, `assist_variants:sigma_points`, `assist_window_then_2body_variants:sigma_points`
  - footprints: `cov_disc`, `cov_disc_reconstructed`
  - detection filters: `innov_ellipse@1`, `innov_ellipse@2`, `innov_ellipse@3`
- **Key outputs**:
  - `metrics.parquet`, `coverage.parquet`
  - `per_target.parquet` (enabled)
  - summary artifacts: `summary_stage4*.parquet`, `viz/*`

### Stage 4 fp-compare: `sample_direct` (best speed + full recovery at @3)
- **Run dir**: `.../artifacts/stage4_fp_compare_n32_sample_direct/20260205T164238Z/`
- **Inputs**:
  - stage3: `.../stage3_fp_compare_n32/20260205T210258Z/20260205T164238Z`
  - strategy: `assist_window_then_2body_variants:sigma_points`
  - footprint: `sample_direct`
  - detection filters: `innov_ellipse@2`, `innov_ellipse@3`
- **Result rows (coverage)**:
  - `innov_ellipse@2`: recall = 0.950413 (115/121)
  - `innov_ellipse@3`: recall = 1.0 (121/121)

### Stage 4 fp-compare: reconstructed footprints (`cov_*_reconstructed`)
- **Run dir**: `.../artifacts/stage4_fp_compare_n32_reconstructed/20260205T164238Z/`
- **Inputs**:
  - stage3: `.../stage3_fp_compare_n32/20260205T210258Z/20260205T164238Z`
  - strategy: `assist_window_then_2body_variants:sigma_points`
  - footprints: `cov_mc_reconstructed`, `cov_polygon_reconstructed_moc`, `cov_disc_reconstructed`
  - detection filters: `innov_ellipse@2`, `innov_ellipse@3`, `innov_ellipse@4`
- **Headline**:
  - `innov_ellipse@3` and `innov_ellipse@4` reach **recall = 1.0 (121/121)** for all three reconstructed footprints.
  - `cov_polygon_reconstructed_moc` is the fastest among these reconstructed options (fewest frames/obs loaded).

### Best-row per-object breakdown (fixed per-target truth accounting)
Purpose: produce per-object/per-orbit scan + truth recovery breakdown for the best-performing Stage 4 row.

- **Run dir**: `.../artifacts/stage4_bestrow_per_object_fixed/20260205T164238Z/`
- **Config**: `assist_window_then_2body_variants:sigma_points`, `sample_direct`, `innov_ellipse@3`
- **Outputs**:
  - `metrics.parquet`, `coverage.parquet`
  - `per_target.parquet` (enabled; includes per-target truth matched/recovered counts)

#### Per-object results table (best row)
Computed from `per_target.parquet`, aggregated to per-orbit totals.

Columns:
- `obs_loaded`: total loaded observations scanned for the orbit
- `accepted`: observations passing `innov_ellipse@3`
- `truth_matched`/`truth_recovered`: truth crossmatch denominator / recovered count (so per-orbit recall is `truth_recovered / truth_matched`)
- `accept_frac`: `accepted / obs_loaded`

Full table (sorted by `obs_loaded` descending):

| orbit_id   | n_targets | frames_loaded | obs_loaded | accepted | truth_matched | truth_recovered | accept_frac | recall   | runtime_total_sec |
| ---------- | --------- | ------------- | ---------- | -------- | ------------- | --------------- | ----------- | -------- | ----------------- |
| 2019 QM125 | 70        | 288           | 309757     | 13       | 2             | 2               | 0.000042    | 1.000000 | 0.142810          |
| 2019 QP125 | 69        | 284           | 309464     | 17       | 9             | 9               | 0.000055    | 1.000000 | 0.063213          |
| 2019 QP87  | 69        | 286           | 308661     | 20       | 13            | 13              | 0.000065    | 1.000000 | 0.056009          |
| 2019 QU127 | 66        | 222           | 286668     | 286668   | 14            | 14              | 1.000000    | 1.000000 | 0.051144          |
| 813492     | 66        | 222           | 286668     | 4        | 3             | 3               | 0.000014    | 1.000000 | 0.045256          |
| 2019 QK125 | 70        | 257           | 203331     | 4        | 2             | 2               | 0.000020    | 1.000000 | 0.297733          |
| 9909       | 67        | 228           | 126320     | 4        | 4             | 4               | 0.000032    | 1.000000 | 0.040815          |
| 7397       | 22        | 92            | 73396      | 4        | 4             | 4               | 0.000054    | 1.000000 | 0.195427          |
| 2512       | 32        | 149           | 73199      | 5        | 4             | 4               | 0.000068    | 1.000000 | 0.153995          |
| 276872     | 26        | 90            | 47317      | 4        | 1             | 1               | 0.000085    | 1.000000 | 0.113150          |
| 92546      | 21        | 72            | 46791      | 4        | 4             | 4               | 0.000085    | 1.000000 | 0.076537          |
| 3460       | 27        | 89            | 46224      | 4        | 3             | 3               | 0.000087    | 1.000000 | 0.099436          |
| 92526      | 23        | 85            | 44216      | 4        | 3             | 3               | 0.000090    | 1.000000 | 0.098160          |
| 5543       | 24        | 99            | 31574      | 7        | 5             | 5               | 0.000222    | 1.000000 | 0.097376          |
| 7886       | 20        | 84            | 29231      | 4        | 4             | 4               | 0.000137    | 1.000000 | 0.081573          |
| 25961      | 9         | 57            | 26750      | 4        | 4             | 4               | 0.000150    | 1.000000 | 0.060654          |
| 129105     | 25        | 89            | 14567      | 4        | 4             | 4               | 0.000275    | 1.000000 | 0.084373          |
| 45889      | 19        | 88            | 13401      | 4        | 4             | 4               | 0.000298    | 1.000000 | 0.041122          |
| 3264       | 19        | 76            | 12103      | 4        | 4             | 4               | 0.000330    | 1.000000 | 0.066508          |
| 115618     | 20        | 87            | 11962      | 4        | 4             | 4               | 0.000334    | 1.000000 | 0.076873          |
| 147465     | 13        | 54            | 11194      | 4        | 4             | 4               | 0.000357    | 1.000000 | 0.046457          |
| 34760      | 10        | 60            | 10234      | 4        | 4             | 4               | 0.000391    | 1.000000 | 0.047930          |
| 3548       | 13        | 62            | 9802       | 3        | 3             | 3               | 0.000306    | 1.000000 | 0.052529          |
| 19642      | 18        | 84            | 9593       | 3        | 3             | 3               | 0.000313    | 1.000000 | 0.059485          |
| 9177       | 9         | 34            | 8750       | 4        | 4             | 4               | 0.000457    | 1.000000 | 0.031228          |
| 90313      | 20        | 83            | 7694       | 4        | 4             | 4               | 0.000520    | 1.000000 | 0.067258          |
| 8396       | 8         | 38            | 5899       | 4        | 4             | 4               | 0.000678    | 1.000000 | 0.037618          |

Top 10 by `accepted` (high-volume / high-accept objects):

| orbit_id   | n_targets | frames_loaded | obs_loaded | accepted | truth_matched | truth_recovered | accept_frac | recall   | runtime_total_sec |
| ---------- | --------- | ------------- | ---------- | -------- | ------------- | --------------- | ----------- | -------- | ----------------- |
| 2019 QU127 | 66        | 222           | 286668     | 286668   | 14            | 14              | 1.000000    | 1.000000 | 0.051144          |
| 2019 QP87  | 69        | 286           | 308661     | 20       | 13            | 13              | 0.000065    | 1.000000 | 0.056009          |
| 2019 QP125 | 69        | 284           | 309464     | 17       | 9             | 9               | 0.000055    | 1.000000 | 0.063213          |
| 2019 QM125 | 70        | 288           | 309757     | 13       | 2             | 2               | 0.000042    | 1.000000 | 0.142810          |
| 5543       | 24        | 99            | 31574      | 7        | 5             | 5               | 0.000222    | 1.000000 | 0.097376          |
| 2512       | 32        | 149           | 73199      | 5        | 4             | 4               | 0.000068    | 1.000000 | 0.153995          |
| 813492     | 66        | 222           | 286668     | 4        | 3             | 3               | 0.000014    | 1.000000 | 0.045256          |
| 2019 QK125 | 70        | 257           | 203331     | 4        | 2             | 2               | 0.000020    | 1.000000 | 0.297733          |
| 9909       | 67        | 228           | 126320     | 4        | 4             | 4               | 0.000032    | 1.000000 | 0.040815          |
| 7397       | 22        | 92            | 73396      | 4        | 4             | 4               | 0.000054    | 1.000000 | 0.195427          |

## Consolidated “best row” summary (nside=32)
Best performing row in **speed + recovery**:
- **strategy**: `assist_window_then_2body_variants:sigma_points`
- **footprint**: `sample_direct`
- **filter**: `innov_ellipse@3`
- **recall**: 1.0 (121/121)
- **volume**:
  - `n_frames_loaded`: 3,359
  - `n_observations_loaded`: 2,364,766
  - `n_after_prefilter`: 286,812

## Outlier: `2019 QU127` dominates scan volume
In `stage4_fast/20260205T164238Z` (strategy `assist_window_then_2body_variants:sigma_points`, footprint `cov_disc_reconstructed`):
- `innov_ellipse@2`: total `n_after_prefilter = 994,616` and `2019 QU127` contributes **994,478** (~99.99%).
- `innov_ellipse@3`: total `n_after_prefilter = 2,567,149` and `2019 QU127` contributes **2,567,005** (~99.99%).

Sigma-point spread diagnostic (from `variants_orbits.parquet`, simple heliocentric position spread):
- `2019 QU127`: `pos_std_au ≈ 2.8`, `max_pairwise_au ≈ 12.6` (extreme outlier vs the rest)

This object likely needs special handling (exclusion/capping/uncertainty gating) for precovery workloads to avoid “ridiculous” scan volumes.

## Notes / caveats
- Stage 4 per-target metrics originally did not populate for the sigma-point “grouped” path used by `sample_direct`; we patched Stage 4 to record per-target rows (including truth matched/recovered counts) for both variant paths so per-object analysis is possible.
