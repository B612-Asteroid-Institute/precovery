# Covariance precovery run notes (2026-02-06)
Scope: 27 orbits over a dense 1-day window, with Stage 4 `innov_ellipse` gating. Results look strong overall; one object (`2019 QU127`) has unusually large uncertainty and dominates scan volume and accept counts.

## Dataset / subset
- **Subset DB (nside=32)**: `experiments/covariance_precovery/local_db/subset_2019-08_atlas-ztf-nsc`
- **Time window**: MJD \([58721.0, 58722.0]\) = UTC \([2019-08-26, 2019-08-27]\)

## Stage 2 (propagation) — 27-orbit run
- **Run dir**: `.../artifacts/stage2/20260205T164238Z/`
- **n_orbits**: 27
- **strategies**:
  - `2body_with_covariance`
  - `assist_variants:sigma_points`
  - `assist_window_then_2body_variants:sigma_points`
- **meta**: `experiments/covariance_precovery/local_db/subset_2019-08_atlas-ztf-nsc/artifacts/stage2/20260205T164238Z/meta.json`

## Stage 3 (healpixel selection / coverage)
### “fast” Stage 3 feeding Stage 4
- **Run dir**: `.../artifacts/stage3_fast/20260205T164238Z/`
- **nside**: 32
- **footprints**: `cov_disc`, `cov_disc_reconstructed`
- **meta**: `experiments/covariance_precovery/local_db/subset_2019-08_atlas-ztf-nsc/artifacts/stage3_fast/20260205T164238Z/meta.json`

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
`experiments/covariance_precovery/local_db/subset_2019-08_atlas-ztf-nsc/artifacts/`

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
