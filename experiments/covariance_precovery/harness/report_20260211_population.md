## Covariance precovery report (truth-recovery runs)

This report summarizes the end-to-end truth-recovery experiments (Stages 2–4) and the follow-up investigations into Stage 3 and Stage 4 failures/outliers.

### Scope and goal
- **Goal**: quantify speed/accuracy of the covariance-based pipeline (Stages 2/3/4) on a long-span subset, with strong emphasis on **truth recall** and identifying **failure modes**.
- **Window**: `2020-01-01` to `2024-01-01` (UTC), obscodes `{I41, T05, T08, W84}`, full production precovery DB subset (healpix nside 32).
- **Strategies tested**:
  - **Stage 2**: `assist_variants:sigma_points` (and a targeted `assist_variants:mc_256` follow-up).
  - **Stage 3**: `cov_polygon_reconstructed_moc`.
  - **Stage 4**: `innov_ellipse@3`.

---

## Population size (what we actually ran)

It’s important to distinguish “selected orbits” vs “truth-matched orbits” vs “covariance-ok orbits”.

### Selection and truth-matched population
- **Selected orbits** (SBDB fetched): **581**
  - Source: `.../artifacts/w_20200101_20240101__I41_T05_T08_W84/orbits_selected_sbdb.parquet`
- **Truth-matched designations in window**: **346**
  - Source: `.../artifacts/w_20200101_20240101__I41_T05_T08_W84/truth_precovery_crossmatch.parquet` (matched==true)

### Stage-2 truth-only benchmarking population
Stage 2 (sigma-points run) was executed in **truth-only** mode and then restricted to covariance-ok orbits for the variants strategy:
- **Stage 2 meta**: `n_orbits = 326`, `n_orbits_covok = 325`
  - Meaning: **326** truth-matched orbits were eligible; **325** had valid covariance for variant propagation.
  - Source: `.../artifacts/stage2/20260211T012705Z/meta.json`
- **Stages 2/4 tables and per-target metrics include**: **325** orbits (the cov-ok truth-orbit set)
  - Sources:
    - `.../artifacts/tables/20260211T012705Z/per_orbit.parquet`
    - `.../artifacts/stage4/20260211T012705Z/per_target.parquet`

---

## What artifacts were created

### Frame data downloaded (lazy blob fetch)
- **Distinct `.data` files (frame blobs) required by truth**: **290**
- **Local `.data` files present**: **290**
- Source: `.../artifacts/w_20200101_20240101__I41_T05_T08_W84/truth_precovery_blob_count.json` and `.../local_db/full_precovery_n32/data/`

### Core pipeline artifacts (paths)
- **Selected designations**: `.../artifacts/w_20200101_20240101__I41_T05_T08_W84/selected_designations.parquet`
- **SBDB orbits**: `.../artifacts/w_20200101_20240101__I41_T05_T08_W84/orbits_selected_sbdb.parquet`
- **Truth crossmatch**: `.../artifacts/w_20200101_20240101__I41_T05_T08_W84/truth_precovery_crossmatch.parquet`

### Stage run directories
- **Stage 2 (sigma points)**: `.../artifacts/stage2/20260211T012705Z/`
- **Stage 3**: `.../artifacts/stage3/20260211T012705Z/`
  - selected healpixels: `.../selected_keys/assist_variants:sigma_points/cov_polygon_reconstructed_moc/selected_keys_unique.parquet`
- **Stage 4**: `.../artifacts/stage4/20260211T012705Z/`
  - per-target metrics: `per_target.parquet`

### Joined analysis tables
- **Full per-orbit table**: `.../artifacts/tables/20260211T012705Z/per_orbit.parquet`
- **Trimmed per-orbit table** (Stage-3-zero-hit removed): `.../artifacts/tables/20260211T012705Z_trim_stage3rec0/per_orbit.parquet`
- **Excluded orbits list**: `.../artifacts/tables/20260211T012705Z_trim_stage3rec0/excluded_orbits.parquet`

### Follow-up analysis artifacts
- **dt_ge_1000 innovation-distance diagnostics**:
  - `.../artifacts/analysis/20260211T012705Z_dt_ge_1000_u35_u6p_truth_vs_innov.parquet`
- **SBDB example JSON payload saved**:
  - `.../artifacts/analysis/sbdb_worst_example.json`

---

## Overall results (full vs without Stage 3 zero-hit outliers)

All metrics below are for **truth recovery** (Stages 3/4 denominators are truth-matched frames/detections).

### Full population (325 orbits)
From `.../tables/20260211T012705Z/per_orbit.parquet`:
- **Stage 3**:
  - truth frames: **18,237**
  - recovered truth frames: **15,198**
  - **weighted coverage**: **0.833**
- **Stage 4**:
  - truth detections: **18,243**
  - recovered truth detections: **13,148**
  - **weighted recall**: **0.721**

### After excluding Stage 3 “zero recovered” outliers (313 orbits)
From `.../tables/20260211T012705Z_trim_stage3rec0/per_orbit.parquet`:
- **Stage 3**:
  - truth frames: **15,198**
  - recovered truth frames: **15,198**
  - **weighted coverage**: **1.000**
- **Stage 4**:
  - truth detections: **15,204**
  - recovered truth detections: **13,148**
  - **weighted recall**: **0.865**

### Immediate interpretation
- A small number of objects produce **complete Stage 3 failure** and dominate the Stage 3 deficit.
- Once removed, Stage 3 is “solved on truth” and Stage 4 becomes the limiting factor.

---

## Outliers: Stage 3 “zero-hit” (complete miss)

### Definition
- **Stage 3 zero-hit outlier**: `stage3_truth_healpix > 0` AND `stage3_recovered_truth_healpix == 0`.

### Count and list
- **Count**: **12 orbits**
- **Orbit IDs**:
  - `C/2010 U3`
  - `C/2017 K2`
  - `C/2019 E3`
  - `C/2019 U5`
  - `C/2020 F2`
  - `C/2021 D2`
  - `C/2021 G2`
  - `C/2022 E2`
  - `C/2022 L2`
  - `C/2022 QE78`
  - `C/2023 H5`
  - `C/2023 R1`

These 12 account for essentially all of the gap between:
- Full Stage 3 truth frames (**18,237**) and recovered (**15,198**) in the full table, because after excluding them Stage 3 becomes **15,198 / 15,198**.

### What we think these are (why they’re bad)
These Stage 3 failures are consistent with **bad orbit state vectors and/or covariance severity** in the specific sense that:
- The SBDB-derived **position covariance severity** is enormous for many of them (e.g. `sigma_pos_rms` in `sigpos_10_100` up through `sigpos_ge_100`) and extreme anisotropy.
- Even though truth detections exist (Stage 4 truth counts are non-zero), the Stage 3 footprint method (`cov_polygon_reconstructed_moc`) selects **no truth healpixels**.

In short: yes, this is the same “bad state vector” category we were referring to earlier — but more precisely, it’s “the propagated mean + reconstructed covariance footprint never overlaps the truth pixel set,” which can happen due to:
- huge uncertainty / pathological covariance geometry, and/or
- mean bias so large that even the covariance footprint doesn’t cover the true location at those target times.

What this is **not**:
- Not a missing-truth alignment bug: Stage 3/4 truth alignment was fixed to join on `(dataset_id, obscode, exposure_id) → exposure_mjd_mid` from `index.db`, and Stage 4 now counts misses even when Stage 3 selects zero pixels.

---

## Stage 4 results after removing Stage 3 zero-hit outliers

After removing the 12 Stage-3-zero-hit orbits:
- Stage 3 weighted coverage becomes **1.000**.
- Stage 4 weighted recall is **0.865** (13,148 / 15,204).

### Stage 4 outliers (in the trimmed population)
Using a practical filter `stage4_truth_detections >= 20` and `stage4_recall < 0.5`, there are **14** low-recall orbits in the trimmed set.

They are dominated by:
- `dt_bin = dt_ge_1000` (12 of 14)
- `u_bin = u_3_5` (10 of 14; plus one `u_6p`)

Representative worst cases in `dt_ge_1000`:
- `155P`: recall **0.000** (0/69)
- `304P`: recall **0.000** (0/46)
- `62P`: recall **0.000** (0/152)
- `104P`: recall **0.227** (51/225)
- `119P`: recall **0.322** (118/367)

---

## Deep dive: why `innov_ellipse@3` fails in `dt_ge_1000`

We focused on the subset `(dt_ge_1000, u_3_5 or u_6p)`:
- **34 orbits**
- **3,435 truth detections** (almost entirely ATLAS)
- Artifact: `.../artifacts/analysis/20260211T012705Z_dt_ge_1000_u35_u6p_truth_vs_innov.parquet`

### How far off are we?
For this regime:
- separation (arcsec) quantiles: median **0.84**, p90 **7.03**, p99 **160.33**
- innovation σ_major (arcsec) quantiles: median **0.30**, p90 **0.52**, p99 **1.70**
- normalized innovation distance \(d\) (≈ “sigmas”) quantiles: median **3.12**, p90 **15.44**, p99 **137.29**

This explains why `@3` is so punishing here: the “typical” truth detection is already around a few sigma, and the tail is extremely heavy.

Worst per-orbit medians (illustrative):
- `325P`: sep_med **74.93″**, σ_med **0.308″**, \(d_{med}\) **243.90**
- `62P`: sep_med **153.39″**, σ_med **1.132″**, \(d_{med}\) **135.72**
- `304P`: sep_med **136.15″**, σ_med **1.593″**, \(d_{med}\) **85.30**

These are not “slightly too tight gating.” The predicted mean is often nowhere near the truth.

### Would loosening the gate help?
Using the computed \(d\) values, the counterfactual recall for this regime is:
- `innov_ellipse@3`: **0.485**
- `innov_ellipse@5`: **0.686**
- `innov_ellipse@7`: **0.761**
- `innov_ellipse@10`: **0.831**

This would materially improve recall for many objects — but some catastrophic comets remain at ~0 even at `@7` because their mean bias is enormous.

---

## What we determined is *not* the problem

### Not sigma-point vs Monte Carlo sampling
We ran a targeted Stage 2 Monte Carlo comparison (`assist_variants:mc_256`) for the 34-orbit dt_ge_1000 regime subset.

Result: the innovation-distance distributions (and thus the gating outcomes) were essentially unchanged. In other words, **sampling method is not the bottleneck** here.

### Not “ATLAS has no sigmas”
ATLAS `.data` observations do include `ra_sigma`/`dec_sigma` and are finite for ATLAS detections in our subset.

However, Stage 4 applies a *floor as a lower bound clamp* for innovation gating:
- even if a detection has a sigma, it uses `max(sigma, floor)`
- the ATLAS floor is **0.30″** for this subset

For truth-matched ATLAS detections in this regime:
- the raw per-detection sigma distribution has median ~**0.23″**
- **~64%** of these sigmas are ≤ **0.30″**, so the clamp binds often

Importantly: clamping to a floor does **not** make recall worse here (it only widens the gate). The dominant problem remains mean bias / model mismatch.

---

## Most likely root cause of Stage 4 dt_ge_1000 failures: non-gravitational forces

We queried SBDB directly (API) for the dt_ge_1000 regime objects and inspected `orbit.model_pars`.

Finding:
- Many worst-offender periodic comets have **non-gravitational model parameters** (A-terms), e.g. **A1/A2/A3**.
- Across the 34-orbit dt_ge_1000 regime:
  - Orbits **without** A1/A2/A3 have much higher Stage 4 recall on average than those **with** A1/A2/A3.
- The most catastrophic failures (`58P`, `62P`, `304P`, `325P`, `155P`) have A-terms present in SBDB’s orbit solution.

Interpretation:
- SBDB’s orbit fit for these comets includes non-gravitational accelerations.
- Our propagation/ephemeris generation in this pipeline does not incorporate those non-grav terms.
- At long dt, the mean prediction drifts far from truth, and `innov_ellipse@3` rejects essentially everything even if Stage 3 selected the right healpixels.

---

## Practical recommendations / next steps

1) **Treat Stage 3 zero-hit outliers as “bad state vector / bad uncertainty” failures**
   - They are a small, explicit list (12 `C/...` objects).
   - They likely require a fundamentally different approach (or exclusion from this covariance-footprint benchmark).

2) For Stage 4 `dt_ge_1000` comet-like failures:
   - **Best fix**: incorporate non-grav model parameters (A1/A2/A3) into propagation for comets where SBDB provides them.
   - **Short-term mitigation**: use a larger `innov_ellipse@n_sigma` for these regimes and rely on downstream multi-detection consistency to control false positives.

3) Use the saved \(d\) distributions to choose a gating policy:
   - `@5` and `@7` yield large recall gains on many objects, but won’t rescue the extreme mean-bias cases.
   - A conditional policy by `(dt_bin, comet/non-grav flag, u_bin)` is likely necessary if we want both high recall and low false positives.

