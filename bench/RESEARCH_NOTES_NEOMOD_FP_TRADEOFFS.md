# NEOMOD FP and Uncertainty Tradeoff Notes

## Scope

This log captures current findings from NEOMOD-style synthetic orbit tests against Rubin-like detections, with emphasis on:

- false-positive behavior in all-exposure month searches,
- interaction between magnitude filtering and on-sky covariance filtering,
- failure modes from broad covariance tails and short-arc-like states.

## Key Findings So Far

1. **Row-level vs unique-detection inflation matters.**
   Candidate and accepted counts are orbit-associated rows, not unique detections. Duplicate assignment of the same detection to multiple orbit hypotheses inflates raw totals.

2. **Detection sigma unit interpretation is critical.**
   A sigma unit mismatch (arcsec vs degree scaling) can massively widen innovation gates and dramatically increase candidate pull-in.

3. **Magnitude gating requires populated photometric parameters.**
   If `H_v`/`G` are null, predicted magnitudes are null and magnitude-based rejection is effectively disabled.

4. **On-sky uncertainty budget is an effective Stage-3 control.**
   The `max_on_sky_sigma_major_arcsec` threshold can suppress broad-uncertainty frame explosion, at a recall tradeoff that depends on orbit uncertainty regime.

5. **Numerical divergence can originate in broad covariance sigma points.**
   A known failure involved a sigma-point variant producing non-convergent universal anomaly behavior and eventually `NaN` light-time values.

## Current Operational Guidance

- Keep strict fail-fast for unexpected numerical divergence (no silent ignore).
- Use covariance-informed pre-propagation viability policy to classify targets into:
  - `full`,
  - `time_limited`,
  - `skip` (explicitly counted).
- Keep on-sky major-axis filtering as the canonical post-propagation spatial uncertainty control.
- Compare policy sweeps by recording at least:
  - truth recall and truth frame coverage,
  - false positives and unknown final detections,
  - Stage-3 rejection counts and runtime.

## Pre-Propagation Threshold Semantics

Three independent controls are now used (no blended viability score for decisions):

1. **`sigma_r_over_r`**
   - Definition: `sigma_r / |r|`, where `sigma_r = sqrt(trace(cov_rr))`.
   - Interpretation: fractional 1-sigma position uncertainty relative to heliocentric/barycentric state radius.
   - Policy behavior: if above `preprop_max_sigma_r_over_r`, mark orbit as `time_limited`.
   - Reason code: `sigma_r_over_r_exceeds_max`.

2. **`covariance_condition`**
   - Definition: matrix condition number `cond(cov_6x6)`.
   - Interpretation: numerical instability / axis anisotropy proxy for 6D covariance inversion and sigma-point spread quality.
   - Policy behavior: if above `preprop_max_covariance_condition`, mark orbit as `skip`.
   - Reason code: `covariance_condition_exceeds_max`.

3. **`short_arc_proxy_days`**
   - Definition: `1 / max(sigma_v_over_v, eps)` where `sigma_v = sqrt(trace(cov_vv))`.
   - Interpretation: heuristic timescale for velocity certainty; smaller values imply short-arc/weakly constrained behavior.
   - Policy behavior (`dynamic_short_arc` mode): if below `preprop_short_arc_days_threshold`, mark orbit as `time_limited` with short-arc limit window.
   - Reason code: `short_arc_proxy_days_below_threshold`.

### Per-orbit audit fields (always returned)

Per-orbit summaries now include:
- policy decision/reason,
- trigger metric/value/threshold,
- targets evaluated before and after policy,
- complete-period coverage flag,
- applied time-limit days and first excluded target time,
- fail-fast stage/reason and timing context (`observation_time_mjd_tdb`, `t0`, `t1`, `dt_days`) when available.

## Tradeoff Interpretation Template

For each run family (e.g., mag-only vs mag+uncertainty, loose/moderate/strict):

- If FP drops with minimal truth loss, tighten that control in the affected uncertainty regime.
- If truth loss rises sharply at little additional FP gain, threshold is too strict for that regime.
- If unknown-final remains low but false-positive-final remains high, prioritize de-dup/linking-aware downstream criteria and uncertainty-regime-specific gating.

## Notes for Future Updates

When appending new results, include:

- input orbit population definition (including covariance source/model),
- detection sigma assumptions and floors,
- policy knobs and thresholds used,
- key totals and derived recall/coverage metrics,
- a short interpretation of FP vs recall tradeoff.
