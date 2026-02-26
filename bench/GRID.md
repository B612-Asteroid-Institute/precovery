# Covariance precovery experiment grid

This document records the **Stage 2 → Stage 3** combination grid: what Stage 2 outputs exist, which Stage 3 footprint representations are valid for those outputs, and what’s currently implemented.

## Stage 2 output shapes

Stage 2 writes one (or more) of these artifacts per strategy:

- **mean ephemeris**: `strategies/<strategy>/mean_ephemeris/part-*.parquet`
  - Contains topocentric `Ephemeris` rows (lon/lat/time) and sometimes covariance.
- **variant ephemeris**: `strategies/<strategy>/<variant_kind>/variants_ephemeris/part-*.parquet`
  - Contains topocentric **`VariantEphemeris`** rows (lon/lat/time + `variant_id`, `weights`, `weights_cov`).
  - This is the particle cloud used directly for sample footprints or for covariance reconstruction via `VariantEphemeris.collapse(...)`.

## Stage 3 footprint representations

Stage 3 can compute frame healpixel selections using any of:

### Mean-only (no covariance required)

- `point`: predicted point pixel

### Covariance-derived (requires covariance on mean ephemeris)

- `cov_disc`: major-axis bound + `query_disc`
- `cov_mc`: sample 2×2 on-sky covariance → pixels

### Sample-derived (requires variant ephemeris)

- `sample_direct`: pixels from samples (+ neighbor/dilation)
- `sample_corridor`: buffered corridor/tube from samples

### Reconstructed covariance (requires variant ephemeris)

These reconstruct an `Ephemeris` covariance from `VariantEphemeris` using `VariantEphemeris.collapse(...)`, then apply the covariance-derived footprint:

- `cov_disc_reconstructed`
- `cov_mc_reconstructed`

## Validity matrix (by Stage 2 output)

Legend:
- ✅ valid + implemented
- ⚠️ valid but can error (recorded in Stage 3 metrics)
- ❌ not applicable (missing required inputs)

| Stage 2 output available | Stage 3 footprint | Status |
|---|---:|---:|
| mean ephem (no cov) | `point` | ✅ |
| mean ephem (no cov) | `cov_disc`, `cov_mc` | ❌ |
| mean ephem **with** cov | `cov_disc` | ✅ |
| mean ephem **with** cov | `cov_mc` | ✅ |
| variant ephem | `sample_direct` | ✅ |
| variant ephem | `sample_corridor` | ✅ |
| variant ephem | `cov_*_reconstructed` | ✅ |

## Implementation notes

- **No silent fallbacks**: Stage 3 records per-(strategy, footprint) errors (`n_errors`, first `error` string) and continues other combinations.
- **Why convexity matters**: `healpy.query_polygon` can hard-abort in C++ if the polygon is non-convex or degenerate. We precheck and raise a Python error before calling into healpy so Stage 3 can record it instead of crashing.
- **Truth-only runs**: Stage 2 supports `--only-truth-orbits` to filter the orbit set down to only objects that have ≥1 crossmatched truth detection in the subset window (currently **144 of 197** for `subset_2019-08_atlas-ztf-nsc`).

## Known upstream TODOs

- Add `VariantEphemeris.collapse_by_object_id()` upstream (adam_core) and switch Stage 3 reconstructed-cov path to use it (avoid constructing a 1-row mean ephemeris in Stage 3).

