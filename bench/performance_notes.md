## Performance notes (covariance precovery harness)

### Stage 2 non-truth target enumeration (critical)
Stage 2’s non-truth workload depends on the number of unique propagation targets:

- targets = distinct `(obscode, exposure_mjd_mid)` in the requested MJD range

On large subsets (e.g. `full_precovery_n32`), the detections parquet dataset can have **tens of millions** of rows in-range.

#### Current implementation (no SQLite)

The legacy SQLite/SQLAlchemy `index.db` path has been removed from `precovery`. Target enumeration is performed via a backend adapter; for local runs we typically use the DuckDB-parquet backend.

Concretely, `DuckDbParquetBackend.enumerate_targets(...)` executes a single DuckDB query of the form:

```sql
SELECT DISTINCT obscode, exposure_mjd_mid_utc, exposure_mjd_mid_key_us, filter
FROM det
WHERE exposure_mjd_mid_utc >= ? AND exposure_mjd_mid_utc < ?
  AND obscode IN (...)
ORDER BY exposure_mjd_mid_utc ASC, obscode ASC;
```

#### What to do

- For large-window, non-truth runs, ensure your `--backend duckdb` selection includes `--duckdb-parquet <dataset>` so the harness can use DuckDB for enumeration.
- Prefer month/window filtering by `exposure_mjd_mid_utc` and keep `exposure_mjd_mid_key_us` materialized in the parquet for stable joins downstream.

