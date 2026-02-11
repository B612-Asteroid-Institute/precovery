## Performance notes (covariance precovery harness)

### Stage 2 non-truth target enumeration (critical)
Stage 2’s non-truth workload depends on the number of unique propagation targets:

- targets = distinct `(obscode, exposure_mjd_mid)` in the requested MJD range

On large subsets (e.g. `full_precovery_n32`), the `frames` table can have **tens of millions** of rows in-range.

#### What to do
- Use a **single SQLite query** to compute the distinct target list:

```sql
SELECT obscode, exposure_mjd_mid
FROM frames
WHERE exposure_mjd_mid >= ? AND exposure_mjd_mid < ?
GROUP BY obscode, exposure_mjd_mid
ORDER BY exposure_mjd_mid ASC, obscode ASC;
```

- Ensure the following index exists (additive + safe):

```sql
CREATE INDEX IF NOT EXISTS frames_exposure_mjd_mid_obscode_idx
  ON frames(exposure_mjd_mid, obscode);
```

#### Why
Avoid streaming the distinct results through SQLAlchemy/Python. That adds avoidable overhead.

With the raw SQLite `GROUP BY` approach + index, the distinct-target extraction for the full 4-year window is **O(seconds)** on our local `index.db`.

