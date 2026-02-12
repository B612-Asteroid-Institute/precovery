from __future__ import annotations

from pathlib import Path

import numpy as np

from precovery.frame_db import HealpixFrame
from precovery.search.types import TargetPixels


def fetch_frames_for_target_triples_sqlite(
    *,
    index_db: Path,
    obscodes: list[str],
    exposure_mjd_mid: np.ndarray,
    healpixels: np.ndarray,
    datasets: set[str] | None = None,
) -> HealpixFrame:
    """
    Fetch all frame rows matching (obscode, exposure_mjd_mid, healpixel) triples.

    This uses a TEMP table join to avoid giant `IN (...)` predicates.
    """
    import sqlite3

    if not Path(index_db).exists():
        raise FileNotFoundError(f"Missing index.db: {index_db}")
    if len(obscodes) == 0:
        return HealpixFrame.empty()

    if not (len(obscodes) == len(exposure_mjd_mid) == len(healpixels)):
        raise ValueError("Triple arrays must have the same length.")

    ds = sorted({str(x).strip() for x in (datasets or set()) if str(x).strip()})

    conn = sqlite3.connect(str(index_db))
    try:
        # Deduplicate triples inside SQLite to avoid Python `set(zip(...))` memory spikes.
        # PRIMARY KEY provides the needed index for the subsequent join.
        conn.execute(
            "CREATE TEMP TABLE tmp_targets ("
            "obscode TEXT NOT NULL, "
            "exposure_mjd_mid REAL NOT NULL, "
            "healpixel INTEGER NOT NULL, "
            "PRIMARY KEY (obscode, exposure_mjd_mid, healpixel)"
            ") WITHOUT ROWID"
        )

        rows = list(
            zip(
                (str(x) for x in obscodes),
                (float(x) for x in np.asarray(exposure_mjd_mid, dtype=np.float64).tolist()),
                (int(x) for x in np.asarray(healpixels, dtype=np.int64).tolist()),
            )
        )
        conn.executemany(
            "INSERT OR IGNORE INTO tmp_targets(obscode, exposure_mjd_mid, healpixel) VALUES (?,?,?)",
            rows,
        )

        q = """
        SELECT
          f.id,
          f.dataset_id,
          f.obscode,
          f.exposure_id,
          f.filter,
          f.exposure_mjd_start,
          f.exposure_mjd_mid,
          f.exposure_duration,
          f.healpixel,
          f.data_uri,
          f.data_offset,
          f.data_length
        FROM frames f
        INNER JOIN tmp_targets t
          ON f.obscode = t.obscode
         AND f.exposure_mjd_mid = t.exposure_mjd_mid
         AND f.healpixel = t.healpixel
        """
        params: list[object] = []
        if ds:
            qs = ",".join(["?"] * len(ds))
            q += f" WHERE f.dataset_id IN ({qs})"
            params.extend(ds)

        cur = conn.execute(q, params)
        out = cur.fetchall()
    finally:
        try:
            conn.execute("DROP TABLE IF EXISTS tmp_targets")
        except Exception:
            pass
        conn.close()

    if not out:
        return HealpixFrame.empty()

    (
        ids,
        dataset_ids,
        obscodes2,
        exposure_ids,
        filters,
        exposure_mjd_starts,
        exposure_mjd_mids,
        exposure_durations,
        hpix,
        data_uris,
        data_offsets,
        data_lengths,
    ) = zip(*out)

    return HealpixFrame.from_kwargs(
        id=ids,
        dataset_id=dataset_ids,
        obscode=obscodes2,
        exposure_id=exposure_ids,
        filter=filters,
        exposure_mjd_start=exposure_mjd_starts,
        exposure_mjd_mid=exposure_mjd_mids,
        exposure_duration=exposure_durations,
        healpixel=hpix,
        data_uri=data_uris,
        data_offset=data_offsets,
        data_length=data_lengths,
    )


def fetch_frames_for_pixels_sqlite(
    *,
    index_db: Path,
    pixels: TargetPixels,
    datasets: set[str] | None = None,
) -> HealpixFrame:
    """
    Convenience wrapper: fetch frames for a `TargetPixels` table.
    """
    if len(pixels) == 0:
        return HealpixFrame.empty()
    return fetch_frames_for_target_triples_sqlite(
        index_db=index_db,
        obscodes=[str(x) for x in pixels.obscode.to_pylist()],
        exposure_mjd_mid=np.asarray(pixels.exposure_mjd_mid.to_pylist(), dtype=np.float64),
        healpixels=np.asarray(pixels.healpixel.to_pylist(), dtype=np.int64),
        datasets=datasets,
    )

