from __future__ import annotations

from array import array
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
    n = int(len(obscodes))
    mjd = np.asarray(exposure_mjd_mid, dtype=np.float64)
    hpix = np.asarray(healpixels, dtype=np.int64)

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

        insert_batch = 100_000
        for start in range(0, n, int(insert_batch)):
            stop = min(n, start + int(insert_batch))
            rows = [
                (str(obscodes[i]), float(mjd[i]), int(hpix[i]))
                for i in range(int(start), int(stop))
            ]
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
        # Stream results to reduce peak memory for large joins.
        fetch_batch = 100_000
        ids: list[object] = []
        dataset_ids: list[object] = []
        obscodes2: list[object] = []
        exposure_ids: list[object] = []
        filters: list[object] = []
        exposure_mjd_starts = array("d")
        exposure_mjd_mids = array("d")
        exposure_durations = array("d")
        hpix_out = array("q")
        data_uris: list[object] = []
        data_offsets = array("q")
        data_lengths = array("q")

        while True:
            out = cur.fetchmany(int(fetch_batch))
            if not out:
                break
            for (
                id_i,
                dataset_id_i,
                obscode_i,
                exposure_id_i,
                filter_i,
                mjd_start_i,
                mjd_mid_i,
                duration_i,
                hp_i,
                uri_i,
                off_i,
                len_i,
            ) in out:
                ids.append(id_i)
                dataset_ids.append(dataset_id_i)
                obscodes2.append(obscode_i)
                exposure_ids.append(exposure_id_i)
                filters.append(filter_i)
                exposure_mjd_starts.append(float(mjd_start_i))
                exposure_mjd_mids.append(float(mjd_mid_i))
                exposure_durations.append(float(duration_i))
                hpix_out.append(int(hp_i))
                data_uris.append(uri_i)
                data_offsets.append(int(off_i))
                data_lengths.append(int(len_i))
    finally:
        try:
            conn.execute("DROP TABLE IF EXISTS tmp_targets")
        except Exception:
            pass
        conn.close()

    if not ids:
        return HealpixFrame.empty()

    return HealpixFrame.from_kwargs(
        id=ids,
        dataset_id=dataset_ids,
        obscode=obscodes2,
        exposure_id=exposure_ids,
        filter=filters,
        exposure_mjd_start=exposure_mjd_starts,
        exposure_mjd_mid=exposure_mjd_mids,
        exposure_duration=exposure_durations,
        healpixel=hpix_out,
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

