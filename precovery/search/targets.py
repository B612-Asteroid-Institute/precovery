from __future__ import annotations

from array import array
from pathlib import Path

import pyarrow as pa

from adam_core.time import Timestamp


def enumerate_distinct_frame_time_targets_sqlite(
    *,
    index_db: Path,
    start_mjd: float,
    end_mjd: float,
    datasets: set[str] | None = None,
    chunk_size: int = 100000,
) -> tuple[pa.Array, pa.Array, Timestamp]:
    """
    Return all distinct (obscode, exposure_mjd_mid) pairs in [start_mjd, end_mjd).

    Why this exists
    ---------------
    Distinct target enumeration is a critical hot path at scale. Using raw sqlite3
    avoids SQLAlchemy reflection/Row object overhead and keeps query planning predictable.

    Performance notes
    -----------------
    - Uses a single GROUP BY query.
    - Assumes the `frames_exposure_mjd_mid_obscode_idx` index exists (created by DB migration).
    """
    import sqlite3

    if not Path(index_db).exists():
        raise FileNotFoundError(f"Missing index.db: {index_db}")

    conn = sqlite3.connect(str(index_db))
    try:
        base = """
        SELECT obscode, exposure_mjd_mid
        FROM frames
        WHERE exposure_mjd_mid >= ?
          AND exposure_mjd_mid < ?
        """
        params: list[object] = [float(start_mjd), float(end_mjd)]
        if datasets:
            ds = sorted({str(x).strip() for x in datasets if str(x).strip()})
            if ds:
                qs = ",".join(["?"] * len(ds))
                base += f" AND dataset_id IN ({qs})\n"
                params.extend(ds)

        q = (
            base
            + """
        GROUP BY obscode, exposure_mjd_mid
        ORDER BY exposure_mjd_mid ASC, obscode ASC
        """
        )

        code_cache: dict[str, str] = {}
        codes: list[str] = []
        mjds = array("d")

        cur = conn.execute(q, params)
        while True:
            rows = cur.fetchmany(int(chunk_size))
            if not rows:
                break
            for code, mjd in rows:
                c = str(code)
                c = code_cache.setdefault(c, c)
                codes.append(c)
                mjds.append(float(mjd))
    finally:
        conn.close()

    if len(mjds) == 0:
        return (
            pa.array([], type=pa.large_string()),
            pa.array([], type=pa.float64()),
            Timestamp.from_mjd([], scale="utc"),
        )
    mjd_arr = pa.array(mjds, type=pa.float64())
    return (
        pa.array(codes, type=pa.large_string()),
        mjd_arr,
        Timestamp.from_mjd(mjd_arr, scale="utc"),
    )

