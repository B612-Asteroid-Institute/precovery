from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from astropy.time import Time

from precovery.precovery_db import PrecoveryDatabase
from precovery.search.frames_sqlite import fetch_frames_for_target_triples_sqlite

from ..time_key import mjd_to_time_key_us
from ...data.lazy_blobs import LazyBlobConfig, ensure_blob_local


@dataclass(frozen=True)
class ExportStats:
    out_dir: Path
    n_frames: int
    n_detections: int


def _year_month_from_mjd_utc(mjd: float) -> str:
    t = Time(float(mjd), format="mjd", scale="utc")
    return str(t.strftime("%Y-%m"))


def export_precovery_subset_to_parquet(
    *,
    subset_dir: Path,
    out_dir: Path,
    start_mjd_utc: float,
    end_mjd_utc: float,
    obscodes: tuple[str, ...],
    max_frames: int | None = None,
    chunk_frames: int = 10_000,
) -> ExportStats:
    """
    Export a slice of the local precovery DB (sqlite+blob) to parquet suitable for DuckDB/ClickHouse.

    The parquet schema matches the DuckDB backend expectations.
    """
    subset_dir = Path(subset_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    db = PrecoveryDatabase.from_dir(str(subset_dir), mode="r", allow_version_mismatch=True)
    idx = db.frames.idx

    # Enumerate frames in-range by scanning sqlite frames table (filter by obscode + time range).
    # We deliberately keep this simple; if needed we can add dataset filters later.
    import sqlite3

    conn = sqlite3.connect(str(idx.db_uri).replace("sqlite:///", "").split("?")[0])
    try:
        qs = ",".join(["?"] * len(obscodes))
        q = f"""
        SELECT obscode, exposure_mjd_mid, healpixel
        FROM frames
        WHERE exposure_mjd_mid >= ?
          AND exposure_mjd_mid < ?
          AND obscode IN ({qs})
        GROUP BY obscode, exposure_mjd_mid, healpixel
        ORDER BY exposure_mjd_mid ASC, obscode ASC
        """
        params = [float(start_mjd_utc), float(end_mjd_utc), *list(map(str, obscodes))]
        rows = conn.execute(q, params).fetchall()
    finally:
        conn.close()

    if not rows:
        return ExportStats(out_dir=out_dir, n_frames=0, n_detections=0)

    if max_frames is not None:
        rows = rows[: int(max_frames)]

    obsc = [str(r[0]) for r in rows]
    mjd_mid = np.asarray([float(r[1]) for r in rows], dtype=np.float64)
    hpix = np.asarray([int(r[2]) for r in rows], dtype=np.int64)

    frames = fetch_frames_for_target_triples_sqlite(
        index_db=Path(str(idx.db_uri).replace("sqlite:///", "").split("?")[0]),
        obscodes=obsc,
        exposure_mjd_mid=mjd_mid,
        healpixels=hpix,
        datasets=None,
    )
    if len(frames) == 0:
        return ExportStats(out_dir=out_dir, n_frames=0, n_detections=0)

    n_frames = int(len(frames))
    n_dets = 0
    part = 0
    for i0 in range(0, n_frames, int(chunk_frames)):
        frames_chunk = frames.take(list(range(i0, min(n_frames, i0 + int(chunk_frames)))))
        obs_list = db.frames.get_observations_many(frames_chunk)

        out_tables: list[pa.Table] = []
        for f, obs in zip(frames_chunk, obs_list):
            if len(obs) == 0:
                continue
            obscode = str(f.obscode[0].as_py())
            mjd_mid_i = float(f.exposure_mjd_mid[0].as_py())
            tkey = int(mjd_to_time_key_us(np.asarray([mjd_mid_i], dtype=np.float64))[0])
            healpixel = int(f.healpixel[0].as_py())
            ym = _year_month_from_mjd_utc(mjd_mid_i)

            out_tables.append(
                pa.table(
                    {
                        "obscode": pa.repeat(obscode, len(obs)),
                        "year_month": pa.repeat(ym, len(obs)),
                        "exposure_mjd_mid_utc": pa.repeat(mjd_mid_i, len(obs)),
                        "exposure_mjd_mid_key_us": pa.repeat(tkey, len(obs)),
                        "filter": pa.repeat(str(f.filter[0].as_py()), len(obs)),
                        "healpixel": pa.repeat(healpixel, len(obs)),
                        # Cast binary ids to strings in Arrow (avoid per-row Python decode).
                        "observation_id": pc.cast(obs.id, pa.large_string()),
                        "obstime_mjd_utc": obs.time.mjd(),
                        "ra_deg": obs.ra,
                        "dec_deg": obs.dec,
                        "ra_sigma_deg": obs.ra_sigma,
                        "dec_sigma_deg": obs.dec_sigma,
                        "mag": obs.mag,
                        "mag_sigma": obs.mag_sigma,
                    }
                )
            )

        if out_tables:
            tbl = pa.concat_tables(out_tables, promote=True)
            # Write as a simple part file; downstream can build dataset partitions if desired.
            pq.write_table(tbl, out_dir / f"part-{part:06d}.parquet")
            n_dets += int(tbl.num_rows)
            part += 1

    return ExportStats(out_dir=out_dir, n_frames=n_frames, n_detections=n_dets)


def export_precovery_keys_to_parquet(
    *,
    subset_dir: Path,
    out_dir: Path,
    obscodes: list[str],
    exposure_mjd_mid_utc: np.ndarray,
    healpixels: np.ndarray,
    chunk_frames: int = 10_000,
    lazy_blob_cfg: LazyBlobConfig | None = None,
) -> ExportStats:
    """
    Export exactly the requested (obscode, exposure_mjd_mid, healpixel) frame keys to parquet.

    This is the recommended path for small correctness/AB tests because it avoids exporting
    an entire month of detections when only a small set of frame keys is needed.
    """
    subset_dir = Path(subset_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if len(obscodes) == 0:
        return ExportStats(out_dir=out_dir, n_frames=0, n_detections=0)
    mjd_mid = np.asarray(exposure_mjd_mid_utc, dtype=np.float64)
    hpix = np.asarray(healpixels, dtype=np.int64)
    if not (len(obscodes) == int(mjd_mid.size) == int(hpix.size)):
        raise ValueError("Key arrays must have the same length.")

    db = PrecoveryDatabase.from_dir(str(subset_dir), mode="r", allow_version_mismatch=True)
    idx = db.frames.idx

    frames = fetch_frames_for_target_triples_sqlite(
        index_db=Path(str(idx.db_uri).replace("sqlite:///", "").split("?")[0]),
        obscodes=[str(x) for x in obscodes],
        exposure_mjd_mid=mjd_mid,
        healpixels=hpix,
        datasets=None,
    )
    if len(frames) == 0:
        return ExportStats(out_dir=out_dir, n_frames=0, n_detections=0)

    if lazy_blob_cfg is not None:
        # Download missing blobs before FrameDB opens files.
        data_uris = [str(x) for x in frames.data_uri.to_pylist()]
        for u in sorted(set(data_uris)):
            ensure_blob_local(db_dir=subset_dir, data_uri=u, cfg=lazy_blob_cfg)

    n_frames = int(len(frames))
    n_dets = 0
    part = 0
    for i0 in range(0, n_frames, int(chunk_frames)):
        frames_chunk = frames.take(list(range(i0, min(n_frames, i0 + int(chunk_frames)))))
        obs_list = db.frames.get_observations_many(frames_chunk)

        out_tables: list[pa.Table] = []
        for f, obs in zip(frames_chunk, obs_list):
            if len(obs) == 0:
                continue
            obscode = str(f.obscode[0].as_py())
            mjd_mid_i = float(f.exposure_mjd_mid[0].as_py())
            tkey = int(mjd_to_time_key_us(np.asarray([mjd_mid_i], dtype=np.float64))[0])
            healpixel = int(f.healpixel[0].as_py())
            ym = _year_month_from_mjd_utc(mjd_mid_i)

            out_tables.append(
                pa.table(
                    {
                        "obscode": pa.repeat(obscode, len(obs)),
                        "year_month": pa.repeat(ym, len(obs)),
                        "exposure_mjd_mid_utc": pa.repeat(mjd_mid_i, len(obs)),
                        "exposure_mjd_mid_key_us": pa.repeat(tkey, len(obs)),
                        "filter": pa.repeat(str(f.filter[0].as_py()), len(obs)),
                        "healpixel": pa.repeat(healpixel, len(obs)),
                        "observation_id": pc.cast(obs.id, pa.large_string()),
                        "obstime_mjd_utc": obs.time.mjd(),
                        "ra_deg": obs.ra,
                        "dec_deg": obs.dec,
                        "ra_sigma_deg": obs.ra_sigma,
                        "dec_sigma_deg": obs.dec_sigma,
                        "mag": obs.mag,
                        "mag_sigma": obs.mag_sigma,
                    }
                )
            )

        if out_tables:
            tbl = pa.concat_tables(out_tables, promote=True)
            pq.write_table(tbl, out_dir / f"part-{part:06d}.parquet")
            n_dets += int(tbl.num_rows)
            part += 1

    return ExportStats(out_dir=out_dir, n_frames=n_frames, n_detections=n_dets)

