from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from precovery.precovery_db import PrecoveryDatabase
from precovery.frame_db import HealpixFrame
from precovery.search.frames_sqlite import fetch_frames_for_target_triples_sqlite

from ..gate import accepted_counts_python
from ..time_key import mjd_to_time_key_us
from ..types import AcceptedCounts, BenchTargets, CandidateDetections, PredictedTargets, PredictedTriples, SubsetPaths
from ...data.lazy_blobs import LazyBlobConfig, ensure_blob_local
from .protocols import BackendCapabilities, BenchBackend, GateParams


def _unique_join_keys_from_triples(triples: PredictedTriples) -> pa.Table:
    """
    Return unique (obscode, exposure_mjd_mid, healpixel) keys to query sqlite.
    """
    if len(triples) == 0:
        return pa.table(
            {
                "obscode": pa.array([], type=pa.large_string()),
                "exposure_mjd_mid_utc": pa.array([], type=pa.float64()),
                "healpixel": pa.array([], type=pa.int64()),
            }
        )
    keys = triples.table.select(["obscode", "exposure_mjd_mid_utc", "healpixel"])
    # Arrow-native dedupe (faster and less memory than Python set(zip(...))).
    return keys.group_by(["obscode", "exposure_mjd_mid_utc", "healpixel"]).aggregate([])


@dataclass
class BaselineSqliteBlobBackend(BenchBackend):
    """
    Baseline backend using the existing precovery subset layout:
      - `index.db` (sqlite frames table)
      - `data/.../*.data` packed observation blobs
    """

    name: str = "baseline_sqlite_blobs"
    capabilities: BackendCapabilities = BackendCapabilities(
        supports_enumerate_targets=True,
        supports_sql_gate_counts=False,
        supports_sql_gate_rows=False,
    )
    lazy_blob_cfg: LazyBlobConfig | None = None

    def enumerate_targets(
        self,
        *,
        subset: SubsetPaths,
        start_mjd_utc: float,
        end_mjd_utc: float,
        obscodes: tuple[str, ...],
    ) -> BenchTargets:
        index_db = Path(subset.index_db)
        if not index_db.exists():
            raise FileNotFoundError(f"Missing index.db: {index_db}")
        if not obscodes:
            raise ValueError("obscodes cannot be empty")

        codes: list[str] = []
        mids: list[float] = []
        filters: list[str] = []

        conn = sqlite3.connect(str(index_db))
        try:
            qs = ",".join(["?"] * len(obscodes))
            q = f"""
            SELECT obscode, exposure_mjd_mid, filter
            FROM frames
            WHERE exposure_mjd_mid >= ?
              AND exposure_mjd_mid < ?
              AND obscode IN ({qs})
            GROUP BY obscode, exposure_mjd_mid, filter
            ORDER BY exposure_mjd_mid ASC, obscode ASC, filter ASC
            """
            params: list[object] = [float(start_mjd_utc), float(end_mjd_utc), *list(obscodes)]
            cur = conn.execute(q, params)
            while True:
                rows = cur.fetchmany(100000)
                if not rows:
                    break
                for code, mjd_mid, filt in rows:
                    codes.append(str(code))
                    mids.append(float(mjd_mid))
                    filters.append(str(filt))
        finally:
            conn.close()

        if not mids:
            return BenchTargets.empty()
        mids_arr = np.asarray(mids, dtype=np.float64)
        keys = mjd_to_time_key_us(mids_arr)
        return BenchTargets.from_kwargs(
            obscode=codes,
            exposure_mjd_mid_utc=mids_arr,
            filter=filters,
            exposure_mjd_mid_key_us=keys,
        )

    def _ensure_blobs_local(self, *, subset_dir: Path, frames: HealpixFrame) -> None:
        if self.lazy_blob_cfg is None:
            return
        # Download missing blobs before FrameDB opens files.
        data_uris = [str(x) for x in frames.data_uri.to_pylist()]
        for u in sorted(set(data_uris)):
            ensure_blob_local(db_dir=subset_dir, data_uri=u, cfg=self.lazy_blob_cfg)

    def fetch_candidates(
        self,
        *,
        subset: SubsetPaths,
        triples: PredictedTriples,
        limit: int | None = None,
    ) -> CandidateDetections:
        if len(triples) == 0:
            return CandidateDetections.empty()

        subset_dir = Path(subset.subset_dir)
        db = PrecoveryDatabase.from_dir(str(subset_dir), mode="r", allow_version_mismatch=True)

        # Query sqlite once for unique frame keys.
        uniq = _unique_join_keys_from_triples(triples)
        frames = fetch_frames_for_target_triples_sqlite(
            index_db=Path(subset.index_db),
            obscodes=[str(x) for x in uniq["obscode"].to_pylist()],
            exposure_mjd_mid=uniq["exposure_mjd_mid_utc"].to_numpy(zero_copy_only=False).astype(np.float64),
            healpixels=uniq["healpixel"].to_numpy(zero_copy_only=False).astype(np.int64),
            datasets=None,
        )
        if len(frames) == 0:
            return CandidateDetections.empty()

        self._ensure_blobs_local(subset_dir=subset_dir, frames=frames)

        # Build a vectorized mapping from triples -> frame_row, joining on stable integer time key.
        frame_row = pa.array(np.arange(len(frames), dtype=np.int64), type=pa.int64())
        frame_mjd = frames.exposure_mjd_mid.to_numpy(zero_copy_only=False).astype(np.float64)
        frame_time_key = pa.array(mjd_to_time_key_us(frame_mjd), type=pa.int64())
        frames_tbl = pa.table(
            {
                "frame_row": frame_row,
                "obscode": frames.obscode,
                "exposure_mjd_mid_key_us": frame_time_key,
                "healpixel": frames.healpixel,
            }
        )
        triples_tbl = triples.table.select(
            ["orbit_id", "target_idx", "obscode", "exposure_mjd_mid_key_us", "healpixel"]
        )
        map_tbl = triples_tbl.join(
            frames_tbl,
            keys=["obscode", "exposure_mjd_mid_key_us", "healpixel"],
            join_type="inner",
        )
        if map_tbl.num_rows == 0:
            return CandidateDetections.empty()

        obs_list = db.frames.get_observations_many(frames)

        # Flatten detections once per frame, then use Arrow joins to replicate per (orbit_id,target_idx).
        det_tables: list[pa.Table] = []
        for i, obs in enumerate(obs_list):
            if len(obs) == 0:
                continue
            t = pa.table(
                {
                    "frame_row": pa.repeat(int(i), len(obs)),
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
            det_tables.append(t)

        if not det_tables:
            return CandidateDetections.empty()

        det_tbl = pa.concat_tables(det_tables, promote=True)
        out_tbl = map_tbl.join(det_tbl, keys=["frame_row"], join_type="inner")
        out_tbl = out_tbl.select(
            [
                "orbit_id",
                "target_idx",
                "observation_id",
                "obstime_mjd_utc",
                "ra_deg",
                "dec_deg",
                "ra_sigma_deg",
                "dec_sigma_deg",
                "mag",
                "mag_sigma",
            ]
        )
        if limit is not None:
            out_tbl = out_tbl.slice(0, int(limit))
        return CandidateDetections.from_pyarrow(out_tbl)

    def count_accepted(
        self,
        *,
        subset: SubsetPaths,
        triples: PredictedTriples,
        preds: PredictedTargets,
        gate: GateParams,
    ) -> AcceptedCounts:
        cands = self.fetch_candidates(subset=subset, triples=triples, limit=None)
        return accepted_counts_python(candidates=cands, preds=preds, gate=gate)

