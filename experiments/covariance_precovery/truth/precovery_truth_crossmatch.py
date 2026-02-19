from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import quivr as qv

from precovery.frame_db import HealpixFrame
from precovery.healpix_geom import radec_to_healpixel
from precovery.precovery_db import PrecoveryDatabase
from precovery.spherical_geom import haversine_distance_deg

from ..selection.subset_designations import read_subset_window
from .subset_truth_observations import TruthObservationByDesignation


class TruthPrecoveryCrossmatch(qv.Table):
    designation = qv.LargeStringColumn()
    obscode = qv.LargeStringColumn()
    truth_obsid = qv.LargeStringColumn()
    truth_time_mjd_utc = qv.Float64Column()
    truth_ra_deg = qv.Float64Column()
    truth_dec_deg = qv.Float64Column()

    matched = qv.BooleanColumn()
    match_exposure_id = qv.LargeStringColumn(nullable=True)
    match_dataset_id = qv.LargeStringColumn(nullable=True)
    match_observation_id = qv.LargeStringColumn(nullable=True)
    match_time_mjd_utc = qv.Float64Column(nullable=True)
    match_ra_deg = qv.Float64Column(nullable=True)
    match_dec_deg = qv.Float64Column(nullable=True)
    delta_t_sec = qv.Float64Column(nullable=True)
    distance_arcsec = qv.Float64Column(nullable=True)
    healpixel_nside = qv.Int64Column()
    healpixel = qv.Int64Column()


@dataclass(frozen=True)
class CrossmatchResult:
    out_parquet: Path
    out_meta_json: Path


def _frames_for_truth(
    *,
    index_db: Path,
    obscode: str,
    healpixel: int,
    mjd: float,
    dt_days: float,
) -> list[dict[str, object]]:
    conn = sqlite3.connect(str(index_db))
    try:
        rows = conn.execute(
            """
            SELECT
              dataset_id, obscode, exposure_id, filter,
              exposure_mjd_start, exposure_mjd_mid, exposure_duration,
              healpixel, data_uri, data_offset, data_length
            FROM frames
            WHERE obscode = ?
              AND healpixel = ?
              AND exposure_mjd_mid >= ?
              AND exposure_mjd_mid <= ?
            """,
            (str(obscode), int(healpixel), float(mjd - dt_days), float(mjd + dt_days)),
        ).fetchall()
    finally:
        conn.close()

    out: list[dict[str, object]] = []
    for r in rows:
        out.append(
            dict(
                dataset_id=r[0],
                obscode=r[1],
                exposure_id=r[2],
                filter=r[3],
                exposure_mjd_start=float(r[4]),
                exposure_mjd_mid=float(r[5]),
                exposure_duration=float(r[6]),
                healpixel=int(r[7]),
                data_uri=r[8],
                data_offset=int(r[9]),
                data_length=int(r[10]),
            )
        )
    return out


def crossmatch_truth_to_precovery_subset(
    *,
    subset_dir: Path,
    truth_parquet: Path | None = None,
    time_tol_sec: float = 60.0,
    dist_tol_arcsec: float = 5.0,
    out_tag: str | None = None,
) -> CrossmatchResult:
    win = read_subset_window(subset_dir)
    win.artifacts_dir.mkdir(parents=True, exist_ok=True)
    db = PrecoveryDatabase.from_dir(str(subset_dir), allow_version_mismatch=True)

    if truth_parquet is None:
        truth_parquet = win.artifacts_dir / "truth_observations_selected.parquet"
    truth = TruthObservationByDesignation.from_parquet(str(truth_parquet))

    nside = int(db.frames.healpix_nside)
    dt_days = float(time_tol_sec) / 86400.0
    dist_tol_deg = float(dist_tol_arcsec) / 3600.0

    # Cache observations by (data_uri, offset, length).
    obs_cache: dict[tuple[str, int, int], tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]] = {}

    out_rows: list[dict[str, object]] = []
    for i in range(len(truth)):
        designation = str(truth.designation[i].as_py())
        obscode = str(truth.obscode[i].as_py())
        truth_obsid = str(truth.obsid[i].as_py())
        mjd = float(truth.time_mjd_utc[i].as_py())
        ra = float(truth.ra_deg[i].as_py())
        dec = float(truth.dec_deg[i].as_py())
        hp = int(radec_to_healpixel(ra, dec, nside))

        frames = _frames_for_truth(
            index_db=Path(subset_dir) / "index.db",
            obscode=obscode,
            healpixel=hp,
            mjd=mjd,
            dt_days=dt_days,
        )

        best: dict[str, object] | None = None
        best_dist = float("inf")
        best_dt = float("inf")

        for fr in frames:
            key = (str(fr["data_uri"]), int(fr["data_offset"]), int(fr["data_length"]))
            if key in obs_cache:
                mjds, ras, decs, ids = obs_cache[key]
            else:
                hf = HealpixFrame.from_kwargs(
                    dataset_id=[str(fr["dataset_id"])],
                    obscode=[str(fr["obscode"])],
                    exposure_id=[str(fr["exposure_id"])],
                    filter=[str(fr["filter"])],
                    exposure_mjd_start=[float(fr["exposure_mjd_start"])],
                    exposure_mjd_mid=[float(fr["exposure_mjd_mid"])],
                    exposure_duration=[float(fr["exposure_duration"])],
                    healpixel=[int(fr["healpixel"])],
                    data_uri=[str(fr["data_uri"])],
                    data_offset=[int(fr["data_offset"])],
                    data_length=[int(fr["data_length"])],
                )
                obs = db.frames.get_observations(hf)
                mjds = obs.time.mjd().to_numpy(zero_copy_only=False).astype(np.float64)
                ras = obs.ra.to_numpy(zero_copy_only=False).astype(np.float64)
                decs = obs.dec.to_numpy(zero_copy_only=False).astype(np.float64)
                ids = [x.decode("utf8") if isinstance(x, (bytes, bytearray)) else str(x) for x in obs.id.to_pylist()]
                obs_cache[key] = (mjds, ras, decs, ids)

            # time window prefilter
            dt_sec = np.abs(mjds - mjd) * 86400.0
            cand_idx = np.where(dt_sec <= float(time_tol_sec))[0]
            if cand_idx.size == 0:
                continue
            d_deg = haversine_distance_deg(ras[cand_idx], ra, decs[cand_idx], dec)
            j = int(np.argmin(d_deg))
            d0 = float(d_deg[j])
            if d0 > dist_tol_deg:
                continue

            k = int(cand_idx[j])
            dt0 = float((mjds[k] - mjd) * 86400.0)
            # tie-break: smallest distance, then smallest |dt|
            if (d0 < best_dist) or (d0 == best_dist and abs(dt0) < abs(best_dt)):
                best_dist = d0
                best_dt = dt0
                best = dict(
                    exposure_id=str(fr["exposure_id"]),
                    dataset_id=str(fr["dataset_id"]),
                    observation_id=str(ids[k]),
                    time_mjd_utc=float(mjds[k]),
                    ra_deg=float(ras[k]),
                    dec_deg=float(decs[k]),
                    delta_t_sec=float(dt0),
                    distance_arcsec=float(d0 * 3600.0),
                )

        out_rows.append(
            dict(
                designation=designation,
                obscode=obscode,
                truth_obsid=truth_obsid,
                truth_time_mjd_utc=mjd,
                truth_ra_deg=ra,
                truth_dec_deg=dec,
                matched=bool(best is not None),
                match_exposure_id=None if best is None else best["exposure_id"],
                match_dataset_id=None if best is None else best["dataset_id"],
                match_observation_id=None if best is None else best["observation_id"],
                match_time_mjd_utc=None if best is None else best["time_mjd_utc"],
                match_ra_deg=None if best is None else best["ra_deg"],
                match_dec_deg=None if best is None else best["dec_deg"],
                delta_t_sec=None if best is None else best["delta_t_sec"],
                distance_arcsec=None if best is None else best["distance_arcsec"],
                healpixel_nside=nside,
                healpixel=hp,
            )
        )

    out_tbl = TruthPrecoveryCrossmatch.from_pyarrow(pa.Table.from_pylist(out_rows))
    tag = None if (out_tag is None or not str(out_tag).strip()) else str(out_tag).strip()
    out_name = "truth_precovery_crossmatch.parquet" if tag is None else f"truth_precovery_crossmatch_{tag}.parquet"
    out_parquet = win.artifacts_dir / out_name
    out_tbl.to_parquet(str(out_parquet))

    n_matched = int(sum(1 for r in out_rows if r["matched"]))
    meta = {
        "subset_dir": str(subset_dir),
        "truth_parquet": str(truth_parquet),
        "n_truth_obs": int(len(truth)),
        "n_matched": n_matched,
        "match_rate": (0.0 if len(truth) == 0 else float(n_matched) / float(len(truth))),
        "time_tol_sec": float(time_tol_sec),
        "dist_tol_arcsec": float(dist_tol_arcsec),
        "healpix_nside": int(nside),
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    meta_name = (
        "truth_precovery_crossmatch_meta.json" if tag is None else f"truth_precovery_crossmatch_meta_{tag}.json"
    )
    out_meta = win.artifacts_dir / meta_name
    out_meta.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")

    return CrossmatchResult(out_parquet=out_parquet, out_meta_json=out_meta)


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Crossmatch truth observations to detections in a local precovery subset.")
    p.add_argument("--subset-dir", type=str, required=True)
    p.add_argument(
        "--truth-parquet",
        type=str,
        default=None,
        help="Optional truth_observations_selected parquet (default: subset artifacts/truth_observations_selected.parquet).",
    )
    p.add_argument("--time-tol-sec", type=float, default=60.0)
    p.add_argument("--dist-tol-arcsec", type=float, default=5.0)
    p.add_argument(
        "--out-tag",
        type=str,
        default=None,
        help="Optional tag to avoid overwriting truth_precovery_crossmatch.parquet.",
    )
    args = p.parse_args()

    out = crossmatch_truth_to_precovery_subset(
        subset_dir=Path(args.subset_dir),
        truth_parquet=(None if args.truth_parquet is None else Path(args.truth_parquet)),
        time_tol_sec=float(args.time_tol_sec),
        dist_tol_arcsec=float(args.dist_tol_arcsec),
        out_tag=args.out_tag,
    )
    print(f"crossmatch_parquet={out.out_parquet}")
    print(f"meta_json={out.out_meta_json}")


if __name__ == "__main__":
    main()

