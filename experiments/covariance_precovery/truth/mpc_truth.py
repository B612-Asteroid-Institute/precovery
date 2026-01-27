from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Sequence

import numpy as np
import pyarrow as pa
import quivr as qv

from precovery.spherical_geom import haversine_distance_deg

from ..selection.bq_select import BqConfig


def _run_bq_json(query: str) -> list[dict[str, Any]]:
    proc = subprocess.run(
        ["bq", "query", "--nouse_legacy_sql", "--format=json", "--max_rows=100000", query],
        check=True,
        capture_output=True,
        text=True,
    )
    out = proc.stdout.strip()
    return [] if not out else json.loads(out)


def _utc_ts(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt = dt.astimezone(timezone.utc)
    return dt.isoformat().replace("+00:00", "Z")


def _unix_to_mjd(unix_seconds: np.ndarray) -> np.ndarray:
    # 1970-01-01 00:00:00 UTC is MJD 40587.0
    return unix_seconds / 86400.0 + 40587.0


class TruthObservation(qv.Table):
    mpc_id = qv.Int64Column()
    obscode = qv.LargeStringColumn()
    obsid = qv.LargeStringColumn()
    time_mjd_utc = qv.Float64Column()
    ra_deg = qv.Float64Column()
    dec_deg = qv.Float64Column()


def fetch_truth_observations(
    *,
    cfg: BqConfig,
    mpc_ids: Sequence[int],
    obscodes: Sequence[str],
    start_utc: datetime,
    end_utc: datetime,
    chunk_size: int = 5000,
) -> TruthObservation:
    """
    Fetch truth observations from MPCQ/BigQuery for the chosen objects, stations, and time range.
    """
    if not mpc_ids:
        return TruthObservation.empty()
    if not obscodes:
        raise ValueError("obscodes cannot be empty")

    stn_list = ", ".join(f"'{s}'" for s in obscodes)
    t0 = _utc_ts(start_utc)
    t1 = _utc_ts(end_utc)

    out = TruthObservation.empty()
    for i0 in range(0, len(mpc_ids), int(chunk_size)):
        chunk = mpc_ids[i0 : i0 + int(chunk_size)]
        id_list = ", ".join(str(int(x)) for x in chunk)
        query = f"""
        SELECT
          id AS mpc_id,
          stn AS obscode,
          obsid,
          obstime,
          ra_f64 AS ra_deg,
          dec_f64 AS dec_deg
        FROM `{cfg.obs_table}`
        WHERE id IN ({id_list})
          AND stn IN ({stn_list})
          AND obstime >= TIMESTAMP('{t0}')
          AND obstime <  TIMESTAMP('{t1}')
        """
        rows = _run_bq_json(query)
        if not rows:
            continue

        obstime = np.array([r["obstime"] for r in rows], dtype="datetime64[ns]")
        unix_s = obstime.astype("datetime64[s]").astype(np.int64).astype(np.float64)
        mjd = _unix_to_mjd(unix_s)

        chunk_tbl = TruthObservation.from_kwargs(
            mpc_id=[int(r["mpc_id"]) for r in rows],
            obscode=[str(r["obscode"]) for r in rows],
            obsid=[str(r["obsid"]) for r in rows],
            time_mjd_utc=mjd,
            ra_deg=[float(r["ra_deg"]) for r in rows],
            dec_deg=[float(r["dec_deg"]) for r in rows],
        )
        out = qv.concatenate([out, chunk_tbl])

    return out


class TruthMatch(qv.Table):
    obscode = qv.LargeStringColumn()
    truth_obsid = qv.LargeStringColumn()
    truth_time_mjd_utc = qv.Float64Column()
    truth_ra_deg = qv.Float64Column()
    truth_dec_deg = qv.Float64Column()
    candidate_observation_id = qv.LargeStringColumn()
    candidate_time_mjd_utc = qv.Float64Column()
    candidate_ra_deg = qv.Float64Column()
    candidate_dec_deg = qv.Float64Column()
    delta_t_sec = qv.Float64Column()
    distance_arcsec = qv.Float64Column()


@dataclass(frozen=True)
class MatchSummary:
    n_truth: int
    n_candidates: int
    n_truth_matched: int
    n_candidates_matched: int

    @property
    def truth_recall(self) -> float:
        return 0.0 if self.n_truth == 0 else self.n_truth_matched / self.n_truth


def match_candidates_to_truth(
    *,
    truth: TruthObservation,
    candidates: qv.Table,  # expected: PrecoveryCandidates-like
    time_tol_sec: float = 2.0,
    dist_tol_arcsec: float = 2.0,
) -> tuple[TruthMatch, MatchSummary]:
    """
    Match precovery candidates to truth observations by (obscode, time, sky position).

    This is intentionally robust to differing observation_id schemes: it does NOT assume
    precovery's `observation_id` equals MPCQ `obsid`.
    """
    # Extract needed candidate columns without importing precovery types.
    for col in ("obscode", "time", "ra_deg", "dec_deg", "observation_id"):
        if col not in candidates.table.column_names:
            raise ValueError(f"Candidates table missing required column {col!r}")

    cand_obscode = candidates.obscode.to_pylist()
    cand_mjd = candidates.time.mjd().to_numpy(zero_copy_only=False).astype(np.float64)
    cand_ra = candidates.ra_deg.to_numpy(zero_copy_only=False).astype(np.float64)
    cand_dec = candidates.dec_deg.to_numpy(zero_copy_only=False).astype(np.float64)
    cand_id = candidates.observation_id.to_pylist()

    dt_days = float(time_tol_sec) / 86400.0
    dist_tol_deg = float(dist_tol_arcsec) / 3600.0

    matches_out = TruthMatch.empty()

    truth_matched = np.zeros(len(truth), dtype=bool)
    cand_matched = np.zeros(len(cand_mjd), dtype=bool)

    # Group by obscode for cheap filtering.
    truth_obscodes = truth.obscode.to_pylist()
    for obscode in sorted(set(truth_obscodes) & set(cand_obscode)):
        t_idx = np.array([i for i, s in enumerate(truth_obscodes) if s == obscode], dtype=np.int64)
        c_idx = np.array([i for i, s in enumerate(cand_obscode) if s == obscode], dtype=np.int64)
        if len(t_idx) == 0 or len(c_idx) == 0:
            continue

        t_mjd = truth.time_mjd_utc.to_numpy(zero_copy_only=False)[t_idx].astype(np.float64)
        t_ra = truth.ra_deg.to_numpy(zero_copy_only=False)[t_idx].astype(np.float64)
        t_dec = truth.dec_deg.to_numpy(zero_copy_only=False)[t_idx].astype(np.float64)

        c_mjd = cand_mjd[c_idx]
        c_ra = cand_ra[c_idx]
        c_dec = cand_dec[c_idx]

        order = np.argsort(c_mjd)
        c_idx = c_idx[order]
        c_mjd = c_mjd[order]
        c_ra = c_ra[order]
        c_dec = c_dec[order]

        for j in range(len(t_idx)):
            mjd0 = t_mjd[j]
            lo = np.searchsorted(c_mjd, mjd0 - dt_days, side="left")
            hi = np.searchsorted(c_mjd, mjd0 + dt_days, side="right")
            if hi <= lo:
                continue

            # Compute angular distances to candidates in the time window.
            win_idx = c_idx[lo:hi]
            d_deg = haversine_distance_deg(
                cand_ra[win_idx], t_ra[j], cand_dec[win_idx], t_dec[j]
            )
            k = int(np.argmin(d_deg))
            best_deg = float(d_deg[k])
            if best_deg > dist_tol_deg:
                continue

            best_cand = int(win_idx[k])
            truth_matched[int(t_idx[j])] = True
            cand_matched[best_cand] = True

            dt_sec = float((cand_mjd[best_cand] - mjd0) * 86400.0)
            matches_out = qv.concatenate(
                [
                    matches_out,
                    TruthMatch.from_kwargs(
                        obscode=[obscode],
                        truth_obsid=[truth.obsid[int(t_idx[j])].as_py()],
                        truth_time_mjd_utc=[mjd0],
                        truth_ra_deg=[float(t_ra[j])],
                        truth_dec_deg=[float(t_dec[j])],
                        candidate_observation_id=[str(cand_id[best_cand])],
                        candidate_time_mjd_utc=[float(cand_mjd[best_cand])],
                        candidate_ra_deg=[float(cand_ra[best_cand])],
                        candidate_dec_deg=[float(cand_dec[best_cand])],
                        delta_t_sec=[dt_sec],
                        distance_arcsec=[best_deg * 3600.0],
                    ),
                ]
            )

    summary = MatchSummary(
        n_truth=int(len(truth)),
        n_candidates=int(len(cand_mjd)),
        n_truth_matched=int(truth_matched.sum()),
        n_candidates_matched=int(cand_matched.sum()),
    )
    return matches_out, summary

