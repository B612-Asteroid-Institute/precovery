from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import json
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import requests


@dataclass(frozen=True)
class StratumModel:
    name: str
    weight: float
    pos_log_mu_au: float
    pos_log_sigma_au: float
    anis_pos_log_mu: float
    anis_pos_log_sigma: float
    vel_log_mu_au_per_d: float
    vel_log_sigma_au_per_d: float
    anis_vel_log_mu: float
    anis_vel_log_sigma: float


@dataclass(frozen=True)
class SyntheticCovarianceModel:
    seed: int
    strata: tuple[StratumModel, ...]
    calibration_generated_at_utc: str
    calibration_rows: int
    sbdb_params: dict[str, Any]

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "seed": int(self.seed),
            "strata": [
                {
                    "name": s.name,
                    "weight": float(s.weight),
                    "pos_log_mu_au": float(s.pos_log_mu_au),
                    "pos_log_sigma_au": float(s.pos_log_sigma_au),
                    "anis_pos_log_mu": float(s.anis_pos_log_mu),
                    "anis_pos_log_sigma": float(s.anis_pos_log_sigma),
                    "vel_log_mu_au_per_d": float(s.vel_log_mu_au_per_d),
                    "vel_log_sigma_au_per_d": float(s.vel_log_sigma_au_per_d),
                    "anis_vel_log_mu": float(s.anis_vel_log_mu),
                    "anis_vel_log_sigma": float(s.anis_vel_log_sigma),
                }
                for s in self.strata
            ],
            "calibration_generated_at_utc": str(self.calibration_generated_at_utc),
            "calibration_rows": int(self.calibration_rows),
            "sbdb_params": dict(self.sbdb_params),
        }


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_float(x: Any) -> float | None:
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None


def _safe_int(x: Any) -> int | None:
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    try:
        return int(float(s))
    except Exception:
        return None


def _fetch_real_neo_candidate_table_from_sbdb(
    *,
    max_rows: int,
    page_size: int,
    timeout_sec: float,
) -> pa.Table:
    """
    Fetch deterministic SBDB candidate metadata for real NEO asteroids.
    """
    want = max(0, int(max_rows))
    if want <= 0:
        return pa.table(
            {
                "pdes": pa.array([], type=pa.large_string()),
                "condition_code": pa.array([], type=pa.int64()),
                "data_arc_days": pa.array([], type=pa.float64()),
                "rms": pa.array([], type=pa.float64()),
                "n_obs_used": pa.array([], type=pa.int64()),
            }
        )

    step = max(1, int(page_size))
    out_pdes: list[str] = []
    out_cc: list[int | None] = []
    out_arc: list[float | None] = []
    out_rms: list[float | None] = []
    out_nobs: list[int | None] = []
    seen: set[str] = set()

    # For larger pulls, spread pages across the full NEO population to avoid
    # low-pdes bias (older, often better-constrained objects).
    starts: list[int] = []
    if want <= 2 * step:
        starts = list(range(0, want, step))
    else:
        total_count: int | None = None
        try:
            resp0 = requests.get(
                "https://ssd-api.jpl.nasa.gov/sbdb_query.api",
                params={"sb-group": "neo", "sb-kind": "a"},
                timeout=float(timeout_sec),
            )
            resp0.raise_for_status()
            c = _safe_int(resp0.json().get("count"))
            if c is not None and c > 0:
                total_count = int(c)
        except Exception:
            total_count = None

        if total_count is None or total_count <= 0:
            starts = list(range(0, want, step))
        else:
            n_pages = int(np.ceil(float(want) / float(step)))
            max_start = max(0, int(total_count) - int(step))
            starts = (
                np.linspace(0, max_start, num=max(1, n_pages), dtype=np.int64)
                .astype(np.int64)
                .tolist()
            )

    for off in starts:
        if len(out_pdes) >= want:
            break
        lim = min(step, want - len(out_pdes))
        params = {
            "fields": "pdes,condition_code,data_arc,rms,n_obs_used",
            "sb-group": "neo",
            "sb-kind": "a",
            "sort": "pdes",
            "limit": str(int(lim)),
            "limit-from": str(int(max(0, off))),
        }
        resp = requests.get(
            "https://ssd-api.jpl.nasa.gov/sbdb_query.api",
            params=params,
            timeout=float(timeout_sec),
        )
        resp.raise_for_status()
        payload = resp.json()
        data = payload.get("data")
        if not isinstance(data, list) or len(data) <= 0:
            break

        n_rows = 0
        for row in data:
            if not isinstance(row, list) or len(row) <= 0:
                continue
            n_rows += 1
            pdes = str(row[0]).strip()
            if not pdes or pdes in seen:
                continue
            seen.add(pdes)
            cc = _safe_int(row[1] if len(row) > 1 else None)
            arc = _safe_float(row[2] if len(row) > 2 else None)
            rms = _safe_float(row[3] if len(row) > 3 else None)
            nobs = _safe_int(row[4] if len(row) > 4 else None)

            out_pdes.append(pdes)
            out_cc.append(cc)
            out_arc.append(arc)
            out_rms.append(rms)
            out_nobs.append(nobs)
            if len(out_pdes) >= want:
                break

        if n_rows <= 0:
            continue

    return pa.table(
        {
            "pdes": pa.array(out_pdes, type=pa.large_string()),
            "condition_code": pa.array(out_cc, type=pa.int64()),
            "data_arc_days": pa.array(out_arc, type=pa.float64()),
            "rms": pa.array(out_rms, type=pa.float64()),
            "n_obs_used": pa.array(out_nobs, type=pa.int64()),
        }
    )


def _rank01(x: np.ndarray) -> np.ndarray:
    """
    Stable [0,1] rank transform with deterministic tie ordering.
    """
    n = int(x.size)
    if n <= 1:
        return np.zeros(n, dtype=np.float64)
    idx = np.argsort(x, kind="mergesort")
    r = np.empty(n, dtype=np.float64)
    r[idx] = np.arange(n, dtype=np.float64)
    return r / float(n - 1)


def _normalize_for_rank(x: np.ndarray, *, fill: float) -> np.ndarray:
    y = np.asarray(x, dtype=np.float64)
    if not np.isfinite(fill):
        fill = 0.0
    return np.where(np.isfinite(y), y, float(fill))


def _allocate_stratum_counts(
    *,
    counts: np.ndarray,
    total: int,
) -> np.ndarray:
    """
    Allocate per-stratum sample sizes proportionally with deterministic rounding.
    """
    c = np.asarray(counts, dtype=np.int64)
    n_total = int(total)
    if n_total <= 0 or c.size <= 0:
        return np.zeros(c.size, dtype=np.int64)

    nonempty = c > 0
    n_nonempty = int(np.count_nonzero(nonempty))
    out = np.zeros(c.size, dtype=np.int64)
    if n_nonempty <= 0:
        return out

    if n_total >= n_nonempty:
        out[nonempty] = 1
        n_total -= n_nonempty
        if n_total <= 0:
            return out

    avail = c - out
    avail = np.where(avail > 0, avail, 0)
    avail_sum = float(np.sum(avail))
    if avail_sum <= 0.0 or n_total <= 0:
        return out

    raw = (avail.astype(np.float64) / avail_sum) * float(n_total)
    add = np.floor(raw).astype(np.int64)
    add = np.minimum(add, avail)
    out += add
    n_left = int(n_total - int(np.sum(add)))
    if n_left <= 0:
        return out

    rem = raw - add.astype(np.float64)
    order = np.argsort(-rem, kind="mergesort")
    for i in order:
        if n_left <= 0:
            break
        if out[i] >= c[i]:
            continue
        out[i] += 1
        n_left -= 1
    return out


def _stratified_select_neo_ids(
    *,
    candidates: pa.Table,
    max_rows: int,
    seed: int,
) -> pa.Table:
    """
    Deterministically select a statistically varied NEO calibration subset.
    """
    if candidates.num_rows <= 0 or int(max_rows) <= 0:
        return candidates.slice(0, 0)

    pdes = [str(x) for x in candidates["pdes"].to_pylist()]
    n = len(pdes)
    k = min(int(max_rows), n)
    if k <= 0:
        return candidates.slice(0, 0)

    cc = np.asarray(candidates["condition_code"].to_numpy(zero_copy_only=False), dtype=np.float64)
    arc = np.asarray(candidates["data_arc_days"].to_numpy(zero_copy_only=False), dtype=np.float64)
    rms = np.asarray(candidates["rms"].to_numpy(zero_copy_only=False), dtype=np.float64)
    nobs = np.asarray(candidates["n_obs_used"].to_numpy(zero_copy_only=False), dtype=np.float64)

    cc_eff = np.where(np.isfinite(cc), np.clip(cc, 0.0, 9.0), 9.0)
    arc_eff = np.where(np.isfinite(arc) & (arc > 0.0), arc, 0.0)
    rms_pos = rms[np.isfinite(rms) & (rms > 0.0)]
    rms_fill = float(np.median(rms_pos)) if rms_pos.size > 0 else 1.0
    rms_eff = np.where(np.isfinite(rms) & (rms > 0.0), rms, rms_fill)
    nobs_eff = np.where(np.isfinite(nobs) & (nobs > 0.0), nobs, 0.0)

    r_cc = cc_eff / 9.0
    r_rms = _rank01(_normalize_for_rank(np.log10(rms_eff), fill=np.log10(rms_fill)))
    r_arc = _rank01(np.log10(arc_eff + 1.0))
    r_nobs = _rank01(np.log10(nobs_eff + 1.0))
    score = 0.40 * r_cc + 0.30 * r_rms + 0.20 * (1.0 - r_arc) + 0.10 * (1.0 - r_nobs)

    cc_bin = np.where(cc_eff <= 2.0, 0, np.where(cc_eff <= 4.0, 1, np.where(cc_eff <= 6.0, 2, 3))).astype(np.int64)
    arc_bin = np.where(arc_eff < 30.0, 0, np.where(arc_eff < 365.0, 1, np.where(arc_eff < 3650.0, 2, 3))).astype(np.int64)
    decile = np.minimum(9, np.floor(_rank01(score) * 10.0).astype(np.int64))

    rng = np.random.default_rng(int(seed))
    all_idx = np.arange(n, dtype=np.int64)
    ord_low = np.argsort(np.stack([score, np.arange(n, dtype=np.float64)], axis=1)[:, 0], kind="mergesort")
    ord_high = ord_low[::-1]

    tail_each = max(1, min(32, k // 20)) if k >= 20 else 1
    reserved: list[int] = []
    for idx in ord_low.tolist():
        if len(reserved) >= tail_each:
            break
        reserved.append(int(idx))
    for idx in ord_high.tolist():
        if len(reserved) >= 2 * tail_each:
            break
        i = int(idx)
        if i not in reserved:
            reserved.append(i)
    reserved = sorted(set(reserved))
    if len(reserved) > k:
        reserved = reserved[:k]

    keep = np.zeros(n, dtype=bool)
    keep[np.asarray(reserved, dtype=np.int64)] = True
    n_rem = k - int(np.count_nonzero(keep))
    if n_rem > 0:
        stratum = cc_bin * 4 + arc_bin
        stratum_avail = np.bincount(stratum[~keep], minlength=16).astype(np.int64)
        alloc = _allocate_stratum_counts(counts=stratum_avail, total=int(n_rem))

        for s in range(16):
            take_n = int(alloc[s])
            if take_n <= 0:
                continue
            idx_s = all_idx[(stratum == s) & (~keep)]
            if idx_s.size <= 0:
                continue
            perm = rng.permutation(idx_s)
            take = perm[: min(take_n, perm.size)]
            keep[take] = True

    sel = all_idx[keep]
    if sel.size > k:
        sel = np.sort(sel)[:k]
    elif sel.size < k:
        remaining = all_idx[~keep]
        if remaining.size > 0:
            fill = remaining[: min(k - sel.size, remaining.size)]
            sel = np.concatenate([sel, fill], axis=0)
    sel = np.sort(sel.astype(np.int64, copy=False))

    return pa.table(
        {
            "pdes": pa.array([pdes[i] for i in sel.tolist()], type=pa.large_string()),
            "condition_code": pa.array(cc_eff[sel].astype(np.int64), type=pa.int64()),
            "data_arc_days": pa.array(arc_eff[sel], type=pa.float64()),
            "rms": pa.array(rms_eff[sel], type=pa.float64()),
            "n_obs_used": pa.array(nobs_eff[sel].astype(np.int64), type=pa.int64()),
            "uncertainty_score": pa.array(score[sel], type=pa.float64()),
            "condition_bin": pa.array(cc_bin[sel], type=pa.int64()),
            "arc_bin": pa.array(arc_bin[sel], type=pa.int64()),
            "uncertainty_decile": pa.array(decile[sel], type=pa.int64()),
        }
    )


def _build_covariance_calibration_table_from_orbits(
    *,
    query_ids: list[str],
    timeout_sec: float,
    max_attempts: int,
    chunk_size: int,
) -> pa.Table:
    from adam_core.orbits.query.sbdb import query_sbdb_new

    qids: list[str] = []
    object_ids: list[str] = []
    pos_sigma_rms_au: list[float] = []
    vel_sigma_rms_au_per_d: list[float] = []

    n = len(query_ids)
    chunk = max(1, int(chunk_size))

    for i in range(0, n, chunk):
        ids = query_ids[i : i + chunk]
        orbits = query_sbdb_new(
            ids,
            max_concurrent_requests=1,
            timeout_s=float(timeout_sec),
            max_attempts=int(max_attempts),
            allow_missing=True,
            orbit_id_from_input=True,
        )
        if len(orbits) <= 0:
            continue

        cov = orbits.coordinates.covariance.to_matrix()
        diag = np.diagonal(cov, axis1=1, axis2=2)

        pos_var = np.nanmean(diag[:, 0:3], axis=1)
        vel_var = np.nanmean(diag[:, 3:6], axis=1)
        pos_sig = np.sqrt(np.where(pos_var > 0.0, pos_var, np.nan))
        vel_sig = np.sqrt(np.where(vel_var > 0.0, vel_var, np.nan))

        qids.extend([str(x) for x in orbits.orbit_id.to_pylist()])
        object_ids.extend([str(x) if x is not None else "" for x in orbits.object_id.to_pylist()])
        pos_sigma_rms_au.extend([float(x) if np.isfinite(x) else np.nan for x in pos_sig])
        vel_sigma_rms_au_per_d.extend([float(x) if np.isfinite(x) else np.nan for x in vel_sig])

    return pa.table(
        {
            "query_id": pa.array(qids, type=pa.large_string()),
            "sbdb_object_id": pa.array(object_ids, type=pa.large_string()),
            "pos_sigma_rms_au": pa.array(pos_sigma_rms_au, type=pa.float64()),
            "vel_sigma_rms_au_per_d": pa.array(vel_sigma_rms_au_per_d, type=pa.float64()),
        }
    )


def fetch_sbdb_neo_calibration_table(
    *,
    source_orbits_parquet: str | None,
    cache_parquet: Path,
    max_rows: int = 4_096,
    page_size: int = 256,
    timeout_sec: float = 60.0,
    max_attempts: int = 5,
    force_refresh: bool = False,
    seed: int = 0,
    candidate_pool_multiplier: int = 4,
) -> pa.Table:
    """
    Fetch a deterministic SBDB calibration sample via `query_sbdb_new` and cache to parquet.

    We sample real NEO asteroid designations from SBDB itself, query their
    states/covariances via `query_sbdb_new`, and derive empirical position/velocity
    covariance scales from returned orbit covariances.
    """
    cache_parquet = Path(cache_parquet)
    if cache_parquet.exists() and not bool(force_refresh):
        return pq.read_table(str(cache_parquet))

    _ = source_orbits_parquet  # kept for backwards-compatible call sites
    candidate_rows = int(max_rows) * max(1, int(candidate_pool_multiplier))
    candidates = _fetch_real_neo_candidate_table_from_sbdb(
        max_rows=int(max(candidate_rows, max_rows)),
        page_size=int(page_size),
        timeout_sec=float(timeout_sec),
    )
    selected = _stratified_select_neo_ids(
        candidates=candidates,
        max_rows=int(max_rows),
        seed=int(seed),
    )
    query_ids = [str(x) for x in selected["pdes"].to_pylist()]
    if len(query_ids) <= 0:
        raise ValueError("No real NEO IDs were returned by SBDB query API for calibration")

    out = _build_covariance_calibration_table_from_orbits(
        query_ids=query_ids,
        timeout_sec=float(timeout_sec),
        max_attempts=int(max_attempts),
        chunk_size=int(page_size),
    )
    if out.num_rows <= 0:
        raise ValueError("No SBDB calibration rows were resolved from query_sbdb_new")

    # Attach sampled uncertainty metadata for downstream diagnostics and manifest reporting.
    sel_meta = selected.rename_columns(
        [
            "query_id",
            "condition_code",
            "data_arc_days",
            "rms",
            "n_obs_used",
            "uncertainty_score",
            "condition_bin",
            "arc_bin",
            "uncertainty_decile",
        ]
    )
    out = out.join(sel_meta, keys=["query_id"], join_type="left outer")

    cache_parquet.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(out, str(cache_parquet))
    return out


def _assign_quality_stratum(*, condition_code: int | None, data_arc_days: float | None) -> int:
    """
    Map SBDB OD quality to one of four uncertainty strata.

    0: well_known
    1: known
    2: constrained
    3: short_arc
    """
    cc = 9 if condition_code is None else int(condition_code)
    arc = 0.0 if data_arc_days is None else float(max(0.0, data_arc_days))

    if cc <= 2 and arc >= 3650.0:
        return 0
    if cc <= 4 and arc >= 365.0:
        return 1
    if cc <= 6 and arc >= 30.0:
        return 2
    return 3


def _arcsec_to_au(theta_arcsec: float) -> float:
    # Small-angle conversion at 1 au.
    return float(theta_arcsec) / 206265.0


def _positive_quantiles(x: np.ndarray) -> tuple[float, float, float]:
    xx = np.asarray(x, dtype=np.float64)
    xx = xx[np.isfinite(xx) & (xx > 0.0)]
    if xx.size == 0:
        return 1e-12, 3e-12, 1e-11
    q25, q50, q75 = np.quantile(xx, [0.25, 0.50, 0.75]).tolist()
    q25 = float(max(q25, 1e-14))
    q50 = float(max(q50, 1e-14))
    q75 = float(max(q75, q50))
    return q25, q50, q75


def build_synthetic_covariance_model(
    *,
    sbdb_table: pa.Table,
    seed: int,
) -> SyntheticCovarianceModel:
    """
    Build a deterministic synthetic covariance model from broad SBDB NEO metadata.

    We calibrate stratum weights and uncertainty scales from SBDB OD quality proxies
    (condition code, data-arc, fit RMS).
    """
    cols = set(sbdb_table.column_names)

    out_strata: list[StratumModel] = []
    names = ("well_known", "known", "constrained", "short_arc")

    if {"pos_sigma_rms_au", "vel_sigma_rms_au_per_d"}.issubset(cols):
        pos = np.asarray(sbdb_table["pos_sigma_rms_au"].to_numpy(zero_copy_only=False), dtype=np.float64)
        vel = np.asarray(
            sbdb_table["vel_sigma_rms_au_per_d"].to_numpy(zero_copy_only=False),
            dtype=np.float64,
        )
        m = np.isfinite(pos) & np.isfinite(vel) & (pos > 0.0) & (vel > 0.0)
        pos = pos[m]
        vel = vel[m]
        if pos.size <= 0:
            raise ValueError("SBDB calibration table has no finite covariance-scale rows")

        q25, q50, q75 = np.quantile(pos, [0.25, 0.50, 0.75]).tolist()
        strata = np.digitize(pos, bins=[q25, q50, q75], right=True).astype(np.int64)
        counts = np.bincount(strata, minlength=4).astype(np.float64)
        weights = np.maximum(counts / max(float(np.sum(counts)), 1.0), 0.01)
        weights = weights / float(np.sum(weights))

        anis_pos_median = np.asarray([2.0, 5.0, 15.0, 60.0], dtype=np.float64)
        anis_pos_log_sigma = np.asarray([0.25, 0.35, 0.50, 0.65], dtype=np.float64)

        g25p, g50p, g75p = _positive_quantiles(pos)
        g25v, g50v, g75v = _positive_quantiles(vel)

        for k, name in enumerate(names):
            mk = strata == int(k)
            pos_k = pos[mk] if np.any(mk) else np.asarray([g50p], dtype=np.float64)
            vel_k = vel[mk] if np.any(mk) else np.asarray([g50v], dtype=np.float64)

            pos_q25, pos_q50, pos_q75 = _positive_quantiles(pos_k)
            vel_q25, vel_q50, vel_q75 = _positive_quantiles(vel_k)

            pos_q25 = float(max(pos_q25, g25p))
            pos_q50 = float(max(pos_q50, pos_q25 * 1.1))
            pos_q75 = float(max(pos_q75, pos_q50 * 1.1))

            vel_q25 = float(max(vel_q25, g25v))
            vel_q50 = float(max(vel_q50, vel_q25 * 1.1))
            vel_q75 = float(max(vel_q75, vel_q50 * 1.1))

            out_strata.append(
                StratumModel(
                    name=str(name),
                    weight=float(weights[k]),
                    pos_log_mu_au=float(np.log(pos_q50)),
                    pos_log_sigma_au=float(max((np.log(pos_q75) - np.log(pos_q25)) / 1.349, 0.20)),
                    anis_pos_log_mu=float(np.log(anis_pos_median[k])),
                    anis_pos_log_sigma=float(anis_pos_log_sigma[k]),
                    vel_log_mu_au_per_d=float(np.log(vel_q50)),
                    vel_log_sigma_au_per_d=float(max((np.log(vel_q75) - np.log(vel_q25)) / 1.349, 0.20)),
                    anis_vel_log_mu=float(np.log(max(anis_pos_median[k] * 0.7, 1.5))),
                    anis_vel_log_sigma=float(max(anis_pos_log_sigma[k] * 0.9, 0.20)),
                )
            )
    else:
        cc_vals = [_safe_int(x) for x in sbdb_table["condition_code"].to_pylist()]
        arc_vals = [_safe_float(x) for x in sbdb_table["data_arc"].to_pylist()]
        rms_vals = np.asarray([_safe_float(x) for x in sbdb_table["rms"].to_pylist()], dtype=np.float64)

        strata = np.asarray(
            [
                _assign_quality_stratum(condition_code=cc, data_arc_days=arc)
                for cc, arc in zip(cc_vals, arc_vals)
            ],
            dtype=np.int64,
        )

        n = int(strata.size)
        if n <= 0:
            raise ValueError("SBDB calibration table is empty")

        counts = np.bincount(strata, minlength=4).astype(np.float64)
        weights = counts / max(float(np.sum(counts)), 1.0)
        # Ensure no stratum collapses to zero probability.
        weights = np.maximum(weights, 0.01)
        weights = weights / float(np.sum(weights))

        # Scale factors map residual RMS to synthetic on-sky positional uncertainty scales.
        # These multipliers intentionally span well-known to short-arc regimes.
        rms_to_pos_factor = np.asarray([80.0, 250.0, 2000.0, 20_000.0], dtype=np.float64)
        vel_tau_days = np.asarray([4000.0, 1000.0, 180.0, 30.0], dtype=np.float64)

        anis_pos_median = np.asarray([3.0, 8.0, 25.0, 120.0], dtype=np.float64)
        anis_pos_log_sigma = np.asarray([0.30, 0.40, 0.55, 0.70], dtype=np.float64)

        for k, name in enumerate(names):
            m = strata == int(k)
            rms_k = rms_vals[m]
            q25_rms, q50_rms, q75_rms = _positive_quantiles(rms_k)

            pos_q25 = _arcsec_to_au(q25_rms * float(rms_to_pos_factor[k]))
            pos_q50 = _arcsec_to_au(q50_rms * float(rms_to_pos_factor[k]))
            pos_q75 = _arcsec_to_au(q75_rms * float(rms_to_pos_factor[k]))

            pos_q25 = float(max(pos_q25, 1e-12))
            pos_q50 = float(max(pos_q50, pos_q25 * 1.1))
            pos_q75 = float(max(pos_q75, pos_q50 * 1.1))

            pos_log_mu = float(np.log(pos_q50))
            pos_log_sigma = float(max((np.log(pos_q75) - np.log(pos_q25)) / 1.349, 0.25))

            vel_log_mu = float(pos_log_mu - np.log(float(vel_tau_days[k])))
            vel_log_sigma = float(max(pos_log_sigma, 0.25))

            anis_pos_mu = float(np.log(anis_pos_median[k]))
            anis_vel_mu = float(np.log(max(anis_pos_median[k] * 0.7, 1.5)))
            anis_vel_sigma = float(max(anis_pos_log_sigma[k] * 0.9, 0.25))

            out_strata.append(
                StratumModel(
                    name=str(name),
                    weight=float(weights[k]),
                    pos_log_mu_au=pos_log_mu,
                    pos_log_sigma_au=pos_log_sigma,
                    anis_pos_log_mu=anis_pos_mu,
                    anis_pos_log_sigma=float(anis_pos_log_sigma[k]),
                    vel_log_mu_au_per_d=vel_log_mu,
                    vel_log_sigma_au_per_d=vel_log_sigma,
                    anis_vel_log_mu=anis_vel_mu,
                    anis_vel_log_sigma=anis_vel_sigma,
                )
            )

    return SyntheticCovarianceModel(
        seed=int(seed),
        strata=tuple(out_strata),
        calibration_generated_at_utc=_utc_now_iso(),
        calibration_rows=int(sbdb_table.num_rows),
        sbdb_params={
            "source": "JPL SBDB API via adam_core.query_sbdb_new",
            "calibration_mode": (
                "covariance_scales"
                if {"pos_sigma_rms_au", "vel_sigma_rms_au_per_d"}.issubset(cols)
                else "metadata_rms"
            ),
        },
    )


def write_model_json(*, model: SyntheticCovarianceModel, out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(model.to_json_dict(), indent=2, sort_keys=True), encoding="utf-8")


def _sample_diag_cov_eigs(
    *,
    sigma_rms: np.ndarray,
    anis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build eigenvalue triples for 3x3 covariance blocks from RMS sigma and anisotropy.

    We use geometric spacing (lambda_min, r*lambda_min, r^2*lambda_min) where
    r==anisotropy in sigma-space (sqrt(lambda_max/lambda_min)).
    """
    s = np.asarray(sigma_rms, dtype=np.float64)
    r = np.asarray(anis, dtype=np.float64)

    s = np.where(np.isfinite(s) & (s > 0.0), s, 1e-10)
    r = np.where(np.isfinite(r) & (r >= 1.0), r, 1.0)

    denom = 1.0 + r + r * r
    lam_min = (s * s) / denom
    lam_mid = r * lam_min
    lam_max = (r * r) * lam_min
    return lam_min, lam_mid, lam_max


def generate_covariance_matrices_for_batch(
    *,
    n_rows: int,
    model: SyntheticCovarianceModel,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate deterministic synthetic 6x6 Cartesian covariance matrices.

    Output shape is (N, 6, 6), finite and PSD by construction.
    """
    n = int(n_rows)
    if n <= 0:
        return np.zeros((0, 6, 6), dtype=np.float64)

    strata = model.strata
    p = np.asarray([float(s.weight) for s in strata], dtype=np.float64)
    p = p / float(np.sum(p))
    k_idx = rng.choice(len(strata), size=n, p=p)

    # Sample per-row scales from stratum-specific lognormal distributions.
    pos_sigma = np.empty(n, dtype=np.float64)
    pos_anis = np.empty(n, dtype=np.float64)
    vel_sigma = np.empty(n, dtype=np.float64)
    vel_anis = np.empty(n, dtype=np.float64)

    for i, s in enumerate(strata):
        m = k_idx == i
        nn = int(np.count_nonzero(m))
        if nn <= 0:
            continue
        pos_sigma[m] = rng.lognormal(mean=float(s.pos_log_mu_au), sigma=float(s.pos_log_sigma_au), size=nn)
        pos_anis[m] = np.maximum(
            1.0,
            rng.lognormal(mean=float(s.anis_pos_log_mu), sigma=float(s.anis_pos_log_sigma), size=nn),
        )
        vel_sigma[m] = rng.lognormal(
            mean=float(s.vel_log_mu_au_per_d),
            sigma=float(s.vel_log_sigma_au_per_d),
            size=nn,
        )
        vel_anis[m] = np.maximum(
            1.0,
            rng.lognormal(mean=float(s.anis_vel_log_mu), sigma=float(s.anis_vel_log_sigma), size=nn),
        )

    # Keep synthetic draws in numerically stable bounds for sigma-point propagation.
    pos_sigma = np.clip(pos_sigma, 1e-13, 2e-3)
    vel_sigma = np.clip(vel_sigma, 1e-16, 2e-4)
    pos_anis = np.clip(pos_anis, 1.0, 5e2)
    vel_anis = np.clip(vel_anis, 1.0, 5e2)

    lam_p0, lam_p1, lam_p2 = _sample_diag_cov_eigs(sigma_rms=pos_sigma, anis=pos_anis)
    lam_v0, lam_v1, lam_v2 = _sample_diag_cov_eigs(sigma_rms=vel_sigma, anis=vel_anis)

    cov = np.zeros((n, 6, 6), dtype=np.float64)

    # Randomly permute axis ordering (cheap orientation diversity without dense rotations).
    perms = np.asarray(
        [
            [0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0],
        ],
        dtype=np.int64,
    )
    p_idx = rng.integers(0, perms.shape[0], size=n, endpoint=False)
    v_idx = rng.integers(0, perms.shape[0], size=n, endpoint=False)

    lam_pos = np.stack([lam_p0, lam_p1, lam_p2], axis=1)
    lam_vel = np.stack([lam_v0, lam_v1, lam_v2], axis=1)

    for axis in range(3):
        cov[np.arange(n), axis, axis] = lam_pos[np.arange(n), perms[p_idx, axis]]
        cov[np.arange(n), axis + 3, axis + 3] = lam_vel[np.arange(n), perms[v_idx, axis]]

    # Symmetrize defensively and guard against numerical drift.
    cov = 0.5 * (cov + np.transpose(cov, (0, 2, 1)))
    cov = np.where(np.isfinite(cov), cov, 0.0)

    tiny = 1e-24
    # NumPy can return a read-only diagonal view; copy so we can patch invalid entries.
    d = np.diagonal(cov, axis1=1, axis2=2).copy()
    bad_diag = ~np.isfinite(d) | (d <= tiny)
    if np.any(bad_diag):
        d[bad_diag] = tiny
        cov[:, np.arange(6), np.arange(6)] = d

    return cov
