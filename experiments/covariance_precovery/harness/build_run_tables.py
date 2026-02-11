from __future__ import annotations

import argparse
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from adam_core.time import Timestamp

from ..selection.designation_normalization import normalize_designation


@dataclass(frozen=True)
class OutlierConfig:
    truth_min: int = 50
    recall_lt: float = 0.5


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _read_stage2_targets(stage2_run_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Return arrays aligned to target_idx:
      - obscode[target_idx] (object dtype)
      - exposure_mjd_mid_utc[target_idx] (float64)
    """
    ft = pq.read_table(stage2_run_dir / "inputs" / "frame_time_targets.parquet")
    codes = np.asarray(ft["obscode"].to_pylist(), dtype=object)
    t = ft.column("time").combine_chunks()
    times = Timestamp.from_kwargs(days=t.field("days"), nanos=t.field("nanos"), scale="utc")
    mjd = np.asarray(times.mjd().to_numpy(zero_copy_only=False), dtype=np.float64)
    return codes, mjd


def _target_idx_map(*, obscode: np.ndarray, exposure_mjd_mid_utc: np.ndarray) -> dict[tuple[str, float], int]:
    out: dict[tuple[str, float], int] = {}
    for i in range(int(len(exposure_mjd_mid_utc))):
        out[(str(obscode[i]), float(exposure_mjd_mid_utc[i]))] = int(i)
    return out


def _read_truth_crossmatch_matched(
    truth_path: Path,
) -> pa.Table:
    truth = pq.read_table(
        truth_path,
        columns=[
            "matched",
            "designation",
            "obscode",
            "match_dataset_id",
            "match_exposure_id",
            "healpixel",
        ],
    )
    truth = truth.filter(pc.equal(truth["matched"], True))
    if truth.num_rows == 0:
        return truth.select(["designation", "obscode", "match_dataset_id", "match_exposure_id", "healpixel"])

    m_has = pc.and_(
        pc.invert(pc.is_null(truth["match_dataset_id"])),
        pc.invert(pc.is_null(truth["match_exposure_id"])),
    )
    truth = truth.filter(m_has)
    return truth.select(["designation", "obscode", "match_dataset_id", "match_exposure_id", "healpixel"])


def _designation_to_orbit_id_map(orbits_selected_sbdb_parquet: Path) -> dict[str, str]:
    """
    Build a mapping from (designation string used in truth/BQ selection) -> orbit_id.

    Note: We normalize `object_id` (SBDB display strings) into canonical designation
    using `normalize_designation`, and map that to the stored `orbit_id`.
    """
    t = pq.read_table(orbits_selected_sbdb_parquet, columns=["orbit_id", "object_id"])
    orbit_id = [str(x) for x in t["orbit_id"].to_pylist()]
    object_id = [str(x) for x in t["object_id"].to_pylist()]
    out: dict[str, str] = {}
    for oid, obj in zip(orbit_id, object_id):
        key = normalize_designation(str(obj))
        if key:
            out[key] = str(oid)
    return out


def _exposure_midpoint_map_from_index_db(
    *,
    index_db: Path,
    keys: list[tuple[str, str, str]],
) -> dict[tuple[str, str, str], float]:
    """
    keys: list[(dataset_id, obscode, exposure_id)]
    returns: dict[(dataset_id, obscode, exposure_id)] -> exposure_mjd_mid
    """
    uniq = sorted(set(keys))
    if not uniq:
        return {}

    conn = sqlite3.connect(str(index_db))
    try:
        conn.execute(
            "CREATE INDEX IF NOT EXISTS frames_truth_join_idx "
            "ON frames(dataset_id, obscode, exposure_id)"
        )
        conn.execute("DROP TABLE IF EXISTS temp.truth_keys")
        conn.execute("CREATE TEMP TABLE truth_keys (dataset_id TEXT, obscode TEXT, exposure_id TEXT)")
        conn.executemany(
            "INSERT INTO truth_keys (dataset_id, obscode, exposure_id) VALUES (?, ?, ?)",
            uniq,
        )
        conn.execute("CREATE INDEX truth_keys_idx ON truth_keys(dataset_id, obscode, exposure_id)")
        rows = conn.execute(
            """
            SELECT t.dataset_id, t.obscode, t.exposure_id, f.exposure_mjd_mid
            FROM frames f
            INNER JOIN truth_keys t
              ON f.dataset_id = t.dataset_id
             AND f.obscode = t.obscode
             AND f.exposure_id = t.exposure_id
            """
        ).fetchall()
    finally:
        conn.close()

    return {(str(r[0]), str(r[1]), str(r[2])): float(r[3]) for r in rows}


def _build_truth_keys_table(
    *,
    subset_dir: Path,
    stage2_run_dir: Path,
    inputs_artifacts_dir: Path,
) -> pd.DataFrame:
    """
    Build the canonical truth-keys table aligned to Stage 2 targets:
      (orbit_id, target_idx, healpixel)
    """
    index_db = subset_dir / "index.db"
    if not index_db.exists():
        raise FileNotFoundError(f"Missing subset index.db: {index_db}")

    truth_path = inputs_artifacts_dir / "truth_precovery_crossmatch.parquet"
    if not truth_path.exists():
        raise FileNotFoundError(f"Missing truth crossmatch parquet: {truth_path}")

    orbits_path = inputs_artifacts_dir / "orbits_selected_sbdb.parquet"
    if not orbits_path.exists():
        raise FileNotFoundError(f"Missing SBDB orbits parquet: {orbits_path}")

    obscode, mjd_mid = _read_stage2_targets(stage2_run_dir)
    tidx = _target_idx_map(obscode=obscode, exposure_mjd_mid_utc=mjd_mid)

    truth = _read_truth_crossmatch_matched(truth_path)
    if truth.num_rows == 0:
        return pd.DataFrame.from_records([], columns=["orbit_id", "target_idx", "healpixel"])

    des_to_orbit = _designation_to_orbit_id_map(orbits_path)
    designation = [str(x) for x in truth["designation"].to_pylist()]
    orbit_id = [des_to_orbit.get(str(d), "") for d in designation]
    ok_map = np.asarray([bool(x) for x in orbit_id], dtype=bool)
    if not ok_map.any():
        return pd.DataFrame.from_records([], columns=["orbit_id", "target_idx", "healpixel"])

    truth = truth.take(pa.array(np.nonzero(ok_map)[0], type=pa.int64()))
    orbit_id = [x for x in orbit_id if x]

    ds = [str(x) for x in truth["match_dataset_id"].to_pylist()]
    oc = [str(x) for x in truth["obscode"].to_pylist()]
    ex = [str(x) for x in truth["match_exposure_id"].to_pylist()]
    hp = np.asarray(truth["healpixel"].to_numpy(zero_copy_only=False), dtype=np.int64)

    keys = list(zip(ds, oc, ex))
    mid_by_key = _exposure_midpoint_map_from_index_db(index_db=index_db, keys=keys)

    mjd = [mid_by_key.get((ds[i], oc[i], ex[i])) for i in range(len(keys))]
    ok_mid = np.asarray([x is not None for x in mjd], dtype=bool)
    if not ok_mid.any():
        return pd.DataFrame.from_records([], columns=["orbit_id", "target_idx", "healpixel"])

    orbit_ok = np.asarray(orbit_id, dtype=object)[ok_mid]
    oc_ok = np.asarray(oc, dtype=object)[ok_mid]
    hp_ok = hp[ok_mid]
    mjd_ok = np.asarray([float(x) for x in np.asarray(mjd, dtype=object)[ok_mid].tolist()], dtype=np.float64)

    target_idx = np.full(int(len(mjd_ok)), -1, dtype=np.int64)
    for i in range(int(len(mjd_ok))):
        target_idx[i] = tidx.get((str(oc_ok[i]), float(mjd_ok[i])), -1)
    ok_t = target_idx >= 0
    if not ok_t.any():
        return pd.DataFrame.from_records([], columns=["orbit_id", "target_idx", "healpixel"])

    out = pd.DataFrame(
        {
            "orbit_id": [str(x) for x in orbit_ok[ok_t].tolist()],
            "target_idx": target_idx[ok_t].astype(np.int64, copy=False),
            "healpixel": hp_ok[ok_t].astype(np.int64, copy=False),
        }
    )
    return out.drop_duplicates(subset=["orbit_id", "target_idx", "healpixel"]).reset_index(drop=True)


def _parse_stratum(s: str | None) -> dict[str, str | None]:
    if s is None:
        return dict(regime=None, arc_bin=None, dt_bin=None, u_bin=None, i_bin=None)
    parts = [p.strip() for p in str(s).split("|") if p.strip()]
    if len(parts) != 5:
        return dict(regime=None, arc_bin=None, dt_bin=None, u_bin=None, i_bin=None)
    return dict(regime=parts[0], arc_bin=parts[1], dt_bin=parts[2], u_bin=parts[3], i_bin=parts[4])


def _bin_edges_label(v: float | None, edges: list[float], prefix: str) -> str:
    if v is None or (not np.isfinite(float(v))):
        return f"{prefix}_unknown"
    x = float(v)
    e = [float(a) for a in edges]
    if x < e[0]:
        return f"{prefix}_lt_{e[0]:g}"
    for i in range(1, len(e)):
        if x < e[i]:
            return f"{prefix}_{e[i-1]:g}_{e[i]:g}"
    return f"{prefix}_ge_{e[-1]:g}"


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    _ensure_dir(path.parent)
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, str(path))


def _group_stats_orbit(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """
    Aggregate per-orbit stats. Input must have:
      - stage3_truth_healpix, stage3_recovered_truth_healpix
      - stage4_truth_detections, stage4_recovered_truth_detections, stage4_recall
      - n_frames_loaded, n_observations_loaded, n_accepted
    """
    if df.empty:
        return pd.DataFrame()

    g = df.groupby(group_cols, dropna=False)

    def _stage4_recall_total(x: pd.DataFrame) -> float:
        t = float(x["stage4_truth_detections"].sum())
        r = float(x["stage4_recovered_truth_detections"].sum())
        return 0.0 if t <= 0 else r / t

    def _stage3_coverage_total(x: pd.DataFrame) -> float:
        t = float(x["stage3_truth_healpix"].sum())
        r = float(x["stage3_recovered_truth_healpix"].sum())
        return 0.0 if t <= 0 else r / t

    out = g.apply(
        lambda x: pd.Series(
            {
                "n_orbits": int(len(x)),
                "stage3_truth_healpix": int(x["stage3_truth_healpix"].fillna(0).sum()),
                "stage3_recovered_truth_healpix": int(x["stage3_recovered_truth_healpix"].fillna(0).sum()),
                "stage3_coverage_total": float(_stage3_coverage_total(x.fillna(0))),
                "stage4_truth_detections": int(x["stage4_truth_detections"].sum()),
                "stage4_recovered_truth_detections": int(x["stage4_recovered_truth_detections"].sum()),
                "stage4_recall_total": float(_stage4_recall_total(x)),
                "stage4_recall_orbit_median": float(x["stage4_recall"].median()),
                "stage4_recall_orbit_mean": float(x["stage4_recall"].mean()),
                "frames_total": int(x["n_frames_loaded"].sum()),
                "obs_total": int(x["n_observations_loaded"].sum()),
                "accepted_total": int(x["n_accepted"].sum()),
                "frames_per_truth_det": float(
                    x["n_frames_loaded"].sum() / max(x["stage4_truth_detections"].sum(), 1)
                ),
                "obs_per_truth_det": float(
                    x["n_observations_loaded"].sum() / max(x["stage4_truth_detections"].sum(), 1)
                ),
                "accepted_per_truth_det": float(
                    x["n_accepted"].sum() / max(x["stage4_truth_detections"].sum(), 1)
                ),
                "frames_per_truth_median": float(x["frames_per_truth"].median()),
            }
        )
    ).reset_index()

    return out


def build_run_tables(
    *,
    subset_dir: Path,
    stage2_run_dir: Path,
    stage3_run_dir: Path,
    stage4_run_dir: Path,
    inputs_artifacts_dir: Path,
    out_dir: Path,
    outlier_cfg: OutlierConfig,
    exclude_stage3_recovered_le: int | None = None,
    exclude_stage3_coverage_le: float | None = None,
) -> dict[str, Path]:
    # ----- Stage 4 per-target (driver) -----
    pt_path = stage4_run_dir / "per_target.parquet"
    if not pt_path.exists():
        raise FileNotFoundError(f"Missing Stage 4 per_target.parquet: {pt_path}")
    pt = pq.read_table(pt_path).to_pandas()

    # Keep only the columns we care about + required keys.
    keep_cols = [
        "strategy",
        "variant_kind",
        "footprint",
        "detection_filter",
        "orbit_id",
        "target_idx",
        "n_frames_loaded",
        "n_observations_loaded",
        "n_accepted",
        "n_truth_matched",
        "n_truth_recovered",
    ]
    pt = pt[[c for c in keep_cols if c in pt.columns]].copy()
    pt["variant_kind"] = pt["variant_kind"].astype(object)

    # ----- Stage 3 selected keys for footprints used in Stage 4 -----
    combos = (
        pt[["strategy", "variant_kind", "footprint"]]
        .drop_duplicates()
        .sort_values(["strategy", "variant_kind", "footprint"], na_position="last")
    )
    selected_rows: list[pd.DataFrame] = []
    for _, r in combos.iterrows():
        strat = str(r["strategy"])
        vk = None if pd.isna(r["variant_kind"]) else str(r["variant_kind"])
        fp = str(r["footprint"])
        strat_key = strat if vk is None else f"{strat}:{vk}"
        p = stage3_run_dir / "selected_keys" / strat_key / fp / "selected_keys_unique.parquet"
        if not p.exists():
            # Incomplete Stage3 outputs: treat as empty selection for this combo.
            continue
        sk = pq.read_table(p, columns=["orbit_id", "target_idx", "healpixel"]).to_pandas()
        sk["strategy"] = strat
        sk["variant_kind"] = vk
        sk["footprint"] = fp
        selected_rows.append(sk)
    sk_all = (
        pd.concat(selected_rows, ignore_index=True)
        if selected_rows
        else pd.DataFrame.from_records([], columns=["orbit_id", "target_idx", "healpixel", "strategy", "variant_kind", "footprint"])
    )

    # ----- Truth keys aligned to Stage 2 targets -----
    truth_keys = _build_truth_keys_table(
        subset_dir=subset_dir,
        stage2_run_dir=stage2_run_dir,
        inputs_artifacts_dir=inputs_artifacts_dir,
    )

    # ----- Stage 3 per-orbit truth/recovered healpixels per (strategy, variant_kind, footprint) -----
    #
    # IMPORTANT: Include *all* truth healpixels per orbit even when Stage 3 selects nothing
    # (stage3_miss). Those orbits must still show stage3_truth>0 and stage3_recovered=0.
    stage3_orbit: pd.DataFrame
    if truth_keys.empty:
        stage3_orbit = pd.DataFrame.from_records(
            [],
            columns=[
                "strategy",
                "variant_kind",
                "footprint",
                "orbit_id",
                "stage3_truth_healpix",
                "stage3_recovered_truth_healpix",
                "stage3_selected_keys",
                "stage3_coverage",
            ],
        )
    else:
        truth_counts_all = truth_keys.groupby("orbit_id", dropna=False).size().rename("stage3_truth_healpix")
        cov_rows: list[pd.DataFrame] = []
        for (strat, vk, fp), sk in sk_all.groupby(["strategy", "variant_kind", "footprint"], dropna=False):
            # Recovered truth healpixels: intersection of truth keys and selected keys.
            covered = truth_keys.merge(
                sk[["orbit_id", "target_idx", "healpixel"]],
                on=["orbit_id", "target_idx", "healpixel"],
                how="inner",
            )
            cov_counts = (
                covered.groupby("orbit_id", dropna=False)
                .size()
                .rename("stage3_recovered_truth_healpix")
            )
            sel_counts = sk.groupby("orbit_id", dropna=False).size().rename("stage3_selected_keys")
            d = (
                pd.concat([truth_counts_all, cov_counts, sel_counts], axis=1)
                .fillna(0)
                .reset_index()
            )
            d["strategy"] = str(strat)
            d["variant_kind"] = (None if (vk is None or (isinstance(vk, float) and np.isnan(vk))) else str(vk))
            d["footprint"] = str(fp)
            d["stage3_coverage"] = d["stage3_recovered_truth_healpix"] / d["stage3_truth_healpix"].clip(lower=1)
            cov_rows.append(
                d[
                    [
                        "strategy",
                        "variant_kind",
                        "footprint",
                        "orbit_id",
                        "stage3_truth_healpix",
                        "stage3_recovered_truth_healpix",
                        "stage3_selected_keys",
                        "stage3_coverage",
                    ]
                ]
            )
        stage3_orbit = pd.concat(cov_rows, ignore_index=True) if cov_rows else pd.DataFrame()

    # ----- Stage 4 per-orbit performance per (strategy, variant_kind, footprint, detection_filter) -----
    group_cols = ["strategy", "variant_kind", "footprint", "detection_filter", "orbit_id"]
    po = (
        pt.groupby(group_cols, dropna=False)
        .agg(
            n_targets=("target_idx", "count"),
            n_targets_with_frames=("n_frames_loaded", lambda x: int((np.asarray(x) > 0).sum())),
            stage4_truth_detections=("n_truth_matched", "sum"),
            stage4_recovered_truth_detections=("n_truth_recovered", "sum"),
            n_frames_loaded=("n_frames_loaded", "sum"),
            n_observations_loaded=("n_observations_loaded", "sum"),
            n_accepted=("n_accepted", "sum"),
        )
        .reset_index()
    )
    po["stage4_recall"] = po["stage4_recovered_truth_detections"] / po["stage4_truth_detections"].clip(lower=1)
    po["frames_per_truth"] = po["n_frames_loaded"] / po["stage4_truth_detections"].clip(lower=1)
    po["obs_per_truth"] = po["n_observations_loaded"] / po["stage4_truth_detections"].clip(lower=1)
    po["accepted_per_truth"] = po["n_accepted"] / po["stage4_truth_detections"].clip(lower=1)

    # Attach Stage 3 coverage (by strategy/vk/fp/orbit_id).
    if not stage3_orbit.empty:
        po = po.merge(
            stage3_orbit,
            on=["strategy", "variant_kind", "footprint", "orbit_id"],
            how="left",
        )
    else:
        po["stage3_truth_healpix"] = np.nan
        po["stage3_recovered_truth_healpix"] = np.nan
        po["stage3_selected_keys"] = np.nan
        po["stage3_coverage"] = np.nan

    # Outlier flags + class.
    po["outlier_low_recall"] = (po["stage4_truth_detections"] >= int(outlier_cfg.truth_min)) & (
        po["stage4_recall"] < float(outlier_cfg.recall_lt)
    )
    po["outlier_stage3_miss"] = po["outlier_low_recall"] & (po["n_frames_loaded"] <= 0) & (
        po["stage4_truth_detections"] > 0
    )
    po["outlier_stage4_reject"] = po["outlier_low_recall"] & (po["n_frames_loaded"] > 0) & (
        po["stage4_truth_detections"] > 0
    )
    po["outlier_class"] = np.where(
        po["outlier_stage3_miss"],
        "stage3_miss",
        np.where(po["outlier_stage4_reject"], "stage4_reject", None),
    )

    # ----- Optional filtering: exclude Stage 3 miss orbits -----
    po_all = po
    excluded_orbit_ids: set[str] = set()
    if exclude_stage3_recovered_le is not None or exclude_stage3_coverage_le is not None:
        m = pd.Series(False, index=po.index)
        if exclude_stage3_recovered_le is not None:
            m = m | (
                (po["stage3_truth_healpix"].fillna(0) > 0)
                & (po["stage3_recovered_truth_healpix"].fillna(0) <= int(exclude_stage3_recovered_le))
            )
        if exclude_stage3_coverage_le is not None:
            m = m | (
                (po["stage3_truth_healpix"].fillna(0) > 0)
                & (po["stage3_coverage"].fillna(0.0) <= float(exclude_stage3_coverage_le))
            )
        excluded_orbit_ids = set(po.loc[m, "orbit_id"].astype(str).tolist())
        po = po.loc[~m].copy()

    # ----- Join sampling strata (if available) -----
    sel_path = inputs_artifacts_dir / "selected_designations.parquet"
    if sel_path.exists():
        sel = pq.read_table(sel_path, columns=["designation", "stratum"]).to_pandas()
        sel = sel.rename(columns={"designation": "orbit_id"})
        po = po.merge(sel, on=["orbit_id"], how="left")
        parsed = po["stratum"].apply(_parse_stratum).apply(pd.Series)
        po = pd.concat([po, parsed], axis=1)
    else:
        po["stratum"] = None
        po["regime"] = None
        po["arc_bin"] = None
        po["dt_bin"] = None
        po["u_bin"] = None
        po["i_bin"] = None

    # ----- Join BQ orbit features (if available) -----
    bq_path = inputs_artifacts_dir / "bq_designation_orbit_features.parquet"
    if bq_path.exists():
        need = sorted(set(po["orbit_id"].astype(str).tolist()))
        bq = pq.read_table(
            bq_path,
            columns=[
                "designation",
                "orbit_type_int",
                "u_param",
                "arc_length_total",
                "nobs_total",
                "n_obs_window",
                "n_stn_window",
                "a",
                "e",
                "i",
                "q",
                "epoch_mjd",
            ],
        )
        mask = pc.is_in(bq["designation"], value_set=pa.array(need, type=pa.large_string()))
        bq = bq.filter(mask).to_pandas()
        bq = bq.rename(columns={"designation": "orbit_id"})
        po = po.merge(bq, on=["orbit_id"], how="left")

    # ----- Join covariance severity (if available) -----
    cov_path = inputs_artifacts_dir / "orbits_selected_sbdb_cov_severity.parquet"
    if cov_path.exists():
        cov = pq.read_table(cov_path).to_pandas()
        cov_cols = [
            "orbit_id",
            "cov_ok",
            "cov_invalid_reason",
            "sigma_pos_rms",
            "sigma_pos_max",
            "anisotropy_pos",
            "sigma_vel_rms",
            "sigma_vel_max",
            "anisotropy_vel",
        ]
        cov = cov[[c for c in cov_cols if c in cov.columns]].copy()
        po = po.merge(cov, on=["orbit_id"], how="left")

        # Simple bins for quick grouping.
        po["sigma_pos_rms_bin"] = po["sigma_pos_rms"].apply(lambda v: _bin_edges_label(v, [1e-6, 1e-4, 1e-2, 1.0, 10.0, 100.0], "sigpos"))
        po["anisotropy_pos_bin"] = po["anisotropy_pos"].apply(lambda v: _bin_edges_label(v, [10.0, 100.0, 1000.0, 10000.0], "aniso"))
    else:
        po["sigma_pos_rms_bin"] = None
        po["anisotropy_pos_bin"] = None

    # ----- Write tables -----
    outputs: dict[str, Path] = {}
    per_orbit_path = out_dir / "per_orbit.parquet"
    _write_parquet(po, per_orbit_path)
    outputs["per_orbit"] = per_orbit_path

    outliers_path = out_dir / "outliers_per_orbit.parquet"
    _write_parquet(po[po["outlier_low_recall"]].copy(), outliers_path)
    outputs["outliers_per_orbit"] = outliers_path

    if excluded_orbit_ids:
        excl = po_all.loc[po_all["orbit_id"].astype(str).isin(sorted(excluded_orbit_ids))].copy()
        if excl.empty:
            excl = pd.DataFrame({"orbit_id": sorted(excluded_orbit_ids)})
        excluded_path = out_dir / "excluded_orbits.parquet"
        _write_parquet(excl, excluded_path)
        outputs["excluded_orbits"] = excluded_path

    # Group tables (per-orbit aggregates).
    group_specs: list[tuple[str, list[str]]] = [
        ("by_stratum", ["stratum"]),
        ("by_regime", ["regime"]),
        ("by_arc_bin", ["arc_bin"]),
        ("by_dt_bin", ["dt_bin"]),
        ("by_u_bin", ["u_bin"]),
        ("by_i_bin", ["i_bin"]),
        ("by_orbit_type_int", ["orbit_type_int"]),
        ("by_sigma_pos_rms_bin", ["sigma_pos_rms_bin"]),
        ("by_anisotropy_pos_bin", ["anisotropy_pos_bin"]),
    ]
    for name, cols in group_specs:
        cols2 = [c for c in cols if c in po.columns]
        if not cols2:
            continue
        g = _group_stats_orbit(po, cols2)
        p = out_dir / f"{name}.parquet"
        _write_parquet(g, p)
        outputs[name] = p

    return outputs


def main() -> None:
    p = argparse.ArgumentParser(description="Build per-orbit + grouped tables from Stage 2/3/4 run artifacts.")
    p.add_argument("--subset-dir", type=str, required=True)
    p.add_argument("--stage2-run-dir", type=str, required=True)
    p.add_argument("--stage3-run-dir", type=str, required=True)
    p.add_argument("--stage4-run-dir", type=str, required=True)
    p.add_argument(
        "--inputs-artifacts-dir",
        type=str,
        required=True,
        help="Directory containing truth_precovery_crossmatch.parquet, selected_designations.parquet, bq_designation_orbit_features.parquet, orbits_selected_sbdb*.parquet.",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory (default: <subset_dir>/artifacts/tables/<stage4_run_dir.name>).",
    )
    p.add_argument("--outlier-truth-min", type=int, default=50)
    p.add_argument("--outlier-recall-lt", type=float, default=0.5)
    p.add_argument(
        "--exclude-stage3-recovered-le",
        type=int,
        default=None,
        help=(
            "Exclude any orbit where stage3_truth_healpix>0 and stage3_recovered_truth_healpix <= this value. "
            "Use 0 to drop Stage-3-miss orbits."
        ),
    )
    p.add_argument(
        "--exclude-stage3-coverage-le",
        type=float,
        default=None,
        help=(
            "Exclude any orbit where stage3_truth_healpix>0 and stage3_coverage <= this value. "
            "Example: 0.05 drops orbits recovering <=5% of truth healpixels in Stage 3."
        ),
    )
    args = p.parse_args()

    subset_dir = Path(args.subset_dir)
    stage2_run_dir = Path(args.stage2_run_dir)
    stage3_run_dir = Path(args.stage3_run_dir)
    stage4_run_dir = Path(args.stage4_run_dir)
    inputs_artifacts_dir = Path(args.inputs_artifacts_dir)

    out_dir = (
        Path(args.out_dir)
        if args.out_dir is not None
        else (subset_dir / "artifacts" / "tables" / stage4_run_dir.name)
    )
    _ensure_dir(out_dir)

    outputs = build_run_tables(
        subset_dir=subset_dir,
        stage2_run_dir=stage2_run_dir,
        stage3_run_dir=stage3_run_dir,
        stage4_run_dir=stage4_run_dir,
        inputs_artifacts_dir=inputs_artifacts_dir,
        out_dir=out_dir,
        outlier_cfg=OutlierConfig(
            truth_min=int(args.outlier_truth_min),
            recall_lt=float(args.outlier_recall_lt),
        ),
        exclude_stage3_recovered_le=args.exclude_stage3_recovered_le,
        exclude_stage3_coverage_le=args.exclude_stage3_coverage_le,
    )

    for k, v in outputs.items():
        print(f"{k}={v}")


if __name__ == "__main__":
    main()

