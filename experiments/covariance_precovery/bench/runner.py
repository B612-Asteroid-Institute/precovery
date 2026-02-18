from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from adam_core.coordinates.covariances import CoordinateCovariances
from adam_core.observers import Observers
from adam_core.orbits import Orbits
from adam_core.photometry.bandpasses import map_to_canonical_filter_bands
from adam_core.photometry.magnitude import convert_magnitude
from adam_core.time import Timestamp

from precovery.search.footprints import CovPolygonReconstructedMoc
from precovery.search.propagation import predict_targets

from .backends.protocols import BenchBackend, GateParams
from .gate import GateTotals, accepted_observation_ids_and_totals_python
from .time_key import mjd_to_time_key_us
from .types import (
    AcceptedCounts,
    BenchTargets,
    PredictedTargets,
    PredictedTriples,
    pack_cov_ll,
)
from .workload import WorkloadSpec


def _arrow_lookup_float(*, needles: pa.Array, keys: pa.Array, values: pa.Array) -> pa.Array:
    if len(keys) == 0:
        return pa.nulls(len(needles), type=pa.float64())
    idx = pc.fill_null(pc.index_in(needles, value_set=keys), -1)
    valid = pc.greater_equal(idx, 0)
    idx_safe = pc.cast(pc.if_else(valid, idx, 0), pa.int64())
    out = pc.take(values, idx_safe)
    return pc.if_else(valid, out, None)


def _load_subset_mag_config(
    *, subset_dir: Path
) -> tuple[pa.Array, pa.Array, float, float | None, float | None]:
    """
    Return (limit_codefid_keys, limit_codefid_vals, faint_margin_mag, max_faint, max_bright).
    """
    cfg_path = Path(subset_dir) / "config.json"
    data: dict[str, object] = {}
    if cfg_path.exists():
        data = json.loads(cfg_path.read_text(encoding="utf-8"))

    faint_margin = float(data.get("faint_frame_skip_margin_mag") or 0.0)
    max_faint = data.get("max_mag_residual_fainter_mag")
    max_bright = data.get("max_mag_residual_brighter_mag")
    max_faint_f = None if max_faint is None else float(max_faint)  # type: ignore[arg-type]
    max_bright_f = None if max_bright is None else float(max_bright)  # type: ignore[arg-type]

    lm_file = data.get("limiting_magnitudes_parquet_file")
    if lm_file is None:
        return pa.array([], type=pa.large_string()), pa.array([], type=pa.float64()), faint_margin, max_faint_f, max_bright_f

    lm_path = Path(subset_dir) / str(lm_file)
    if not lm_path.exists():
        return pa.array([], type=pa.large_string()), pa.array([], type=pa.float64()), faint_margin, max_faint_f, max_bright_f

    t = pq.read_table(str(lm_path), columns=["obscode", "filter_id", "limiting_mag"])
    if t.num_rows == 0:
        return pa.array([], type=pa.large_string()), pa.array([], type=pa.float64()), faint_margin, max_faint_f, max_bright_f

    sep = pa.scalar("|", type=pa.large_string())
    codefid = pc.binary_join_element_wise(t["obscode"], t["filter_id"], sep)
    base = pa.table({"codefid": codefid, "limiting_mag": pc.cast(t["limiting_mag"], pa.float64())})
    # De-duplicate if needed. Prefer max to be conservative (avoid skipping too aggressively).
    gb = base.group_by(["codefid"], use_threads=False).aggregate([("limiting_mag", "max")])
    return (
        pc.cast(gb["codefid"], pa.large_string()),
        pc.cast(gb["limiting_mag_max"], pa.float64()),
        faint_margin,
        max_faint_f,
        max_bright_f,
    )


@dataclass(frozen=True)
class TruthMetrics:
    n_truth: int
    n_hit: int

    @property
    def recall(self) -> float:
        if self.n_truth <= 0:
            return 0.0
        return float(self.n_hit) / float(self.n_truth)


@dataclass(frozen=True)
class BackendRunResult:
    backend: str
    elapsed_s: float
    bytes_estimate: int | None
    counts: AcceptedCounts
    truth: TruthMetrics | None
    gate_totals: GateTotals | None


@dataclass(frozen=True)
class BenchmarkResult:
    workload_label: str
    n_orbits: int
    n_targets: int
    n_triples: int
    build_elapsed_s: float
    build_predict_elapsed_s: float
    build_pred_mag_elapsed_s: float
    build_footprint_elapsed_s: float
    build_triples_elapsed_s: float
    n_pairs_skipped_faint: int
    backend_results: tuple[BackendRunResult, ...]


def _discover_stage2_bad_orbit_ids(*, subset_dir: Path) -> set[str]:
    """
    Discover bad orbit IDs recorded by Stage-2 runs under:
      <subset_dir>/artifacts/stage2/**/bad_orbits.json
    """
    root = Path(subset_dir) / "artifacts" / "stage2"
    if not root.exists():
        return set()
    out: set[str] = set()
    for p in root.glob("**/bad_orbits.json"):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        ids = d.get("bad_orbit_ids") or []
        for x in ids:
            if x is None:
                continue
            s = str(x).strip()
            if s:
                out.add(s)
    return out


def _drop_orbits_by_id(orbits: Orbits, *, drop_ids: set[str]) -> Orbits:
    if len(orbits) == 0 or not drop_ids:
        return orbits
    value_set = pa.array(sorted(drop_ids), type=pa.large_string())
    mask = pc.invert(pc.is_in(orbits.orbit_id, value_set=value_set))
    idx = np.nonzero(mask.to_numpy(zero_copy_only=False).astype(bool))[0]
    return orbits.take(idx.tolist())


def _repair_orbits_covariance_psd_for_sampling(
    orbits: Orbits,
    *,
    abs_tol: float = 1e-15,
    rel_tol: float = 1e-10,
) -> Orbits:
    """
    Make orbit covariances PSD-enough for sigma-point sampling.

    This mirrors the Stage-2 harness policy:
    - symmetrize
    - if min eigenvalue is only slightly negative (within tolerance), clip to 0
    - otherwise drop the orbit
    """
    if len(orbits) == 0:
        return orbits
    cov = getattr(orbits.coordinates, "covariance", None)
    if cov is None or cov.is_all_nan():
        return Orbits.empty()

    m_raw = cov.to_matrix().astype(np.float64, copy=False)

    keep: list[int] = []
    cov_out: list[np.ndarray] = []
    sym_tol = 1e-12
    for i in range(int(m_raw.shape[0])):
        c_raw = m_raw[i]
        if not np.isfinite(c_raw).all():
            continue

        # Always evaluate PSD-ness on a symmetrized copy (required for `eigh`).
        sym_err = float(np.max(np.abs(c_raw - c_raw.T)))
        c = c_raw if sym_err <= float(sym_tol) else 0.5 * (c_raw + c_raw.T)
        modified = sym_err > float(sym_tol)
        try:
            w, v = np.linalg.eigh(c)
        except Exception:  # noqa: BLE001
            continue
        if not np.isfinite(w).all():
            continue
        w_min = float(w.min())
        w_max = float(w.max())
        tol = float(max(float(abs_tol), float(rel_tol) * float(w_max)))
        if w_min < -tol:
            continue
        if w_min < 0.0:
            w = np.where(w < 0.0, 0.0, w)
            c = (v * w) @ v.T
            modified = True

        c_out_i = c_raw
        if modified:
            c_out_i = 0.5 * (c + c.T)
            # Only add diagonal jitter if strict eigval checks still see a tiny negative.
            min_ev = float(np.min(np.real(np.linalg.eigvals(c_out_i))))
            if min_ev < 0.0:
                c_out_i = c_out_i + np.eye(6, dtype=np.float64) * (
                    -min_ev + float(abs_tol) * 10.0
                )
        keep.append(int(i))
        cov_out.append(c_out_i.astype(np.float64, copy=False))

    if not keep:
        return Orbits.empty()

    out = orbits.take(keep)
    cov_fixed = np.stack(cov_out, axis=0)
    out = out.set_column("coordinates.covariance", CoordinateCovariances.from_matrix(cov_fixed))
    return out


def _sample_orbits(orbits: Orbits, *, max_orbits: int | None) -> Orbits:
    if max_orbits is None or len(orbits) <= int(max_orbits):
        return orbits
    return orbits.take(list(range(int(max_orbits))))


def _sample_targets(targets: BenchTargets, *, max_targets: int | None) -> BenchTargets:
    if max_targets is None or len(targets) <= int(max_targets):
        return targets
    return targets.take(list(range(int(max_targets))))


def _build_predictions_and_triples(
    *,
    orbits: Orbits,
    targets: BenchTargets,
    workload: WorkloadSpec,
    start_mjd_utc: float,
    window_size_days: int = 30,
    max_processes: int | None = None,
    limit_codefid_keys: pa.Array,
    limit_codefid_vals: pa.Array,
    faint_margin_mag: float,
    max_mag_residual_fainter_mag: float | None,
    max_mag_residual_brighter_mag: float | None,
) -> tuple[PredictedTargets, PredictedTriples, float, float, float, float, int]:
    """
    Build `PredictedTargets` + exploded `PredictedTriples` for all (orbit, target_idx).

    This uses the production `predict_targets` + `CovPolygonReconstructedMoc`.
    """
    if len(targets) == 0 or len(orbits) == 0:
        return PredictedTargets.empty(), PredictedTriples.empty(), 0.0, 0.0, 0.0, 0.0, 0

    # Drop known-bad orbit IDs (recorded by previous Stage-2 runs) to avoid hard failures.
    bad_ids = _discover_stage2_bad_orbit_ids(subset_dir=Path(workload.subset_dir))
    orbits = _drop_orbits_by_id(orbits, drop_ids=bad_ids)
    if len(orbits) == 0:
        return PredictedTargets.empty(), PredictedTriples.empty(), 0.0, 0.0, 0.0, 0.0, 0

    # Ensure covariances are PSD enough for sigma-point sampling to avoid hard failures.
    orbits = _repair_orbits_covariance_psd_for_sampling(orbits)
    if len(orbits) == 0:
        return PredictedTargets.empty(), PredictedTriples.empty(), 0.0, 0.0, 0.0, 0.0, 0

    footprint = CovPolygonReconstructedMoc(n_sigma=float(workload.n_sigma))
    nside = int(workload.healpix_nside)

    times = Timestamp.from_mjd(targets.exposure_mjd_mid_utc, scale="utc")
    obsc = targets.obscode  # pyarrow array

    canonical = map_to_canonical_filter_bands(
        targets.obscode,
        targets.filter,
        allow_fallback_filters=True,
    )
    canon_arr = pa.array(canonical, type=pa.large_string())

    # Pre-build bandpass conversion IDs once (reused across orbits).
    src_filter_id = np.full(len(targets), "V", dtype=object)
    tgt_filter_id = np.asarray(canonical, dtype=object)

    # Precompute observers aligned to targets once; reused across orbits in stage2.
    observers_all = Observers.from_codes(pa.array(obsc, type=pa.large_string()), times)

    # Precompute limiting-magnitude lookup needles once (aligned to targets).
    sep = pa.scalar("|", type=pa.large_string())
    target_codefid = pc.binary_join_element_wise(targets.obscode, canon_arr, sep)
    limit_for_target = _arrow_lookup_float(
        needles=target_codefid, keys=limit_codefid_keys, values=limit_codefid_vals
    )
    limit_with_margin = pc.add(limit_for_target, float(faint_margin_mag))
    need_pred_mag = (
        len(limit_codefid_keys) > 0
        or max_mag_residual_fainter_mag is not None
        or max_mag_residual_brighter_mag is not None
    )

    # Build outputs as column-oriented Python lists (far cheaper than per-row dicts).
    pred_orbit_id: list[str] = []
    pred_target_idx: list[int] = []
    pred_obscode: list[str] = []
    pred_exposure_mjd_mid_utc: list[float] = []
    pred_exposure_mjd_mid_key_us: list[int] = []
    pred_canonical_filter_id: list[str | None] = []
    pred_lon_deg: list[float] = []
    pred_lat_deg: list[float] = []
    pred_cov00: list[float] = []
    pred_cov01: list[float] = []
    pred_cov11: list[float] = []
    pred_mag_out: list[float | None] = []

    trip_orbit_id: list[str] = []
    trip_target_idx: list[int] = []
    trip_obscode: list[str] = []
    trip_exposure_mjd_mid_utc: list[float] = []
    trip_exposure_mjd_mid_key_us: list[int] = []
    trip_healpixel: list[int] = []

    build_predict_s = 0.0
    build_pred_mag_s = 0.0
    build_footprint_s = 0.0
    build_triples_s = 0.0
    n_pairs_skipped_faint = 0

    # Reduce per-chunk overhead inside `predict_targets` by using larger time chunks in benchmarks.
    # Cap to keep K*N ephemeris materialization bounded.
    time_chunk_size = int(min(20_000, max(2_000, int(len(targets)))))

    for i_orb in range(int(len(orbits))):
        orbit_single = orbits.take([int(i_orb)])
        oid = str(orbit_single.orbit_id[0].as_py())

        try:
            t_pred0 = time.perf_counter()
            pred = predict_targets(
                orbit=orbit_single,
                obscode=obsc,
                times_utc=times,
                start_mjd=float(start_mjd_utc),
                window_size_days=int(window_size_days),
                strategy=str(workload.stage2_strategy),
                max_processes=max_processes,
                time_chunk_size=time_chunk_size,
                observers_all=observers_all,
            )
            build_predict_s += time.perf_counter() - t_pred0
        except Exception:
            # Treat as a bad orbit for this workload; skip rather than aborting the benchmark.
            continue
        eph = pred.ephem
        if len(eph) != len(targets):
            raise RuntimeError("predict_targets returned unexpected length vs targets")

        lon = eph.coordinates.lon.to_numpy(zero_copy_only=False).astype(np.float64)
        lat = eph.coordinates.lat.to_numpy(zero_copy_only=False).astype(np.float64)

        # Predicted apparent magnitudes (canonical band) for this orbit, aligned to targets.
        pred_mag_arr: pa.Array = pa.nulls(len(targets), type=pa.float64())
        if need_pred_mag:
            if (
                hasattr(eph, "predicted_magnitude_v")
                and not pc.all(pc.is_null(eph.predicted_magnitude_v)).as_py()
            ):
                t_mag0 = time.perf_counter()
                mag_v = pc.cast(eph.predicted_magnitude_v, pa.float64())
                valid = pc.is_valid(mag_v)
                mags_v_np = pc.fill_null(
                    mag_v, pa.scalar(np.nan, type=pa.float64())
                ).to_numpy(zero_copy_only=False).astype(np.float64)
                pred_mag_np = convert_magnitude(
                    mags_v_np,
                    source_filter_id=src_filter_id,
                    target_filter_id=tgt_filter_id,
                    composition="C",
                )
                pred_mag_arr = pc.if_else(
                    valid, pa.array(pred_mag_np, type=pa.float64()), None
                )
                build_pred_mag_s += time.perf_counter() - t_mag0

        pred_mag_list = pred_mag_arr.to_pylist()

        too_faint = None
        if len(limit_codefid_keys) > 0 and not pc.all(pc.is_null(pred_mag_arr)).as_py():
            too_faint = pc.fill_null(
                pc.and_(
                    pc.is_valid(limit_for_target),
                    pc.greater(pred_mag_arr, limit_with_margin),
                ),
                False,
            )

        for i in range(len(targets)):
            cov_ll = pred.cov_ll_deg2[i]
            c00, c01, c11 = pack_cov_ll(cov_ll)
            pred_orbit_id.append(oid)
            pred_target_idx.append(int(i))
            pred_obscode.append(str(targets.obscode[i].as_py()))
            pred_exposure_mjd_mid_utc.append(float(targets.exposure_mjd_mid_utc[i].as_py()))
            pred_exposure_mjd_mid_key_us.append(int(targets.exposure_mjd_mid_key_us[i].as_py()))
            pred_canonical_filter_id.append(
                str(canon_arr[i].as_py()) if canon_arr[i].as_py() is not None else None
            )
            pred_lon_deg.append(float(lon[i]))
            pred_lat_deg.append(float(lat[i]))
            pred_cov00.append(float(c00))
            pred_cov01.append(float(c01))
            pred_cov11.append(float(c11))
            pm = pred_mag_list[i]
            pred_mag_out.append(None if pm is None else float(pm))

            if too_faint is not None and bool(too_faint[i].as_py()):
                # Limiting-magnitude skip: do not generate triples, so backends never fetch candidates.
                n_pairs_skipped_faint += 1
                continue
            t_fp0 = time.perf_counter()
            pix = footprint.pixels_for_prediction(
                lon0_deg=float(lon[i]),
                lat0_deg=float(lat[i]),
                cov_ll_deg2=cov_ll,
                nside=nside,
            )
            build_footprint_s += time.perf_counter() - t_fp0

            pix_arr = np.asarray(pix, dtype=np.int64)
            n_pix = int(pix_arr.size)
            if n_pix <= 0:
                continue

            t_tr0 = time.perf_counter()
            trip_orbit_id.extend([oid] * n_pix)
            trip_target_idx.extend([int(i)] * n_pix)
            oc = str(targets.obscode[i].as_py())
            mjd = float(targets.exposure_mjd_mid_utc[i].as_py())
            key_us = int(targets.exposure_mjd_mid_key_us[i].as_py())
            trip_obscode.extend([oc] * n_pix)
            trip_exposure_mjd_mid_utc.extend([mjd] * n_pix)
            trip_exposure_mjd_mid_key_us.extend([key_us] * n_pix)
            trip_healpixel.extend(pix_arr.tolist())
            build_triples_s += time.perf_counter() - t_tr0

    if pred_orbit_id:
        pred_tbl = pa.table(
            {
                "orbit_id": pa.array(pred_orbit_id, type=pa.large_string()),
                "target_idx": pa.array(pred_target_idx, type=pa.int64()),
                "obscode": pa.array(pred_obscode, type=pa.large_string()),
                "exposure_mjd_mid_utc": pa.array(pred_exposure_mjd_mid_utc, type=pa.float64()),
                "exposure_mjd_mid_key_us": pa.array(pred_exposure_mjd_mid_key_us, type=pa.int64()),
                "canonical_filter_id": pa.array(pred_canonical_filter_id, type=pa.large_string()),
                "pred_lon_deg": pa.array(pred_lon_deg, type=pa.float64()),
                "pred_lat_deg": pa.array(pred_lat_deg, type=pa.float64()),
                "cov_ll_00": pa.array(pred_cov00, type=pa.float64()),
                "cov_ll_01": pa.array(pred_cov01, type=pa.float64()),
                "cov_ll_11": pa.array(pred_cov11, type=pa.float64()),
                "pred_mag": pa.array(pred_mag_out, type=pa.float64()),
            }
        )
    else:
        pred_tbl = PredictedTargets.empty().table

    if trip_orbit_id:
        triples_tbl = pa.table(
            {
                "orbit_id": pa.array(trip_orbit_id, type=pa.large_string()),
                "target_idx": pa.array(trip_target_idx, type=pa.int64()),
                "obscode": pa.array(trip_obscode, type=pa.large_string()),
                "exposure_mjd_mid_utc": pa.array(trip_exposure_mjd_mid_utc, type=pa.float64()),
                "exposure_mjd_mid_key_us": pa.array(trip_exposure_mjd_mid_key_us, type=pa.int64()),
                "healpixel": pa.array(trip_healpixel, type=pa.int64()),
            }
        )
    else:
        triples_tbl = PredictedTriples.empty().table

    return (
        PredictedTargets.from_pyarrow(pred_tbl),
        PredictedTriples.from_pyarrow(triples_tbl),
        float(build_predict_s),
        float(build_pred_mag_s),
        float(build_footprint_s),
        float(build_triples_s),
        int(n_pairs_skipped_faint),
    )


def _load_truth_table(truth_path: Path) -> pa.Table:
    t = pq.read_table(str(truth_path))
    cols = set(t.column_names)

    orbit_col = "orbit_id" if "orbit_id" in cols else ("designation" if "designation" in cols else None)
    if orbit_col is None:
        raise ValueError(f"Truth table missing expected orbit id column. Found: {sorted(cols)}")

    # Prefer crossmatch-to-precovery IDs when present: these are the observation IDs our
    # candidate backends will emit/recover.
    if "match_observation_id" in cols:
        obs_col = "match_observation_id"
    elif "observation_id" in cols:
        obs_col = "observation_id"
    elif "obsid" in cols:
        obs_col = "obsid"
    elif "truth_obsid" in cols:
        # Fallback: truth observation IDs (may be a different namespace than precovery obs IDs).
        obs_col = "truth_obsid"
    else:
        raise ValueError(f"Truth table missing expected observation id column. Found: {sorted(cols)}")

    # Time column for window filtering (prefer matched time when present).
    time_col = (
        "match_time_mjd_utc"
        if "match_time_mjd_utc" in cols
        else ("truth_time_mjd_utc" if "truth_time_mjd_utc" in cols else None)
    )
    if time_col is None:
        raise ValueError(f"Truth table missing expected time column. Found: {sorted(cols)}")

    select_cols = [orbit_col, obs_col, "obscode", time_col]
    if "matched" in cols:
        select_cols.append("matched")
    t2 = t.select(select_cols).rename_columns(
        ["orbit_id", "observation_id", "obscode", "time_mjd_utc", *(['matched'] if 'matched' in cols else [])]
    )

    if "matched" in t2.column_names:
        m = pc.fill_null(t2["matched"], False)
        t2 = t2.filter(m).drop(["matched"])

    t2 = t2.filter(pc.invert(pc.is_null(t2["observation_id"])))
    t2 = t2.filter(pc.invert(pc.is_null(t2["time_mjd_utc"])))
    return t2


def _truth_targets_for_workload(*, truth: pa.Table, start_mjd_utc: float, end_mjd_utc: float) -> BenchTargets:
    """
    Build target exposure midpoints from truth crossmatch rows within the workload time bounds.

    This avoids enumerating all exposures in the month when we specifically want truth scoring.
    """
    if truth.num_rows == 0:
        return BenchTargets.empty()
    m = pc.greater_equal(truth["time_mjd_utc"], pa.scalar(float(start_mjd_utc)))
    m = pc.and_(m, pc.less(truth["time_mjd_utc"], pa.scalar(float(end_mjd_utc))))
    t2 = truth.filter(m).select(["obscode", "time_mjd_utc"])
    if t2.num_rows == 0:
        return BenchTargets.empty()
    # Unique (obscode,time) targets.
    t2 = t2.group_by(["obscode", "time_mjd_utc"]).aggregate([]).sort_by([("time_mjd_utc", "ascending")])
    mids = np.asarray(t2["time_mjd_utc"].to_numpy(zero_copy_only=False), dtype=np.float64)
    return BenchTargets.from_kwargs(
        obscode=[str(x) for x in t2["obscode"].to_pylist()],
        exposure_mjd_mid_utc=mids,
        # Truth tables do not reliably carry the dataset filter band; use a placeholder.
        filter=["V"] * int(mids.size),
        exposure_mjd_mid_key_us=mjd_to_time_key_us(mids),
    )


def _truth_metrics(*, truth: pa.Table, accepted: pa.Table) -> TruthMetrics:
    if truth.num_rows == 0 or accepted.num_rows == 0:
        return TruthMetrics(n_truth=int(truth.num_rows), n_hit=0)
    joined = accepted.join(truth, keys=["orbit_id", "observation_id"], join_type="inner")
    return TruthMetrics(n_truth=int(truth.num_rows), n_hit=int(joined.num_rows))


def run_benchmark(
    *,
    workload: WorkloadSpec,
    orbits: Orbits,
    enumerate_backend: BenchBackend,
    backends: Iterable[BenchBackend],
    max_orbits: int | None = 25,
    max_targets: int | None = 2000,
    truth_path: Path | None = None,
    targets_from_truth: bool = False,
    det_sigma_floor_arcsec: float = 0.10,
    max_processes: int | None = None,
    compute_gate_totals: bool = False,
) -> BenchmarkResult:
    """
    Run the canonical workload across backends and collect comparable metrics.
    """
    subset = workload.subset_paths()
    start_mjd, end_mjd = workload.bounds_mjd_utc()

    limit_keys, limit_vals, faint_margin, max_faint, max_bright = _load_subset_mag_config(
        subset_dir=Path(workload.subset_dir)
    )

    orbits_use = _sample_orbits(orbits, max_orbits=max_orbits)
    truth_tbl = _load_truth_table(truth_path) if truth_path is not None and truth_path.exists() else None
    if truth_tbl is not None:
        # Restrict truth rows to the current window obscodes, orbit sample, and time range.
        m = pc.is_in(
            truth_tbl["obscode"],
            value_set=pa.array(list(workload.window.obscodes), type=pa.large_string()),
        )
        m = pc.and_(
            m,
            pc.is_in(
                truth_tbl["orbit_id"],
                value_set=pa.array(
                    [str(x) for x in orbits_use.orbit_id.to_pylist()], type=pa.large_string()
                ),
            ),
        )
        m = pc.and_(m, pc.greater_equal(truth_tbl["time_mjd_utc"], pa.scalar(float(start_mjd))))
        m = pc.and_(m, pc.less(truth_tbl["time_mjd_utc"], pa.scalar(float(end_mjd))))
        truth_tbl = truth_tbl.filter(m)
        if truth_tbl.num_rows == 0:
            truth_tbl = None

    if truth_tbl is not None and bool(targets_from_truth):
        targets = _truth_targets_for_workload(
            truth=truth_tbl, start_mjd_utc=float(start_mjd), end_mjd_utc=float(end_mjd)
        )
        if len(targets) == 0:
            targets = enumerate_backend.enumerate_targets(
                subset=subset,
                start_mjd_utc=float(start_mjd),
                end_mjd_utc=float(end_mjd),
                obscodes=workload.window.obscodes,
            )
    else:
        targets = enumerate_backend.enumerate_targets(
            subset=subset,
            start_mjd_utc=float(start_mjd),
            end_mjd_utc=float(end_mjd),
            obscodes=workload.window.obscodes,
        )
    targets = _sample_targets(targets, max_targets=max_targets)

    t0 = time.perf_counter()
    preds, triples, t_pred, t_mag, t_fp, t_tri, n_skipped_faint = _build_predictions_and_triples(
        orbits=orbits_use,
        targets=targets,
        workload=workload,
        start_mjd_utc=float(start_mjd),
        window_size_days=30,
        max_processes=max_processes,
        limit_codefid_keys=limit_keys,
        limit_codefid_vals=limit_vals,
        faint_margin_mag=float(faint_margin),
        max_mag_residual_fainter_mag=max_faint,
        max_mag_residual_brighter_mag=max_bright,
    )
    build_elapsed = time.perf_counter() - t0

    gate = GateParams(
        n_sigma=float(workload.n_sigma),
        det_sigma_floor_arcsec=float(det_sigma_floor_arcsec),
        max_mag_residual_fainter_mag=max_faint,
        max_mag_residual_brighter_mag=max_bright,
    )

    backend_results: list[BackendRunResult] = []
    for b in backends:
        t1 = time.perf_counter()
        counts = b.count_accepted(subset=subset, triples=triples, preds=preds, gate=gate)
        elapsed = time.perf_counter() - t1

        bytes_est = getattr(b, "last_bytes_estimate", None)
        truth_m: TruthMetrics | None = None
        gate_totals: GateTotals | None = None
        # Only compute truth metrics for backends that can fetch rows (BigQuery virtual skips).
        if truth_tbl is not None:
            try:
                cands = b.fetch_candidates(subset=subset, triples=triples, limit=None)
                accepted, gate_totals = accepted_observation_ids_and_totals_python(
                    candidates=cands, preds=preds, gate=gate
                )
                truth_m = _truth_metrics(truth=truth_tbl, accepted=accepted)
            except NotImplementedError:
                truth_m = None
        elif bool(compute_gate_totals):
            # Optional: compute rejection totals even in throughput mode (no truth table).
            # This re-fetches candidate rows, so keep it off for large runs unless needed.
            try:
                cands = b.fetch_candidates(subset=subset, triples=triples, limit=None)
                _, gate_totals = accepted_observation_ids_and_totals_python(
                    candidates=cands, preds=preds, gate=gate
                )
            except NotImplementedError:
                gate_totals = None
        backend_results.append(
            BackendRunResult(
                backend=str(b.name),
                elapsed_s=float(elapsed),
                bytes_estimate=None if bytes_est is None else int(bytes_est),
                counts=counts,
                truth=truth_m,
                gate_totals=gate_totals,
            )
        )

    return BenchmarkResult(
        workload_label=workload.window.label(),
        n_orbits=int(len(orbits_use)),
        n_targets=int(len(targets)),
        n_triples=int(len(triples)),
        build_elapsed_s=float(build_elapsed),
        build_predict_elapsed_s=float(t_pred),
        build_pred_mag_elapsed_s=float(t_mag),
        build_footprint_elapsed_s=float(t_fp),
        build_triples_elapsed_s=float(t_tri),
        n_pairs_skipped_faint=int(n_skipped_faint),
        backend_results=tuple(backend_results),
    )

