from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import quivr as qv
import pyarrow.parquet as pq
from adam_assist import ASSISTPropagator
from adam_core.coordinates.covariances import CoordinateCovariances
from adam_core.dynamics.ephemeris import generate_ephemeris_2body
from adam_core.dynamics.propagation import propagate_2body
from adam_core.observers import Observers
from adam_core.orbits import Orbits
from adam_core.orbits.ephemeris import Ephemeris
from adam_core.orbits.variants import VariantOrbits
from adam_core.time import Timestamp

from precovery.precovery_db import PrecoveryDatabase

from ..selection.subset_designations import read_subset_window


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _utc_run_id() -> str:
    # e.g. 20260127T184501Z
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _write_json(path: Path, obj: dict[str, object]) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def _pair_observers_for_propagated_orbits(*, observers: Observers, n_orbits: int) -> Observers:
    """
    `generate_ephemeris_2body` expects `propagated_orbits` and `observers` to be paired 1:1.

    `propagate_2body(orbits, times)` expands to (N_orbits * N_times) rows in "orbit-major" order:
      [orbit0@t0..tM, orbit1@t0..tM, ...]

    So we repeat the (code,time) window-centers for each orbit in that same order.
    """
    n_times = int(len(observers))
    idx = np.tile(np.arange(n_times, dtype=np.int64), int(n_orbits))
    # Reuse already-computed observer states; just repeat rows in the right order.
    return observers.take(idx)


def _cov_ok_mask(orbits: Orbits) -> np.ndarray:
    if orbits.coordinates.covariance is None:
        return np.zeros(len(orbits), dtype=bool)
    m = orbits.coordinates.covariance.to_matrix()
    return np.isfinite(m).all(axis=(1, 2))


def _designation_from_object_id(object_id: str) -> str:
    s = str(object_id).strip()
    if s.startswith("(") and s.endswith(")") and len(s) >= 3:
        return s[1:-1].strip()
    return s.split()[0].strip()


def _truth_orbit_ids_in_subset(subset_dir: Path) -> set[str]:
    """
    Return orbit_ids (designation-normalized) that have matched truth detections in this subset window.
    """
    truth_path = subset_dir / "artifacts" / "truth_precovery_crossmatch.parquet"
    orbits_path = subset_dir / "artifacts" / "orbits_selected_sbdb.parquet"
    if not truth_path.exists() or not orbits_path.exists():
        return set()

    truth = pq.read_table(str(truth_path), columns=["matched", "designation"])
    truth = truth.filter(pc.equal(truth["matched"], True))
    truth_des = set(str(x) for x in pc.unique(truth["designation"]).to_pylist())

    orbits = pq.read_table(str(orbits_path), columns=["orbit_id", "object_id"])
    orbit_id = [str(x) for x in orbits["orbit_id"].to_pylist()]
    object_id = [str(x) for x in orbits["object_id"].to_pylist()]
    des = [_designation_from_object_id(x) for x in object_id]
    des_to_orbit = {d: oid for d, oid in zip(des, orbit_id)}

    return {des_to_orbit[d] for d in truth_des if d in des_to_orbit}


def _run_2body_ephemeris(*, orbits: Orbits, window_observers_tdb: Observers) -> Ephemeris:
    """
    2-body ephemeris for all (orbit, window_center) pairs with a single vectorized
    propagation + ephemeris call (no Python loops over times).
    """
    times = window_observers_tdb.coordinates.time
    prop = propagate_2body(orbits, times)  # (N_orbits * N_times)
    obs_nm = _pair_observers_for_propagated_orbits(observers=window_observers_tdb, n_orbits=int(len(orbits)))
    ephem = generate_ephemeris_2body(prop, obs_nm)
    # Store ephemeris in UTC for downstream joins (Stage 3), even if propagation
    # is performed in TDB internally.
    return _ephem_to_utc(ephem)


def _run_assist_window_then_2body(*, orbits: Orbits, window_observers_tdb: Observers) -> Ephemeris:
    """
    ASSIST propagate to reference epoch (no covariance), then 2-body to each
    window-center time (paired, vectorized over orbits and times).
    """
    mean_mjd_tdb = float(np.mean(window_observers_tdb.coordinates.time.mjd().to_numpy(zero_copy_only=False)))
    t_ref = Timestamp.from_mjd([mean_mjd_tdb], scale="tdb")
    prop_assist = ASSISTPropagator()
    orb_ref = prop_assist.propagate_orbits(orbits, t_ref, covariance=False)
    return _run_2body_ephemeris(orbits=orb_ref, window_observers_tdb=window_observers_tdb)


def _ephem_to_utc(ephem: Ephemeris) -> Ephemeris:
    """
    Normalize ephemeris timestamps to UTC so downstream joins can be performed in a
    single timescale (Stage 3 is keyed on exposure midpoints in UTC MJD).

    We intentionally keep this as a "best-effort" transformation: if a given column path
    is absent (or already UTC), we simply leave it unchanged.
    """
    try:
        ephem = ephem.set_column("coordinates.time", ephem.coordinates.time.rescale("utc"))
    except Exception:  # noqa: BLE001
        pass
    try:
        ephem = ephem.set_column(
            "aberrated_coordinates.time", ephem.aberrated_coordinates.time.rescale("utc")
        )
    except Exception:  # noqa: BLE001
        pass
    return ephem

def _parse_strategies_arg(s: str | None) -> list[str] | None:
    if s is None:
        return None
    items = [p.strip() for p in s.split(",")]
    items = [p for p in items if p]
    return items or None


def _parse_mc_samples_arg(s: str | None, *, default: list[int]) -> list[int]:
    if s is None:
        return list(default)
    items: list[int] = []
    for p in s.split(","):
        p = p.strip()
        if not p:
            continue
        items.append(int(p))
    if not items:
        return list(default)
    if any(n <= 0 for n in items):
        raise ValueError("mc_samples must all be > 0")
    # de-dupe but preserve order
    out: list[int] = []
    seen: set[int] = set()
    for n in items:
        if n in seen:
            continue
        seen.add(n)
        out.append(n)
    return out


def _fetch_unique_exposure_midpoints(
    db: PrecoveryDatabase,
    *,
    start_mjd: float,
    end_mjd: float,
) -> tuple[pa.Array, Timestamp]:
    """
    Return all distinct (obscode, exposure_mjd_mid) pairs in [start_mjd, end_mjd).
    """
    import sqlalchemy as sq

    idx = db.frames.idx
    query = (
        sq.select(idx.frames.c.obscode, idx.frames.c.exposure_mjd_mid)
        .where((idx.frames.c.exposure_mjd_mid < end_mjd) & (idx.frames.c.exposure_mjd_mid >= start_mjd))
        .distinct()
        .order_by(idx.frames.c.exposure_mjd_mid.asc(), idx.frames.c.obscode.asc())
    )

    chunk_size = 100000
    obscodes: list[str] = []
    mjds: list[float] = []
    result = idx.dbconn.execution_options(stream_results=True).execute(query)
    while True:
        chunk = result.fetchmany(chunk_size)
        if not chunk:
            break
        c_obs, c_mjd = zip(*chunk)
        obscodes.extend(c_obs)
        mjds.extend(c_mjd)

    if not mjds:
        return pa.array([], type=pa.large_string()), Timestamp.from_mjd([], scale="utc")
    return pa.array(obscodes, type=pa.large_string()), Timestamp.from_mjd(mjds, scale="utc")


def _write_quivr_part(out_dir: Path, *, qt: qv.Table, part_idx: int) -> None:
    _ensure_dir(out_dir)
    qt.to_parquet(str(out_dir / f"part-{part_idx:06d}.parquet"))


class FrameTimeTargets(qv.Table):
    obscode = qv.LargeStringColumn()
    time = Timestamp.as_column()


def run_stage2_propagation_bench(
    *,
    subset_dir: Path,
    orbits_parquet: Path,
    window_size_days: int = 7,
    out_dir: Path | None = None,
    max_orbits: int | None = None,
    max_window_centers: int | None = None,
    strategies: list[str] | None = None,
    mc_samples: list[int] | None = None,
    time_chunk_size: int = 2048,
    max_processes: int | None = 8,
    only_truth_orbits: bool = False,
    mean_with_covariance: bool = False,
    write_ephemeris: bool = True,
    write_variants_orbits: bool = True,
) -> Path:
    win = read_subset_window(subset_dir)
    db = PrecoveryDatabase.from_dir(str(subset_dir), allow_version_mismatch=True)

    if out_dir is None:
        out_dir = win.artifacts_dir / "stage2"
    _ensure_dir(out_dir)
    run_dir = out_dir / _utc_run_id()
    _ensure_dir(run_dir)

    # Full propagation targets: all distinct (obscode, exposure_mjd_mid) pairs in the subset range.
    # This is the true workload for per-frame ephemeris generation.
    target_codes, target_times_utc = _fetch_unique_exposure_midpoints(
        db, start_mjd=float(win.min_mjd), end_mjd=float(win.max_mjd) + 1e-9
    )
    if max_window_centers is not None:
        # Backwards-compat: this flag now caps the number of (obscode,time) targets.
        n = int(max_window_centers)
        target_codes = target_codes.slice(0, n)
        target_times_utc = target_times_utc[:n]

    # Use SBDB-provided orbits by default if present.
    sbdb_default = win.artifacts_dir / "orbits_selected_sbdb.parquet"
    orbits_path = sbdb_default if sbdb_default.exists() else orbits_parquet
    orbits = Orbits.from_parquet(str(orbits_path))
    truth_orbit_ids: set[str] | None = None
    if bool(only_truth_orbits):
        truth_orbit_ids = _truth_orbit_ids_in_subset(subset_dir)
        if truth_orbit_ids:
            mask = pc.is_in(orbits.orbit_id, value_set=pa.array(sorted(truth_orbit_ids), pa.large_string()))
            idx = np.nonzero(mask.to_numpy(zero_copy_only=False).astype(bool))[0]
            orbits = orbits.take(idx.tolist())
    if max_orbits is not None:
        orbits = orbits[: int(max_orbits)]
    cov_ok = _cov_ok_mask(orbits)
    n_cov_ok = int(cov_ok.sum())
    orbits_covok = orbits.take(np.where(cov_ok)[0]) if n_cov_ok > 0 else Orbits.empty()
    # For mean-only ephemerides, we generally do NOT want covariance propagation costs.
    # If desired, enable via `mean_with_covariance=True`.
    if bool(mean_with_covariance):
        orbits_mean = orbits
    else:
        orbits_mean = orbits.set_column(
            "coordinates.covariance", CoordinateCovariances.nulls(len(orbits))
        )

    # Target observers at full frame times.
    target_observers_utc = Observers.from_codes(target_codes, target_times_utc)
    target_observers_tdb = Observers.from_codes(target_codes, target_times_utc.rescale("tdb"))

    metrics_rows: list[dict[str, object]] = []

    inputs_dir = run_dir / "inputs"
    _ensure_dir(inputs_dir)
    FrameTimeTargets.from_kwargs(obscode=target_codes, time=target_times_utc).to_parquet(
        str(inputs_dir / "frame_time_targets.parquet")
    )
    target_observers_utc.to_parquet(str(inputs_dir / "target_observers_utc.parquet"))
    target_observers_tdb.to_parquet(str(inputs_dir / "target_observers_tdb.parquet"))

    # Window centers are now used *only* for the mixed strategy.
    windows = db.frames.idx.window_centers(win.min_mjd, win.max_mjd, int(window_size_days))
    windows.to_parquet(str(inputs_dir / "window_centers.parquet"))

    # Warm up JIT compilation / kernels (do not include in benchmark timings).
    # Without this, the first strategy to run can look artificially slow.
    if len(orbits_mean) > 0 and len(target_observers_utc) > 0:
        o1 = orbits_mean[:1]
        obs_utc_2 = target_observers_utc[: min(2, len(target_observers_utc))]
        obs_tdb_2 = target_observers_tdb[: min(2, len(target_observers_tdb))]
        try:
            _ = _run_2body_ephemeris(orbits=o1, window_observers_tdb=obs_tdb_2)
        except Exception:
            pass
        try:
            _ = ASSISTPropagator().generate_ephemeris(o1, obs_utc_2, covariance=False)
        except Exception:
            pass

    strategies_dir = run_dir / "strategies"
    _ensure_dir(strategies_dir)

    def _write_strategy_meta(strategy_dir: Path, meta: dict[str, object]) -> None:
        _ensure_dir(strategy_dir)
        _write_json(strategy_dir / "meta.json", meta)

    def _enabled(name: str) -> bool:
        return strategies is None or name in strategies

    # MC defaults:
    # - In capped runs (debugging / development), we default to MC=256 so we always have a
    #   representative Monte Carlo benchmark.
    # - In full runs (no caps), MC must be explicitly requested (otherwise it can explode).
    if mc_samples is None:
        mc_samples = [256] if (max_orbits is not None or max_window_centers is not None) else []

    # 2-body baselines (full target times).
    for name, include_covariance in [
        ("2body_only", False),
        ("2body_with_covariance", True),
    ]:
        if not _enabled(name):
            continue
        strat_dir = strategies_dir / name
        # Benchmark time should cover only propagation + ephemeris generation (compute),
        # not frame/target fetching, window computation, or disk I/O.
        err = None
        n_rows: int | None = None
        compute_sec = 0.0
        io_sec = 0.0
        try:
            _ensure_dir(strat_dir)
            out_ephem_dir = strat_dir / "mean_ephemeris"
            _ensure_dir(out_ephem_dir)

            orbits_for_mean = orbits if bool(include_covariance) else orbits_mean

            # Chunk over target times to avoid huge in-memory tables.
            n_targets = int(len(target_observers_tdb))
            total_rows = 0
            part = 0
            for i0 in range(0, n_targets, int(time_chunk_size)):
                i1 = min(i0 + int(time_chunk_size), n_targets)
                obs_tdb = target_observers_tdb[i0:i1]
                t_compute0 = time.perf_counter()
                ephem = _run_2body_ephemeris(orbits=orbits_for_mean, window_observers_tdb=obs_tdb)
                compute_sec += time.perf_counter() - t_compute0
                total_rows += int(len(ephem))
                if write_ephemeris:
                    t_io0 = time.perf_counter()
                    _write_quivr_part(out_ephem_dir, qt=ephem, part_idx=part)
                    io_sec += time.perf_counter() - t_io0
                part += 1
            n_rows = total_rows
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
        _write_strategy_meta(
            strat_dir,
            dict(
                strategy=name,
                kind="mean_ephemeris",
                window_size_days=int(window_size_days),
                n_orbits=int(len(orbits)),
                n_time_targets=int(len(target_observers_utc)),
                n_window_centers=int(len(windows)),
                n_rows=n_rows,
                time_chunk_size=int(time_chunk_size),
                mean_with_covariance=bool(include_covariance),
                runtime_sec=float(compute_sec),
                io_sec=float(io_sec),
                runtime_total_sec=float(compute_sec + io_sec),
                error=err,
            ),
        )
        metrics_rows.append(
            dict(
                subset_dir=str(subset_dir),
                strategy=name,
                variant_kind=None,
                n_orbits=int(len(orbits)),
                n_orbits_covok=n_cov_ok,
                n_time_targets=int(len(target_observers_utc)),
                n_window_centers=int(len(windows)),
                n_ephem_rows_mean=n_rows,
                n_variant_orbits=None,
                n_ephem_rows_variants=None,
                runtime_sec=float(compute_sec),
                io_sec=float(io_sec),
                runtime_total_sec=float(compute_sec + io_sec),
                error=err,
            )
        )

    # Mixed strategy: ASSIST to window centers, then 2-body to all targets within each window.
    if _enabled("assist_window_then_2body"):
        name = "assist_window_then_2body"
        strat_dir = strategies_dir / name
        # Benchmark time should cover only propagation + ephemeris generation (compute),
        # not frame/target fetching, window computation, or disk I/O.
        err = None
        n_rows: int | None = None
        compute_sec = 0.0
        io_sec = 0.0
        try:
            _ensure_dir(strat_dir)
            out_ephem_dir = strat_dir / "mean_ephemeris"
            _ensure_dir(out_ephem_dir)

            # Prepare center times (unique) and propagate once with ASSIST.
            center_times_utc = windows.time.unique().sort_by(["days", "nanos"])
            T = int(len(center_times_utc))
            if T == 0:
                raise ValueError("No window centers found; cannot run mixed strategy.")

            assist = ASSISTPropagator()
            t_compute0 = time.perf_counter()
            orbits_at_centers = assist.propagate_orbits(
                orbits_mean,
                center_times_utc,
                covariance=False,
                max_processes=max_processes,
            )  # (N*T)
            compute_sec += time.perf_counter() - t_compute0

            # Map center time -> index in center_times_utc via (days,nanos)
            center_days = center_times_utc.days.to_numpy(zero_copy_only=False)
            center_nanos = center_times_utc.nanos.to_numpy(zero_copy_only=False)
            center_key_to_idx = {(int(d), int(n)): i for i, (d, n) in enumerate(zip(center_days, center_nanos))}

            # Target mjds (UTC) for window membership tests.
            target_mjd = target_times_utc.mjd().to_numpy(zero_copy_only=False).astype(np.float64)
            target_code = target_codes.to_numpy(zero_copy_only=False)

            total_rows = 0
            part = 0
            n_orb = int(len(orbits))

            # Loop over windows (small) but do vectorized propagation within each.
            for w in windows:
                obscode = str(w.obscode[0].as_py())
                w0 = float(w.window_start().mjd()[0].as_py())
                w1 = float(w.window_end().mjd()[0].as_py())
                # select targets in this window + obscode
                mask = (target_code == obscode) & (target_mjd >= w0) & (target_mjd <= w1)
                idx = np.nonzero(mask)[0]
                if idx.size == 0:
                    continue

                # Get the center index for this window.
                key = (int(w.time.days[0].as_py()), int(w.time.nanos[0].as_py()))
                cidx = center_key_to_idx.get(key)
                if cidx is None:
                    continue

                # Slice orbits at this center: positions are cidx + arange(N)*T
                take_idx = (cidx + np.arange(n_orb, dtype=np.int64) * T).tolist()
                orbits_center = orbits_at_centers.take(take_idx)

                # Chunk within window targets to bound memory.
                for j0 in range(0, int(idx.size), int(time_chunk_size)):
                    j1 = min(j0 + int(time_chunk_size), int(idx.size))
                    sub = idx[j0:j1].tolist()
                    times_tdb = target_times_utc.take(sub).rescale("tdb")
                    # 2-body from center to each target time (cross product over times)
                    t_compute0 = time.perf_counter()
                    prop = propagate_2body(orbits_center, times_tdb)
                    obs_tdb = Observers.from_codes([obscode] * len(times_tdb), times_tdb)
                    obs_nm = _pair_observers_for_propagated_orbits(observers=obs_tdb, n_orbits=n_orb)
                    ephem = _ephem_to_utc(generate_ephemeris_2body(prop, obs_nm))
                    compute_sec += time.perf_counter() - t_compute0
                    total_rows += int(len(ephem))
                    if write_ephemeris:
                        t_io0 = time.perf_counter()
                        _write_quivr_part(out_ephem_dir, qt=ephem, part_idx=part)
                        io_sec += time.perf_counter() - t_io0
                    part += 1

            n_rows = total_rows
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
        _write_strategy_meta(
            strat_dir,
            dict(
                strategy=name,
                kind="mean_ephemeris",
                window_size_days=int(window_size_days),
                n_orbits=int(len(orbits)),
                n_time_targets=int(len(target_observers_utc)),
                n_window_centers=int(len(windows)),
                n_rows=n_rows,
                time_chunk_size=int(time_chunk_size),
                runtime_sec=float(compute_sec),
                io_sec=float(io_sec),
                runtime_total_sec=float(compute_sec + io_sec),
                error=err,
            ),
        )
        metrics_rows.append(
            dict(
                subset_dir=str(subset_dir),
                strategy=name,
                variant_kind=None,
                n_orbits=int(len(orbits)),
                n_orbits_covok=n_cov_ok,
                n_time_targets=int(len(target_observers_utc)),
                n_window_centers=int(len(windows)),
                n_ephem_rows_mean=n_rows,
                n_variant_orbits=None,
                n_ephem_rows_variants=None,
                runtime_sec=float(compute_sec),
                io_sec=float(io_sec),
                runtime_total_sec=float(compute_sec + io_sec),
                error=err,
            )
        )

    # Mixed strategy with covariance carried via persistent particles:
    #  - pre-sample (sigma points / MC) once at the input epoch,
    #  - propagate those same particles with ASSIST to each window center,
    #  - then propagate those particles with 2-body to all exposure times within the window.
    #
    # This is the intended "C" behavior when we want covariance-aware footprints while still
    # using windows to reduce n-body work.
    mixed_variant_specs: list[tuple[str, str, int | None]] = [("sigma_points", "sigma-point", None)]
    mixed_variant_specs.extend([(f"mc_{int(n)}", "monte-carlo", int(n)) for n in mc_samples])
    for variant_kind, method, num_samples in mixed_variant_specs:
        strat_name = f"assist_window_then_2body_variants:{variant_kind}"
        if not _enabled(strat_name):
            continue
        name = "assist_window_then_2body_variants"
        strat_dir = strategies_dir / name / variant_kind
        err = None
        n_rows: int | None = None
        n_var: int | None = None
        compute_sec = 0.0
        io_sec = 0.0
        try:
            if n_cov_ok == 0:
                raise ValueError("No orbits have fully-defined 6x6 covariance; cannot generate variants.")
            _ensure_dir(strat_dir)

            # Prepare center times (unique) and propagate variants to centers with ASSIST.
            center_times_utc = windows.time.unique().sort_by(["days", "nanos"])
            T = int(len(center_times_utc))
            if T == 0:
                raise ValueError("No window centers found; cannot run mixed strategy.")

            t_compute0 = time.perf_counter()
            variants = (
                VariantOrbits.create(orbits_covok, method=method)
                if num_samples is None
                else VariantOrbits.create(
                    orbits_covok, method=method, num_samples=int(num_samples), seed=0
                )
            )
            compute_sec += time.perf_counter() - t_compute0
            n_var = int(len(variants))

            if write_variants_orbits:
                t_io0 = time.perf_counter()
                variants.to_parquet(str(strat_dir / "variants_orbits.parquet"))
                io_sec += time.perf_counter() - t_io0

            assist = ASSISTPropagator()
            t_compute0 = time.perf_counter()
            # NOTE: Upstream limitation/bug in `adam_core.propagator.Propagator.propagate_orbits`
            # when `max_processes > 1` AND the input is a `VariantOrbits`.
            #
            # The Ray worker returns a `VariantOrbits` chunk, but the parallel dispatcher treats
            # all `VariantOrbits` results as "internal covariance variants", leaving the main
            # `propagated_list` empty, then calling `qv.concatenate(propagated_list)` which raises
            # `ValueError: No values to concatenate`.
            #
            # This is reproducible even with `covariance=False`. Until fixed upstream, force
            # single-process for this specific call.
            variants_at_centers = assist.propagate_orbits(
                variants,
                center_times_utc,
                covariance=False,
                max_processes=1,
            )  # (n_var * T)
            compute_sec += time.perf_counter() - t_compute0

            # Map center time -> index in center_times_utc via (days,nanos)
            center_days = center_times_utc.days.to_numpy(zero_copy_only=False)
            center_nanos = center_times_utc.nanos.to_numpy(zero_copy_only=False)
            center_key_to_idx = {(int(d), int(n)): i for i, (d, n) in enumerate(zip(center_days, center_nanos))}

            # Target mjds (UTC) for window membership tests.
            target_mjd = target_times_utc.mjd().to_numpy(zero_copy_only=False).astype(np.float64)
            target_code = target_codes.to_numpy(zero_copy_only=False)

            out_ephem_dir = strat_dir / "variants_ephemeris"
            _ensure_dir(out_ephem_dir)

            total_rows = 0
            part = 0

            # Loop over windows (small) but do vectorized propagation within each.
            for w in windows:
                obscode = str(w.obscode[0].as_py())
                w0 = float(w.window_start().mjd()[0].as_py())
                w1 = float(w.window_end().mjd()[0].as_py())
                mask = (target_code == obscode) & (target_mjd >= w0) & (target_mjd <= w1)
                idx = np.nonzero(mask)[0]
                if idx.size == 0:
                    continue

                key = (int(w.time.days[0].as_py()), int(w.time.nanos[0].as_py()))
                cidx = center_key_to_idx.get(key)
                if cidx is None:
                    continue

                # Slice variants at this center: positions are cidx + arange(n_var)*T
                take_idx = (cidx + np.arange(n_var, dtype=np.int64) * T).tolist()
                variants_center = variants_at_centers.take(take_idx)

                # Chunk within window targets to bound memory.
                for j0 in range(0, int(idx.size), int(time_chunk_size)):
                    j1 = min(j0 + int(time_chunk_size), int(idx.size))
                    sub = idx[j0:j1].tolist()
                    times_tdb = target_times_utc.take(sub).rescale("tdb")

                    t_compute0 = time.perf_counter()
                    prop = propagate_2body(variants_center, times_tdb)
                    obs_tdb = Observers.from_codes([obscode] * len(times_tdb), times_tdb)
                    obs_nm = _pair_observers_for_propagated_orbits(observers=obs_tdb, n_orbits=int(n_var))
                    ephem = _ephem_to_utc(generate_ephemeris_2body(prop, obs_nm))
                    compute_sec += time.perf_counter() - t_compute0

                    total_rows += int(len(ephem))
                    if write_ephemeris:
                        t_io0 = time.perf_counter()
                        _write_quivr_part(out_ephem_dir, qt=ephem, part_idx=part)
                        io_sec += time.perf_counter() - t_io0
                    part += 1

            n_rows = total_rows
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
        _write_strategy_meta(
            strat_dir,
            dict(
                strategy=name,
                variant_kind=variant_kind,
                variant_method=str(method),
                n_variant_orbits=n_var,
                kind="variants_ephemeris",
                window_size_days=int(window_size_days),
                n_orbits=int(len(orbits)),
                n_orbits_covok=n_cov_ok,
                n_time_targets=int(len(target_observers_utc)),
                n_window_centers=int(len(windows)),
                n_rows=n_rows,
                time_chunk_size=int(time_chunk_size),
                layout_note="Carry the same particles through ASSIST-to-centers then 2-body-to-targets; each ephemeris part is a cross product (variants_center × time_chunk).",
                assist_max_processes=None if max_processes is None else int(max_processes),
                assist_max_processes_variants_propagate_orbits=1,
                runtime_sec=float(compute_sec),
                io_sec=float(io_sec),
                runtime_total_sec=float(compute_sec + io_sec),
                error=err,
            ),
        )
        metrics_rows.append(
            dict(
                subset_dir=str(subset_dir),
                strategy=name,
                variant_kind=variant_kind,
                n_orbits=int(len(orbits)),
                n_orbits_covok=n_cov_ok,
                n_time_targets=int(len(target_observers_utc)),
                n_window_centers=int(len(windows)),
                n_ephem_rows_mean=None,
                n_variant_orbits=n_var,
                n_ephem_rows_variants=n_rows,
                runtime_sec=float(compute_sec),
                io_sec=float(io_sec),
                runtime_total_sec=float(compute_sec + io_sec),
                error=err,
            )
        )

    # ASSIST mean (no covariance) for all orbits.
    if _enabled("assist_mean"):
        assist_dir = strategies_dir / "assist_mean"
        # Benchmark time should cover only propagation + ephemeris generation (compute),
        # not frame/target fetching, window computation, or disk I/O.
        err = None
        n_rows: int | None = None
        compute_sec = 0.0
        io_sec = 0.0
        try:
            _ensure_dir(assist_dir)
            out_ephem_dir = assist_dir / "mean_ephemeris"
            _ensure_dir(out_ephem_dir)
            n_targets = int(len(target_observers_utc))
            total_rows = 0
            part = 0
            assist = ASSISTPropagator()
            for i0 in range(0, n_targets, int(time_chunk_size)):
                i1 = min(i0 + int(time_chunk_size), n_targets)
                obs_utc = target_observers_utc[i0:i1]
                t_compute0 = time.perf_counter()
                ephem = assist.generate_ephemeris(
                    orbits_mean,
                    obs_utc,
                    covariance=False,
                    max_processes=max_processes,
                )
                compute_sec += time.perf_counter() - t_compute0
                total_rows += int(len(ephem))
                if write_ephemeris:
                    t_io0 = time.perf_counter()
                    _write_quivr_part(out_ephem_dir, qt=ephem, part_idx=part)
                    io_sec += time.perf_counter() - t_io0
                part += 1
            n_rows = total_rows
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
        _write_strategy_meta(
            assist_dir,
            dict(
                strategy="assist_mean",
                kind="mean_ephemeris",
                window_size_days=int(window_size_days),
                n_orbits=int(len(orbits)),
                n_time_targets=int(len(target_observers_utc)),
                n_window_centers=int(len(windows)),
                n_rows=n_rows,
                time_chunk_size=int(time_chunk_size),
                runtime_sec=float(compute_sec),
                io_sec=float(io_sec),
                runtime_total_sec=float(compute_sec + io_sec),
                error=err,
            ),
        )
        metrics_rows.append(
            dict(
                subset_dir=str(subset_dir),
                strategy="assist_mean",
                variant_kind=None,
                n_orbits=int(len(orbits)),
                n_orbits_covok=n_cov_ok,
                n_time_targets=int(len(target_observers_utc)),
                n_window_centers=int(len(windows)),
                n_ephem_rows_mean=n_rows,
                n_variant_orbits=None,
                n_ephem_rows_variants=None,
                runtime_sec=float(compute_sec),
                io_sec=float(io_sec),
                runtime_total_sec=float(compute_sec + io_sec),
                error=err,
            )
        )

    # ASSIST variants (only for cov-ok orbits).
    variant_specs: list[tuple[str, str, int | None]] = [("sigma_points", "sigma-point", None)]
    variant_specs.extend([(f"mc_{int(n)}", "monte-carlo", int(n)) for n in mc_samples])

    for variant_kind, method, num_samples in variant_specs:
        strat_name = f"assist_variants:{variant_kind}"
        if not _enabled(strat_name):
            continue
        strat_dir = strategies_dir / "assist_variants" / variant_kind
        # Benchmark time should cover only variant creation + propagation + ephemeris generation (compute),
        # not frame/target fetching, window computation, or disk I/O.
        err = None
        n_rows: int | None = None
        n_var = None
        compute_sec = 0.0
        io_sec = 0.0
        try:
            if n_cov_ok == 0:
                raise ValueError("No orbits have fully-defined 6x6 covariance; cannot generate variants.")
            t_compute0 = time.perf_counter()
            variants = (
                VariantOrbits.create(orbits_covok, method=method)
                if num_samples is None
                else VariantOrbits.create(orbits_covok, method=method, num_samples=int(num_samples), seed=0)
            )
            compute_sec += time.perf_counter() - t_compute0
            n_var = int(len(variants))
            _ensure_dir(strat_dir)
            if write_variants_orbits:
                t_io0 = time.perf_counter()
                variants.to_parquet(str(strat_dir / "variants_orbits.parquet"))
                io_sec += time.perf_counter() - t_io0
            out_ephem_dir = strat_dir / "variants_ephemeris"
            _ensure_dir(out_ephem_dir)
            assist = ASSISTPropagator()

            # Chunk over target observers; this can still be huge for MC and is intentionally opt-in.
            n_targets = int(len(target_observers_utc))
            total_rows = 0
            part = 0
            for i0 in range(0, n_targets, int(time_chunk_size)):
                i1 = min(i0 + int(time_chunk_size), n_targets)
                obs_utc = target_observers_utc[i0:i1]
                t_compute0 = time.perf_counter()
                ephem = assist.generate_ephemeris(
                    variants,
                    obs_utc,
                    covariance=False,
                    max_processes=max_processes,
                )
                compute_sec += time.perf_counter() - t_compute0
                total_rows += int(len(ephem))
                if write_ephemeris:
                    t_io0 = time.perf_counter()
                    _write_quivr_part(out_ephem_dir, qt=ephem, part_idx=part)
                    io_sec += time.perf_counter() - t_io0
                part += 1
            n_rows = total_rows
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
        _write_strategy_meta(
            strat_dir,
            dict(
                strategy="assist_variants",
                variant_kind=variant_kind,
                variant_method=str(method),
                n_variant_orbits=n_var,
                n_time_targets=int(len(target_observers_utc)),
                n_window_centers=int(len(windows)),
                n_rows=n_rows,
                time_chunk_size=int(time_chunk_size),
                layout_note="Store VariantOrbits and Ephemeris separately; each ephemeris part is a cross product (variants × time_chunk).",
                runtime_sec=float(compute_sec),
                io_sec=float(io_sec),
                runtime_total_sec=float(compute_sec + io_sec),
                error=err,
            ),
        )
        metrics_rows.append(
            dict(
                subset_dir=str(subset_dir),
                strategy="assist_variants",
                variant_kind=variant_kind,
                n_orbits=int(len(orbits)),
                n_orbits_covok=n_cov_ok,
                n_time_targets=int(len(target_observers_utc)),
                n_window_centers=int(len(windows)),
                n_ephem_rows_mean=None,
                n_variant_orbits=n_var,
                n_ephem_rows_variants=n_rows,
                runtime_sec=float(compute_sec),
                io_sec=float(io_sec),
                runtime_total_sec=float(compute_sec + io_sec),
                error=err,
            )
        )

    pq.write_table(pa.Table.from_pylist(metrics_rows), str(run_dir / "metrics.parquet"))

    meta = {
        "subset_dir": str(subset_dir),
        "orbits_parquet_arg": str(orbits_parquet),
        "orbits_parquet_used": str(orbits_path),
        "n_orbits": int(len(orbits)),
        "n_orbits_covok": n_cov_ok,
        "window_size_days": int(window_size_days),
        "max_orbits": None if max_orbits is None else int(max_orbits),
        "max_window_centers": None if max_window_centers is None else int(max_window_centers),
        "strategies_requested": None if strategies is None else list(strategies),
        "mc_samples": [int(x) for x in mc_samples],
        "time_chunk_size": int(time_chunk_size),
        "max_processes": None if max_processes is None else int(max_processes),
        "only_truth_orbits": bool(only_truth_orbits),
        "n_orbits_truth_matched": None if truth_orbit_ids is None else int(len(truth_orbit_ids)),
        "mean_with_covariance": bool(mean_with_covariance),
        "write_ephemeris": bool(write_ephemeris),
        "write_variants_orbits": bool(write_variants_orbits),
        "n_time_targets": int(len(target_observers_utc)),
        "obscodes": [str(x) for x in pc.unique(target_codes).to_pylist()],
        "n_window_centers": int(len(windows)),
        "strategies_ran": [r["strategy"] if r["variant_kind"] is None else f"{r['strategy']}:{r['variant_kind']}" for r in metrics_rows],
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    _write_json(run_dir / "meta.json", meta)
    return run_dir


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Stage 2: atomic propagation benchmark runner (cached ephemerides).")
    p.add_argument("--subset-dir", type=str, required=True)
    p.add_argument("--orbits-parquet", type=str, required=True)
    p.add_argument("--window-size-days", type=int, default=7)
    p.add_argument("--max-orbits", type=int, default=None)
    p.add_argument(
        "--max-window-centers",
        type=int,
        default=None,
        help="Deprecated name: now caps number of (obscode,time) targets used for benchmarking.",
    )
    p.add_argument("--time-chunk-size", type=int, default=2048)
    p.add_argument(
        "--max-processes",
        type=int,
        default=8,
        help="Max processes for ASSISTPropagator calls (propagate_orbits/generate_ephemeris). Use 0 to disable.",
    )
    p.add_argument(
        "--only-truth-orbits",
        action="store_true",
        help="Filter input orbits to those with ≥1 matched truth detection in this subset window.",
    )
    p.add_argument(
        "--mean-with-covariance",
        action="store_true",
        help="Include covariance propagation costs in mean-only strategies (default: off).",
    )
    p.add_argument(
        "--no-write-ephemeris",
        action="store_true",
        help="Do not write ephemeris parquet parts (benchmark compute only).",
    )
    p.add_argument(
        "--no-write-variants-orbits",
        action="store_true",
        help="Do not write VariantOrbits parquet (benchmark compute only).",
    )
    p.add_argument(
        "--mc-samples",
        type=str,
        default=None,
        help="Comma-separated list of Monte Carlo sample counts (default: none; MC is opt-in). Example: '256,1024'.",
    )
    p.add_argument(
        "--strategies",
        type=str,
        default=None,
        help=(
            "Comma-separated list. Examples: '2body_only', 'assist_window_then_2body', 'assist_mean', "
            "'assist_variants:sigma_points', 'assist_variants:mc_256', 'assist_variants:mc_1024', "
            "'assist_window_then_2body_variants:sigma_points', 'assist_window_then_2body_variants:mc_256'. "
            "If omitted, runs all."
        ),
    )
    args = p.parse_args()

    run_dir = run_stage2_propagation_bench(
        subset_dir=Path(args.subset_dir),
        orbits_parquet=Path(args.orbits_parquet),
        window_size_days=int(args.window_size_days),
        max_orbits=args.max_orbits,
        max_window_centers=args.max_window_centers,
        strategies=_parse_strategies_arg(args.strategies),
        mc_samples=_parse_mc_samples_arg(args.mc_samples, default=[]),
        time_chunk_size=int(args.time_chunk_size),
        max_processes=(None if int(args.max_processes) <= 0 else int(args.max_processes)),
        only_truth_orbits=bool(args.only_truth_orbits),
        mean_with_covariance=bool(args.mean_with_covariance),
        write_ephemeris=not bool(args.no_write_ephemeris),
        write_variants_orbits=not bool(args.no_write_variants_orbits),
    )
    print(f"run_dir={run_dir}")
    print(f"meta_json={run_dir / 'meta.json'}")
    print(f"metrics_parquet={run_dir / 'metrics.parquet'}")


if __name__ == "__main__":
    main()

