from __future__ import annotations

import json
import logging
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

from precovery.frame_db import WindowCenters
from precovery.precovery_db import PrecoveryDatabase

from ..selection.subset_designations import read_subset_window
from ..selection.designation_normalization import normalize_designation


_LOG = logging.getLogger(__name__)
_PROGRESS_LOG_EVERY_SEC = 30.0


def _progress_maybe_log(
    *,
    label: str,
    i: int,
    n: int,
    t0: float,
    last_log_t: float,
    extra: str = "",
    every_sec: float = _PROGRESS_LOG_EVERY_SEC,
) -> float:
    """
    Periodic progress logging helper.

    Returns updated `last_log_t`.
    """
    if n <= 0:
        return last_log_t
    now = time.perf_counter()
    if (now - float(last_log_t)) < float(every_sec) and i < n:
        return last_log_t
    frac = float(i) / float(n)
    _LOG.debug(
        "%s progress %d/%d (%.1f%%) elapsed=%.1fs%s",
        str(label),
        int(i),
        int(n),
        100.0 * frac,
        float(now - float(t0)),
        ("" if not extra else f" {extra}"),
    )
    return float(now)


def _run_with_strategy_logs(
    *,
    label: str,
    fn: callable[[], dict[str, object]],
) -> dict[str, object]:
    """
    Log start/end of a strategy run.

    We keep this intentionally minimal so it's easy to track progress in long runs.
    """
    _LOG.info("stage2 START %s", str(label))
    t0 = time.perf_counter()
    out = fn()
    dt = time.perf_counter() - t0
    err = out.get("error")
    extra = "" if not err else f" error={err!s}"

    # Optional breakdowns.
    t_prop = out.get("t_2body_propagate_sec")
    t_rep = out.get("t_2body_observer_repeat_sec")
    t_eph = out.get("t_2body_generate_ephemeris_sec")
    t_utc = out.get("t_2body_rescale_utc_sec")
    if all(x is not None for x in (t_prop, t_rep, t_eph, t_utc)):
        try:
            extra += (
                f" 2body(propagate={float(t_prop):.1f}s"
                f" repeat_obs={float(t_rep):.1f}s"
                f" ephem={float(t_eph):.1f}s"
                f" utc={float(t_utc):.1f}s)"
            )
        except Exception:  # noqa: BLE001
            pass

    t_vcreate = out.get("t_variants_create_sec")
    t_assist_cent = out.get("t_assist_propagate_centers_sec")
    t_obs_codes = out.get("t_observers_from_codes_sec")
    if any(x is not None for x in (t_vcreate, t_assist_cent, t_obs_codes)):
        parts: list[str] = []
        try:
            if t_vcreate is not None:
                parts.append(f"variants_create={float(t_vcreate):.1f}s")
            if t_assist_cent is not None:
                parts.append(f"assist_centers={float(t_assist_cent):.1f}s")
            if t_obs_codes is not None:
                parts.append(f"observers_from_codes={float(t_obs_codes):.1f}s")
            if parts:
                extra += " " + " ".join(parts)
        except Exception:  # noqa: BLE001
            pass

    _LOG.info("stage2 END   %s elapsed=%.1fs%s", str(label), float(dt), extra)
    return out


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _center_time_index(
    *,
    center_times: Timestamp,
    time: Timestamp,
    precision: str = "ns",
) -> int | None:
    """
    Return the index of `time` within `center_times`, or None if absent.

    Uses `Timestamp.equals()` so we can tolerate small differences (e.g. nanosecond-level)
    if desired via `precision`.
    """
    if len(time) != 1:
        raise ValueError("time must be a length-1 Timestamp")
    m = center_times.equals(time, precision=str(precision))
    hit = pc.indices_nonzero(m).to_numpy(zero_copy_only=False)
    if hit.size == 0:
        return None
    return int(hit[0])

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


def _repair_orbits_covariance_psd_for_sampling(
    orbits: Orbits,
    *,
    abs_tol: float = 1e-15,
    rel_tol: float = 1e-10,
) -> tuple[Orbits, dict[str, object]]:
    """
    Ensure coordinate covariances are positive-semidefinite enough for sigma-point sampling.

    Why:
    - Sigma-point sampling uses a matrix square-root; if the covariance is not PSD, SciPy's
      sqrtm can yield complex results. Those can silently get cast back to real, producing
      invalid variant states that later yield NaNs in ephemeris generation.

    Policy:
    - Symmetrize covariances.
    - If the most-negative eigenvalue is small compared to the largest eigenvalue
      (|min_eig| <= max(abs_tol, rel_tol * max_eig)), clip negative eigenvalues to 0.
    - Otherwise, drop the orbit from the returned table.
    """
    if len(orbits) == 0:
        return orbits, {
            "n_cov_psd_repaired": 0,
            "n_cov_psd_dropped": 0,
            "first_cov_psd_dropped_orbit_id": None,
            "first_cov_psd_dropped_min_eig": None,
            "cov_psd_abs_tol": float(abs_tol),
            "cov_psd_rel_tol": float(rel_tol),
        }

    cov = orbits.coordinates.covariance.to_matrix()
    cov = 0.5 * (cov + np.swapaxes(cov, 1, 2))

    keep: list[int] = []
    cov_out: list[np.ndarray] = []
    n_repaired = 0
    first_drop_orbit: str | None = None
    first_drop_min_eig: float | None = None

    for i in range(int(cov.shape[0])):
        c = cov[i]
        try:
            w, v = np.linalg.eigh(c)
        except Exception:  # noqa: BLE001
            w = np.full(6, np.nan)
            v = None

        if not np.isfinite(w).all() or v is None:
            if first_drop_orbit is None:
                first_drop_orbit = str(orbits.orbit_id[i].as_py())
                first_drop_min_eig = None
            continue

        w_min = float(w.min())
        w_max = float(w.max())
        tol = float(max(float(abs_tol), float(rel_tol) * float(w_max)))

        if w_min < -tol:
            if first_drop_orbit is None:
                first_drop_orbit = str(orbits.orbit_id[i].as_py())
                first_drop_min_eig = float(w_min)
            continue

        if w_min < 0.0:
            w = np.where(w < 0.0, 0.0, w)
            n_repaired += 1

        c_psd = (v * w) @ v.T
        cov_out.append(c_psd.astype(np.float64, copy=False))
        keep.append(int(i))

    n_dropped = int(len(orbits) - len(keep))
    if n_dropped > 0:
        orbits = orbits.take(keep)
    if len(cov_out) != int(len(orbits)):
        raise RuntimeError(
            "Internal error while repairing covariances: "
            f"cov_out={len(cov_out)} rows != orbits={int(len(orbits))} rows"
        )

    if len(orbits) > 0:
        cov_fixed = np.stack(cov_out, axis=0)
        orbits = orbits.set_column(
            "coordinates.covariance",
            CoordinateCovariances.from_matrix(cov_fixed),
        )

    return orbits, {
        "n_cov_psd_repaired": int(n_repaired),
        "n_cov_psd_dropped": int(n_dropped),
        "first_cov_psd_dropped_orbit_id": first_drop_orbit,
        "first_cov_psd_dropped_min_eig": first_drop_min_eig,
        "cov_psd_abs_tol": float(abs_tol),
        "cov_psd_rel_tol": float(rel_tol),
    }


def _designation_from_object_id(object_id: str) -> str:
    return normalize_designation(str(object_id))


def _truth_orbit_ids_in_subset(subset_dir: Path, *, inputs_artifacts_dir: Path | None = None) -> set[str]:
    """
    Return orbit_ids (designation-normalized) that have matched truth detections in this subset window.
    """
    artifacts_dir = (
        Path(inputs_artifacts_dir).expanduser().resolve()
        if inputs_artifacts_dir is not None
        else (Path(subset_dir) / "artifacts")
    )
    truth_path = artifacts_dir / "truth_precovery_crossmatch.parquet"
    orbits_path = artifacts_dir / "orbits_selected_sbdb.parquet"
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


def _run_2body_ephemeris(
    *,
    orbits: Orbits,
    window_observers_tdb: Observers,
    max_processes: int | None = None,
    ephem_ray_chunk_size: int = 5000,
    timing: dict[str, float] | None = None,
) -> Ephemeris:
    """
    2-body ephemeris for all (orbit, window_center) pairs with a single vectorized
    propagation + ephemeris call (no Python loops over times).
    """
    times = window_observers_tdb.coordinates.time
    t0 = time.perf_counter()
    prop = (
        propagate_2body(orbits, times, max_processes=int(max_processes))
        if max_processes is not None
        else propagate_2body(orbits, times)
    )  # (N_orbits * N_times)
    t1 = time.perf_counter()
    obs_nm = _pair_observers_for_propagated_orbits(
        observers=window_observers_tdb, n_orbits=int(len(orbits))
    )
    t2 = time.perf_counter()
    ephem = (
        generate_ephemeris_2body(
            prop,
            obs_nm,
            max_processes=int(max_processes),
            chunk_size=int(ephem_ray_chunk_size),
        )
        if max_processes is not None
        else generate_ephemeris_2body(prop, obs_nm)
    )
    t3 = time.perf_counter()
    # Store ephemeris in UTC for downstream joins (Stage 3), even if propagation
    # is performed in TDB internally.
    ephem = _ephem_to_utc(ephem)
    t4 = time.perf_counter()

    if timing is not None:
        timing["t_2body_total_sec"] = timing.get("t_2body_total_sec", 0.0) + float(t4 - t0)
        timing["t_2body_propagate_sec"] = timing.get("t_2body_propagate_sec", 0.0) + float(t1 - t0)
        timing["t_2body_observer_repeat_sec"] = timing.get("t_2body_observer_repeat_sec", 0.0) + float(t2 - t1)
        timing["t_2body_generate_ephemeris_sec"] = timing.get("t_2body_generate_ephemeris_sec", 0.0) + float(t3 - t2)
        timing["t_2body_rescale_utc_sec"] = timing.get("t_2body_rescale_utc_sec", 0.0) + float(t4 - t3)

    return ephem


def _run_2body_propagation_only(
    *,
    orbits: Orbits,
    window_observers_tdb: Observers,
    max_processes: int | None = None,
    timing: dict[str, float] | None = None,
) -> tuple[Orbits, Observers]:
    """
    Propagate (2-body) to a set of observer times, but do NOT generate ephemeris yet.

    Returns:
    - propagated orbits (length = len(orbits) * len(window_observers_tdb))
    - paired observers repeated to match propagated row order

    This is used to batch many small propagation chunks into fewer, larger ephemeris calls.
    """
    times = window_observers_tdb.coordinates.time
    t0 = time.perf_counter()
    prop = (
        propagate_2body(orbits, times, max_processes=int(max_processes))
        if max_processes is not None
        else propagate_2body(orbits, times)
    )
    t1 = time.perf_counter()
    obs_nm = _pair_observers_for_propagated_orbits(
        observers=window_observers_tdb, n_orbits=int(len(orbits))
    )
    t2 = time.perf_counter()

    if timing is not None:
        timing["t_2body_propagate_sec"] = timing.get("t_2body_propagate_sec", 0.0) + float(t1 - t0)
        timing["t_2body_observer_repeat_sec"] = timing.get("t_2body_observer_repeat_sec", 0.0) + float(
            t2 - t1
        )
        # Keep compatibility with prior "total" semantics by accumulating the propagation-side work
        # into the same key that ephemeris-only batching uses.
        timing["t_2body_total_sec"] = timing.get("t_2body_total_sec", 0.0) + float(t2 - t0)

    return prop, obs_nm


def _run_2body_ephemeris_only(
    *,
    propagated_orbits: Orbits,
    paired_observers_tdb: Observers,
    max_processes: int | None = None,
    ephem_ray_chunk_size: int = 5000,
    timing: dict[str, float] | None = None,
) -> Ephemeris:
    """
    Generate ephemeris (2-body) for already-propagated orbit rows + paired observers.

    This is separated from propagation so we can batch many small propagation chunks into a
    single larger ephemeris call (reducing Ray scheduling/object-store overhead).
    """
    t0 = time.perf_counter()
    ephem = (
        generate_ephemeris_2body(
            propagated_orbits,
            paired_observers_tdb,
            max_processes=int(max_processes),
            chunk_size=int(ephem_ray_chunk_size),
        )
        if max_processes is not None
        else generate_ephemeris_2body(propagated_orbits, paired_observers_tdb)
    )
    t1 = time.perf_counter()
    ephem = _ephem_to_utc(ephem)
    t2 = time.perf_counter()

    if timing is not None:
        timing["t_2body_generate_ephemeris_sec"] = timing.get("t_2body_generate_ephemeris_sec", 0.0) + float(
            t1 - t0
        )
        timing["t_2body_rescale_utc_sec"] = timing.get("t_2body_rescale_utc_sec", 0.0) + float(t2 - t1)
        timing["t_2body_total_sec"] = timing.get("t_2body_total_sec", 0.0) + float(t2 - t0)

    return ephem


def _flush_2body_ephemeris_batches(
    *,
    prop_batches: list[Orbits],
    obs_batches: list[Observers],
    out_ephem_dir: Path,
    write_ephemeris: bool,
    part_idx: int,
    max_processes: int | None,
    ephem_ray_chunk_size: int,
    timing: dict[str, float] | None,
    bad_orbit_ids: set[str] | None = None,
    bad_orbit_errors: dict[str, str] | None = None,
) -> tuple[int, int, float, float]:
    """
    Flush accumulated propagation batches into one ephemeris call + optional parquet write.

    Returns:
    - next part_idx
    - n_rows written/generated
    - compute_sec for ephemeris generation (includes UTC rescale)
    - io_sec for parquet write
    """
    if not prop_batches:
        return int(part_idx), 0, 0.0, 0.0

    if len(prop_batches) == 1:
        prop_all = prop_batches[0]
        obs_all = obs_batches[0]
    else:
        prop_all = qv.concatenate(prop_batches)
        obs_all = qv.concatenate(obs_batches)

    def _subset_paired_by_orbit_ids(
        *,
        prop: Orbits,
        obs: Observers,
        orbit_ids: list[str],
    ) -> tuple[Orbits, Observers]:
        # Filter both tables by matching orbit_id rows.
        m = pc.is_in(
            prop.table.column("orbit_id").combine_chunks(),
            value_set=pa.array(orbit_ids, type=pa.large_string()),
        )
        idx = pc.indices_nonzero(m).to_numpy(zero_copy_only=False).astype(np.int64)
        return prop.take(idx.tolist()), obs.take(idx.tolist())

    def _try_ephem(
        *,
        prop: Orbits,
        obs: Observers,
    ) -> None:
        # Use serial mode during isolation to avoid Ray noise and reduce scheduling overhead.
        _run_2body_ephemeris_only(
            propagated_orbits=prop,
            paired_observers_tdb=obs,
            max_processes=1,
            ephem_ray_chunk_size=int(ephem_ray_chunk_size),
            timing=None,
        )

    def _isolate_bad_orbit_ids(
        *,
        prop: Orbits,
        obs: Observers,
    ) -> tuple[list[str], str]:
        """
        Return (bad_orbit_ids, error_string) for a failing ephemeris batch.

        Uses a recursive bisection over unique orbit_id values. Each test runs a serial
        2-body ephemeris call on the subset.
        """
        # Unique orbit IDs in this batch.
        orbit_ids = sorted({str(x) for x in prop.table.column("orbit_id").to_pylist() if str(x).strip()})
        if not orbit_ids:
            return [], "unknown: empty orbit_id set"

        # Cache to avoid repeating identical subset tests.
        cache: dict[tuple[str, ...], bool] = {}

        def fails(ids: list[str]) -> bool:
            key = tuple(ids)
            hit = cache.get(key)
            if hit is not None:
                return hit
            p_sub, o_sub = _subset_paired_by_orbit_ids(prop=prop, obs=obs, orbit_ids=list(ids))
            try:
                _try_ephem(prop=p_sub, obs=o_sub)
                cache[key] = False
            except Exception:
                cache[key] = True
            return cache[key]

        # If the whole batch doesn't fail in serial, treat as non-isolatable (likely Ray-only issue).
        if not fails(orbit_ids):
            return [], "non-deterministic failure (did not reproduce in serial isolation)"

        bad: list[str] = []

        def bisect(ids: list[str]) -> None:
            if not ids:
                return
            if not fails(ids):
                return
            if len(ids) == 1:
                bad.append(ids[0])
                return
            mid = len(ids) // 2
            bisect(ids[:mid])
            bisect(ids[mid:])

        bisect(orbit_ids)
        return sorted(set(bad)), "ephemeris generation failed for at least one orbit_id"

    # Attempt ephemeris generation; if it fails, isolate orbit_ids, drop them, and retry.
    compute_sec = 0.0
    n_rows = 0
    ephem: Ephemeris | None = None
    last_err: str | None = None
    max_isolation_rounds = 10
    for _round in range(int(max_isolation_rounds)):
        try:
            t_compute0 = time.perf_counter()
            ephem = _run_2body_ephemeris_only(
                propagated_orbits=prop_all,
                paired_observers_tdb=obs_all,
                max_processes=max_processes,
                ephem_ray_chunk_size=int(ephem_ray_chunk_size),
                timing=timing,
            )
            compute_sec += float(time.perf_counter() - t_compute0)
            n_rows = int(len(ephem))
            last_err = None
            break
        except Exception as e:  # noqa: BLE001
            last_err = f"{type(e).__name__}: {e}"
            bad_ids, _why = _isolate_bad_orbit_ids(prop=prop_all, obs=obs_all)
            if not bad_ids:
                break
            if bad_orbit_ids is not None:
                bad_orbit_ids.update(bad_ids)
            if bad_orbit_errors is not None:
                for oid in bad_ids:
                    bad_orbit_errors.setdefault(oid, last_err)
            # Drop bad orbit_ids from this batch and retry.
            keep_mask = pc.invert(
                pc.is_in(
                    prop_all.table.column("orbit_id").combine_chunks(),
                    value_set=pa.array(bad_ids, type=pa.large_string()),
                )
            )
            keep_idx = pc.indices_nonzero(keep_mask).to_numpy(zero_copy_only=False).astype(np.int64)
            prop_all = prop_all.take(keep_idx.tolist())
            obs_all = obs_all.take(keep_idx.tolist())
            if int(len(prop_all)) == 0:
                break

    if ephem is None:
        raise RuntimeError(f"2-body ephemeris generation failed: {last_err}")

    io_sec = 0.0
    if bool(write_ephemeris):
        t_io0 = time.perf_counter()
        _write_quivr_part(out_ephem_dir, qt=ephem, part_idx=int(part_idx))
        io_sec = float(time.perf_counter() - t_io0)

    # Clear batches eagerly to free memory.
    prop_batches.clear()
    obs_batches.clear()

    return int(part_idx) + 1, int(n_rows), float(compute_sec), float(io_sec)


def _run_assist_window_then_2body(
    *,
    orbits: Orbits,
    window_observers_tdb: Observers,
    max_processes: int | None = None,
) -> Ephemeris:
    """
    ASSIST propagate to reference epoch (no covariance), then 2-body to each
    window-center time (paired, vectorized over orbits and times).
    """
    mean_mjd_tdb = float(np.mean(window_observers_tdb.coordinates.time.mjd().to_numpy(zero_copy_only=False)))
    t_ref = Timestamp.from_mjd([mean_mjd_tdb], scale="tdb")
    prop_assist = ASSISTPropagator()
    orb_ref = prop_assist.propagate_orbits(orbits, t_ref, covariance=False)
    return _run_2body_ephemeris(
        orbits=orb_ref,
        window_observers_tdb=window_observers_tdb,
        max_processes=max_processes,
    )


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
    # PERFORMANCE NOTE
    #
    # This query is on the hot path for non-truth Stage 2 runs: it determines the
    # number of propagation targets. Using SQLAlchemy reflection + streaming can
    # add substantial overhead on large subsets (tens of millions of frame rows).
    #
    # The fast path is to execute a single SQLite GROUP BY query directly against
    # `index.db`, backed by an index on (exposure_mjd_mid, obscode).
    #
    # On the full_precovery_n32 subset (≈92M frames), this reduces the full 4-year
    # distinct-target extraction to O(seconds) rather than O(minutes).
    return _fetch_unique_exposure_midpoints_sqlite(
        index_db=Path(db.frames.idx.db_uri.replace("sqlite:///", "").split("?")[0]),
        start_mjd=float(start_mjd),
        end_mjd=float(end_mjd),
    )


def _fetch_unique_exposure_midpoints_sqlite(
    *,
    index_db: Path,
    start_mjd: float,
    end_mjd: float,
    chunk_size: int = 100000,
) -> tuple[pa.Array, Timestamp]:
    """
    Fast distinct-target enumeration using raw sqlite3.

    Returns all distinct (obscode, exposure_mjd_mid) pairs in [start_mjd, end_mjd)
    ordered by (exposure_mjd_mid, obscode).
    """
    import sqlite3
    from array import array

    if not index_db.exists():
        raise FileNotFoundError(f"Missing subset index.db: {index_db}")

    # Use sqlite3 for minimal overhead and predictable query planning.
    conn = sqlite3.connect(str(index_db))
    try:
        # Additive, safe, and critical for performance.
        conn.execute(
            "CREATE INDEX IF NOT EXISTS frames_exposure_mjd_mid_obscode_idx "
            "ON frames(exposure_mjd_mid, obscode)"
        )

        # The GROUP BY form is typically faster than DISTINCT for SQLite here,
        # and avoids SQLAlchemy reflection costs.
        q = """
        SELECT obscode, exposure_mjd_mid
        FROM frames
        WHERE exposure_mjd_mid >= ?
          AND exposure_mjd_mid < ?
        GROUP BY obscode, exposure_mjd_mid
        ORDER BY exposure_mjd_mid ASC, obscode ASC
        """

        # Avoid creating millions of duplicate string objects for the same few
        # obscodes by interning via a tiny cache.
        code_cache: dict[str, str] = {}
        codes: list[str] = []
        mjds = array("d")

        cur = conn.execute(q, (float(start_mjd), float(end_mjd)))
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
        return pa.array([], type=pa.large_string()), Timestamp.from_mjd([], scale="utc")
    return (
        pa.array(codes, type=pa.large_string()),
        Timestamp.from_mjd(pa.array(mjds, type=pa.float64()), scale="utc"),
    )


def _fetch_truth_matched_exposure_midpoints(
    *,
    subset_dir: Path,
    index_db: Path,
    inputs_artifacts_dir: Path | None = None,
    restrict_designations: set[str] | None = None,
) -> tuple[pa.Array, Timestamp]:
    """
    Return distinct (obscode, exposure_mjd_mid) pairs for *truth-matched* detections.

    Stage 3 and Stage 4 use `exposure_mjd_mid` as the join key into the subset index:
    - Stage 3 materializes (target_idx, healpixel) by joining frames.exposure_mjd_mid
      to Stage 2's input targets by *exact equality*.
    - Stage 4 queries frames by (obscode, exposure_mjd_mid) when loading observations.

    So in truth-only workflows we must target exposure midpoints, not detection times.
    We derive these midpoints by joining the truth crossmatch's matched exposure IDs
    back to `index.db`.
    """
    artifacts_dir = (
        Path(inputs_artifacts_dir).expanduser().resolve()
        if inputs_artifacts_dir is not None
        else (Path(subset_dir) / "artifacts")
    )
    truth_path = artifacts_dir / "truth_precovery_crossmatch.parquet"
    if not truth_path.exists():
        raise FileNotFoundError(f"Missing truth crossmatch parquet: {truth_path}")
    if not Path(index_db).exists():
        raise FileNotFoundError(f"Missing subset index.db: {index_db}")

    truth = pq.read_table(
        str(truth_path),
        columns=[
            "matched",
            "designation",
            "obscode",
            "match_dataset_id",
            "match_exposure_id",
        ],
    )
    truth = truth.filter(pc.equal(truth["matched"], True))
    if restrict_designations is not None and truth.num_rows > 0:
        wanted = sorted({str(x).strip() for x in restrict_designations if str(x).strip()})
        if wanted:
            truth = truth.filter(
                pc.is_in(
                    truth["designation"],
                    value_set=pa.array(wanted, type=pa.large_string()),
                )
            )
    if truth.num_rows == 0:
        return pa.array([], type=pa.large_string()), Timestamp.from_mjd([], scale="utc")

    # Keep only rows with exposure identifiers.
    m_has = pc.and_(
        pc.invert(pc.is_null(truth["match_exposure_id"])),
        pc.invert(pc.is_null(truth["match_dataset_id"])),
    )
    truth = truth.filter(m_has)
    if truth.num_rows == 0:
        return pa.array([], type=pa.large_string()), Timestamp.from_mjd([], scale="utc")

    # Build a compact set of unique keys to look up in index.db.
    # Note: keys are small enough that an in-DB temp table join is fastest and simplest.
    ds = [str(x) for x in truth["match_dataset_id"].to_pylist()]
    code = [str(x) for x in truth["obscode"].to_pylist()]
    exp = [str(x) for x in truth["match_exposure_id"].to_pylist()]

    keys = sorted(set(zip(ds, code, exp)))
    if not keys:
        return pa.array([], type=pa.large_string()), Timestamp.from_mjd([], scale="utc")

    import sqlite3

    conn = sqlite3.connect(str(index_db))
    try:
        # Performance: Stage 2 truth-target mode joins `frames` on (dataset_id, obscode, exposure_id).
        # The full index ships with indices optimized for (exposure_mjd_mid, obscode, healpixel)
        # workloads, but not this exact join key. Create a persistent index once to avoid
        # repeated full-table scans on large subsets.
        conn.execute(
            "CREATE INDEX IF NOT EXISTS frames_truth_join_idx "
            "ON frames(dataset_id, obscode, exposure_id)"
        )
        conn.execute("CREATE TEMP TABLE truth_keys (dataset_id TEXT, obscode TEXT, exposure_id TEXT)")
        conn.executemany(
            "INSERT INTO truth_keys (dataset_id, obscode, exposure_id) VALUES (?, ?, ?)",
            keys,
        )
        conn.execute("CREATE INDEX truth_keys_idx ON truth_keys (dataset_id, obscode, exposure_id)")
        rows = conn.execute(
            """
            SELECT f.obscode, f.exposure_mjd_mid
            FROM frames f
            INNER JOIN truth_keys t
              ON f.dataset_id = t.dataset_id
             AND f.obscode = t.obscode
             AND f.exposure_id = t.exposure_id
            """
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        return pa.array([], type=pa.large_string()), Timestamp.from_mjd([], scale="utc")

    obsc = [str(r[0]) for r in rows]
    mjd = [float(r[1]) for r in rows]

    # De-dupe and stable sort by (mjd, obscode) so targets are deterministic.
    uniq = sorted(set(zip(mjd, obsc)))
    mjd_u = [x[0] for x in uniq]
    obsc_u = [x[1] for x in uniq]
    return pa.array(obsc_u, type=pa.large_string()), Timestamp.from_mjd(mjd_u, scale="utc")


def _fetch_truth_orbit_target_idx_map(
    *,
    subset_dir: Path,
    index_db: Path,
    inputs_artifacts_dir: Path | None = None,
    restrict_designations: set[str] | None = None,
) -> tuple[pa.Array, Timestamp, dict[str, list[int]]]:
    """
    Truth-only helper that returns:
    - the global Stage2 target list (obscode, exposure_mjd_mid) as (codes, times_utc), and
    - a mapping orbit_id -> sorted list of target_idx values where that orbit has ≥1 matched truth.

    This is the efficient mode for truth recovery runs: Stage2 ephemerides are generated
    only for the orbit-target pairs that actually appear in truth, rather than the full
    Cartesian product (orbits × all truth targets).
    """
    artifacts_dir = (
        Path(inputs_artifacts_dir).expanduser().resolve()
        if inputs_artifacts_dir is not None
        else (Path(subset_dir) / "artifacts")
    )
    truth_path = artifacts_dir / "truth_precovery_crossmatch.parquet"
    if not truth_path.exists():
        raise FileNotFoundError(f"Missing truth crossmatch parquet: {truth_path}")
    if not Path(index_db).exists():
        raise FileNotFoundError(f"Missing subset index.db: {index_db}")

    truth = pq.read_table(
        str(truth_path),
        columns=[
            "matched",
            "designation",
            "obscode",
            "match_dataset_id",
            "match_exposure_id",
        ],
    )
    truth = truth.filter(pc.equal(truth["matched"], True))
    if restrict_designations is not None and truth.num_rows > 0:
        wanted = sorted({str(x).strip() for x in restrict_designations if str(x).strip()})
        if wanted:
            truth = truth.filter(
                pc.is_in(
                    truth["designation"],
                    value_set=pa.array(wanted, type=pa.large_string()),
                )
            )
    if truth.num_rows == 0:
        return pa.array([], type=pa.large_string()), Timestamp.from_mjd([], scale="utc"), {}

    m_has = pc.and_(
        pc.invert(pc.is_null(truth["match_exposure_id"])),
        pc.invert(pc.is_null(truth["match_dataset_id"])),
    )
    truth = truth.filter(m_has)
    if truth.num_rows == 0:
        return pa.array([], type=pa.large_string()), Timestamp.from_mjd([], scale="utc"), {}

    # Build unique exposure keys annotated with designation.
    ds = [str(x) for x in truth["match_dataset_id"].to_pylist()]
    code = [str(x) for x in truth["obscode"].to_pylist()]
    exp = [str(x) for x in truth["match_exposure_id"].to_pylist()]
    des = [str(x) for x in truth["designation"].to_pylist()]
    keys = sorted(set(zip(ds, code, exp, des)))
    if not keys:
        return pa.array([], type=pa.large_string()), Timestamp.from_mjd([], scale="utc"), {}

    import sqlite3

    conn = sqlite3.connect(str(index_db))
    try:
        conn.execute(
            "CREATE INDEX IF NOT EXISTS frames_truth_join_idx "
            "ON frames(dataset_id, obscode, exposure_id)"
        )
        conn.execute(
            "CREATE TEMP TABLE truth_keys (dataset_id TEXT, obscode TEXT, exposure_id TEXT, designation TEXT)"
        )
        conn.executemany(
            "INSERT INTO truth_keys (dataset_id, obscode, exposure_id, designation) VALUES (?, ?, ?, ?)",
            keys,
        )
        conn.execute("CREATE INDEX truth_keys_idx ON truth_keys (dataset_id, obscode, exposure_id)")
        rows = conn.execute(
            """
            SELECT t.designation, f.obscode, f.exposure_mjd_mid
            FROM frames f
            INNER JOIN truth_keys t
              ON f.dataset_id = t.dataset_id
             AND f.obscode = t.obscode
             AND f.exposure_id = t.exposure_id
            """
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        return pa.array([], type=pa.large_string()), Timestamp.from_mjd([], scale="utc"), {}

    # Global targets: stable sort by (mjd, obscode) for deterministic target_idx.
    pairs = sorted(set((float(mjd), str(obs)) for _, obs, mjd in rows))
    mjd_u = [p[0] for p in pairs]
    obsc_u = [p[1] for p in pairs]
    codes = pa.array(obsc_u, type=pa.large_string())
    times = Timestamp.from_mjd(mjd_u, scale="utc")

    # Build lookup from (obscode, exposure_mjd_mid) -> target_idx.
    idx_by_key: dict[tuple[str, float], int] = {}
    for i, (mjd, obs) in enumerate(pairs):
        idx_by_key[(str(obs), float(mjd))] = int(i)

    # Map designation -> target indices (de-dupe + sort).
    tmp: dict[str, set[int]] = {}
    for d, obs, mjd in rows:
        k = (str(obs), float(mjd))
        tidx = idx_by_key.get(k)
        if tidx is None:
            continue
        tmp.setdefault(str(d), set()).add(int(tidx))
    out = {k: sorted(v) for k, v in tmp.items()}

    return codes, times, out


def _write_quivr_part(out_dir: Path, *, qt: qv.Table, part_idx: int) -> None:
    _ensure_dir(out_dir)
    qt.to_parquet(str(out_dir / f"part-{part_idx:06d}.parquet"))


def _write_strategy_meta(strategy_dir: Path, meta: dict[str, object]) -> None:
    _ensure_dir(strategy_dir)
    _write_json(strategy_dir / "meta.json", meta)


def _run_chunked_2body_mean_ephemeris(
    *,
    orbits: Orbits,
    target_observers_tdb: Observers,
    time_chunk_size: int,
    max_processes: int | None,
    out_ephem_dir: Path,
    write_ephemeris: bool,
) -> tuple[int, float, float]:
    """
    Vectorized 2-body propagation + ephemeris generation, chunked only over time to
    bound memory.
    """
    compute_sec = 0.0
    io_sec = 0.0
    timing: dict[str, float] = {}
    # Batch ephemeris generation across many small propagation chunks to avoid
    # Ray scheduling / object-store overhead when windows are tiny.
    ephem_batch_max_rows = 1_000_000
    ephem_ray_chunk_size = 5000
    n_targets = int(len(target_observers_tdb))
    n_chunks = int((n_targets + int(time_chunk_size) - 1) // int(time_chunk_size))
    total_rows = 0
    part = 0
    pending_rows = 0
    prop_batches: list[Orbits] = []
    obs_batches: list[Observers] = []
    t0 = time.perf_counter()
    last_log_t = t0
    chunk_i = 0
    for i0 in range(0, n_targets, int(time_chunk_size)):
        i1 = min(i0 + int(time_chunk_size), n_targets)
        obs_tdb = target_observers_tdb[i0:i1]
        t_compute0 = time.perf_counter()
        prop, obs_nm = _run_2body_propagation_only(
            orbits=orbits,
            window_observers_tdb=obs_tdb,
            max_processes=max_processes,
            timing=timing,
        )
        compute_sec += time.perf_counter() - t_compute0
        prop_batches.append(prop)
        obs_batches.append(obs_nm)
        pending_rows += int(len(prop))

        if pending_rows >= int(ephem_batch_max_rows):
            part, n_i, compute_i, io_i = _flush_2body_ephemeris_batches(
                prop_batches=prop_batches,
                obs_batches=obs_batches,
                out_ephem_dir=out_ephem_dir,
                write_ephemeris=bool(write_ephemeris),
                part_idx=int(part),
                max_processes=max_processes,
                ephem_ray_chunk_size=int(ephem_ray_chunk_size),
                timing=timing,
            )
            compute_sec += float(compute_i)
            io_sec += float(io_i)
            total_rows += int(n_i)
            pending_rows = 0

        chunk_i += 1
        last_log_t = _progress_maybe_log(
            label="stage2 2body mean_ephem",
            i=int(chunk_i),
            n=int(n_chunks),
            t0=float(t0),
            last_log_t=float(last_log_t),
            extra=(
                f"rows={int(total_rows)} pending_rows={int(pending_rows)} "
                f"compute_sec={compute_sec:.1f} io_sec={io_sec:.1f}"
            ),
        )
    # Final flush.
    part, n_i, compute_i, io_i = _flush_2body_ephemeris_batches(
        prop_batches=prop_batches,
        obs_batches=obs_batches,
        out_ephem_dir=out_ephem_dir,
        write_ephemeris=bool(write_ephemeris),
        part_idx=int(part),
        max_processes=max_processes,
        ephem_ray_chunk_size=int(ephem_ray_chunk_size),
        timing=timing,
    )
    compute_sec += float(compute_i)
    io_sec += float(io_i)
    total_rows += int(n_i)

    # Expose the breakdown to the caller by attaching to function attributes (simple and local).
    _run_chunked_2body_mean_ephemeris._timing = timing  # type: ignore[attr-defined]
    return int(total_rows), float(compute_sec), float(io_sec)


def _run_strategy_2body(
    *,
    subset_dir: Path,
    strategies_dir: Path,
    orbits_all: Orbits,
    orbits_mean: Orbits,
    target_observers_utc: Observers,
    target_observers_tdb: Observers,
    windows: qv.Table,
    window_size_days: int,
    time_chunk_size: int,
    max_processes: int | None,
    write_ephemeris: bool,
    include_covariance: bool,
    n_cov_ok: int,
) -> dict[str, object]:
    name = "2body_with_covariance" if bool(include_covariance) else "2body_only"
    strat_dir = strategies_dir / name
    err: str | None = None
    n_rows: int | None = None
    compute_sec = 0.0
    io_sec = 0.0
    try:
        _ensure_dir(strat_dir)
        out_ephem_dir = strat_dir / "mean_ephemeris"
        _ensure_dir(out_ephem_dir)
        orbits_for_mean = orbits_all if bool(include_covariance) else orbits_mean
        n_rows, compute_sec, io_sec = _run_chunked_2body_mean_ephemeris(
            orbits=orbits_for_mean,
            target_observers_tdb=target_observers_tdb,
            time_chunk_size=int(time_chunk_size),
            max_processes=max_processes,
            out_ephem_dir=out_ephem_dir,
            write_ephemeris=bool(write_ephemeris),
        )
        timing = getattr(_run_chunked_2body_mean_ephemeris, "_timing", {})  # type: ignore[attr-defined]
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {e}"
        timing = {}

    _ensure_dir(strat_dir)
    _write_json(
        strat_dir / "meta.json",
        dict(
            strategy=name,
            kind="mean_ephemeris",
            window_size_days=int(window_size_days),
            n_orbits=int(len(orbits_all)),
            n_time_targets=int(len(target_observers_utc)),
            n_window_centers=int(len(windows)),
            n_rows=n_rows,
            time_chunk_size=int(time_chunk_size),
            mean_with_covariance=bool(include_covariance),
            two_body_max_processes=None if max_processes is None else int(max_processes),
            runtime_sec=float(compute_sec),
            io_sec=float(io_sec),
            runtime_total_sec=float(compute_sec + io_sec),
            **{k: float(v) for k, v in dict(timing).items()},
            error=err,
        ),
    )
    return dict(
        subset_dir=str(subset_dir),
        strategy=name,
        variant_kind=None,
        n_orbits=int(len(orbits_all)),
        n_orbits_covok=int(n_cov_ok),
        n_time_targets=int(len(target_observers_utc)),
        n_window_centers=int(len(windows)),
        n_ephem_rows_mean=n_rows,
        n_variant_orbits=None,
        n_ephem_rows_variants=None,
        runtime_sec=float(compute_sec),
        io_sec=float(io_sec),
        runtime_total_sec=float(compute_sec + io_sec),
        **{k: float(v) for k, v in dict(timing).items()},
        error=err,
    )


def _run_chunked_assist_ephemeris(
    *,
    assist: ASSISTPropagator,
    orbits: Orbits | VariantOrbits,
    target_observers_utc: Observers,
    time_chunk_size: int,
    max_processes: int | None,
    covariance: bool,
    out_ephem_dir: Path,
    write_ephemeris: bool,
) -> tuple[int, float, float]:
    """
    Chunked ASSIST ephemeris generation over (obscode,time) targets.
    """
    compute_sec = 0.0
    io_sec = 0.0
    n_targets = int(len(target_observers_utc))
    n_chunks = int((n_targets + int(time_chunk_size) - 1) // int(time_chunk_size))
    total_rows = 0
    part = 0
    t0 = time.perf_counter()
    last_log_t = t0
    for i0 in range(0, n_targets, int(time_chunk_size)):
        i1 = min(i0 + int(time_chunk_size), n_targets)
        obs_utc = target_observers_utc[i0:i1]
        t_compute0 = time.perf_counter()
        ephem = assist.generate_ephemeris(
            orbits,
            obs_utc,
            covariance=bool(covariance),
            max_processes=max_processes,
        )
        compute_sec += time.perf_counter() - t_compute0
        total_rows += int(len(ephem))
        if bool(write_ephemeris):
            t_io0 = time.perf_counter()
            _write_quivr_part(out_ephem_dir, qt=ephem, part_idx=part)
            io_sec += time.perf_counter() - t_io0
        part += 1
        last_log_t = _progress_maybe_log(
            label="stage2 assist ephem",
            i=int(part),
            n=int(n_chunks),
            t0=float(t0),
            last_log_t=float(last_log_t),
            extra=f"rows={int(total_rows)} compute_sec={compute_sec:.1f} io_sec={io_sec:.1f}",
        )
    return int(total_rows), float(compute_sec), float(io_sec)


def _run_strategy_assist_window_then_2body(
    *,
    subset_dir: Path,
    strategies_dir: Path,
    orbits_all: Orbits,
    orbits_mean: Orbits,
    target_codes: pa.Array,
    target_times_utc: Timestamp,
    target_observers_utc: Observers,
    target_observers_tdb: Observers,
    windows: qv.Table,
    window_size_days: int,
    time_chunk_size: int,
    max_processes: int | None,
    write_ephemeris: bool,
    n_cov_ok: int,
) -> dict[str, object]:
    name = "assist_window_then_2body"
    strat_dir = strategies_dir / name
    err: str | None = None
    n_rows: int | None = None
    compute_sec = 0.0
    io_sec = 0.0
    timing: dict[str, float] = {}
    ephem_batch_max_rows = 1_000_000
    ephem_ray_chunk_size = 5000
    try:
        _ensure_dir(strat_dir)
        out_ephem_dir = strat_dir / "mean_ephemeris"
        _ensure_dir(out_ephem_dir)

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

        target_mjd = target_times_utc.mjd().to_numpy(zero_copy_only=False).astype(np.float64)
        target_code = target_codes.to_numpy(zero_copy_only=False)

        total_rows = 0
        part = 0
        n_orb = int(len(orbits_all))
        pending_rows = 0
        prop_batches: list[Orbits] = []
        obs_batches: list[Observers] = []

        for w in windows:
            obscode = str(w.obscode[0].as_py())
            w0 = float(w.window_start().mjd()[0].as_py())
            w1 = float(w.window_end().mjd()[0].as_py())
            mask = (target_code == obscode) & (target_mjd >= w0) & (target_mjd <= w1)
            idx = np.nonzero(mask)[0].tolist()
            if not idx:
                continue

            cidx = _center_time_index(
                center_times=center_times_utc,
                time=w.time,
                precision="ns",
            )
            if cidx is None:
                continue

            # `ASSISTPropagator.propagate_orbits(orbits, center_times)` returns results in
            # time-major order:
            #   [all_orbits@t0, all_orbits@t1, ...]
            # so the slice for one center time is a contiguous block.
            take_idx = (int(cidx) * int(n_orb) + np.arange(n_orb, dtype=np.int64)).tolist()
            orbits_center = orbits_at_centers.take(take_idx)

            for j0 in range(0, int(len(idx)), int(time_chunk_size)):
                j1 = min(j0 + int(time_chunk_size), int(len(idx)))
                sub = idx[j0:j1]
                t_compute0 = time.perf_counter()
                obs_tdb = target_observers_tdb.take(sub)
                prop, obs_nm = _run_2body_propagation_only(
                    orbits=orbits_center,
                    window_observers_tdb=obs_tdb,
                    max_processes=max_processes,
                    timing=timing,
                )
                compute_sec += time.perf_counter() - t_compute0
                prop_batches.append(prop)
                obs_batches.append(obs_nm)
                pending_rows += int(len(prop))

                if pending_rows >= int(ephem_batch_max_rows):
                    part, n_i, compute_i, io_i = _flush_2body_ephemeris_batches(
                        prop_batches=prop_batches,
                        obs_batches=obs_batches,
                        out_ephem_dir=out_ephem_dir,
                        write_ephemeris=bool(write_ephemeris),
                        part_idx=int(part),
                        max_processes=max_processes,
                        ephem_ray_chunk_size=int(ephem_ray_chunk_size),
                        timing=timing,
                    )
                    compute_sec += float(compute_i)
                    io_sec += float(io_i)
                    total_rows += int(n_i)
                    pending_rows = 0

        # Final flush.
        part, n_i, compute_i, io_i = _flush_2body_ephemeris_batches(
            prop_batches=prop_batches,
            obs_batches=obs_batches,
            out_ephem_dir=out_ephem_dir,
            write_ephemeris=bool(write_ephemeris),
            part_idx=int(part),
            max_processes=max_processes,
            ephem_ray_chunk_size=int(ephem_ray_chunk_size),
            timing=timing,
        )
        compute_sec += float(compute_i)
        io_sec += float(io_i)
        total_rows += int(n_i)

        n_rows = int(total_rows)
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {e}"

    _write_strategy_meta(
        strat_dir,
        dict(
            strategy=name,
            kind="mean_ephemeris",
            window_size_days=int(window_size_days),
            n_orbits=int(len(orbits_all)),
            n_time_targets=int(len(target_observers_utc)),
            n_window_centers=int(len(windows)),
            n_rows=n_rows,
            time_chunk_size=int(time_chunk_size),
            two_body_max_processes=None if max_processes is None else int(max_processes),
            ephem_batch_max_rows=int(ephem_batch_max_rows),
            ephem_ray_chunk_size=int(ephem_ray_chunk_size),
            runtime_sec=float(compute_sec),
            io_sec=float(io_sec),
            runtime_total_sec=float(compute_sec + io_sec),
            **{k: float(v) for k, v in dict(timing).items()},
            error=err,
        ),
    )
    return dict(
        subset_dir=str(subset_dir),
        strategy=name,
        variant_kind=None,
        n_orbits=int(len(orbits_all)),
        n_orbits_covok=int(n_cov_ok),
        n_time_targets=int(len(target_observers_utc)),
        n_window_centers=int(len(windows)),
        n_ephem_rows_mean=n_rows,
        n_variant_orbits=None,
        n_ephem_rows_variants=None,
        runtime_sec=float(compute_sec),
        io_sec=float(io_sec),
        runtime_total_sec=float(compute_sec + io_sec),
        two_body_max_processes=None if max_processes is None else int(max_processes),
        error=err,
    )


def _run_strategy_assist_window_then_2body_variants(
    *,
    subset_dir: Path,
    strategies_dir: Path,
    variant_kind: str,
    method: str,
    num_samples: int | None,
    orbits_all: Orbits,
    orbits_covok: Orbits,
    n_cov_ok: int,
    target_codes: pa.Array,
    target_times_utc: Timestamp,
    target_observers_utc: Observers,
    target_observers_tdb: Observers,
    windows: qv.Table,
    window_size_days: int,
    time_chunk_size: int,
    max_processes: int | None,
    write_ephemeris: bool,
    write_variants_orbits: bool,
) -> dict[str, object]:
    name = "assist_window_then_2body_variants"
    strat_dir = strategies_dir / name / str(variant_kind)
    err: str | None = None
    n_rows: int | None = None
    n_var: int | None = None
    compute_sec = 0.0
    io_sec = 0.0
    timing: dict[str, float] = {}
    ephem_batch_max_rows = 1_000_000
    ephem_ray_chunk_size = 5000
    bad_orbit_ids: set[str] = set()
    bad_orbit_errors: dict[str, str] = {}
    try:
        if int(n_cov_ok) == 0:
            raise ValueError("No orbits have fully-defined 6x6 covariance; cannot generate variants.")
        _ensure_dir(strat_dir)

        center_times_utc = windows.time.unique().sort_by(["days", "nanos"])
        T = int(len(center_times_utc))
        if T == 0:
            raise ValueError("No window centers found; cannot run mixed strategy.")

        t_compute0 = time.perf_counter()
        variants = (
            VariantOrbits.create(orbits_covok, method=str(method))
            if num_samples is None
            else VariantOrbits.create(
                orbits_covok, method=str(method), num_samples=int(num_samples), seed=0
            )
        )
        dt = time.perf_counter() - t_compute0
        compute_sec += dt
        timing["t_variants_create_sec"] = timing.get("t_variants_create_sec", 0.0) + float(dt)
        # Drop any orbit_id whose sampled variant coordinates are not finite.
        # (If sigma-point sampling produced invalid states, we prefer to drop those orbits
        # and record them rather than crash downstream.)
        v_vals = variants.coordinates.values
        finite_rows = np.isfinite(v_vals).all(axis=1)
        if not finite_rows.all():
            bad_idx = np.nonzero(~finite_rows)[0].tolist()
            bad_ids = sorted({str(x) for x in variants.take(bad_idx).orbit_id.to_pylist()})
            for oid in bad_ids:
                bad_orbit_ids.add(oid)
                bad_orbit_errors.setdefault(oid, "Variant sampling produced non-finite state vector")
            keep_idx = np.nonzero(finite_rows)[0].tolist()
            variants = variants.take(keep_idx)

        n_var = int(len(variants))

        if bool(write_variants_orbits):
            t_io0 = time.perf_counter()
            variants.to_parquet(str(strat_dir / "variants_orbits.parquet"))
            io_sec += time.perf_counter() - t_io0

        assist = ASSISTPropagator()
        t_compute0 = time.perf_counter()
        variants_at_centers = assist.propagate_orbits(
            variants,
            center_times_utc,
            covariance=False,
            max_processes=max_processes,
        )  # (n_var * T)
        dt = time.perf_counter() - t_compute0
        compute_sec += dt
        timing["t_assist_propagate_centers_sec"] = timing.get("t_assist_propagate_centers_sec", 0.0) + float(dt)

        # `ASSISTPropagator.propagate_orbits(..., center_times)` does not guarantee a stable
        # output ordering (time-major vs orbit-major) across implementations, and it may
        # emit times that differ slightly (µs) from the requested `center_times`.
        #
        # To robustly select the propagated variant block for a given center time `w.time`,
        # precompute a mapping from each requested `center_times_utc[i]` to the closest
        # unique propagated time value, then select rows by exact (days,nanos) equality
        # against that propagated time.
        prop_time = variants_at_centers.coordinates.time
        try:
            center_times_match = center_times_utc.rescale(str(prop_time.scale))
        except Exception:  # noqa: BLE001
            center_times_match = center_times_utc
        prop_unique = prop_time.unique().sort_by(["days", "nanos"])
        if int(len(prop_unique)) == 0:
            raise RuntimeError("ASSIST propagate_orbits produced no unique times.")
        if int(len(prop_unique)) != int(len(center_times_match)):
            raise RuntimeError(
                "Unexpected ASSIST propagate_orbits time grid: "
                f"got {int(len(prop_unique))} unique times, expected {int(len(center_times_match))}."
            )
        day_ns = 86_400 * 1_000_000_000
        pu_key = (
            prop_unique.days.to_numpy(zero_copy_only=False).astype(np.int64, copy=False) * day_ns
            + prop_unique.nanos.to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
        )
        ct_key = (
            center_times_match.days.to_numpy(zero_copy_only=False).astype(np.int64, copy=False) * day_ns
            + center_times_match.nanos.to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
        )
        prop_idx_for_center: list[int] = []
        for k in ct_key.tolist():
            j = int(np.argmin(np.abs(pu_key - int(k))))
            dt_ns = int(np.abs(int(pu_key[j]) - int(k)))
            # We expect these to be extremely close (typically µs); guard against gross mismatches.
            if dt_ns > 5_000_000_000:  # 5 seconds
                raise RuntimeError(
                    "Could not align ASSIST propagated times to requested centers: "
                    f"closest dt={float(dt_ns)/1e9:.6f}s."
                )
            prop_idx_for_center.append(j)

        target_mjd = target_times_utc.mjd().to_numpy(zero_copy_only=False).astype(np.float64)
        target_code = target_codes.to_numpy(zero_copy_only=False)

        out_ephem_dir = strat_dir / "variants_ephemeris"
        _ensure_dir(out_ephem_dir)

        total_rows = 0
        part = 0
        pending_rows = 0
        prop_batches: list[Orbits] = []
        obs_batches: list[Observers] = []
        for w in windows:
            obscode = str(w.obscode[0].as_py())
            w0 = float(w.window_start().mjd()[0].as_py())
            w1 = float(w.window_end().mjd()[0].as_py())
            mask = (target_code == obscode) & (target_mjd >= w0) & (target_mjd <= w1)
            idx = np.nonzero(mask)[0].tolist()
            if not idx:
                continue

            cidx = _center_time_index(
                center_times=center_times_utc,
                time=w.time,
                precision="ns",
            )
            if cidx is None:
                continue

            center_time_prop = prop_unique.take([int(prop_idx_for_center[int(cidx)])])
            m = prop_time.equals(center_time_prop, precision="ns")
            take_idx = pc.indices_nonzero(m).to_numpy(zero_copy_only=False).astype(np.int64)
            if take_idx.size != int(n_var):
                raise RuntimeError(
                    "Unexpected ASSIST propagate_orbits layout: "
                    f"center time produced {int(take_idx.size)} rows, expected {int(n_var)}."
                )
            variants_center = variants_at_centers.take(take_idx.tolist())
            if bad_orbit_ids:
                m_keep = pc.invert(
                    pc.is_in(
                        variants_center.table.column("orbit_id").combine_chunks(),
                        value_set=pa.array(sorted(bad_orbit_ids), type=pa.large_string()),
                    )
                )
                keep = pc.indices_nonzero(m_keep).to_numpy(zero_copy_only=False).astype(np.int64)
                if keep.size == 0:
                    continue
                variants_center = variants_center.take(keep.tolist())

            for j0 in range(0, int(len(idx)), int(time_chunk_size)):
                j1 = min(j0 + int(time_chunk_size), int(len(idx)))
                sub = idx[j0:j1]
                obs_tdb = target_observers_tdb.take(sub)

                t_compute0 = time.perf_counter()
                prop, obs_nm = _run_2body_propagation_only(
                    orbits=variants_center,
                    window_observers_tdb=obs_tdb,
                    max_processes=max_processes,
                    timing=timing,
                )
                compute_sec += time.perf_counter() - t_compute0
                prop_batches.append(prop)
                obs_batches.append(obs_nm)
                pending_rows += int(len(prop))

                if pending_rows >= int(ephem_batch_max_rows):
                    part, n_i, compute_i, io_i = _flush_2body_ephemeris_batches(
                        prop_batches=prop_batches,
                        obs_batches=obs_batches,
                        out_ephem_dir=out_ephem_dir,
                        write_ephemeris=bool(write_ephemeris),
                        part_idx=int(part),
                        max_processes=max_processes,
                        ephem_ray_chunk_size=int(ephem_ray_chunk_size),
                        timing=timing,
                        bad_orbit_ids=bad_orbit_ids,
                        bad_orbit_errors=bad_orbit_errors,
                    )
                    compute_sec += float(compute_i)
                    io_sec += float(io_i)
                    total_rows += int(n_i)
                    pending_rows = 0

        # Final flush.
        part, n_i, compute_i, io_i = _flush_2body_ephemeris_batches(
            prop_batches=prop_batches,
            obs_batches=obs_batches,
            out_ephem_dir=out_ephem_dir,
            write_ephemeris=bool(write_ephemeris),
            part_idx=int(part),
            max_processes=max_processes,
            ephem_ray_chunk_size=int(ephem_ray_chunk_size),
            timing=timing,
            bad_orbit_ids=bad_orbit_ids,
            bad_orbit_errors=bad_orbit_errors,
        )
        compute_sec += float(compute_i)
        io_sec += float(io_i)
        total_rows += int(n_i)

        n_rows = int(total_rows)
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {e}"

    # Persist bad-orbit report (if any) for later rollups / debugging.
    bad_orbits_json: str | None = None
    if bad_orbit_ids:
        p = strat_dir / "bad_orbits.json"
        _write_json(
            p,
            {
                "n_bad_orbits": int(len(bad_orbit_ids)),
                "bad_orbit_ids": sorted(bad_orbit_ids),
                "bad_orbit_errors": {k: str(v) for k, v in sorted(bad_orbit_errors.items())},
            },
        )
        bad_orbits_json = str(p)

    _write_strategy_meta(
        strat_dir,
        dict(
            strategy=name,
            variant_kind=str(variant_kind),
            variant_method=str(method),
            n_variant_orbits=n_var,
            kind="variants_ephemeris",
            window_size_days=int(window_size_days),
            n_orbits=int(len(orbits_all)),
            n_orbits_covok=int(n_cov_ok),
            n_time_targets=int(len(target_observers_utc)),
            n_window_centers=int(len(windows)),
            n_rows=n_rows,
            time_chunk_size=int(time_chunk_size),
            ephem_batch_max_rows=int(ephem_batch_max_rows),
            ephem_ray_chunk_size=int(ephem_ray_chunk_size),
            n_bad_orbits=int(len(bad_orbit_ids)),
            bad_orbits_json=bad_orbits_json,
            layout_note="Carry the same particles through ASSIST-to-centers then 2-body-to-targets; each ephemeris part is a cross product (variants_center × time_chunk).",
            assist_max_processes=None if max_processes is None else int(max_processes),
            assist_max_processes_variants_propagate_orbits=None if max_processes is None else int(max_processes),
            two_body_max_processes=None if max_processes is None else int(max_processes),
            runtime_sec=float(compute_sec),
            io_sec=float(io_sec),
            runtime_total_sec=float(compute_sec + io_sec),
            **{k: float(v) for k, v in dict(timing).items()},
            error=err,
        ),
    )
    return dict(
        subset_dir=str(subset_dir),
        strategy=name,
        variant_kind=str(variant_kind),
        n_orbits=int(len(orbits_all)),
        n_orbits_covok=int(n_cov_ok),
        n_time_targets=int(len(target_observers_utc)),
        n_window_centers=int(len(windows)),
        n_ephem_rows_mean=None,
        n_variant_orbits=n_var,
        n_ephem_rows_variants=n_rows,
        runtime_sec=float(compute_sec),
        io_sec=float(io_sec),
        runtime_total_sec=float(compute_sec + io_sec),
        two_body_max_processes=None if max_processes is None else int(max_processes),
        **{k: float(v) for k, v in dict(timing).items()},
        error=err,
    )


def _run_strategy_assist_mean(
    *,
    subset_dir: Path,
    strategies_dir: Path,
    orbits_all: Orbits,
    orbits_mean: Orbits,
    target_observers_utc: Observers,
    windows: qv.Table,
    window_size_days: int,
    time_chunk_size: int,
    max_processes: int | None,
    write_ephemeris: bool,
    n_cov_ok: int,
) -> dict[str, object]:
    strat_dir = strategies_dir / "assist_mean"
    err: str | None = None
    n_rows: int | None = None
    compute_sec = 0.0
    io_sec = 0.0
    try:
        _ensure_dir(strat_dir)
        out_ephem_dir = strat_dir / "mean_ephemeris"
        _ensure_dir(out_ephem_dir)
        assist = ASSISTPropagator()
        n_rows, compute_sec, io_sec = _run_chunked_assist_ephemeris(
            assist=assist,
            orbits=orbits_mean,
            target_observers_utc=target_observers_utc,
            time_chunk_size=int(time_chunk_size),
            max_processes=max_processes,
            covariance=False,
            out_ephem_dir=out_ephem_dir,
            write_ephemeris=bool(write_ephemeris),
        )
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {e}"

    _write_strategy_meta(
        strat_dir,
        dict(
            strategy="assist_mean",
            kind="mean_ephemeris",
            window_size_days=int(window_size_days),
            n_orbits=int(len(orbits_all)),
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
    return dict(
        subset_dir=str(subset_dir),
        strategy="assist_mean",
        variant_kind=None,
        n_orbits=int(len(orbits_all)),
        n_orbits_covok=int(n_cov_ok),
        n_time_targets=int(len(target_observers_utc)),
        n_window_centers=int(len(windows)),
        n_ephem_rows_mean=n_rows,
        n_variant_orbits=None,
        n_ephem_rows_variants=None,
        runtime_sec=float(compute_sec),
        io_sec=float(io_sec),
        runtime_total_sec=float(compute_sec + io_sec),
        two_body_max_processes=None if max_processes is None else int(max_processes),
        error=err,
    )


def _run_strategy_assist_variants(
    *,
    subset_dir: Path,
    strategies_dir: Path,
    variant_kind: str,
    method: str,
    num_samples: int | None,
    orbits_all: Orbits,
    orbits_covok: Orbits,
    n_cov_ok: int,
    target_observers_utc: Observers,
    windows: qv.Table,
    time_chunk_size: int,
    max_processes: int | None,
    write_ephemeris: bool,
    write_variants_orbits: bool,
    truth_targets_by_orbit: dict[str, list[int]] | None = None,
    target_codes: pa.Array | None = None,
    target_times_utc: Timestamp | None = None,
) -> dict[str, object]:
    strat_dir = strategies_dir / "assist_variants" / str(variant_kind)
    err: str | None = None
    n_rows: int | None = None
    n_var: int | None = None
    compute_sec = 0.0
    io_sec = 0.0
    n_fail_orbits = 0
    first_fail_orbit: str | None = None
    first_fail_error: str | None = None
    try:
        if int(n_cov_ok) == 0:
            raise ValueError("No orbits have fully-defined 6x6 covariance; cannot generate variants.")
        _ensure_dir(strat_dir)
        out_ephem_dir = strat_dir / "variants_ephemeris"
        _ensure_dir(out_ephem_dir)
        assist = ASSISTPropagator()

        if truth_targets_by_orbit is not None:
            if target_codes is None or target_times_utc is None:
                raise ValueError("truth_targets_by_orbit requires target_codes and target_times_utc")

            variants_all: VariantOrbits | None = None
            total_rows = 0
            part = 0
            for i in range(int(len(orbits_covok))):
                oid = str(orbits_covok.orbit_id[i].as_py())
                tidx = truth_targets_by_orbit.get(oid)
                if not tidx:
                    continue

                o1 = orbits_covok.take([i])
                t_compute0 = time.perf_counter()
                v1 = (
                    VariantOrbits.create(o1, method=str(method))
                    if num_samples is None
                    else VariantOrbits.create(o1, method=str(method), num_samples=int(num_samples), seed=0)
                )
                compute_sec += time.perf_counter() - t_compute0
                variants_all = v1 if variants_all is None else qv.concatenate([variants_all, v1])

                codes_i = pc.take(target_codes, pa.array(list(tidx), type=pa.int64()))
                times_i = target_times_utc.take(list(tidx))
                obs_i = Observers.from_codes(codes_i, times_i)

                try:
                    t_compute0 = time.perf_counter()
                    eph = assist.generate_ephemeris(
                        v1,
                        obs_i,
                        covariance=False,
                        max_processes=max_processes,
                    )
                    compute_sec += time.perf_counter() - t_compute0
                except Exception as e:  # noqa: BLE001
                    n_fail_orbits += 1
                    if first_fail_orbit is None:
                        first_fail_orbit = oid
                        first_fail_error = f"{type(e).__name__}: {e}"
                    continue

                total_rows += int(len(eph))
                if bool(write_ephemeris):
                    t_io0 = time.perf_counter()
                    _write_quivr_part(out_ephem_dir, qt=eph, part_idx=int(part))
                    io_sec += time.perf_counter() - t_io0
                part += 1

            if variants_all is None:
                n_var = 0
            else:
                n_var = int(len(variants_all))
                if bool(write_variants_orbits):
                    t_io0 = time.perf_counter()
                    variants_all.to_parquet(str(strat_dir / "variants_orbits.parquet"))
                    io_sec += time.perf_counter() - t_io0
            n_rows = int(total_rows)
        else:
            t_compute0 = time.perf_counter()
            variants = (
                VariantOrbits.create(orbits_covok, method=str(method))
                if num_samples is None
                else VariantOrbits.create(orbits_covok, method=str(method), num_samples=int(num_samples), seed=0)
            )
            compute_sec += time.perf_counter() - t_compute0
            n_var = int(len(variants))

            if bool(write_variants_orbits):
                t_io0 = time.perf_counter()
                variants.to_parquet(str(strat_dir / "variants_orbits.parquet"))
                io_sec += time.perf_counter() - t_io0

            n_rows, c_sec, io2_sec = _run_chunked_assist_ephemeris(
                assist=assist,
                orbits=variants,
                target_observers_utc=target_observers_utc,
                time_chunk_size=int(time_chunk_size),
                max_processes=max_processes,
                covariance=False,
                out_ephem_dir=out_ephem_dir,
                write_ephemeris=bool(write_ephemeris),
            )
            compute_sec += float(c_sec)
            io_sec += float(io2_sec)
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {e}"

    _write_strategy_meta(
        strat_dir,
        dict(
            strategy="assist_variants",
            variant_kind=str(variant_kind),
            variant_method=str(method),
            n_variant_orbits=n_var,
            n_time_targets=int(len(target_observers_utc)),
            n_window_centers=int(len(windows)),
            n_rows=n_rows,
            time_chunk_size=int(time_chunk_size),
            layout_note=(
                "Store VariantOrbits and Ephemeris separately. "
                "Default: each ephemeris part is (variants × time_chunk). "
                "Truth-orbit-targets: parts are per-orbit (variants × that orbit's truth targets)."
            ),
            truth_orbit_targets=bool(truth_targets_by_orbit is not None),
            n_failed_orbits=int(n_fail_orbits),
            first_failed_orbit=first_fail_orbit,
            first_failed_orbit_error=first_fail_error,
            runtime_sec=float(compute_sec),
            io_sec=float(io_sec),
            runtime_total_sec=float(compute_sec + io_sec),
            error=err,
        ),
    )
    return dict(
        subset_dir=str(subset_dir),
        strategy="assist_variants",
        variant_kind=str(variant_kind),
        n_orbits=int(len(orbits_all)),
        n_orbits_covok=int(n_cov_ok),
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


class FrameTimeTargets(qv.Table):
    obscode = qv.LargeStringColumn()
    time = Timestamp.as_column()


def run_stage2_propagation_bench(
    *,
    subset_dir: Path,
    orbits_parquet: Path,
    window_size_days: int = 7,
    out_dir: Path | None = None,
    inputs_artifacts_dir: Path | None = None,
    max_orbits: int | None = None,
    orbit_ids: list[str] | None = None,
    max_window_centers: int | None = None,
    targets_mjd_min: float | None = None,
    targets_mjd_max: float | None = None,
    strategies: list[str] | None = None,
    mc_samples: list[int] | None = None,
    time_chunk_size: int = 0,
    max_processes: int | None = 8,
    only_truth_orbits: bool = False,
    only_truth_targets: bool = False,
    truth_orbit_targets: bool = False,
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

    resolved_inputs_artifacts_dir = (
        Path(inputs_artifacts_dir).expanduser().resolve()
        if inputs_artifacts_dir is not None
        else (Path(subset_dir) / "artifacts")
    )

    mjd_min = float(win.min_mjd) if targets_mjd_min is None else float(targets_mjd_min)
    mjd_max = float(win.max_mjd) if targets_mjd_max is None else float(targets_mjd_max)
    if not np.isfinite(mjd_min) or not np.isfinite(mjd_max) or mjd_max <= mjd_min:
        raise ValueError(f"Invalid target MJD range: mjd_min={mjd_min} mjd_max={mjd_max}")

    # Propagation targets: by default, all distinct (obscode, exposure_mjd_mid) pairs in the subset range.
    # In truth-only recovery runs, restrict targets to only the exposure midpoints that are known
    # (from the truth crossmatch) to contain a matched truth detection.
    truth_targets_by_orbit: dict[str, list[int]] | None = None
    if bool(truth_orbit_targets):
        if (targets_mjd_min is not None) or (targets_mjd_max is not None) or (max_window_centers is not None):
            raise ValueError(
                "--truth-orbit-targets is not compatible with target slicing/window flags "
                "(targets_mjd_min/targets_mjd_max/max_window_centers); it would invalidate target_idx mappings."
            )
        if not bool(only_truth_targets):
            raise ValueError("--truth-orbit-targets requires --only-truth-targets")
        target_codes, target_times_utc, truth_targets_by_orbit = _fetch_truth_orbit_target_idx_map(
            subset_dir=Path(subset_dir),
            index_db=Path(win.index_db),
            inputs_artifacts_dir=resolved_inputs_artifacts_dir,
            restrict_designations=(
                None
                if orbit_ids is None
                else {str(x).strip() for x in orbit_ids if str(x).strip()}
            ),
        )
    elif bool(only_truth_targets):
        target_codes, target_times_utc = _fetch_truth_matched_exposure_midpoints(
            subset_dir=Path(subset_dir),
            index_db=Path(win.index_db),
            inputs_artifacts_dir=resolved_inputs_artifacts_dir,
            restrict_designations=(
                None
                if orbit_ids is None
                else {str(x).strip() for x in orbit_ids if str(x).strip()}
            ),
        )
    else:
        # This is the true workload for per-frame ephemeris generation.
        target_codes, target_times_utc = _fetch_unique_exposure_midpoints(
            db, start_mjd=float(mjd_min), end_mjd=float(mjd_max) + 1e-9
        )

    # Optional target-time window filter (applies to both truth-target and full-target modes).
    if len(target_times_utc) > 0 and (
        (targets_mjd_min is not None) or (targets_mjd_max is not None)
    ):
        mjd = target_times_utc.mjd().to_numpy(zero_copy_only=False).astype(np.float64)
        m = (mjd >= float(mjd_min)) & (mjd < float(mjd_max) + 1e-9)
        if not m.any():
            target_codes = pa.array([], type=pa.large_string())
            target_times_utc = Timestamp.from_mjd([], scale="utc")
        elif not m.all():
            idx = np.nonzero(m)[0]
            target_codes = pc.take(target_codes, pa.array(idx, type=pa.int64()))
            target_times_utc = target_times_utc.take(idx.tolist())
    if max_window_centers is not None:
        # Backwards-compat: this flag now caps the number of (obscode,time) targets.
        n = int(max_window_centers)
        target_codes = target_codes.slice(0, n)
        target_times_utc = target_times_utc[:n]

    # Default: disable time chunking (single batch over all targets). Chunking can be
    # re-enabled by passing a positive time_chunk_size if memory becomes a problem.
    if int(time_chunk_size) <= 0:
        time_chunk_size = int(len(target_times_utc))
    else:
        time_chunk_size = int(min(int(time_chunk_size), int(len(target_times_utc))))

    # Use the provided orbits parquet path (do not silently replace it with a subset default).
    # This lets callers run controlled samples (e.g., 20 truth-matched orbits) for benchmarking.
    orbits_path = orbits_parquet
    orbits = Orbits.from_parquet(str(orbits_path))
    truth_orbit_ids: set[str] | None = None
    if bool(only_truth_orbits):
        truth_orbit_ids = _truth_orbit_ids_in_subset(
            subset_dir, inputs_artifacts_dir=resolved_inputs_artifacts_dir
        )
        if truth_orbit_ids:
            mask = pc.is_in(
                orbits.orbit_id, value_set=pa.array(sorted(truth_orbit_ids), pa.large_string())
            )
            idx = np.nonzero(mask.to_numpy(zero_copy_only=False).astype(bool))[0]
            orbits = orbits.take(idx.tolist())
    if orbit_ids is not None:
        wanted = sorted({str(x).strip() for x in orbit_ids if str(x).strip()})
        if wanted:
            mask = pc.is_in(orbits.orbit_id, value_set=pa.array(wanted, pa.large_string()))
            idx = np.nonzero(mask.to_numpy(zero_copy_only=False).astype(bool))[0]
            orbits = orbits.take(idx.tolist())
    if max_orbits is not None:
        orbits = orbits[: int(max_orbits)]
    cov_ok = _cov_ok_mask(orbits)
    n_cov_ok_finite = int(cov_ok.sum())
    orbits_covok = orbits.take(np.where(cov_ok)[0]) if n_cov_ok_finite > 0 else Orbits.empty()
    orbits_covok, cov_psd_meta = _repair_orbits_covariance_psd_for_sampling(orbits_covok)
    n_cov_ok = int(len(orbits_covok))
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

    # Window centers are only needed for the windowed mixed strategies. Computing them can be
    # very expensive on large subsets because it scans `frames` rows in the time range.
    # If strategies is None, we will run all strategies (including windowed ones).
    # If strategies are specified, compute windows only when any requested strategy
    # is the windowed family.
    needs_windows = (
        (strategies is None)
        or any(str(s).strip().startswith("assist_window_then_2body") for s in strategies)
    )
    if needs_windows:
        windows = db.frames.idx.window_centers(float(mjd_min), float(mjd_max), int(window_size_days))
        windows.to_parquet(str(inputs_dir / "window_centers.parquet"))
    else:
        windows = WindowCenters.empty()

    # Warm up JIT compilation / kernels (do not include in benchmark timings).
    # Without this, the first strategy to run can look artificially slow.
    if len(orbits_mean) > 0 and len(target_observers_utc) > 0:
        o1 = orbits_mean[:1]
        obs_utc_2 = target_observers_utc[: min(2, len(target_observers_utc))]
        obs_tdb_2 = target_observers_tdb[: min(2, len(target_observers_tdb))]
        try:
            # Avoid implicit multiprocessing defaults during warmup (can hang/crash on some platforms).
            _ = _run_2body_ephemeris(orbits=o1, window_observers_tdb=obs_tdb_2, max_processes=1)
        except Exception:
            pass
        try:
            # Avoid implicit multiprocessing defaults during warmup (can hang/crash on some platforms).
            _ = ASSISTPropagator().generate_ephemeris(o1, obs_utc_2, covariance=False, max_processes=1)
        except Exception:
            pass

    def _enabled(name: str) -> bool:
        return strategies is None or name in strategies

    # MC defaults:
    # - In capped runs (debugging / development), we default to MC=256 so we always have a
    #   representative Monte Carlo benchmark.
    # - In full runs (no caps), MC must be explicitly requested (otherwise it can explode).
    if mc_samples is None:
        mc_samples = [256] if (max_orbits is not None or max_window_centers is not None) else []

    strategies_dir = run_dir / "strategies"
    _ensure_dir(strategies_dir)

    # 2-body baselines (full target times).
    if _enabled("2body_only"):
        metrics_rows.append(
            _run_with_strategy_logs(
                label="2body_only",
                fn=lambda: _run_strategy_2body(
                    subset_dir=subset_dir,
                    strategies_dir=strategies_dir,
                    orbits_all=orbits,
                    orbits_mean=orbits_mean,
                    target_observers_utc=target_observers_utc,
                    target_observers_tdb=target_observers_tdb,
                    windows=windows,
                    window_size_days=int(window_size_days),
                    time_chunk_size=int(time_chunk_size),
                    max_processes=max_processes,
                    write_ephemeris=bool(write_ephemeris),
                    include_covariance=False,
                    n_cov_ok=int(n_cov_ok),
                ),
            )
        )
    if _enabled("2body_with_covariance"):
        metrics_rows.append(
            _run_with_strategy_logs(
                label="2body_with_covariance",
                fn=lambda: _run_strategy_2body(
                    subset_dir=subset_dir,
                    strategies_dir=strategies_dir,
                    orbits_all=orbits,
                    orbits_mean=orbits_mean,
                    target_observers_utc=target_observers_utc,
                    target_observers_tdb=target_observers_tdb,
                    windows=windows,
                    window_size_days=int(window_size_days),
                    time_chunk_size=int(time_chunk_size),
                    max_processes=max_processes,
                    write_ephemeris=bool(write_ephemeris),
                    include_covariance=True,
                    n_cov_ok=int(n_cov_ok),
                ),
            )
        )

    # Mixed strategy: ASSIST to window centers, then 2-body to all targets within each window.
    if _enabled("assist_window_then_2body"):
        metrics_rows.append(
            _run_with_strategy_logs(
                label="assist_window_then_2body",
                fn=lambda: _run_strategy_assist_window_then_2body(
                    subset_dir=subset_dir,
                    strategies_dir=strategies_dir,
                    orbits_all=orbits,
                    orbits_mean=orbits_mean,
                    target_codes=target_codes,
                    target_times_utc=target_times_utc,
                    target_observers_utc=target_observers_utc,
                    target_observers_tdb=target_observers_tdb,
                    windows=windows,
                    window_size_days=int(window_size_days),
                    time_chunk_size=int(time_chunk_size),
                    max_processes=max_processes,
                    write_ephemeris=bool(write_ephemeris),
                    n_cov_ok=int(n_cov_ok),
                ),
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
        metrics_rows.append(
            _run_with_strategy_logs(
                label=strat_name,
                fn=lambda variant_kind=variant_kind, method=method, num_samples=num_samples: _run_strategy_assist_window_then_2body_variants(
                    subset_dir=subset_dir,
                    strategies_dir=strategies_dir,
                    variant_kind=str(variant_kind),
                    method=str(method),
                    num_samples=num_samples,
                    orbits_all=orbits,
                    orbits_covok=orbits_covok,
                    n_cov_ok=int(n_cov_ok),
                    target_codes=target_codes,
                    target_times_utc=target_times_utc,
                    target_observers_utc=target_observers_utc,
                    target_observers_tdb=target_observers_tdb,
                    windows=windows,
                    window_size_days=int(window_size_days),
                    time_chunk_size=int(time_chunk_size),
                    max_processes=max_processes,
                    write_ephemeris=bool(write_ephemeris),
                    write_variants_orbits=bool(write_variants_orbits),
                ),
            )
        )

    # ASSIST mean (no covariance) for all orbits.
    if _enabled("assist_mean"):
        metrics_rows.append(
            _run_with_strategy_logs(
                label="assist_mean",
                fn=lambda: _run_strategy_assist_mean(
                    subset_dir=subset_dir,
                    strategies_dir=strategies_dir,
                    orbits_all=orbits,
                    orbits_mean=orbits_mean,
                    target_observers_utc=target_observers_utc,
                    windows=windows,
                    window_size_days=int(window_size_days),
                    time_chunk_size=int(time_chunk_size),
                    max_processes=max_processes,
                    write_ephemeris=bool(write_ephemeris),
                    n_cov_ok=int(n_cov_ok),
                ),
            )
        )

    # ASSIST variants (only for cov-ok orbits).
    variant_specs: list[tuple[str, str, int | None]] = [("sigma_points", "sigma-point", None)]
    variant_specs.extend([(f"mc_{int(n)}", "monte-carlo", int(n)) for n in mc_samples])

    for variant_kind, method, num_samples in variant_specs:
        strat_name = f"assist_variants:{variant_kind}"
        if not _enabled(strat_name):
            continue
        metrics_rows.append(
            _run_with_strategy_logs(
                label=strat_name,
                fn=lambda variant_kind=variant_kind, method=method, num_samples=num_samples: _run_strategy_assist_variants(
                    subset_dir=subset_dir,
                    strategies_dir=strategies_dir,
                    variant_kind=str(variant_kind),
                    method=str(method),
                    num_samples=num_samples,
                    orbits_all=orbits,
                    orbits_covok=orbits_covok,
                    n_cov_ok=int(n_cov_ok),
                    target_observers_utc=target_observers_utc,
                    windows=windows,
                    time_chunk_size=int(time_chunk_size),
                    max_processes=max_processes,
                    write_ephemeris=bool(write_ephemeris),
                    write_variants_orbits=bool(write_variants_orbits),
                    truth_targets_by_orbit=truth_targets_by_orbit,
                    target_codes=target_codes,
                    target_times_utc=target_times_utc,
                ),
            )
        )

    pq.write_table(pa.Table.from_pylist(metrics_rows), str(run_dir / "metrics.parquet"))

    meta = {
        "subset_dir": str(subset_dir),
        "inputs_artifacts_dir": str(resolved_inputs_artifacts_dir),
        "orbits_parquet_arg": str(orbits_parquet),
        "orbits_parquet_used": str(orbits_path),
        "n_orbits": int(len(orbits)),
        "n_orbits_covok": n_cov_ok,
        "n_orbits_covok_finite": int(n_cov_ok_finite),
        **{k: v for k, v in dict(cov_psd_meta).items()},
        "window_size_days": int(window_size_days),
        "max_orbits": None if max_orbits is None else int(max_orbits),
        "max_window_centers": None if max_window_centers is None else int(max_window_centers),
        "targets_mjd_min": float(mjd_min),
        "targets_mjd_max": float(mjd_max),
        "strategies_requested": None if strategies is None else list(strategies),
        "mc_samples": [int(x) for x in mc_samples],
        "time_chunk_size": int(time_chunk_size),
        "max_processes": None if max_processes is None else int(max_processes),
        "only_truth_orbits": bool(only_truth_orbits),
        "only_truth_targets": bool(only_truth_targets),
        "truth_orbit_targets": bool(truth_orbit_targets),
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

    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s")
    # Ensure INFO logs are emitted even if something else configured logging earlier.
    root.setLevel(logging.INFO)

    p = argparse.ArgumentParser(description="Stage 2: atomic propagation benchmark runner (cached ephemerides).")
    p.add_argument("--subset-dir", type=str, required=True)
    p.add_argument("--orbits-parquet", type=str, required=True)
    p.add_argument(
        "--inputs-artifacts-dir",
        type=str,
        default=None,
        help=(
            "Directory containing truth_precovery_crossmatch.parquet (for --only-truth-targets) and "
            "orbits_selected_sbdb.parquet (for --only-truth-orbits). Default: <subset_dir>/artifacts."
        ),
    )
    p.add_argument("--window-size-days", type=int, default=7)
    p.add_argument("--max-orbits", type=int, default=None)
    p.add_argument(
        "--orbit-ids",
        type=str,
        default=None,
        help="Comma-separated list of orbit_id values to run (debugging / 1-orbit comparisons).",
    )
    p.add_argument(
        "--max-window-centers",
        type=int,
        default=None,
        help="Deprecated name: now caps number of (obscode,time) targets used for benchmarking.",
    )
    p.add_argument(
        "--targets-mjd-min",
        type=float,
        default=None,
        help=(
            "Minimum exposure midpoint MJD (UTC) to include as Stage 2 targets. "
            "If omitted, uses the subset window minimum."
        ),
    )
    p.add_argument(
        "--targets-mjd-max",
        type=float,
        default=None,
        help=(
            "Maximum exposure midpoint MJD (UTC) to include as Stage 2 targets. "
            "Targets are filtered to mjd < targets_mjd_max. "
            "If omitted, uses the subset window maximum."
        ),
    )
    p.add_argument(
        "--time-chunk-size",
        type=int,
        default=0,
        help=(
            "Chunk size over (obscode,time) targets (default: 0 disables chunking; run all targets in one batch). "
            "Use a positive value only if memory becomes a problem."
        ),
    )
    p.add_argument(
        "--max-processes",
        type=int,
        default=8,
        help=(
            "Max processes for multiprocessing-enabled propagation/ephemeris calls in this benchmark "
            "(ASSISTPropagator + 2-body propagate_2body/generate_ephemeris_2body). Use 0 to disable."
        ),
    )
    p.add_argument(
        "--only-truth-orbits",
        action="store_true",
        help="Filter input orbits to those with ≥1 matched truth detection in this subset window.",
    )
    p.add_argument(
        "--only-truth-targets",
        action="store_true",
        help=(
            "Restrict propagation targets to only those exposure midpoints that are known (from the "
            "truth crossmatch) to contain a matched truth detection. This avoids propagating to all "
            "unique exposure times in the month partition."
        ),
    )
    p.add_argument(
        "--truth-orbit-targets",
        action="store_true",
        help=(
            "Truth-only optimization: generate ephemerides only for orbit-target pairs present in truth "
            "(per orbit), rather than (all orbits × all truth targets). Requires --only-truth-targets. "
            "Not compatible with target slicing/window flags."
        ),
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
        inputs_artifacts_dir=None if args.inputs_artifacts_dir is None else Path(args.inputs_artifacts_dir),
        max_orbits=args.max_orbits,
        orbit_ids=(None if args.orbit_ids is None else [s.strip() for s in str(args.orbit_ids).split(',') if s.strip()]),
        max_window_centers=args.max_window_centers,
        targets_mjd_min=args.targets_mjd_min,
        targets_mjd_max=args.targets_mjd_max,
        strategies=_parse_strategies_arg(args.strategies),
        mc_samples=_parse_mc_samples_arg(args.mc_samples, default=[]),
        time_chunk_size=int(args.time_chunk_size),
        max_processes=(None if int(args.max_processes) <= 0 else int(args.max_processes)),
        only_truth_orbits=bool(args.only_truth_orbits),
        only_truth_targets=bool(args.only_truth_targets),
        truth_orbit_targets=bool(args.truth_orbit_targets),
        mean_with_covariance=bool(args.mean_with_covariance),
        write_ephemeris=not bool(args.no_write_ephemeris),
        write_variants_orbits=not bool(args.no_write_variants_orbits),
    )
    print(f"run_dir={run_dir}")
    print(f"meta_json={run_dir / 'meta.json'}")
    print(f"metrics_parquet={run_dir / 'metrics.parquet'}")


if __name__ == "__main__":
    main()

