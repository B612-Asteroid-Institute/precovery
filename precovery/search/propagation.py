from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import quivr as qv
import time
import warnings
import os
import ray
from adam_assist import ASSISTPropagator
from adam_core.coordinates.cartesian import CartesianCoordinates
from adam_core.coordinates.covariances import CoordinateCovariances
from adam_core.coordinates.origin import Origin, OriginCodes
from adam_core.coordinates.spherical import SphericalCoordinates
from adam_core.coordinates.transform import transform_coordinates
from adam_core.dynamics.ephemeris import generate_ephemeris_2body
from adam_core.dynamics.propagation import propagate_2body
from adam_core.observers import Observers
from adam_core.orbits import Orbits
from adam_core.orbits.ephemeris import Ephemeris
from adam_core.orbits.variants import VariantOrbits, VariantEphemeris
from adam_core.photometry.magnitude import calculate_apparent_magnitude_v_and_phase_angle
from adam_core.time import Timestamp
from adam_core.ray_cluster import initialize_use_ray

from .chunking import resolve_time_chunk_size_from_ram_budget
from .reconstruction import reconstruct_cov_ll_from_sigma_point_cloud
from .covariance import attach_cov_ll_to_ephemeris
from .covariance_psd import repair_covariance_matrix_psd_with_reason


PropagationStrategy = Literal[
    "assist_window_then_2body_variants:sigma_points",
    "assist_variants:sigma_points",
]


@lru_cache(maxsize=1)
def _assist_propagator() -> ASSISTPropagator:
    # ASSISTPropagator construction can be expensive (SPICE kernel loading, etc.).
    # Cache one instance per process to avoid repeated setup in tight loops/tests.
    return ASSISTPropagator()


def _effective_max_processes(max_processes: int | None) -> int | None:
    """
    Normalize max_processes for libraries that treat `max_processes=1` as a special
    multiprocessing path (creating a worker pool of size 1), which can be much slower
    than a true in-process implementation for small workloads.
    """
    if max_processes is None:
        return None
    mp = int(max_processes)
    if mp <= 1:
        return None
    return mp


def _propagate_2body_variant_orbits_preserve_metadata(
    *,
    variants: VariantOrbits,
    times_utc: Timestamp,
    max_processes: int | None,
) -> VariantOrbits:
    """
    Propagate sigma-point VariantOrbits using adam-core's 2-body propagator while preserving
    VariantOrbits metadata columns (variant_id, weights, weights_cov).
    """
    if len(variants) == 0 or len(times_utc) == 0:
        return VariantOrbits.empty()

    # `propagate_2body` operates on `Orbits` and does not preserve VariantOrbits-specific columns,
    # so we round-trip through `Orbits` and then reattach variant metadata (repeat per time).
    base = Orbits.from_kwargs(
        orbit_id=variants.orbit_id,
        object_id=variants.object_id,
        coordinates=variants.coordinates,
        physical_parameters=variants.physical_parameters,
    )
    prop = propagate_2body(base, times_utc, max_processes=_effective_max_processes(max_processes))

    n_variants = int(len(variants))
    n_times = int(len(times_utc))
    if int(len(prop)) != int(n_variants * n_times):
        raise RuntimeError(
            "propagate_2body returned unexpected length for variants. "
            f"expected={int(n_variants*n_times)} got={int(len(prop))}"
        )

    rep = int(n_times)
    variant_id = pc.cast(variants.variant_id, pa.large_string()).to_numpy(zero_copy_only=False).astype(object)
    w = pc.cast(pc.fill_null(variants.weights, pa.scalar(np.nan, type=pa.float64())), pa.float64()).to_numpy(
        zero_copy_only=False
    ).astype(np.float64, copy=False)
    w_cov = pc.cast(
        pc.fill_null(variants.weights_cov, pa.scalar(np.nan, type=pa.float64())), pa.float64()
    ).to_numpy(zero_copy_only=False).astype(np.float64, copy=False)

    variant_id_rep = np.repeat(variant_id, rep)
    w_rep = np.repeat(w, rep)
    w_cov_rep = np.repeat(w_cov, rep)

    return VariantOrbits.from_kwargs(
        orbit_id=prop.orbit_id,
        object_id=prop.object_id,
        variant_id=pa.array(variant_id_rep, type=pa.large_string()),
        weights=pa.array(w_rep, type=pa.float64()),
        weights_cov=pa.array(w_cov_rep, type=pa.float64()),
        coordinates=prop.coordinates,
        physical_parameters=prop.physical_parameters,
    )


def _resolve_ray_num_cpus(max_processes: int | None) -> int | None:
    if max_processes is None:
        return None
    mp = int(max_processes)
    if mp <= 0:
        return None
    return mp


def _stage2_windowed_sigma_points_center_time_chunk(
    *,
    orb_center: Orbits,
    variants: VariantOrbits,
    times_chunk: Timestamp,
    obs_chunk: Observers,
    aberration_mode: str,
    compute_predicted_magnitudes: bool,
) -> tuple[Ephemeris, dict[str, float]]:
    """
    Compute the collapsed mean ephemeris (+ cov) for one (center, time_chunk, orbit_chunk).

    This runs *serially* (no Ray inside) to avoid Ray object-store thrash from returning
    large intermediate variant ephemerides.
    """
    timings_local: dict[str, float] = {}
    if len(variants) == 0 or len(times_chunk) == 0:
        return Ephemeris.empty(), timings_local

    # Propagate variants via 2-body (serial), preserving variant metadata.
    t_pv0 = time.perf_counter()
    base = Orbits.from_kwargs(
        orbit_id=variants.orbit_id,
        object_id=variants.object_id,
        coordinates=variants.coordinates,
        physical_parameters=variants.physical_parameters,
    )
    prop = propagate_2body(base, times_chunk, max_processes=1)
    timings_local["stage2.propagate2body_variants_elapsed_s"] = time.perf_counter() - t_pv0

    n_variants = int(len(variants))
    n_times = int(len(times_chunk))
    rep = int(n_times)
    variant_id = (
        pc.cast(variants.variant_id, pa.large_string())
        .to_numpy(zero_copy_only=False)
        .astype(object)
    )
    w = (
        pc.cast(
            pc.fill_null(variants.weights, pa.scalar(np.nan, type=pa.float64())),
            pa.float64(),
        )
        .to_numpy(zero_copy_only=False)
        .astype(np.float64, copy=False)
    )
    w_cov = (
        pc.cast(
            pc.fill_null(variants.weights_cov, pa.scalar(np.nan, type=pa.float64())),
            pa.float64(),
        )
        .to_numpy(zero_copy_only=False)
        .astype(np.float64, copy=False)
    )

    variants_prop = VariantOrbits.from_kwargs(
        orbit_id=prop.orbit_id,
        object_id=prop.object_id,
        variant_id=pa.array(np.repeat(variant_id, rep), type=pa.large_string()),
        weights=pa.array(np.repeat(w, rep), type=pa.float64()),
        weights_cov=pa.array(np.repeat(w_cov, rep), type=pa.float64()),
        coordinates=prop.coordinates,
        physical_parameters=prop.physical_parameters,
    )

    # Pair observers with the cartesian product layout (base-variant-major blocks).
    n_base = int(n_variants)
    rep_obs = qv.concatenate([obs_chunk] * int(n_base))

    # Variant ephemerides (serial) and UT-mean collapse.
    t_ev0 = time.perf_counter()
    eph_var = generate_ephemeris_2body(
        Orbits.from_kwargs(
            orbit_id=variants_prop.orbit_id,
            object_id=variants_prop.object_id,
            coordinates=variants_prop.coordinates,
            physical_parameters=variants_prop.physical_parameters,
        ),
        rep_obs,
        predict_magnitudes=False,
        predict_phase_angle=False,
        max_processes=1,
        chunk_size=20_000,
    )
    timings_local["stage2.ephemeris_variants_elapsed_s"] = time.perf_counter() - t_ev0

    vephem = VariantEphemeris.from_kwargs(
        orbit_id=variants_prop.orbit_id,
        object_id=variants_prop.object_id,
        variant_id=variants_prop.variant_id,
        weights=variants_prop.weights,
        weights_cov=variants_prop.weights_cov,
        coordinates=eph_var.coordinates,
        aberrated_coordinates=None,
        predicted_magnitude_v=None,
        alpha=None,
        light_time=None,
    )

    t_col0 = time.perf_counter()
    aberration_mode = str(aberration_mode).strip().lower()
    if aberration_mode == "none":
        mean = vephem.collapse_sigma_points_orbit_major(
            n_times=int(len(times_chunk)), n_variants=13
        )
    else:
        mean = vephem.collapse_by_object_id(aberration_mode=aberration_mode)
    timings_local["stage2.variant_collapse_elapsed_s"] = time.perf_counter() - t_col0

    if compute_predicted_magnitudes:
        # Nominal predicted magnitudes (serial). With the orbit-major output of
        # collapse_sigma_points_orbit_major, row order matches nominal ephemeris.
        t_pn0 = time.perf_counter()
        prop_nom = propagate_2body(orb_center, times_chunk, max_processes=1)
        timings_local["stage2.propagate2body_nominal_elapsed_s"] = time.perf_counter() - t_pn0

        rep_obs_nom = qv.concatenate([obs_chunk] * int(len(orb_center)))
        t_en0 = time.perf_counter()
        eph_nom = generate_ephemeris_2body(
            prop_nom,
            rep_obs_nom,
            predict_magnitudes=True,
            predict_phase_angle=False,
            max_processes=1,
            chunk_size=20_000,
        )
        timings_local["stage2.ephemeris_nominal_elapsed_s"] = time.perf_counter() - t_en0

        if len(eph_nom) == len(mean):
            # Only do an index-aligned assignment when the collapsed ephemeris is orbit-major
            # (same order as `propagate_2body` cartesian product output).
            idx0 = [int(i * n_times) for i in range(int(len(orb_center)))]
            try:
                mean_orbit_id0 = pc.take(mean.orbit_id, idx0).to_pylist()
                expected_orbit_id0 = pc.take(orb_center.orbit_id, list(range(int(len(orb_center))))).to_pylist()
                index_aligned = mean_orbit_id0 == expected_orbit_id0
            except Exception:
                index_aligned = False

            if index_aligned:
                mean = mean.set_column(
                    "predicted_magnitude_v",
                    pc.cast(eph_nom.predicted_magnitude_v, pa.float64()),
                )
            else:
                # Robust (slower) key join fallback.
                mean_keys = mean.table.select(
                    [
                        "orbit_id",
                        "coordinates.time.days",
                        "coordinates.time.nanos",
                        "coordinates.origin.code",
                    ]
                ).append_column(
                    "_row_idx",
                    pa.array(np.arange(int(len(mean)), dtype=np.int64), type=pa.int64()),
                )
                eph_keys = eph_nom.table.select(
                    [
                        "orbit_id",
                        "coordinates.time.days",
                        "coordinates.time.nanos",
                        "coordinates.origin.code",
                        "predicted_magnitude_v",
                    ]
                )
                joined = mean_keys.join(
                    eph_keys,
                    keys=[
                        "orbit_id",
                        "coordinates.time.days",
                        "coordinates.time.nanos",
                        "coordinates.origin.code",
                    ],
                    join_type="left outer",
                )
                if int(joined.num_rows) == int(len(mean)):
                    joined = joined.sort_by([("_row_idx", "ascending")])
                    mean = mean.set_column(
                        "predicted_magnitude_v",
                        pc.cast(joined["predicted_magnitude_v"], pa.float64()),
                    )

    return mean, timings_local


@ray.remote(num_cpus=1)
def _stage2_windowed_sigma_points_center_time_chunk_remote(
    orb_center: Orbits,
    variants: VariantOrbits,
    times_chunk: Timestamp,
    obs_chunk: Observers,
    aberration_mode: str,
    compute_predicted_magnitudes: bool,
) -> tuple[Ephemeris, dict[str, float]]:
    return _stage2_windowed_sigma_points_center_time_chunk(
        orb_center=orb_center,
        variants=variants,
        times_chunk=times_chunk,
        obs_chunk=obs_chunk,
        aberration_mode=aberration_mode,
        compute_predicted_magnitudes=compute_predicted_magnitudes,
    )


def predict_targets_batched_assist_window_then_2body_variants_sigma_points(
    *,
    orbits: Orbits,
    obscode: np.ndarray,
    times_utc: Timestamp,
    start_mjd: float,
    window_size_days: int,
    observers_all: Observers | None = None,
    max_processes: int | None = None,
    orbit_chunk_size: int = 4,
    time_chunk_size: int = 0,
    aberration_mode: Literal["none", "collapse", "recompute"] = "none",
    default_mag_slope_G: float | None = None,
    timings: dict[str, float] | None = None,
) -> tuple[Ephemeris, np.ndarray]:
    """
    Batched prediction for the high-accuracy windowed sigma-point strategy:

      ASSIST propagate (centers) → sigma-point VariantOrbits → 2-body propagate (variants) →
      2-body ephemeris (variants) → VariantEphemeris.collapse_by_object_id (UT mean + cov)

    This is designed to avoid per-orbit Python loops: it chunks by orbit count and time count
    to bound memory, and uses adam-core multiprocessing (Ray) for propagation and ephemeris where possible.
    """
    if len(orbits) == 0 or len(times_utc) == 0:
        return Ephemeris.empty(), np.zeros((0, 2, 2), dtype=np.float64)

    if default_mag_slope_G is not None:
        g0 = float(default_mag_slope_G)
        if not np.isfinite(g0):
            raise ValueError("default_mag_slope_G must be finite when provided.")
        try:
            phys = orbits.physical_parameters
            H_v = phys.H_v
            G = phys.G
            has_H = pc.fill_null(pc.is_finite(H_v), False)
            has_G = pc.fill_null(pc.is_finite(G), False)
            need = pc.and_(has_H, pc.invert(has_G))
            if bool(pc.any(need).as_py()):
                G2 = pc.if_else(
                    need,
                    pa.scalar(g0, type=pa.float64()),
                    pc.cast(G, pa.float64()),
                )
                orbits = orbits.set_column("physical_parameters.G", G2)
        except Exception:
            pass

    orbit_chunk_size = int(orbit_chunk_size)
    if orbit_chunk_size <= 0:
        raise ValueError("orbit_chunk_size must be > 0")
    time_chunk_size = int(time_chunk_size)

    # Sigma points for 6D -> 13 variants per orbit.
    K_SIGMA = 13

    # Auto time chunk sizing based on RAM budget, accounting for variant expansion.
    auto_time_chunk = time_chunk_size == 0
    if auto_time_chunk:
        rows_per_time_target = int(orbit_chunk_size * K_SIGMA)
        time_chunk_size, _meta = resolve_time_chunk_size_from_ram_budget(
            n_time_targets=int(len(times_utc)),
            rows_per_time_target=int(rows_per_time_target),
            ram_budget_frac=0.10,
            effective_bytes_per_row=200.0,
            min_chunk=1,
        )
        # Keep windowed Stage-2 result objects small enough for Ray object-store stability.
        time_chunk_size = int(min(int(time_chunk_size), 2_000))
    if time_chunk_size < 0:
        time_chunk_size = int(len(times_utc))

    t = times_utc
    codes = pc.cast(pa.array(obscode), pa.large_string())
    if observers_all is not None and len(observers_all) != len(t):
        raise ValueError(
            "observers_all must be aligned to (obscode, times_utc). "
            f"Expected len={len(t)} but got len={len(observers_all)}."
        )
    if observers_all is None:
        t_obs0 = time.perf_counter() if timings is not None else None
        observers_all = Observers.from_codes(codes, t)
        if timings is not None and t_obs0 is not None:
            timings["stage2.observers_elapsed_s"] = timings.get("stage2.observers_elapsed_s", 0.0) + (
                time.perf_counter() - t_obs0
            )

    # Window centers shared across all orbits.
    t_wc0 = time.perf_counter() if timings is not None else None
    mjd = t.mjd().to_numpy(zero_copy_only=False).astype(np.float64)
    center_mjd = _window_center_mjd(
        start_mjd=float(start_mjd), mjd=mjd, window_size_days=int(window_size_days)
    )
    center_times = Timestamp.from_mjd(pa.array(np.unique(center_mjd), type=pa.float64()), scale="utc")
    center_times = center_times.sort_by(["days", "nanos"])
    if timings is not None and t_wc0 is not None:
        timings["stage2.window_centers_elapsed_s"] = timings.get("stage2.window_centers_elapsed_s", 0.0) + (
            time.perf_counter() - t_wc0
        )

    center_mjd_unique = center_times.mjd().to_numpy(zero_copy_only=False).astype(np.float64)
    center_map = {float(x): int(i) for i, x in enumerate(center_mjd_unique.tolist())}
    center_idx = np.array([center_map[float(x)] for x in center_mjd.tolist()], dtype=np.int64)

    assist = _assist_propagator()
    out_parts: list[Ephemeris] = []

    # Use Ray only as an outer parallel scheduler to avoid Ray object-store spilling
    # on large intermediate variant ephemeris objects.
    num_cpus = _resolve_ray_num_cpus(max_processes)
    use_ray = initialize_use_ray(num_cpus=num_cpus)
    pending: list[ray.ObjectRef] = []
    pending_rows: dict[ray.ObjectRef, int] = {}
    n_workers = int(num_cpus) if (num_cpus is not None and int(num_cpus) > 0) else int(os.cpu_count() or 4)
    max_pending = int(max(2, n_workers))
    # Bound in-flight result rows (mean ephemeris rows) to avoid plasma backpressure.
    max_inflight_rows = int(max(40_000, n_workers * 40_000))
    inflight_rows = 0

    for o0 in range(0, int(len(orbits)), int(orbit_chunk_size)):
        o1 = int(min(int(len(orbits)), int(o0) + int(orbit_chunk_size)))
        orb = orbits.take(list(range(o0, o1)))
        if len(orb) == 0:
            continue

        # Propagate nominal orbits to each window center (covariance needed for sigma-point creation).
        t_as0 = time.perf_counter() if timings is not None else None
        orb_at_centers = assist.propagate_orbits(
            orb,
            center_times,
            covariance=True,
            max_processes=_effective_max_processes(max_processes),
        )
        if timings is not None and t_as0 is not None:
            timings["stage2.assist_propagate_centers_elapsed_s"] = timings.get(
                "stage2.assist_propagate_centers_elapsed_s", 0.0
            ) + (time.perf_counter() - t_as0)

        n_centers = int(len(center_times))
        n_orb = int(len(orb))
        if int(len(orb_at_centers)) != int(n_orb * n_centers):
            raise RuntimeError(
                "assist.propagate_orbits returned unexpected length for center propagation. "
                f"expected={int(n_orb*n_centers)} got={int(len(orb_at_centers))}"
            )

        out_time = orb_at_centers.coordinates.time

        for c_i in range(int(n_centers)):
            hit = np.nonzero(center_idx == int(c_i))[0]
            if hit.size == 0:
                continue

            # ASSIST does not guarantee a specific row order for the cartesian product
            # (orbits × center_times). To stay correct for n_centers>1, select the center
            # slice by (days,nanos) match within tolerance and then reorder rows to match the input
            # orbit chunk order.
            cd = int(center_times.days[int(c_i)].as_py())
            cn = int(center_times.nanos[int(c_i)].as_py())
            m = out_time.equals_scalar(cd, cn, precision="ms")
            idx = np.nonzero(
                pc.cast(m, pa.bool_()).to_numpy(zero_copy_only=False)
            )[0].astype(np.int64)
            if int(idx.size) != int(n_orb):
                raise RuntimeError(
                    "assist.propagate_orbits returned unexpected center slice size. "
                    f"expected={int(n_orb)} got={int(idx.size)} c_i={int(c_i)}"
                )
            orb_center = orb_at_centers.take(idx.tolist())

            # Reorder to match the input `orb` ordering.
            order = pc.index_in(
                pc.cast(orb_center.orbit_id, pa.large_string()),
                value_set=pc.cast(orb.orbit_id, pa.large_string()),
            )
            order_np = np.asarray(order.to_numpy(zero_copy_only=False), dtype=np.int64)
            if not np.all(np.isfinite(order_np)) or np.any(order_np < 0):
                raise RuntimeError(
                    "assist.propagate_orbits center slice contains orbit_id not present in input chunk."
                )
            perm = np.argsort(order_np, kind="stable")
            orb_center = orb_center.take(perm.tolist())

            t_sv0 = time.perf_counter() if timings is not None else None
            variants = VariantOrbits.create(orb_center, method="sigma-point")
            if timings is not None and t_sv0 is not None:
                timings["stage2.sigma_point_variants_elapsed_s"] = timings.get(
                    "stage2.sigma_point_variants_elapsed_s", 0.0
                ) + (time.perf_counter() - t_sv0)

            variants_ref = ray.put(variants) if use_ray else None

            # Chunk by target times for this center.
            for t0i in range(0, int(hit.size), int(time_chunk_size)):
                idx_chunk = hit[t0i : t0i + int(time_chunk_size)]
                times_chunk = t.take(idx_chunk.tolist())
                obs_chunk = observers_all.take(idx_chunk.tolist())

                if use_ray:
                    chunk_rows = int(len(orb_center)) * int(len(times_chunk))
                    # Backpressure: keep both task count and object volume bounded.
                    while pending and (
                        len(pending) >= max_pending
                        or (inflight_rows + int(chunk_rows)) > max_inflight_rows
                    ):
                        ready, pending = ray.wait(pending, num_returns=1)
                        ref = ready[0]
                        mean, tloc = ray.get(ref)
                        inflight_rows = int(max(0, inflight_rows - int(pending_rows.pop(ref, 0))))
                        out_parts.append(mean)
                        if timings is not None:
                            for k, v in tloc.items():
                                timings[k] = timings.get(k, 0.0) + float(v)

                    fut = _stage2_windowed_sigma_points_center_time_chunk_remote.remote(
                        orb_center,
                        variants_ref,
                        times_chunk,
                        obs_chunk,
                        str(aberration_mode),
                        bool(default_mag_slope_G is not None),
                    )
                    pending.append(fut)
                    pending_rows[fut] = int(chunk_rows)
                    inflight_rows += int(chunk_rows)
                else:
                    mean, tloc = _stage2_windowed_sigma_points_center_time_chunk(
                        orb_center=orb_center,
                        variants=variants,
                        times_chunk=times_chunk,
                        obs_chunk=obs_chunk,
                        aberration_mode=str(aberration_mode),
                        compute_predicted_magnitudes=bool(default_mag_slope_G is not None),
                    )
                    out_parts.append(mean)
                    if timings is not None:
                        for k, v in tloc.items():
                            timings[k] = timings.get(k, 0.0) + float(v)

    if use_ray and pending:
        while pending:
            ready, pending = ray.wait(pending, num_returns=1)
            ref = ready[0]
            mean, tloc = ray.get(ref)
            inflight_rows = int(max(0, inflight_rows - int(pending_rows.pop(ref, 0))))
            out_parts.append(mean)
            if timings is not None:
                for k, v in tloc.items():
                    timings[k] = timings.get(k, 0.0) + float(v)

    ephem_mean = out_parts[0] if len(out_parts) == 1 else qv.concatenate(out_parts)
    cov_ll = _cov_ll_from_ephemeris_spherical_cov(ephem_mean)
    return ephem_mean, cov_ll

@dataclass(frozen=True)
class TargetPrediction:
    """
    Nominal target ephemeris plus reconstructed (lon,lat) covariance.
    """

    ephem: Ephemeris  # (N)
    cov_ll_deg2: np.ndarray  # (N,2,2)


def _cov_ll_from_ephemeris_spherical_cov(ephem: Ephemeris) -> np.ndarray:
    """
    Extract (lon,lat) 2x2 covariance (deg^2) from an Ephemeris' spherical 6x6 covariance.
    """
    if len(ephem) == 0:
        return np.zeros((0, 2, 2), dtype=np.float64)
    cov = ephem.coordinates.covariance.to_matrix().astype(np.float64, copy=False)
    # SphericalCoordinates.values ordering: [rho, lon, lat, vrho, vlon, vlat]
    c00 = cov[:, 1, 1]
    c01 = cov[:, 1, 2]
    c11 = cov[:, 2, 2]
    out = np.empty((cov.shape[0], 2, 2), dtype=np.float64)
    out[:, 0, 0] = c00
    out[:, 0, 1] = c01
    out[:, 1, 0] = c01
    out[:, 1, 1] = c11
    return out


def _mean_ephemeris_from_sigma_point_variant_0(*, variant_ephem) -> Ephemeris | None:
    """
    Build a mean Ephemeris from the sigma-point variant ephemeris by taking variant_id == "0".

    Notes
    -----
    - This avoids generating a separate "mean ephemeris" when the sigma-point variant set
      already includes the nominal (mean) state.
    - If the expected variant_id is not present or does not match the observer grid, callers
      should fall back to explicitly generating a mean ephemeris.
    """
    if variant_ephem is None or len(variant_ephem) == 0:
        return None
    try:
        m0 = pc.equal(
            pc.cast(variant_ephem.variant_id, pa.large_string()),
            pa.scalar("0", type=pa.large_string()),
        )
        t0 = variant_ephem.table.filter(m0)
        if t0.num_rows == 0:
            return None
        # Keep exactly the Ephemeris schema columns.
        cols = Ephemeris.empty().table.column_names
        keep = [c for c in cols if c in t0.column_names]
        if len(keep) != len(cols):
            return None
        t0 = t0.select(keep).sort_by(
            [
                ("coordinates.time.days", "ascending"),
                ("coordinates.time.nanos", "ascending"),
                ("coordinates.origin.code", "ascending"),
            ]
        )
        return Ephemeris.from_pyarrow(t0)
    except Exception:
        return None


def _generate_ephemeris_2body_helio_fast(
    *,
    propagated_orbits: Orbits,  # (N), SUN/ecliptic
    observers: Observers,  # (N), SUN/ecliptic
    times_utc: Timestamp,  # (N)
    max_processes: int | None,
) -> Ephemeris:
    """
    Fast path for 2-body ephemeris generation when inputs are already heliocentric (SUN origin)
    in ecliptic frame.

    This avoids the expensive SUN->SSB transforms inside `generate_ephemeris_2body` which dominate
    wall time for large N.
    """
    if len(propagated_orbits) != len(observers) or len(propagated_orbits) != len(times_utc):
        raise ValueError("propagated_orbits, observers, and times_utc must be aligned (same length).")
    if len(propagated_orbits) == 0:
        return Ephemeris.empty()

    compute_pred_mags = False
    phys = getattr(propagated_orbits, "physical_parameters", None)
    if phys is not None and hasattr(phys, "H_v") and hasattr(phys, "G"):
        try:
            H_arr = pc.cast(phys.H_v, pa.float64())
            G_arr = pc.cast(phys.G, pa.float64())
            has_params = pc.fill_null(pc.and_(pc.is_finite(H_arr), pc.is_finite(G_arr)), False)
            compute_pred_mags = bool(pc.any(has_params).as_py())
        except Exception:
            compute_pred_mags = False

    # Only safe if both coordinates are already in the same heliocentric inertial frame.
    try:
        o_origin = str(propagated_orbits.coordinates.origin.code[0].as_py())
        obs_origin = str(observers.coordinates.origin.code[0].as_py())
    except Exception:
        o_origin = ""
        obs_origin = ""
    if o_origin != "SUN" or obs_origin != "SUN" or str(propagated_orbits.coordinates.frame) != "ecliptic":
        # Fallback to the canonical implementation.
        return generate_ephemeris_2body(
            propagated_orbits,
            observers,
            predict_magnitudes=bool(compute_pred_mags),
            max_processes=max_processes,
        )

    try:
        from adam_core.dynamics.ephemeris import _generate_ephemeris_2body_vmap  # type: ignore
    except Exception:
        return generate_ephemeris_2body(
            propagated_orbits,
            observers,
            predict_magnitudes=bool(compute_pred_mags),
            max_processes=max_processes,
        )

    orbits_vals = np.asarray(propagated_orbits.coordinates.values, dtype=np.float64)
    obs_vals = np.asarray(observers.coordinates.values, dtype=np.float64)
    mu = np.asarray(observers.coordinates.origin.mu(), dtype=np.float64)
    times_mjd = times_utc.mjd().to_numpy(zero_copy_only=False).astype(np.float64)

    eph_sph, light_time, aberrated_orbits = _generate_ephemeris_2body_vmap(  # noqa: SLF001
        orbits_vals,
        times_mjd,
        obs_vals,
        mu,
        1e-10,  # lt_tol (match default)
        1000,  # max_iter (match default)
        1e-15,  # tol (match default)
        False,  # stellar_aberration (match default)
    )
    sph_np = np.asarray(eph_sph, dtype=np.float64)
    lt_np = np.asarray(light_time, dtype=np.float64)
    ab_np = np.asarray(aberrated_orbits, dtype=np.float64)

    n = int(len(propagated_orbits))
    emission_times = propagated_orbits.coordinates.time.add_fractional_days(pa.array(-lt_np))

    coords = SphericalCoordinates.from_kwargs(
        rho=sph_np[:, 0],
        lon=sph_np[:, 1],
        lat=sph_np[:, 2],
        vrho=sph_np[:, 3],
        vlon=sph_np[:, 4],
        vlat=sph_np[:, 5],
        time=propagated_orbits.coordinates.time,
        origin=Origin.from_kwargs(
            code=observers.code.to_numpy(zero_copy_only=False).astype(object)
        ),
        frame="ecliptic",
    )
    # Match `generate_ephemeris_2body` convention: output on-sky in equatorial frame.
    coords = transform_coordinates(coords, SphericalCoordinates, frame_out="equatorial")
    aberrated_coords = CartesianCoordinates.from_kwargs(
        x=ab_np[:, 0],
        y=ab_np[:, 1],
        z=ab_np[:, 2],
        vx=ab_np[:, 3],
        vy=ab_np[:, 4],
        vz=ab_np[:, 5],
        time=emission_times,
        origin=Origin.from_kwargs(code=np.full(n, OriginCodes.SUN.name)),
        frame="ecliptic",
    )

    pred_mag_v = pa.nulls(n, type=pa.float64())
    alpha = pa.nulls(n, type=pa.float64())
    if phys is not None and hasattr(phys, "H_v") and hasattr(phys, "G"):
        H_arr = pc.cast(phys.H_v, pa.float64())
        G_arr = pc.cast(phys.G, pa.float64())
        has_params = pc.fill_null(pc.and_(pc.is_finite(H_arr), pc.is_finite(G_arr)), False)
        if bool(pc.any(has_params).as_py()):
            nan = pa.scalar(np.nan, type=pa.float64())
            H_np = pc.fill_null(H_arr, nan).to_numpy(zero_copy_only=False).astype(np.float64)
            G_np = pc.fill_null(G_arr, nan).to_numpy(zero_copy_only=False).astype(np.float64)

            mags_v_np, alpha_np = calculate_apparent_magnitude_v_and_phase_angle(
                H_v=H_np,
                object_coords=aberrated_coords,
                observer=observers,
                G=G_np,
            )
            pred_mag_v = pc.if_else(
                has_params, pa.array(mags_v_np, type=pa.float64()), None
            )
            alpha = pc.if_else(has_params, pa.array(alpha_np, type=pa.float64()), None)

    return Ephemeris.from_kwargs(
        orbit_id=propagated_orbits.orbit_id,
        object_id=propagated_orbits.object_id,
        coordinates=coords,
        predicted_magnitude_v=pred_mag_v,
        alpha=alpha,
        light_time=pa.array(lt_np, type=pa.float64()),
        aberrated_coordinates=aberrated_coords,
    )


def _repair_orbit_covariance_psd_for_sampling(
    orbit: Orbits,
    *,
    abs_tol: float = 1e-15,
    rel_tol: float = 1e-10,
) -> Orbits:
    """
    Repair a single-orbit covariance to be PSD enough for downstream sampling.

    Policy:
    - symmetrize
    - if min eigenvalue is only slightly negative (within tolerance), clip to 0
    - otherwise raise
    - enforce symmetry again + add tiny diagonal jitter to avoid strict-absolute PSD checks
    """
    if len(orbit) != 1:
        raise ValueError("_repair_orbit_covariance_psd_for_sampling expects len(orbit)==1.")
    cov = orbit.coordinates.covariance
    if cov is None or cov.is_all_nan():
        raise ValueError("Orbit covariance is required but missing.")
    oid = ""
    try:
        oid = str(orbit.orbit_id[0].as_py())
    except Exception:
        oid = ""
    c_raw = np.asarray(cov.to_matrix()[0], dtype=np.float64)
    c_out, modified, reason = repair_covariance_matrix_psd_with_reason(
        c_raw, abs_tol=float(abs_tol), rel_tol=float(rel_tol)
    )
    if c_out is None:
        msg = "Orbit covariance is invalid for sampling"
        if reason is not None:
            msg += f" ({reason})"
        if oid:
            msg += f" orbit_id={oid}"
        raise ValueError(msg)
    if not modified:
        return orbit
    warnings.warn(
        f"Orbit covariance repaired to PSD for sampling orbit_id={oid}" if oid else "Orbit covariance repaired to PSD for sampling",
        RuntimeWarning,
        stacklevel=2,
    )
    cov_fixed = CoordinateCovariances.from_matrix(c_out[None, :, :])
    return orbit.set_column("coordinates.covariance", cov_fixed)


def _window_center_mjd(*, start_mjd: float, mjd: np.ndarray, window_size_days: int) -> np.ndarray:
    w = float(window_size_days)
    wid = np.floor((mjd - float(start_mjd)) / w).astype(np.int64)
    return float(start_mjd) + (wid.astype(np.float64) + 0.5) * w


def _circular_weighted_mean_lon_deg(*, lon_deg: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Circular weighted mean of longitude in degrees.

    Parameters
    ----------
    lon_deg
        Array shaped (K,N) in degrees.
    weights
        Array shaped (K,) (will be normalized to sum to 1).
    """
    lon = np.asarray(lon_deg, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if lon.ndim != 2:
        raise ValueError("lon_deg must be (K,N)")
    if w.shape[0] != lon.shape[0]:
        raise ValueError("weights must be shape (K,) matching lon_deg first dimension")
    s = float(np.sum(w))
    if not np.isfinite(s) or s == 0.0:
        w = np.full_like(w, 1.0 / float(w.shape[0]), dtype=np.float64)
    else:
        w = w / s
    w2 = w.reshape(-1, 1)
    rad = np.deg2rad(lon)
    s_sin = np.sum(w2 * np.sin(rad), axis=0)
    s_cos = np.sum(w2 * np.cos(rad), axis=0)
    return (np.degrees(np.arctan2(s_sin, s_cos)) + 360.0) % 360.0


def _sigma_point_variants_at_epoch(
    *, orbit_at_epoch: Orbits
) -> tuple[Orbits, np.ndarray, np.ndarray]:
    """
    Create sigma-point variants at a single epoch, returning them as `Orbits` plus covariance weights.
    """
    orbit_at_epoch = _repair_orbit_covariance_psd_for_sampling(orbit_at_epoch)
    variants = VariantOrbits.create(orbit_at_epoch, method="sigma-point")
    w_mean = variants.weights.to_numpy(zero_copy_only=False).astype(np.float64)
    w_cov = variants.weights_cov.to_numpy(zero_copy_only=False).astype(np.float64)
    # Convert VariantOrbits -> Orbits for `propagate_2body`.
    v_orbits = Orbits.from_kwargs(
        orbit_id=variants.orbit_id,
        object_id=variants.object_id,
        coordinates=variants.coordinates,
        physical_parameters=variants.physical_parameters,
    )
    # Defensive: keep variant count constant for JAX shape stability.
    #
    # If some variant states are non-finite (pathological covariances), we replace those
    # states with the nominal orbit state and set their covariance weight to zero. This
    # preserves K while preventing NaNs/Infs from poisoning propagation or covariance
    # reconstruction.
    vals = v_orbits.coordinates.values
    finite = np.isfinite(vals).all(axis=1)
    if not bool(np.all(finite)):
        nominal = orbit_at_epoch.coordinates.values[0]
        vals2 = np.asarray(vals, dtype=np.float64).copy()
        bad = np.nonzero(~finite)[0]
        vals2[bad, :] = nominal[None, :]
        # Rebuild coordinates with sanitized values (CartesianCoordinates.values is a property).
        coords = v_orbits.coordinates
        coords2 = CartesianCoordinates.from_kwargs(
            x=vals2[:, 0],
            y=vals2[:, 1],
            z=vals2[:, 2],
            vx=vals2[:, 3],
            vy=vals2[:, 4],
            vz=vals2[:, 5],
            time=coords.time,
            covariance=coords.covariance,
            origin=coords.origin,
            frame=coords.frame,
        )
        v_orbits = v_orbits.set_column("coordinates", coords2)
        w_mean = np.asarray(w_mean, dtype=np.float64).copy()
        w_cov = np.asarray(w_cov, dtype=np.float64).copy()
        w_mean[bad] = 0.0
        w_cov[bad] = 0.0
    return v_orbits, w_mean, w_cov


def _variant_cloud_lonlat_matrix(
    *,
    mean_ephem: Ephemeris,  # (N)
    variant_ephem,  # VariantEphemeris (K*N)
    n_variants: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Robustly reshape a variant ephemeris cloud to (K,N) lon/lat matrices.

    Why this exists
    ---------------
    ASSIST ephemeris output ordering is not guaranteed to be stable across implementations.
    We therefore avoid relying on `.sort_by('variant_id')` (lexicographic issues) or implicit
    ordering. Instead, we map each variant row to the corresponding observer index by
    matching (time, origin_code) against the nominal `mean_ephem` grid.
    """
    N = int(len(mean_ephem))
    K = int(n_variants)
    if N == 0 or K == 0:
        return np.zeros((K, N), dtype=np.float64), np.zeros((K, N), dtype=np.float64)

    mean_code = mean_ephem.coordinates.origin.code.to_numpy(zero_copy_only=False)
    mean_key = mean_ephem.coordinates.time.key(scale=None)
    mean_map: dict[tuple[str, int], int] = {
        (str(mean_code[i]), int(mean_key[i])): int(i) for i in range(N)
    }

    var_code = variant_ephem.coordinates.origin.code.to_numpy(zero_copy_only=False)
    var_key = variant_ephem.coordinates.time.key(scale=None)

    # Variant indices: "0".."12" for sigma points.
    var_id = variant_ephem.variant_id.to_numpy(zero_copy_only=False)
    v_idx = np.array([int(x) if x is not None else -1 for x in var_id], dtype=np.int64)
    if np.any((v_idx < 0) | (v_idx >= K)):
        raise RuntimeError("Variant ephemeris contained invalid variant_id values.")

    lon = variant_ephem.coordinates.lon.to_numpy(zero_copy_only=False).astype(np.float64)
    lat = variant_ephem.coordinates.lat.to_numpy(zero_copy_only=False).astype(np.float64)

    lon_m = np.full((K, N), np.nan, dtype=np.float64)
    lat_m = np.full((K, N), np.nan, dtype=np.float64)
    for r in range(int(len(variant_ephem))):
        o = mean_map.get((str(var_code[r]), int(var_key[r])))
        if o is None:
            continue
        lon_m[int(v_idx[r]), int(o)] = float(lon[r])
        lat_m[int(v_idx[r]), int(o)] = float(lat[r])

    if not np.isfinite(lon_m).all() or not np.isfinite(lat_m).all():
        raise RuntimeError("Failed to align variant ephemeris rows to the nominal grid.")
    return lon_m, lat_m


def _sigma_point_lonlat_cloud_2body_fast(
    *,
    propagated_variants: Orbits,  # (K*N) orbit-major
    observers: Observers,  # (N)
    times_utc: Timestamp,  # (N)
    n_variants: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute (K,N) lon/lat sample matrices without building a full Ephemeris table.

    This avoids the pyarrow/quivr object construction costs in `generate_ephemeris_2body`
    for the variant cloud (we only need lon/lat to reconstruct covariance).
    """
    try:
        # Private but stable in practice; falls back to the slower path if absent.
        from adam_core.dynamics.ephemeris import _generate_ephemeris_2body_vmap  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("Fast 2-body lon/lat path unavailable") from e

    K = int(n_variants)
    N = int(len(times_utc))
    if K <= 0 or N <= 0:
        return np.zeros((K, N), dtype=np.float64), np.zeros((K, N), dtype=np.float64)
    if int(len(propagated_variants)) != K * N:
        raise ValueError("propagated_variants must have length K*N (orbit-major ordering).")

    # Prefer fast heliocentric path when inputs are already SUN/ecliptic.
    try:
        o_origin = str(propagated_variants.coordinates.origin.code[0].as_py())
        obs_origin = str(observers.coordinates.origin.code[0].as_py())
    except Exception:
        o_origin = ""
        obs_origin = ""

    if o_origin == "SUN" and obs_origin == "SUN" and str(propagated_variants.coordinates.frame) == "ecliptic":
        orbits_vals = np.asarray(propagated_variants.coordinates.values, dtype=np.float64)
        obs_vals = np.asarray(observers.coordinates.values, dtype=np.float64)
        times_mjd = times_utc.mjd().to_numpy(zero_copy_only=False).astype(np.float64)
        mu = np.asarray(observers.coordinates.origin.mu(), dtype=np.float64)

        obs_rep = np.tile(obs_vals, (K, 1))
        times_rep = np.tile(times_mjd, K)
        mu_rep = np.tile(mu, K)
        eph_sph, _, _ = _generate_ephemeris_2body_vmap(  # noqa: SLF001
            orbits_vals,
            times_rep,
            obs_rep,
            mu_rep,
            1e-10,  # lt_tol (match default)
            1000,  # max_iter (match default)
            1e-15,  # tol (match default)
            False,  # stellar_aberration (match default)
        )
    else:
        # Fallback: transform to barycentric ecliptic, matching `generate_ephemeris_2body`'s convention.
        prop_bary = transform_coordinates(
            propagated_variants.coordinates,
            CartesianCoordinates,
            frame_out="ecliptic",
            origin_out=OriginCodes.SOLAR_SYSTEM_BARYCENTER,
        )
        obs_bary = transform_coordinates(
            observers.coordinates,
            CartesianCoordinates,
            frame_out="ecliptic",
            origin_out=OriginCodes.SOLAR_SYSTEM_BARYCENTER,
        )

        orbits_vals = np.asarray(prop_bary.values, dtype=np.float64)
        obs_vals = np.asarray(obs_bary.values, dtype=np.float64)
        times_mjd = times_utc.mjd().to_numpy(zero_copy_only=False).astype(np.float64)
        mu = np.asarray(obs_bary.origin.mu(), dtype=np.float64)

        obs_rep = np.tile(obs_vals, (K, 1))
        times_rep = np.tile(times_mjd, K)
        mu_rep = np.tile(mu, K)
        eph_sph, _, _ = _generate_ephemeris_2body_vmap(  # noqa: SLF001
            orbits_vals,
            times_rep,
            obs_rep,
            mu_rep,
            1e-10,  # lt_tol (match default)
            1000,  # max_iter (match default)
            1e-15,  # tol (match default)
            False,  # stellar_aberration (match default)
        )

    eph_np = np.asarray(eph_sph, dtype=np.float64)
    lon_s = eph_np[:, 1].reshape(K, N)
    lat_s = eph_np[:, 2].reshape(K, N)
    return lon_s, lat_s


def predict_targets(
    *,
    orbit: Orbits,
    obscode: pa.Array,
    times_utc: Timestamp,
    start_mjd: float,
    window_size_days: int,
    strategy: PropagationStrategy = "assist_window_then_2body_variants:sigma_points",
    max_processes: int | None = None,
    time_chunk_size: int = 0,
    observers_all: Observers | None = None,
    timings: dict[str, float] | None = None,
    default_mag_slope_G: float | None = None,
) -> TargetPrediction:
    """
    Predict nominal ephemerides and reconstructed covariances for frame-time targets.

    This function is designed to be called on moderately sized chunks to keep memory bounded.
    """
    if len(orbit) != 1:
        raise ValueError("predict_targets expects a single-orbit Orbits table (len==1).")
    if len(times_utc) == 0:
        return TargetPrediction(ephem=Ephemeris.empty(), cov_ll_deg2=np.zeros((0, 2, 2)))

    if default_mag_slope_G is not None:
        g0 = float(default_mag_slope_G)
        if not np.isfinite(g0):
            raise ValueError("default_mag_slope_G must be finite when provided.")
        # SBDB often provides H_v but not G for asteroids/TNOs. If H_v is present but G is missing,
        # explicitly fill G with a conventional default so magnitude-based filtering can run.
        try:
            phys = orbit.physical_parameters
            H_v = phys.H_v
            G = phys.G
            has_H = pc.fill_null(pc.is_finite(H_v), False)
            has_G = pc.fill_null(pc.is_finite(G), False)
            need = pc.and_(has_H, pc.invert(has_G))
            if bool(pc.any(need).as_py()):
                G2 = pc.if_else(
                    need,
                    pa.scalar(g0, type=pa.float64()),
                    pc.cast(G, pa.float64()),
                )
                orbit = orbit.set_column("physical_parameters.G", G2)
        except Exception:
            # If physical parameters are missing/unexpected, leave orbit unchanged.
            pass

    t = times_utc
    codes = pc.cast(pa.array(obscode), pa.large_string())
    if observers_all is not None and len(observers_all) != len(t):
        raise ValueError(
            "observers_all must be aligned to (obscode, times_utc). "
            f"Expected len={len(t)} but got len={len(observers_all)}."
        )

    if strategy == "assist_variants:sigma_points":
        use_cov = orbit.coordinates.covariance is not None and not orbit.coordinates.covariance.is_all_nan()
        if not use_cov:
            raise ValueError("assist_variants:sigma_points requires a finite orbit covariance.")

        if observers_all is not None:
            observers = observers_all
        else:
            t_obs0 = time.perf_counter() if timings is not None else None
            observers = Observers.from_codes(codes, t)
            if timings is not None and t_obs0 is not None:
                timings["stage2.observers_elapsed_s"] = timings.get("stage2.observers_elapsed_s", 0.0) + (
                    time.perf_counter() - t_obs0
                )
        assist = _assist_propagator()
        orbit = _repair_orbit_covariance_psd_for_sampling(orbit)
        t_var0 = time.perf_counter() if timings is not None else None
        variants = VariantOrbits.create(orbit, method="sigma-point")
        if timings is not None and t_var0 is not None:
            timings["stage2.sigma_point_variants_elapsed_s"] = timings.get(
                "stage2.sigma_point_variants_elapsed_s", 0.0
            ) + (time.perf_counter() - t_var0)
        t_ve0 = time.perf_counter() if timings is not None else None
        var_ephem = assist.generate_ephemeris(
            variants,
            observers,
            covariance=False,
            max_processes=_effective_max_processes(max_processes),
        )
        if timings is not None and t_ve0 is not None:
            timings["stage2.assist_variant_ephem_elapsed_s"] = timings.get(
                "stage2.assist_variant_ephem_elapsed_s", 0.0
            ) + (time.perf_counter() - t_ve0)
        # Idiomatic UT mean + covariance: collapse variants using weights/weights_cov.
        t_col0 = time.perf_counter() if timings is not None else None
        mean_ephem = VariantEphemeris.from_pyarrow(var_ephem.table).collapse_by_object_id(
            aberration_mode="none"
        )
        if timings is not None and t_col0 is not None:
            timings["stage2.variant_collapse_elapsed_s"] = timings.get(
                "stage2.variant_collapse_elapsed_s", 0.0
            ) + (time.perf_counter() - t_col0)
        cov_ll = _cov_ll_from_ephemeris_spherical_cov(mean_ephem)
        return TargetPrediction(ephem=mean_ephem, cov_ll_deg2=cov_ll)

    # Default: assist_window_then_2body_variants:sigma_points
    t_wc0 = time.perf_counter() if timings is not None else None
    mjd = t.mjd().to_numpy(zero_copy_only=False).astype(np.float64)
    center_mjd = _window_center_mjd(
        start_mjd=float(start_mjd), mjd=mjd, window_size_days=int(window_size_days)
    )
    center_times = Timestamp.from_mjd(pa.array(np.unique(center_mjd), type=pa.float64()), scale="utc")
    center_times = center_times.sort_by(["days", "nanos"])
    if timings is not None and t_wc0 is not None:
        timings["stage2.window_centers_elapsed_s"] = timings.get("stage2.window_centers_elapsed_s", 0.0) + (
            time.perf_counter() - t_wc0
        )

    assist = _assist_propagator()
    use_cov = orbit.coordinates.covariance is not None and not orbit.coordinates.covariance.is_all_nan()
    if not use_cov:
        raise ValueError("assist_window_then_2body_variants:sigma_points requires a finite orbit covariance.")

    if observers_all is None:
        t_obs0 = time.perf_counter() if timings is not None else None
        observers_all = Observers.from_codes(codes, t)
        if timings is not None and t_obs0 is not None:
            timings["stage2.observers_elapsed_s"] = timings.get("stage2.observers_elapsed_s", 0.0) + (
                time.perf_counter() - t_obs0
            )
    orbit = _repair_orbit_covariance_psd_for_sampling(orbit)
    # Propagate nominal orbit to each window center (covariance needed for variant sampling).
    t_as0 = time.perf_counter() if timings is not None else None
    orbit_at_centers = assist.propagate_orbits(
        orbit,
        center_times,
        covariance=True,
        max_processes=_effective_max_processes(max_processes),
    )
    if timings is not None and t_as0 is not None:
        timings["stage2.assist_propagate_centers_elapsed_s"] = timings.get(
            "stage2.assist_propagate_centers_elapsed_s", 0.0
        ) + (time.perf_counter() - t_as0)

    # Map each target time -> center index.
    center_mjd_unique = center_times.mjd().to_numpy(zero_copy_only=False).astype(np.float64)
    center_map = {float(x): int(i) for i, x in enumerate(center_mjd_unique.tolist())}
    center_idx = np.array([center_map[float(x)] for x in center_mjd.tolist()], dtype=np.int64)

    # For each center, process its targets in manageable chunks to bound K*N memory.
    idx_parts: list[np.ndarray] = []
    out_ephem_parts: list[Ephemeris] = []
    out_cov_parts: list[np.ndarray] = []

    # Resolve per-center chunk size (auto mode) based on RAM budget.
    #
    # Worst-case expansion is K sigma points per time target during variant propagation.
    if int(time_chunk_size) == 0:
        time_chunk_size, _meta = resolve_time_chunk_size_from_ram_budget(
            n_time_targets=int(len(t)),
            rows_per_time_target=13,
            ram_budget_frac=0.10,
            effective_bytes_per_row=200.0,
            min_chunk=1,
        )

    for c_i, _c_time in enumerate(center_times):
        hit = np.nonzero(center_idx == int(c_i))[0]
        if hit.size == 0:
            continue

        orbit_ref = orbit_at_centers.take([int(c_i)])
        t_sv0 = time.perf_counter() if timings is not None else None
        variants_orbits, _w_mean, w_cov = _sigma_point_variants_at_epoch(orbit_at_epoch=orbit_ref)
        if timings is not None and t_sv0 is not None:
            timings["stage2.sigma_point_variants_elapsed_s"] = timings.get(
                "stage2.sigma_point_variants_elapsed_s", 0.0
            ) + (time.perf_counter() - t_sv0)

        # Chunk the targets for this center by time to cap memory.
        #
        # Performance optimization: we batch multiple propagate_2body chunks and flush them
        # through a smaller number of ephemeris generation calls (reduces scheduling/object-store
        # overhead when generate_ephemeris_2body uses parallel execution).
        buf_idx: list[np.ndarray] = []
        buf_n: list[int] = []
        buf_prop_nom: list[Orbits] = []
        buf_obs: list[Observers] = []
        buf_lon_s: list[np.ndarray] = []
        buf_lat_s: list[np.ndarray] = []
        pending_rows = 0
        flush_max_rows = int(min(200_000, max(10_000, int(time_chunk_size) * 10)))

        def _flush_nominal_batches() -> None:
            nonlocal pending_rows
            if not buf_prop_nom:
                return

            prop_all = buf_prop_nom[0] if len(buf_prop_nom) == 1 else qv.concatenate(buf_prop_nom)
            obs_all = buf_obs[0] if len(buf_obs) == 1 else qv.concatenate(buf_obs)
            times_all = prop_all.coordinates.time

            t_en0 = time.perf_counter() if timings is not None else None
            ephem_all = _generate_ephemeris_2body_helio_fast(
                propagated_orbits=prop_all,
                observers=obs_all,
                times_utc=times_all,
                max_processes=max_processes,
            )
            if timings is not None and t_en0 is not None:
                timings["stage2.ephemeris_nominal_elapsed_s"] = timings.get(
                    "stage2.ephemeris_nominal_elapsed_s", 0.0
                ) + (time.perf_counter() - t_en0)

            offset = 0
            for idx_chunk, n_i, lon_s, lat_s in zip(buf_idx, buf_n, buf_lon_s, buf_lat_s, strict=True):
                # Slice ephemeris rows for this chunk.
                ephem_nom = Ephemeris.from_pyarrow(ephem_all.table.slice(int(offset), int(n_i)))
                offset += int(n_i)

                lon0 = ephem_nom.coordinates.lon.to_numpy(zero_copy_only=False).astype(np.float64)
                lat0 = ephem_nom.coordinates.lat.to_numpy(zero_copy_only=False).astype(np.float64)
                t_rc0 = time.perf_counter() if timings is not None else None
                cov_ll = reconstruct_cov_ll_from_sigma_point_cloud(
                    lon0_deg=lon0,
                    lat0_deg=lat0,
                    lon_samples_deg=lon_s,
                    lat_samples_deg=lat_s,
                    weights_cov=w_cov,
                )
                if timings is not None and t_rc0 is not None:
                    timings["stage2.reconstruct_cov_elapsed_s"] = timings.get(
                        "stage2.reconstruct_cov_elapsed_s", 0.0
                    ) + (time.perf_counter() - t_rc0)

                ephem_nom = attach_cov_ll_to_ephemeris(ephem=ephem_nom, cov_ll_deg2=cov_ll)
                idx_parts.append(np.asarray(idx_chunk, dtype=np.int64))
                out_ephem_parts.append(ephem_nom)
                out_cov_parts.append(cov_ll)

            buf_idx.clear()
            buf_n.clear()
            buf_prop_nom.clear()
            buf_obs.clear()
            buf_lon_s.clear()
            buf_lat_s.clear()
            pending_rows = 0

        for start in range(0, int(hit.size), int(time_chunk_size)):
            idx_chunk = hit[start : start + int(time_chunk_size)]
            times_chunk = t.take(idx_chunk.tolist())

            observers = observers_all.take(idx_chunk.tolist())

            # Nominal propagation via 2-body within the window (ephemeris is batched).
            t_pn0 = time.perf_counter() if timings is not None else None
            prop_nom = propagate_2body(
                orbit_ref, times_chunk, max_processes=_effective_max_processes(max_processes)
            )
            if timings is not None and t_pn0 is not None:
                timings["stage2.propagate2body_nominal_elapsed_s"] = timings.get(
                    "stage2.propagate2body_nominal_elapsed_s", 0.0
                ) + (time.perf_counter() - t_pn0)

            # Variant ephemeris: propagate K variants to each time and generate ephemeris.
            t_pv0 = time.perf_counter() if timings is not None else None
            prop_var = propagate_2body(
                variants_orbits, times_chunk, max_processes=_effective_max_processes(max_processes)
            )
            if timings is not None and t_pv0 is not None:
                timings["stage2.propagate2body_variants_elapsed_s"] = timings.get(
                    "stage2.propagate2body_variants_elapsed_s", 0.0
                ) + (time.perf_counter() - t_pv0)

            K = int(len(variants_orbits))
            N = int(len(times_chunk))
            t_ll0 = time.perf_counter() if timings is not None else None
            try:
                lon_s, lat_s = _sigma_point_lonlat_cloud_2body_fast(
                    propagated_variants=prop_var,
                    observers=observers,
                    times_utc=times_chunk,
                    n_variants=K,
                )
            except Exception:
                # Slow fallback: build full variant ephemeris table.
                if timings is not None:
                    timings["stage2.variant_lonlat_fallback_count"] = timings.get(
                        "stage2.variant_lonlat_fallback_count", 0.0
                    ) + 1.0
                rep_obs = qv.concatenate([observers] * K)
                ephem_var = generate_ephemeris_2body(
                    prop_var, rep_obs, predict_magnitudes=False, max_processes=max_processes
                )
                lon_s = (
                    ephem_var.coordinates.lon.to_numpy(zero_copy_only=False).astype(np.float64).reshape(K, N)
                )
                lat_s = (
                    ephem_var.coordinates.lat.to_numpy(zero_copy_only=False).astype(np.float64).reshape(K, N)
                )
            if timings is not None and t_ll0 is not None:
                timings["stage2.variant_lonlat_elapsed_s"] = timings.get(
                    "stage2.variant_lonlat_elapsed_s", 0.0
                ) + (time.perf_counter() - t_ll0)
            # Buffer until ephemeris flush.
            n_i = int(len(times_chunk))
            buf_idx.append(np.asarray(idx_chunk, dtype=np.int64))
            buf_n.append(int(n_i))
            buf_prop_nom.append(prop_nom)
            buf_obs.append(observers)
            buf_lon_s.append(np.asarray(lon_s, dtype=np.float64))
            buf_lat_s.append(np.asarray(lat_s, dtype=np.float64))
            pending_rows += int(n_i)
            if pending_rows >= int(flush_max_rows):
                _flush_nominal_batches()

        # Final flush for this center.
        _flush_nominal_batches()

    if not out_ephem_parts:
        return TargetPrediction(ephem=Ephemeris.empty(), cov_ll_deg2=np.zeros((0, 2, 2)))

    idx_all = np.concatenate(idx_parts, axis=0)
    ephem_all = qv.concatenate(out_ephem_parts)
    cov_all = np.concatenate(out_cov_parts, axis=0)
    order = np.argsort(idx_all)
    ephem_out = ephem_all.take(order.tolist())
    cov_out = cov_all[order]
    return TargetPrediction(ephem=ephem_out, cov_ll_deg2=cov_out)


def predict_targets_batched_assist_variants_sigma_points(
    *,
    orbits: Orbits,
    observers_utc: Observers,
    max_processes: int | None = None,
    orbit_chunk_size: int = 16,
    time_chunk_size: int = 0,
    aberration_mode: Literal["none", "collapse", "recompute"] = "none",
    default_mag_slope_G: float | None = None,
    timings: dict[str, float] | None = None,
) -> tuple[Ephemeris, np.ndarray]:
    """
    Batched sigma-point prediction for many orbits at once using:

    - VariantOrbits.create(method="sigma-point")
    - ASSISTPropagator.generate_ephemeris(variants, observers)
    - deterministic (orbit, variant, time) ordering + covariance reconstruction from sigma cloud

    This path is intended to be the *idiomatic + performant* implementation when callers have
    many orbits and many shared observer times/codes: it amortizes SPICE/kernel lookups and
    avoids per-orbit Python loops.

    Returns
    -------
    ephem_mean
        Ephemeris with one row per (orbit, observer) pair (UT mean + covariance).
    cov_ll_deg2
        Extracted (lon,lat) 2x2 covariance (deg^2), aligned to ephem_mean rows.
    """
    if len(orbits) == 0 or len(observers_utc) == 0:
        return Ephemeris.empty(), np.zeros((0, 2, 2), dtype=np.float64)

    if default_mag_slope_G is not None:
        g0 = float(default_mag_slope_G)
        if not np.isfinite(g0):
            raise ValueError("default_mag_slope_G must be finite when provided.")
        try:
            phys = orbits.physical_parameters
            H_v = phys.H_v
            G = phys.G
            has_H = pc.fill_null(pc.is_finite(H_v), False)
            has_G = pc.fill_null(pc.is_finite(G), False)
            need = pc.and_(has_H, pc.invert(has_G))
            if bool(pc.any(need).as_py()):
                G2 = pc.if_else(
                    need,
                    pa.scalar(g0, type=pa.float64()),
                    pc.cast(G, pa.float64()),
                )
                orbits = orbits.set_column("physical_parameters.G", G2)
        except Exception:
            pass

    orbit_chunk_size = int(orbit_chunk_size)
    if orbit_chunk_size <= 0:
        raise ValueError("orbit_chunk_size must be > 0")
    time_chunk_size = int(time_chunk_size)

    # Sigma-point count for 6D states is fixed (13), but we keep this as a constant here
    # so we can size time chunks without creating variants just to count.
    K_SIGMA = 13

    # Simple memory guard: cap variant ephemeris rows per chunk.
    # Rows per chunk ≈ (orbit_chunk_size * K_SIGMA) * time_chunk_size
    # If time_chunk_size==0, resolve chunk from RAM budget.
    if time_chunk_size == 0:
        rows_per_time = int(orbit_chunk_size * K_SIGMA)
        time_chunk_size, _meta = resolve_time_chunk_size_from_ram_budget(
            n_time_targets=int(len(observers_utc)),
            rows_per_time_target=int(rows_per_time),
            ram_budget_frac=0.10,
            # ASSIST variant ephemeris rows are much heavier than a minimal lon/lat table,
            # so use a more conservative bytes-per-row estimate to avoid building huge
            # intermediate tables (K*N) in memory.
            effective_bytes_per_row=2000.0,
            min_chunk=1,
        )
    if time_chunk_size < 0:
        # Disable chunking when explicitly requested (<0).
        time_chunk_size = int(len(observers_utc))

    assist = _assist_propagator()
    out_parts: list[Ephemeris] = []
    cov_parts: list[np.ndarray] = []

    # Chunk over orbits, then over observers (time targets).
    for o0 in range(0, int(len(orbits)), int(orbit_chunk_size)):
        o1 = int(min(int(len(orbits)), int(o0) + int(orbit_chunk_size)))
        orb = orbits.take(list(range(o0, o1)))

        t_var0 = time.perf_counter() if timings is not None else None
        variants = VariantOrbits.create(orb, method="sigma-point")
        # Identify the sigma-point variant that is exactly (or closest to) the nominal orbit
        # state for each orbit. This lets us reuse the single variant ephemeris propagation
        # for both the nominal prediction (mean orbit) and the covariance reconstruction,
        # without assuming a fixed sigma-point index.
        orb_ids = pc.cast(orb.orbit_id, pa.large_string()).to_pylist()
        orb_map = {str(oid): int(i) for i, oid in enumerate(orb_ids)}
        var_orb_ids = pc.cast(variants.orbit_id, pa.large_string()).to_pylist()
        var_to_orb = np.array([orb_map[str(x)] for x in var_orb_ids], dtype=np.int64)
        orb_state = np.asarray(orb.coordinates.values, dtype=np.float64)
        var_state = np.asarray(variants.coordinates.values, dtype=np.float64)
        if var_state.shape[0] != int(len(variants)) or orb_state.shape[0] != int(len(orb)):
            raise RuntimeError("Unexpected orbit/variant state array shapes.")
        d = var_state - orb_state[var_to_orb]
        dist2 = np.sum(d * d, axis=1)
        var_vid_int = pc.cast(pc.cast(variants.variant_id, pa.large_string()), pa.int64()).to_numpy(
            zero_copy_only=False
        ).astype(np.int64)
        mean_vid_int_by_orb = np.full(int(len(orb)), -1, dtype=np.int64)
        for i in range(int(len(orb))):
            m = var_to_orb == i
            if not np.any(m):
                raise RuntimeError("Missing sigma-point variants for an orbit chunk.")
            j = int(np.argmin(np.where(m, dist2, np.inf)))
            mean_vid_int_by_orb[i] = int(var_vid_int[j])
        if np.any(mean_vid_int_by_orb < 0):
            raise RuntimeError("Failed to identify nominal sigma-point variant per orbit.")
        mean_vid_int_by_orbit_id = {str(orb_ids[i]): int(mean_vid_int_by_orb[i]) for i in range(int(len(orb)))}
        if timings is not None and t_var0 is not None:
            timings["stage2.sigma_point_variants_elapsed_s"] = timings.get(
                "stage2.sigma_point_variants_elapsed_s", 0.0
            ) + (time.perf_counter() - t_var0)

        for t0 in range(0, int(len(observers_utc)), int(time_chunk_size)):
            t1 = int(min(int(len(observers_utc)), int(t0) + int(time_chunk_size)))
            obs = observers_utc.take(list(range(t0, t1)))

            compute_mag = bool(default_mag_slope_G is not None)
            assist_max_processes = (
                None if max_processes is None else max(1, int(max_processes))
            )
            t_ev0 = time.perf_counter() if timings is not None else None
            var_ephem = assist.generate_ephemeris(
                variants,
                obs,
                covariance=False,
                max_processes=assist_max_processes,
                predict_magnitudes=compute_mag,
                predict_phase_angle=False,
            )
            if timings is not None and t_ev0 is not None:
                timings["stage2.ephemeris_variants_elapsed_s"] = timings.get(
                    "stage2.ephemeris_variants_elapsed_s", 0.0
                ) + (time.perf_counter() - t_ev0)

            # Normalize to deterministic numeric ordering by (orbit, variant, time, origin) before reshape.
            t_c0 = time.perf_counter() if timings is not None else None

            # Ensure deterministic numeric ordering by variant_id within each orbit.
            # ASSIST often emits `variant_id` as strings and may order them lexicographically,
            # which interleaves different orbits once ids reach multiple digits (e.g. "130" < "14").
            v_tbl = var_ephem.table
            v_tbl = v_tbl.append_column("_variant_id_int", pc.cast(pc.cast(v_tbl["variant_id"], pa.large_string()), pa.int64()))
            coord = v_tbl["coordinates"]
            t_struct = pc.struct_field(coord, "time")
            v_tbl = v_tbl.append_column("_time_days", pc.cast(pc.struct_field(t_struct, "days"), pa.int64()))
            v_tbl = v_tbl.append_column("_time_nanos", pc.cast(pc.struct_field(t_struct, "nanos"), pa.int64()))
            o_struct = pc.struct_field(coord, "origin")
            v_tbl = v_tbl.append_column(
                "_origin_code",
                pc.cast(pc.struct_field(o_struct, "code"), pa.large_string()),
            )
            v_tbl = v_tbl.sort_by(
                [
                    ("orbit_id", "ascending"),
                    ("_variant_id_int", "ascending"),
                    ("_time_days", "ascending"),
                    ("_time_nanos", "ascending"),
                    ("_origin_code", "ascending"),
                ]
            ).drop(["_variant_id_int", "_time_days", "_time_nanos", "_origin_code"])
            v = VariantEphemeris.from_pyarrow(v_tbl)
            n_orb = int(len(orb))
            K = int(len(variants)) // int(len(orb))
            N = int(len(obs))
            if int(len(v)) != int(n_orb * K * N):
                raise RuntimeError(
                    "Unexpected VariantEphemeris length for batched assist_variants. "
                    f"expected={int(n_orb*K*N)} got={int(len(v))}"
                )

            lon = (
                v.coordinates.lon.to_numpy(zero_copy_only=False)
                .astype(np.float64)
                .reshape(n_orb, K, N)
            )
            lat = (
                v.coordinates.lat.to_numpy(zero_copy_only=False)
                .astype(np.float64)
                .reshape(n_orb, K, N)
            )
            w_cov = (
                v.weights_cov.to_numpy(zero_copy_only=False)
                .astype(np.float64)
                .reshape(n_orb, K, N)
            )

            # Nominal lon/lat per (orbit,time): select the sigma point corresponding to the
            # nominal orbit state at the orbit epoch.
            vid_int = pc.cast(pc.cast(v.variant_id, pa.large_string()), pa.int64()).to_numpy(
                zero_copy_only=False
            ).astype(np.int64).reshape(n_orb, K, N)
            vid0 = vid_int[:, :, 0]
            k0 = np.full(n_orb, -1, dtype=np.int64)
            orbit_id0 = (
                pc.cast(v.orbit_id, pa.large_string())
                .to_numpy(zero_copy_only=False)
                .astype(object)
                .reshape(n_orb, K, N)[:, 0, 0]
            )
            for i in range(n_orb):
                want = mean_vid_int_by_orbit_id.get(str(orbit_id0[i]))
                if want is None:
                    raise RuntimeError("Missing nominal sigma-point mapping for an orbit_id in ephemeris.")
                hit = np.nonzero(vid0[i] == int(want))[0]
                if hit.size != 1:
                    raise RuntimeError("Failed to locate nominal sigma-point ephemeris slice.")
                k0[i] = int(hit[0])
            lon0 = np.stack([lon[i, int(k0[i]), :] for i in range(n_orb)], axis=0)
            lat0 = np.stack([lat[i, int(k0[i]), :] for i in range(n_orb)], axis=0)

            # Covariance in tangent plane per (orbit,time).
            cov_ll = np.empty((n_orb * N, 2, 2), dtype=np.float64)
            for i in range(n_orb):
                cov_ll[i * N : (i + 1) * N] = reconstruct_cov_ll_from_sigma_point_cloud(
                    lon0_deg=lon0[i],
                    lat0_deg=lat0[i],
                    lon_samples_deg=lon[i],
                    lat_samples_deg=lat[i],
                    weights_cov=w_cov[i, :, 0],
                )

            # Predicted magnitude: take the nominal sigma point's predicted magnitude.
            pred_mag_v = None
            if compute_mag:
                mags = v.predicted_magnitude_v.to_numpy(zero_copy_only=False).astype(np.float64).reshape(
                    n_orb, K, N
                )
                pred_mag_v = np.stack([mags[i, int(k0[i]), :] for i in range(n_orb)], axis=0)

            # Build mean Ephemeris rows aligned to the (time, origin_code) grid used by ASSIST.
            days = (
                v.coordinates.time.days.to_numpy(zero_copy_only=False)
                .astype(np.int64)
                .reshape(n_orb, K, N)[:, 0, :]
                .reshape(-1)
            )
            nanos = (
                v.coordinates.time.nanos.to_numpy(zero_copy_only=False)
                .astype(np.int64)
                .reshape(n_orb, K, N)[:, 0, :]
                .reshape(-1)
            )
            out_time = Timestamp.from_kwargs(days=pa.array(days, pa.int64()), nanos=pa.array(nanos, pa.int64()), scale=v.coordinates.time.scale)
            origin_code = (
                v.coordinates.origin.code.to_numpy(zero_copy_only=False)
                .reshape(n_orb, K, N)[:, 0, :]
                .astype(object)
                .reshape(-1)
            )
            out_origin = Origin.from_kwargs(code=pa.array(origin_code, type=pa.large_string()))
            orbit_id = pc.cast(v.orbit_id, pa.large_string()).to_numpy(zero_copy_only=False).astype(object).reshape(n_orb, K, N)[:, 0, :].reshape(-1)
            object_id = pc.cast(v.object_id, pa.large_string()).to_numpy(zero_copy_only=False).astype(object).reshape(n_orb, K, N)[:, 0, :].reshape(-1)
            coords = SphericalCoordinates.from_kwargs(
                rho=np.full(n_orb * N, np.nan, dtype=np.float64),
                lon=lon0.reshape(-1),
                lat=lat0.reshape(-1),
                vrho=np.full(n_orb * N, np.nan, dtype=np.float64),
                vlon=np.full(n_orb * N, np.nan, dtype=np.float64),
                vlat=np.full(n_orb * N, np.nan, dtype=np.float64),
                time=out_time,
                origin=out_origin,
                frame=v.coordinates.frame,
                covariance=CoordinateCovariances.nulls(n_orb * N),
            )
            mean = Ephemeris.from_kwargs(
                orbit_id=pa.array(orbit_id, type=pa.large_string()),
                object_id=pa.array(object_id, type=pa.large_string()),
                coordinates=coords,
                predicted_magnitude_v=(None if pred_mag_v is None else pred_mag_v.reshape(-1).tolist()),
            )

            if timings is not None and t_c0 is not None:
                timings["stage2.variant_collapse_elapsed_s"] = timings.get(
                    "stage2.variant_collapse_elapsed_s", 0.0
                ) + (time.perf_counter() - t_c0)

            out_parts.append(mean)
            cov_parts.append(cov_ll)

    ephem_mean = out_parts[0] if len(out_parts) == 1 else qv.concatenate(out_parts)
    cov_ll = cov_parts[0] if len(cov_parts) == 1 else np.concatenate(cov_parts, axis=0)
    return ephem_mean, cov_ll


def predict_observations_in_frame(
    *,
    orbit: Orbits,
    obscode: str,
    times_utc: Timestamp,
    strategy: PropagationStrategy,
    max_processes: int | None = None,
) -> TargetPrediction:
    """
    Predict nominal ephemerides + reconstructed covariance at per-observation timestamps.

    This is used in the frame-level detection gate.
    """
    # For per-frame prediction we use a single reference epoch (mean of observation times) to
    # amortize ASSIST propagation costs, then do 2-body within that small time span.
    if len(orbit) != 1:
        raise ValueError("predict_observations_in_frame expects a single-orbit Orbits table (len==1).")
    if len(times_utc) == 0:
        return TargetPrediction(ephem=Ephemeris.empty(), cov_ll_deg2=np.zeros((0, 2, 2)))

    if strategy == "assist_variants:sigma_points":
        # Direct-assist variant ephemerides at observation times.
        assist = _assist_propagator()
        obs = Observers.from_code(obscode, times_utc)
        variants = VariantOrbits.create(orbit, method="sigma-point")
        var_ephem = assist.generate_ephemeris(
            variants,
            obs,
            covariance=False,
            max_processes=_effective_max_processes(max_processes),
        )
        mean_ephem = VariantEphemeris.from_pyarrow(var_ephem.table).collapse_by_object_id(
            aberration_mode="none"
        )
        cov_ll = _cov_ll_from_ephemeris_spherical_cov(mean_ephem)
        return TargetPrediction(ephem=mean_ephem, cov_ll_deg2=cov_ll)

    # Window-then-2body variants, but for a single frame we choose reference epoch = mean time.
    mjd = times_utc.mjd().to_numpy(zero_copy_only=False).astype(np.float64)
    t_ref = Timestamp.from_mjd([float(np.mean(mjd))], scale="utc")
    assist = _assist_propagator()
    orbit_ref = assist.propagate_orbits(
        orbit,
        t_ref,
        covariance=True,
        max_processes=_effective_max_processes(max_processes),
    )
    variants_orbits, _w_mean, w_cov = _sigma_point_variants_at_epoch(orbit_at_epoch=orbit_ref)

    obs = Observers.from_code(obscode, times_utc)
    prop_nom = propagate_2body(
        orbit_ref, times_utc, max_processes=_effective_max_processes(max_processes)
    )
    eph_nom = _generate_ephemeris_2body_helio_fast(
        propagated_orbits=prop_nom,
        observers=obs,
        times_utc=times_utc,
        max_processes=max_processes,
    )

    prop_var = propagate_2body(
        variants_orbits, times_utc, max_processes=_effective_max_processes(max_processes)
    )
    K = int(len(variants_orbits))
    N = int(len(times_utc))
    try:
        lon_s, lat_s = _sigma_point_lonlat_cloud_2body_fast(
            propagated_variants=prop_var,
            observers=obs,
            times_utc=times_utc,
            n_variants=K,
        )
    except Exception:
        # Slow fallback: build full variant ephemeris table.
        rep_obs = qv.concatenate([obs] * K)
        eph_var = generate_ephemeris_2body(
            prop_var, rep_obs, predict_magnitudes=False, max_processes=max_processes
        )
        lon_s = eph_var.coordinates.lon.to_numpy(zero_copy_only=False).astype(np.float64).reshape(K, N)
        lat_s = eph_var.coordinates.lat.to_numpy(zero_copy_only=False).astype(np.float64).reshape(K, N)
    lon0 = eph_nom.coordinates.lon.to_numpy(zero_copy_only=False).astype(np.float64)
    lat0 = eph_nom.coordinates.lat.to_numpy(zero_copy_only=False).astype(np.float64)
    cov_ll = reconstruct_cov_ll_from_sigma_point_cloud(
        lon0_deg=lon0, lat0_deg=lat0, lon_samples_deg=lon_s, lat_samples_deg=lat_s, weights_cov=w_cov
    )
    eph_nom = attach_cov_ll_to_ephemeris(ephem=eph_nom, cov_ll_deg2=cov_ll)
    return TargetPrediction(ephem=eph_nom, cov_ll_deg2=cov_ll)
