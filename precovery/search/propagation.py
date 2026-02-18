from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import quivr as qv
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
from adam_core.orbits.variants import VariantOrbits
from adam_core.photometry.magnitude import calculate_apparent_magnitude_v_and_phase_angle
from adam_core.time import Timestamp

from .reconstruction import reconstruct_cov_ll_from_sigma_point_cloud
from .covariance import attach_cov_ll_to_ephemeris
from .covariance_psd import repair_covariance_matrix_psd


PropagationStrategy = Literal[
    "assist_window_then_2body_variants:sigma_points",
    "assist_variants:sigma_points",
]


@dataclass(frozen=True)
class TargetPrediction:
    """
    Nominal target ephemeris plus reconstructed (lon,lat) covariance.
    """

    ephem: Ephemeris  # (N)
    cov_ll_deg2: np.ndarray  # (N,2,2)


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
    if phys is not None and hasattr(phys, "H_v"):
        try:
            H_arr = pc.cast(phys.H_v, pa.float64())
            compute_pred_mags = not pc.all(pc.is_null(H_arr)).as_py()
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
    if phys is not None and hasattr(phys, "H_v"):
        H_arr = pc.cast(phys.H_v, pa.float64())
        if not pc.all(pc.is_null(H_arr)).as_py():
            # Fill missing H with NaN; we'll mask back to null after compute.
            nan = pa.scalar(np.nan, type=pa.float64())
            H_np = pc.fill_null(H_arr, nan).to_numpy(zero_copy_only=False).astype(np.float64)

            if hasattr(phys, "G"):
                G_arr = pc.cast(phys.G, pa.float64())
                G_np = pc.fill_null(G_arr, pa.scalar(0.15, type=pa.float64())).to_numpy(
                    zero_copy_only=False
                ).astype(np.float64)
            else:
                G_np = np.full(n, 0.15, dtype=np.float64)

            mags_v_np, alpha_np = calculate_apparent_magnitude_v_and_phase_angle(
                H_v=H_np,
                object_coords=aberrated_coords,
                observer=observers,
                G=G_np,
            )
            valid_h = pc.is_valid(H_arr)
            pred_mag_v = pc.if_else(
                valid_h, pa.array(mags_v_np, type=pa.float64()), None
            )
            alpha = pc.if_else(valid_h, pa.array(alpha_np, type=pa.float64()), None)

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
    c_raw = np.asarray(cov.to_matrix()[0], dtype=np.float64)
    c_out, modified = repair_covariance_matrix_psd(c_raw, abs_tol=float(abs_tol), rel_tol=float(rel_tol))
    if c_out is None:
        raise ValueError("Orbit covariance is too non-PSD for repair.")
    if not modified:
        return orbit
    cov_fixed = CoordinateCovariances.from_matrix(c_out[None, :, :])
    return orbit.set_column("coordinates.covariance", cov_fixed)


def _window_center_mjd(*, start_mjd: float, mjd: np.ndarray, window_size_days: int) -> np.ndarray:
    w = float(window_size_days)
    wid = np.floor((mjd - float(start_mjd)) / w).astype(np.int64)
    return float(start_mjd) + (wid.astype(np.float64) + 0.5) * w


def _sigma_point_variants_at_epoch(*, orbit_at_epoch: Orbits) -> tuple[Orbits, np.ndarray]:
    """
    Create sigma-point variants at a single epoch, returning them as `Orbits` plus covariance weights.
    """
    orbit_at_epoch = _repair_orbit_covariance_psd_for_sampling(orbit_at_epoch)
    variants = VariantOrbits.create(orbit_at_epoch, method="sigma-point")
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
        w_cov = np.asarray(w_cov, dtype=np.float64).copy()
        w_cov[bad] = 0.0
    return v_orbits, w_cov


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

    mean_days = mean_ephem.coordinates.time.days.to_numpy(zero_copy_only=False).astype(np.int64)
    mean_nanos = mean_ephem.coordinates.time.nanos.to_numpy(zero_copy_only=False).astype(np.int64)
    mean_code = mean_ephem.coordinates.origin.code.to_numpy(zero_copy_only=False)

    nanos_in_day = 86_400_000_000_000
    mean_key = mean_days * nanos_in_day + mean_nanos
    mean_map: dict[tuple[str, int], int] = {
        (str(mean_code[i]), int(mean_key[i])): int(i) for i in range(N)
    }

    var_days = variant_ephem.coordinates.time.days.to_numpy(zero_copy_only=False).astype(np.int64)
    var_nanos = variant_ephem.coordinates.time.nanos.to_numpy(zero_copy_only=False).astype(np.int64)
    var_code = variant_ephem.coordinates.origin.code.to_numpy(zero_copy_only=False)
    var_key = var_days * nanos_in_day + var_nanos

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
    time_chunk_size: int = 2000,
    observers_all: Observers | None = None,
) -> TargetPrediction:
    """
    Predict nominal ephemerides and reconstructed covariances for frame-time targets.

    This function is designed to be called on moderately sized chunks to keep memory bounded.
    """
    if len(orbit) != 1:
        raise ValueError("predict_targets expects a single-orbit Orbits table (len==1).")
    if len(times_utc) == 0:
        return TargetPrediction(ephem=Ephemeris.empty(), cov_ll_deg2=np.zeros((0, 2, 2)))

    codes = pa.array(obscode, type=pa.large_string())
    t = times_utc

    if strategy == "assist_variants:sigma_points":
        assist = ASSISTPropagator()
        observers = observers_all if observers_all is not None else Observers.from_codes(codes, t)
        use_cov = orbit.coordinates.covariance is not None and not orbit.coordinates.covariance.is_all_nan()
        if not use_cov:
            raise ValueError("assist_variants:sigma_points requires a finite orbit covariance.")

        orbit = _repair_orbit_covariance_psd_for_sampling(orbit)
        # We reconstruct sky covariance from sigma-point variants; avoid internal covariance
        # propagation (which can trigger strict PSD checks and additional sampling cost).
        mean_ephem = assist.generate_ephemeris(orbit, observers, covariance=False, max_processes=max_processes)
        variants = VariantOrbits.create(orbit, method="sigma-point")
        w_cov = variants.weights_cov.to_numpy(zero_copy_only=False).astype(np.float64)
        var_ephem = assist.generate_ephemeris(variants, observers, covariance=False, max_processes=max_processes)

        K = int(len(variants))
        N = int(len(t))
        if K * N != int(len(var_ephem)):
            raise RuntimeError(
                "Unexpected variant ephemeris shape: expected K*N rows. "
                f"K={K} N={N} got={int(len(var_ephem))}"
            )
        lon_s, lat_s = _variant_cloud_lonlat_matrix(
            mean_ephem=mean_ephem, variant_ephem=var_ephem, n_variants=K
        )
        lon0 = mean_ephem.coordinates.lon.to_numpy(zero_copy_only=False).astype(np.float64)
        lat0 = mean_ephem.coordinates.lat.to_numpy(zero_copy_only=False).astype(np.float64)
        cov_ll = reconstruct_cov_ll_from_sigma_point_cloud(
            lon0_deg=lon0, lat0_deg=lat0, lon_samples_deg=lon_s, lat_samples_deg=lat_s, weights_cov=w_cov
        )
        mean_ephem = attach_cov_ll_to_ephemeris(ephem=mean_ephem, cov_ll_deg2=cov_ll)
        return TargetPrediction(ephem=mean_ephem, cov_ll_deg2=cov_ll)

    # Default: assist_window_then_2body_variants:sigma_points
    mjd = t.mjd().to_numpy(zero_copy_only=False).astype(np.float64)
    center_mjd = _window_center_mjd(
        start_mjd=float(start_mjd), mjd=mjd, window_size_days=int(window_size_days)
    )
    center_times = Timestamp.from_mjd(pa.array(np.unique(center_mjd), type=pa.float64()), scale="utc")
    center_times = center_times.sort_by(["days", "nanos"])

    assist = ASSISTPropagator()
    use_cov = orbit.coordinates.covariance is not None and not orbit.coordinates.covariance.is_all_nan()
    if not use_cov:
        raise ValueError("assist_window_then_2body_variants:sigma_points requires a finite orbit covariance.")

    orbit = _repair_orbit_covariance_psd_for_sampling(orbit)
    # Propagate nominal orbit to each window center (covariance needed for variant sampling).
    orbit_at_centers = assist.propagate_orbits(orbit, center_times, covariance=True, max_processes=max_processes)

    # Map each target time -> center index.
    center_mjd_unique = center_times.mjd().to_numpy(zero_copy_only=False).astype(np.float64)
    center_map = {float(x): int(i) for i, x in enumerate(center_mjd_unique.tolist())}
    center_idx = np.array([center_map[float(x)] for x in center_mjd.tolist()], dtype=np.int64)

    # For each center, process its targets in manageable chunks to bound K*N memory.
    idx_parts: list[np.ndarray] = []
    out_ephem_parts: list[Ephemeris] = []
    out_cov_parts: list[np.ndarray] = []

    for c_i, c_time in enumerate(center_times):
        hit = np.nonzero(center_idx == int(c_i))[0]
        if hit.size == 0:
            continue

        orbit_ref = orbit_at_centers.take([int(c_i)])
        variants_orbits, w_cov = _sigma_point_variants_at_epoch(
            orbit_at_epoch=orbit_at_centers.take([int(c_i)])
        )

        # Chunk the targets for this center by time to cap memory.
        for start in range(0, int(hit.size), int(time_chunk_size)):
            idx_chunk = hit[start : start + int(time_chunk_size)]
            codes_chunk = pc.take(codes, pa.array(idx_chunk, type=pa.int64()))
            times_chunk = t.take(idx_chunk.tolist())

            observers = (
                observers_all.take(idx_chunk.tolist())
                if observers_all is not None
                else Observers.from_codes(codes_chunk, times_chunk)
            )

            # Nominal ephem via 2-body within the window.
            prop_nom = propagate_2body(orbit_ref, times_chunk, max_processes=max_processes)
            ephem_nom = _generate_ephemeris_2body_helio_fast(
                propagated_orbits=prop_nom,
                observers=observers,
                times_utc=times_chunk,
                max_processes=max_processes,
            )

            # Variant ephemeris: propagate K variants to each time and generate ephemeris.
            prop_var = propagate_2body(variants_orbits, times_chunk, max_processes=max_processes)

            K = int(len(variants_orbits))
            N = int(len(times_chunk))
            try:
                lon_s, lat_s = _sigma_point_lonlat_cloud_2body_fast(
                    propagated_variants=prop_var,
                    observers=observers,
                    times_utc=times_chunk,
                    n_variants=K,
                )
            except Exception:
                # Slow fallback: build full variant ephemeris table.
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
            lon0 = ephem_nom.coordinates.lon.to_numpy(zero_copy_only=False).astype(np.float64)
            lat0 = ephem_nom.coordinates.lat.to_numpy(zero_copy_only=False).astype(np.float64)
            cov_ll = reconstruct_cov_ll_from_sigma_point_cloud(
                lon0_deg=lon0, lat0_deg=lat0, lon_samples_deg=lon_s, lat_samples_deg=lat_s, weights_cov=w_cov
            )
            ephem_nom = attach_cov_ll_to_ephemeris(ephem=ephem_nom, cov_ll_deg2=cov_ll)

            idx_parts.append(np.asarray(idx_chunk, dtype=np.int64))
            out_ephem_parts.append(ephem_nom)
            out_cov_parts.append(cov_ll)

    if not out_ephem_parts:
        return TargetPrediction(ephem=Ephemeris.empty(), cov_ll_deg2=np.zeros((0, 2, 2)))

    idx_all = np.concatenate(idx_parts, axis=0)
    ephem_all = qv.concatenate(out_ephem_parts)
    cov_all = np.concatenate(out_cov_parts, axis=0)
    order = np.argsort(idx_all)
    ephem_out = ephem_all.take(order.tolist())
    cov_out = cov_all[order]
    return TargetPrediction(ephem=ephem_out, cov_ll_deg2=cov_out)


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
        assist = ASSISTPropagator()
        obs = Observers.from_code(obscode, times_utc)
        mean_ephem = assist.generate_ephemeris(orbit, obs, covariance=True, max_processes=max_processes)
        variants = VariantOrbits.create(orbit, method="sigma-point")
        w_cov = variants.weights_cov.to_numpy(zero_copy_only=False).astype(np.float64)
        var_ephem = assist.generate_ephemeris(variants, obs, covariance=False, max_processes=max_processes)
        K = int(len(variants))
        N = int(len(times_utc))
        if K * N != int(len(var_ephem)):
            raise RuntimeError(
                f"Unexpected variant ephemeris shape: expected K*N rows. K={K} N={N} got={int(len(var_ephem))}"
            )
        lon_s, lat_s = _variant_cloud_lonlat_matrix(
            mean_ephem=mean_ephem, variant_ephem=var_ephem, n_variants=K
        )
        lon0 = mean_ephem.coordinates.lon.to_numpy(zero_copy_only=False).astype(np.float64)
        lat0 = mean_ephem.coordinates.lat.to_numpy(zero_copy_only=False).astype(np.float64)
        cov_ll = reconstruct_cov_ll_from_sigma_point_cloud(
            lon0_deg=lon0, lat0_deg=lat0, lon_samples_deg=lon_s, lat_samples_deg=lat_s, weights_cov=w_cov
        )
        mean_ephem = attach_cov_ll_to_ephemeris(ephem=mean_ephem, cov_ll_deg2=cov_ll)
        return TargetPrediction(ephem=mean_ephem, cov_ll_deg2=cov_ll)

    # Window-then-2body variants, but for a single frame we choose reference epoch = mean time.
    mjd = times_utc.mjd().to_numpy(zero_copy_only=False).astype(np.float64)
    t_ref = Timestamp.from_mjd([float(np.mean(mjd))], scale="utc")
    assist = ASSISTPropagator()
    orbit_ref = assist.propagate_orbits(orbit, t_ref, covariance=True, max_processes=max_processes)
    variants_orbits, w_cov = _sigma_point_variants_at_epoch(orbit_at_epoch=orbit_ref)

    obs = Observers.from_code(obscode, times_utc)
    prop_nom = propagate_2body(orbit_ref, times_utc, max_processes=max_processes)
    eph_nom = _generate_ephemeris_2body_helio_fast(
        propagated_orbits=prop_nom,
        observers=obs,
        times_utc=times_utc,
        max_processes=max_processes,
    )

    prop_var = propagate_2body(variants_orbits, times_utc, max_processes=max_processes)
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

