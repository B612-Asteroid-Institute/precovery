import logging
import os
from typing import Any, Optional, Tuple, Type, Union, overload

import healpy as hp
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import quivr as qv
import ray
from adam_core.coordinates import CoordinateCovariances
from adam_core.coordinates.cartesian import CartesianCoordinates
from adam_core.coordinates.origin import Origin, OriginCodes
from adam_core.coordinates.residuals import Residuals
from adam_core.coordinates.spherical import SphericalCoordinates
from adam_core.coordinates.transform import transform_coordinates
from adam_core.dynamics.ephemeris import generate_ephemeris_2body
from adam_core.dynamics.propagation import propagate_2body
from adam_core.observations import Exposures, PointSourceDetections
from adam_core.observers import Observers
from adam_core.orbits import Orbits
from adam_core.orbits.ephemeris import Ephemeris
from adam_core.photometry.bandpasses import (
    bandpass_delta_mag,
    map_to_canonical_filter_bands,
)
from adam_core.photometry.magnitude import predict_magnitudes
from adam_core.propagator import Propagator
from adam_core.ray_cluster import initialize_use_ray
from adam_core.time import Timestamp

try:
    from ._version import __version__
except ImportError:
    __version__ = "unknown"
from .config import Config, DefaultConfig
from .filter_limiting_magnitudes import FilterLimitingMagnitudes
from .frame_db import FrameDB, FrameIndex, GenericFrame, HealpixFrame, WindowCenters
from .healpix_geom import radec_to_healpixel
from .observation import ObservationsTable
from .spherical_geom import haversine_distance_deg

DEGREE = 1.0
ARCMIN = DEGREE / 60
ARCSEC = ARCMIN / 60

CANDIDATE_K = 15
CANDIDATE_NSIDE = 2**CANDIDATE_K

logging.basicConfig()
logger = logging.getLogger("precovery")

REJECT_REASON_LIMITING_MAGNITUDE = "limiting_magnitude"
REJECT_REASON_MAG_RESIDUAL = "mag_residual"


class PrecoveryCandidates(qv.Table):

    time = Timestamp.as_column()
    ra_deg = qv.Float64Column()
    dec_deg = qv.Float64Column()
    ra_sigma_arcsec = qv.Float64Column()
    dec_sigma_arcsec = qv.Float64Column()
    mag = qv.Float64Column()
    mag_sigma = qv.Float64Column()
    exposure_time_start = Timestamp.as_column()
    exposure_time_mid = Timestamp.as_column()
    filter = qv.LargeStringColumn()
    obscode = qv.LargeStringColumn()
    exposure_id = qv.LargeStringColumn()
    exposure_duration = qv.Float64Column()
    observation_id = qv.LargeStringColumn()
    healpix_id = qv.Int64Column()
    pred_ra_deg = qv.Float64Column()
    pred_dec_deg = qv.Float64Column()
    pred_vra_degpday = qv.Float64Column()
    pred_vdec_degpday = qv.Float64Column()
    delta_ra_arcsec = qv.Float64Column()
    delta_dec_arcsec = qv.Float64Column()
    distance_arcsec = qv.Float64Column()
    pred_mag = qv.Float64Column(nullable=True)
    mag_residual = qv.Float64Column(nullable=True)
    rejected = qv.BooleanColumn()
    rejected_reason = qv.LargeStringColumn(nullable=True)
    aberrated_coordinates = CartesianCoordinates.as_column(nullable=True)
    dataset_id = qv.LargeStringColumn()
    orbit_id = qv.LargeStringColumn()

    def point_source_detections(self) -> PointSourceDetections:
        return PointSourceDetections.from_kwargs(
            id=self.observation_id,
            exposure_id=self.exposure_id,
            ra=self.ra_deg,
            dec=self.dec_deg,
            ra_sigma=self.ra_sigma_arcsec,
            dec_sigma=self.dec_sigma_arcsec,
            mag=self.mag,
            mag_sigma=self.mag_sigma,
            time=self.time,
        )

    def exposures(self) -> Exposures:
        unique = self.drop_duplicates(subset=["exposure_id"])

        return Exposures.from_kwargs(
            id=unique.exposure_id,
            start_time=unique.exposure_time_start,
            observatory_code=unique.obscode,
            filter=unique.filter,
            duration=unique.exposure_duration,
        )

    def as_exposures(self) -> Exposures:
        return Exposures.from_kwargs(
            id=self.exposure_id,
            start_time=self.exposure_time_start,
            observatory_code=self.obscode,
            filter=self.filter,
            duration=self.exposure_duration,
        )

    def predicted_ephemeris(self, orbit_ids=None) -> Ephemeris:
        """
        Return the predicted ephemeris for these candidates.

        Parameters
        ----------
        orbit_ids : Optional[List[str]], optional
            Orbit IDs to use for the predicted ephemeris. If None, a unique
            orbit ID will be generated for each candidate.

        Returns
        -------
        Ephemeris
            Predicted ephemeris for these candidates.
        """
        if orbit_ids is None:
            orbit_ids = [str(i) for i in range(len(self.time))]
        return Ephemeris.from_kwargs(
            orbit_id=orbit_ids,
            coordinates=SphericalCoordinates.from_kwargs(
                lon=self.pred_ra_deg,
                lat=self.pred_dec_deg,
                vlon=self.pred_vra_degpday,
                vlat=self.pred_vdec_degpday,
                time=self.time,
                origin=Origin.from_kwargs(
                    code=self.obscode,
                ),
                frame="equatorial",
            ),
            aberrated_coordinates=self.aberrated_coordinates,
        )

    def to_residuals(self) -> Residuals:
        """
        Compute the residuals between the observations and the predicted ephemeris.

        Returns
        -------
        Residuals
            Residuals between the observations and the predicted ephemeris.
        """
        return Residuals.calculate(
            self.to_spherical_coordinates(), self.predicted_ephemeris().coordinates
        )

    def to_spherical_coordinates(self) -> SphericalCoordinates:
        """
        Convert the observations to a SphericalCoordinates object.

        Returns
        -------
        SphericalCoordinates
            Observations represented as a SphericalCoordinates object.
        """
        # Create a 2D array of sigmas for the observations
        # Convert arcseconds to degrees
        sigmas = np.full((len(self.time), 6), np.nan)
        sigmas[:, 1] = self.ra_sigma_arcsec.to_numpy(zero_copy_only=False) / 3600
        sigmas[:, 2] = self.dec_sigma_arcsec.to_numpy(zero_copy_only=False) / 3600

        # Create a Coordinates object for the observations - we need
        # these to calculate residuals
        return SphericalCoordinates.from_kwargs(
            lon=self.ra_deg,
            lat=self.dec_deg,
            time=self.time,
            covariance=CoordinateCovariances.from_sigmas(sigmas),
            origin=Origin.from_kwargs(
                code=self.obscode,
            ),
            frame="equatorial",
        )

    def get_observers(self) -> Observers:
        """
        Get the sorted observers for these candidates. Observers
        can be used with an `~adam_core.propagator.Propagator` to
        generate predicted ephemerides at the same time as the
        observations.

        Returns
        -------
        Observers (N)
            Observers for these candidates sorted by time and
            observatory code.
        """
        observers = Observers.empty()
        for obscode in self.obscode.unique():
            self_obs = self.select("obscode", obscode)
            times = self_obs.time
            observers = qv.concatenate(
                [observers, Observers.from_code(obscode.as_py(), times)]
            )

        return observers.sort_by(
            [
                "coordinates.time.days",
                "coordinates.time.nanos",
                "code",
            ]
        )


class FrameCandidates(qv.Table):

    exposure_time_start = Timestamp.as_column()
    exposure_time_mid = Timestamp.as_column()
    filter = qv.LargeStringColumn()
    obscode = qv.LargeStringColumn()
    exposure_id = qv.LargeStringColumn()
    exposure_duration = qv.Float64Column()
    healpix_id = qv.Int64Column()
    pred_ra_deg = qv.Float64Column()
    pred_dec_deg = qv.Float64Column()
    pred_vra_degpday = qv.Float64Column()
    pred_vdec_degpday = qv.Float64Column()
    pred_mag = qv.Float64Column(nullable=True)
    rejected = qv.BooleanColumn()
    rejected_reason = qv.LargeStringColumn(nullable=True)
    aberrated_coordinates = CartesianCoordinates.as_column(nullable=True)
    dataset_id = qv.LargeStringColumn()
    orbit_id = qv.LargeStringColumn()

    def exposures(self) -> Exposures:
        unique = self.drop_duplicates(subset=["exposure_id"])

        return Exposures.from_kwargs(
            id=unique.exposure_id,
            start_time=unique.exposure_time_start,
            observatory_code=unique.obscode,
            filter=unique.filter,
            duration=unique.exposure_duration,
        )

    def as_exposures(self) -> Exposures:
        return Exposures.from_kwargs(
            id=self.exposure_id,
            start_time=self.exposure_time_start,
            observatory_code=self.obscode,
            filter=self.filter,
            duration=self.exposure_duration,
        )

    def predicted_ephemeris(self, orbit_ids=None) -> Ephemeris:
        origin = Origin.from_kwargs(
            code=["SUN" for i in range(len(self.exposure_time_mid))]
        )
        frame = "ecliptic"
        if orbit_ids is None:
            orbit_ids = [str(i) for i in range(len(self.exposure_time_mid))]
        return Ephemeris.from_kwargs(
            orbit_id=orbit_ids,
            coordinates=SphericalCoordinates.from_kwargs(
                lon=self.pred_ra_deg,
                lat=self.pred_dec_deg,
                vlon=self.pred_vra_degpday,
                vlat=self.pred_vdec_degpday,
                time=self.exposure_time_mid,
                origin=origin,
                frame=frame,
            ),
        )


def candidates_from_ephem(
    obs: ObservationsTable,
    ephem: Ephemeris,
    frame: HealpixFrame,
    *,
    pred_mag: pa.Array | None = None,
    mag_residual: pa.Array | None = None,
    rejected: pa.Array | None = None,
    rejected_reason: pa.Array | None = None,
) -> PrecoveryCandidates:
    """
    Generates PrecoveryCandidates from constituent observations, ephem, and frame data

    Parameters
    ----------
    obs : ObservationsTable
        Observations that matched and are now candidates
    ephem : Ephemeris
        Matching ephemeris for each observation from which the residuals are calculated
    frame : HealpixFrame
        Frames include the exposure metadata which is part of the candidate

    Returns
    -------
    PrecoveryCandidates
        PrecoveryCandidates table with the matching observations
    """
    assert len(frame) == 1, "frame should have only one entry"
    assert len(obs) == len(
        ephem
    ), "Observations and ephemeris must be the same length. If ephem is identical, use ephem.take() to repeat the same ephemeris for each observation."  # noqa: E501

    healpix = radec_to_healpixel(
        ephem.coordinates.lon.to_numpy(),
        ephem.coordinates.lat.to_numpy(),
        nside=CANDIDATE_NSIDE,
    )

    exposure_time_start = Timestamp.from_mjd(
        pa.repeat(frame.exposure_mjd_start[0].as_py(), len(obs)), scale="utc"
    )

    exposure_time_mid = Timestamp.from_mjd(
        pa.repeat(frame.exposure_mjd_mid[0].as_py(), len(obs)), scale="utc"
    )

    frame_filter = pa.repeat(frame.filter[0].as_py(), len(obs))
    obscode = pa.repeat(frame.obscode[0].as_py(), len(obs))
    exposure_id = pa.repeat(frame.exposure_id[0].as_py(), len(obs))
    exposure_duration = pa.repeat(frame.exposure_duration[0].as_py(), len(obs))
    delta_ra_arcsec = pc.divide(pc.subtract(obs.ra, ephem.coordinates.lon), ARCSEC)
    delta_dec_arcsec = pc.divide(pc.subtract(obs.dec, ephem.coordinates.lat), ARCSEC)

    distance_arcsec = (
        haversine_distance_deg(
            obs.ra.to_numpy(),
            ephem.coordinates.lon.to_numpy(),
            obs.dec.to_numpy(),
            ephem.coordinates.lat.to_numpy(),
        )
        / ARCSEC
    )

    dataset_id = pa.repeat(frame.dataset_id[0].as_py(), len(obs))
    orbit_id = pa.repeat(ephem.orbit_id[0].as_py(), len(obs))

    if pred_mag is None:
        pred_mag = pa.nulls(len(obs), type=pa.float64())
    if mag_residual is None:
        mag_residual = pa.nulls(len(obs), type=pa.float64())
    if rejected is None:
        rejected = pa.repeat(False, len(obs))
    if rejected_reason is None:
        rejected_reason = pa.array([None] * len(obs), type=pa.large_string())

    return PrecoveryCandidates.from_kwargs(
        time=obs.time,
        ra_deg=obs.ra,
        dec_deg=obs.dec,
        ra_sigma_arcsec=pc.divide(obs.ra_sigma, ARCSEC),
        dec_sigma_arcsec=pc.divide(obs.dec_sigma, ARCSEC),
        mag=obs.mag,
        mag_sigma=obs.mag_sigma,
        exposure_time_start=exposure_time_start,
        exposure_time_mid=exposure_time_mid,
        filter=frame_filter,
        obscode=obscode,
        exposure_id=exposure_id,
        exposure_duration=exposure_duration,
        observation_id=obs.id,
        healpix_id=healpix,
        pred_ra_deg=ephem.coordinates.lon,
        pred_dec_deg=ephem.coordinates.lat,
        pred_vra_degpday=ephem.coordinates.vlon,
        pred_vdec_degpday=ephem.coordinates.vlat,
        delta_ra_arcsec=delta_ra_arcsec,
        delta_dec_arcsec=delta_dec_arcsec,
        distance_arcsec=distance_arcsec,
        pred_mag=pred_mag,
        mag_residual=mag_residual,
        rejected=rejected,
        rejected_reason=rejected_reason,
        aberrated_coordinates=ephem.aberrated_coordinates,
        dataset_id=dataset_id,
        orbit_id=orbit_id,
    )


def frame_candidates_from_frame(
    frame: HealpixFrame,
    ephem: Ephemeris,
    *,
    pred_mag: float | None = None,
    rejected: bool = False,
    rejected_reason: str | None = None,
):
    # Calculate the HEALpixel ID for the predicted ephemeris of
    # the orbit with a high nside value (k=15, nside=2**15) The
    # indexed observations are indexed to a much lower nside but
    # we may decide in the future to re-index the database using
    # different values for that parameter. As long as we return a
    # Healpix ID generated with nside greater than the indexed
    # database then we can always down-sample the ID to a lower
    # nside value
    healpix_id = int(
        radec_to_healpixel(
            ephem.coordinates.lon[0].as_py(),
            ephem.coordinates.lat[0].as_py(),
            nside=CANDIDATE_NSIDE,
        )
    )

    return FrameCandidates.from_kwargs(
        exposure_time_start=Timestamp.from_mjd(frame.exposure_mjd_start, scale="utc"),
        exposure_time_mid=Timestamp.from_mjd(frame.exposure_mjd_mid, scale="utc"),
        filter=frame.filter,
        obscode=frame.obscode,
        exposure_id=frame.exposure_id,
        exposure_duration=frame.exposure_duration,
        dataset_id=frame.dataset_id,
        healpix_id=[healpix_id],
        pred_ra_deg=ephem.coordinates.lon,
        pred_dec_deg=ephem.coordinates.lat,
        pred_vra_degpday=ephem.coordinates.vlon,
        pred_vdec_degpday=ephem.coordinates.vlat,
        pred_mag=[pred_mag],
        rejected=[bool(rejected)],
        rejected_reason=[rejected_reason],
        aberrated_coordinates=ephem.aberrated_coordinates,
        orbit_id=ephem.orbit_id,
    )


def _as_bool_array(x: pa.Array) -> pa.Array:
    return pc.cast(x, pa.bool_())


REJECT_REASON_LIMITING_MAGNITUDE = "limiting_magnitude"
REJECT_REASON_MAG_RESIDUAL = "mag_residual"


def find_healpixel_matches(
    propagation_targets: GenericFrame, ephems: Ephemeris, nside: int
) -> GenericFrame:
    """
    Find the healpixels that match between the propagation targets and the 2 body ephemeris

    Parameters
    ----------
    propagation_targets : PropagationTargets
        Propagation targets to match
    ephems : Ephemeris
        Ephemeris to match

    Returns
    -------
    PropagationTargets
        Propagation targets that match the ephemeris
    """
    # Sort them both by time
    propagation_targets = propagation_targets.sort_by(["time.days", "time.nanos"])
    ephems = ephems.sort_by(["coordinates.time.days", "coordinates.time.nanos"])

    propagation_target_times = propagation_targets.time.rescale("utc")
    ephem_times = ephems.coordinates.time.rescale("utc")

    # quickly check to make sure times are equal
    assert pc.all(
        propagation_target_times.equals(ephem_times, precision="ms")
    ).as_py(), "Propagation targets and ephemeris must have matching times"

    # Calculate the healpixels for the ephemeris
    ephem_healpixels = radec_to_healpixel(
        ephems.coordinates.lon.to_numpy(),
        ephems.coordinates.lat.to_numpy(),
        nside=nside,
    )

    # Find the matching healpixels
    mask = pc.equal(propagation_targets.healpixel, ephem_healpixels)
    filtered_targets = propagation_targets.apply_mask(mask)

    return filtered_targets


def find_healpixel_matches_covariance(
    propagation_targets: GenericFrame,
    ephems: Ephemeris,
    nside: int,
    n_sigma: float,
) -> GenericFrame:
    """
    Covariance-aware healpixel selection.

    Instead of requiring the predicted ephemeris point to fall in the exact same
    healpixel as the frame, we expand the prediction into an N-sigma region and
    keep any frame healpixels that intersect that region.

    Fast approach: use a conservative disc radius based on the *major axis* of the
    local tangent-plane covariance ellipse.
    """
    # Sort them both by time
    propagation_targets = propagation_targets.sort_by(["time.days", "time.nanos"])
    ephems = ephems.sort_by(["coordinates.time.days", "coordinates.time.nanos"])

    propagation_target_times = propagation_targets.time.rescale("utc")
    ephem_times = ephems.coordinates.time.rescale("utc")

    # quickly check to make sure times are equal
    assert pc.all(
        propagation_target_times.equals(ephem_times, precision="ms")
    ).as_py(), "Propagation targets and ephemeris must have matching times"

    if ephems.coordinates.covariance is None or ephems.coordinates.covariance.is_all_nan():
        # Fall back to exact-pixel matching if covariances aren't available.
        logger.warning(
            "Ephemeris has no covariances; falling back to exact healpixel matching."
        )
        return find_healpixel_matches(propagation_targets, ephems, nside)

    lon = ephems.coordinates.lon.to_numpy(zero_copy_only=False).astype(np.float64)
    lat = ephems.coordinates.lat.to_numpy(zero_copy_only=False).astype(np.float64)

    # Extract (lon, lat) covariance block (degrees^2) and apply cos(lat) scaling to lon,
    # matching the tangent-plane convention used elsewhere (and in adam_core Residuals).
    cov6 = ephems.coordinates.covariance.to_matrix()
    cov_ll = cov6[:, 1:3, 1:3]
    cos_lat = np.cos(np.deg2rad(lat))
    a = (cos_lat**2) * cov_ll[:, 0, 0]  # var(x) where x = lon*cos(lat)
    d = cov_ll[:, 1, 1]  # var(y) where y = lat
    b = 0.5 * cos_lat * (cov_ll[:, 0, 1] + cov_ll[:, 1, 0])  # cov(x, y)

    # Major-axis variance of 2x2 covariance: lambda_max = 0.5*(tr + sqrt((a-d)^2 + 4b^2))
    tr = a + d
    disc = np.sqrt(np.maximum((a - d) ** 2 + 4.0 * (b**2), 0.0))
    lambda_max = 0.5 * (tr + disc)
    sigma_major_deg = np.sqrt(np.maximum(lambda_max, 0.0))

    # Inflate by N-sigma and add a healpix "pixel radius" safety margin so we don't drop
    # boundary cases due to discretization.
    pix_margin_rad = float(hp.max_pixrad(nside))
    radii_rad = np.deg2rad(float(n_sigma) * sigma_major_deg) + pix_margin_rad
    radii_rad = np.where(np.isfinite(radii_rad) & (radii_rad > 0), radii_rad, pix_margin_rad)

    # Many rows can share the same timestamp. Cache query_disc results per unique time
    # to avoid recomputing pixel sets repeatedly.
    days = ephems.coordinates.time.days.to_numpy(zero_copy_only=False)
    nanos = ephems.coordinates.time.nanos.to_numpy(zero_copy_only=False)

    pixels_by_time: dict[tuple[int, int], set[int]] = {}
    for i in range(len(ephems)):
        key = (int(days[i]), int(nanos[i]))
        if key in pixels_by_time:
            continue
        vec = hp.ang2vec(lon[i], lat[i], lonlat=True)
        pix = hp.query_disc(
            nside,
            vec,
            float(radii_rad[i]),
            inclusive=True,
            nest=True,
        )
        pixels_by_time[key] = set(int(p) for p in pix.tolist())

    # Filter propagation targets by membership in the per-time pixel set.
    target_days = propagation_targets.time.days.to_numpy(zero_copy_only=False)
    target_nanos = propagation_targets.time.nanos.to_numpy(zero_copy_only=False)
    target_pixels = propagation_targets.healpixel.to_numpy(zero_copy_only=False)
    keep = np.zeros(len(propagation_targets), dtype=bool)
    for i in range(len(propagation_targets)):
        key = (int(target_days[i]), int(target_nanos[i]))
        keep[i] = int(target_pixels[i]) in pixels_by_time.get(key, set())

    return propagation_targets.apply_mask(pa.array(keep))


def find_healpixel_matches_covariance_mc(
    propagation_targets: GenericFrame,
    ephems: Ephemeris,
    nside: int,
    n_sigma: float,
    num_samples: int = 64,
    seed: int = 0,
) -> GenericFrame:
    """
    Covariance-aware healpixel selection using sampling ("MC variants").

    Compared to `find_healpixel_matches_covariance` (fast disc approximation), this method
    draws samples from the predicted (lon, lat) covariance (in a local tangent plane),
    converts those samples to (lon, lat) points, and keeps any frame healpixels touched
    by the samples (plus immediate neighbors as a safety margin).

    Notes
    -----
    - This is still based on the (already-propagated) ephemeris covariance; it does *not*
      propagate a full set of orbit clones.
    - Runtime scales ~O(N_unique_times * num_samples).
    """
    # Sort them both by time
    propagation_targets = propagation_targets.sort_by(["time.days", "time.nanos"])
    ephems = ephems.sort_by(["coordinates.time.days", "coordinates.time.nanos"])

    propagation_target_times = propagation_targets.time.rescale("utc")
    ephem_times = ephems.coordinates.time.rescale("utc")

    # quickly check to make sure times are equal
    assert pc.all(
        propagation_target_times.equals(ephem_times, precision="ms")
    ).as_py(), "Propagation targets and ephemeris must have matching times"

    if ephems.coordinates.covariance is None or ephems.coordinates.covariance.is_all_nan():
        logger.warning(
            "Ephemeris has no covariances; falling back to exact healpixel matching."
        )
        return find_healpixel_matches(propagation_targets, ephems, nside)

    # If num_samples is invalid, fall back to the fast disc method.
    if int(num_samples) <= 0:
        return find_healpixel_matches_covariance(
            propagation_targets, ephems, nside=nside, n_sigma=n_sigma
        )

    lon = ephems.coordinates.lon.to_numpy(zero_copy_only=False).astype(np.float64)
    lat = ephems.coordinates.lat.to_numpy(zero_copy_only=False).astype(np.float64)

    cov6 = ephems.coordinates.covariance.to_matrix()
    cov_ll = cov6[:, 1:3, 1:3]  # degrees^2 in (lon, lat)

    days = ephems.coordinates.time.days.to_numpy(zero_copy_only=False)
    nanos = ephems.coordinates.time.nanos.to_numpy(zero_copy_only=False)

    pix_margin_rad = float(hp.max_pixrad(nside))

    def _safe_cholesky_2x2(c: np.ndarray) -> np.ndarray:
        # Add a small jitter and retry if needed.
        jitter = 1e-18
        for _ in range(3):
            try:
                return np.linalg.cholesky(c + np.eye(2) * jitter)
            except np.linalg.LinAlgError:
                jitter *= 100.0
        # Last resort: eigenvalue clamp.
        w, v = np.linalg.eigh(c)
        w = np.maximum(w, 0.0)
        return v @ np.diag(np.sqrt(w))

    pixels_by_time: dict[tuple[int, int], set[int]] = {}
    base_rng = np.random.default_rng(int(seed))
    for i in range(len(ephems)):
        key = (int(days[i]), int(nanos[i]))
        if key in pixels_by_time:
            continue

        # Tangent-plane convention: x = dlon*cos(lat), y = dlat
        cos_lat = float(np.cos(np.deg2rad(lat[i])))
        cos_lat = cos_lat if np.isfinite(cos_lat) and abs(cos_lat) > 1e-12 else 1e-12

        c_ll = cov_ll[i].astype(np.float64)
        c_ll = 0.5 * (c_ll + c_ll.T)  # symmetrize
        A = np.array([[cos_lat, 0.0], [0.0, 1.0]], dtype=np.float64)
        c_xy = A @ c_ll @ A.T

        # Sample in tangent plane at N-sigma.
        # Seed per-time deterministically so results don't depend on iteration order.
        rng = np.random.default_rng(base_rng.integers(0, 2**32 - 1) ^ (key[0] & 0xFFFFFFFF))
        L = _safe_cholesky_2x2(c_xy)
        z = rng.standard_normal((int(num_samples), 2))
        dxy = (z @ L.T) * float(n_sigma)

        dx = dxy[:, 0]
        dy = dxy[:, 1]

        dlon = dx / cos_lat
        dlat = dy

        lon_s = (lon[i] + dlon) % 360.0
        lat_s = np.clip(lat[i] + dlat, -89.999999, 89.999999)

        # Always include the nominal predicted point too.
        lon_all = np.concatenate([lon_s, np.array([lon[i]], dtype=np.float64)])
        lat_all = np.concatenate([lat_s, np.array([lat[i]], dtype=np.float64)])

        pix = hp.ang2pix(nside, lon_all, lat_all, lonlat=True, nest=True)
        pix_set = set(int(p) for p in np.asarray(pix).ravel().tolist())

        # Safety margin: include neighbors (8-connected) and any pixels intersecting a single-pixel disc
        # (helps with boundary discretization).
        neigh = hp.get_all_neighbours(nside, np.asarray(list(pix_set), dtype=np.int64), nest=True)
        if neigh is not None:
            for p in np.asarray(neigh).ravel().tolist():
                if int(p) >= 0:
                    pix_set.add(int(p))

        vec = hp.ang2vec(float(lon[i]), float(lat[i]), lonlat=True)
        disc_pix = hp.query_disc(
            nside, vec, pix_margin_rad, inclusive=True, nest=True
        )
        pix_set.update(int(p) for p in disc_pix.tolist())

        pixels_by_time[key] = pix_set

    # Filter propagation targets by membership in the per-time pixel set.
    target_days = propagation_targets.time.days.to_numpy(zero_copy_only=False)
    target_nanos = propagation_targets.time.nanos.to_numpy(zero_copy_only=False)
    target_pixels = propagation_targets.healpixel.to_numpy(zero_copy_only=False)
    keep = np.zeros(len(propagation_targets), dtype=bool)
    for i in range(len(propagation_targets)):
        key = (int(target_days[i]), int(target_nanos[i]))
        keep[i] = int(target_pixels[i]) in pixels_by_time.get(key, set())

    return propagation_targets.apply_mask(pa.array(keep))


def find_healpixel_matches_covariance_polygon(
    propagation_targets: GenericFrame,
    ephems: Ephemeris,
    nside: int,
    n_sigma: float,
    num_vertices: int = 32,
) -> GenericFrame:
    """
    Covariance-aware healpixel selection using an N-sigma ellipse boundary polygon.

    This is tighter than the fast-disc approximation: we approximate the N-sigma uncertainty
    ellipse in a local tangent plane with `num_vertices` points, map those points back to
    (lon, lat), then call `healpy.query_polygon` to get the intersecting pixels.

    Notes
    -----
    - Uses the ephemeris-provided spherical covariance (lon/lat block). This is still a
      first-order (Gaussian) approximation, but it reduces false pixels vs the disc bound.
    - Runtime scales ~O(N_unique_times * num_vertices) plus healpy polygon queries.
    """
    # Sort them both by time
    propagation_targets = propagation_targets.sort_by(["time.days", "time.nanos"])
    ephems = ephems.sort_by(["coordinates.time.days", "coordinates.time.nanos"])

    propagation_target_times = propagation_targets.time.rescale("utc")
    ephem_times = ephems.coordinates.time.rescale("utc")

    # quickly check to make sure times are equal
    assert pc.all(
        propagation_target_times.equals(ephem_times, precision="ms")
    ).as_py(), "Propagation targets and ephemeris must have matching times"

    if ephems.coordinates.covariance is None or ephems.coordinates.covariance.is_all_nan():
        logger.warning(
            "Ephemeris has no covariances; falling back to exact healpixel matching."
        )
        return find_healpixel_matches(propagation_targets, ephems, nside)

    if int(num_vertices) < 8:
        num_vertices = 8

    lon = ephems.coordinates.lon.to_numpy(zero_copy_only=False).astype(np.float64)
    lat = ephems.coordinates.lat.to_numpy(zero_copy_only=False).astype(np.float64)

    cov6 = ephems.coordinates.covariance.to_matrix()
    cov_ll = cov6[:, 1:3, 1:3]  # degrees^2 in (lon, lat)

    days = ephems.coordinates.time.days.to_numpy(zero_copy_only=False)
    nanos = ephems.coordinates.time.nanos.to_numpy(zero_copy_only=False)

    pix_margin_rad = float(hp.max_pixrad(nside))

    # Unit circle points for ellipse boundary.
    angles = np.linspace(0.0, 2.0 * np.pi, int(num_vertices), endpoint=False)
    unit = np.stack([np.cos(angles), np.sin(angles)], axis=1)  # (V, 2)

    def _safe_sqrtm_2x2(c: np.ndarray) -> np.ndarray:
        # Symmetrize and clamp eigenvalues to avoid tiny negative due to numerical noise.
        c = 0.5 * (c + c.T)
        w, v = np.linalg.eigh(c)
        w = np.maximum(w, 0.0)
        return v @ np.diag(np.sqrt(w)) @ v.T

    pixels_by_time: dict[tuple[int, int], set[int]] = {}
    for i in range(len(ephems)):
        key = (int(days[i]), int(nanos[i]))
        if key in pixels_by_time:
            continue

        # Tangent-plane convention: x = dlon*cos(lat), y = dlat
        cos_lat = float(np.cos(np.deg2rad(lat[i])))
        cos_lat = cos_lat if np.isfinite(cos_lat) and abs(cos_lat) > 1e-12 else 1e-12

        c_ll = cov_ll[i].astype(np.float64)
        A = np.array([[cos_lat, 0.0], [0.0, 1.0]], dtype=np.float64)
        c_xy = A @ c_ll @ A.T

        S = _safe_sqrtm_2x2(c_xy)
        dxy = (unit @ S.T) * float(n_sigma)  # (V, 2)
        dx = dxy[:, 0]
        dy = dxy[:, 1]

        dlon = dx / cos_lat
        dlat = dy

        lon_poly = (lon[i] + dlon) % 360.0
        lat_poly = np.clip(lat[i] + dlat, -89.999999, 89.999999)

        verts = hp.ang2vec(lon_poly, lat_poly, lonlat=True)
        pix = hp.query_polygon(nside, verts, inclusive=True, nest=True)
        pix_set = set(int(p) for p in np.asarray(pix).ravel().tolist())

        # Safety margin: include neighbors and the nominal center pixel disc.
        if pix_set:
            neigh = hp.get_all_neighbours(
                nside, np.asarray(list(pix_set), dtype=np.int64), nest=True
            )
            if neigh is not None:
                for p in np.asarray(neigh).ravel().tolist():
                    if int(p) >= 0:
                        pix_set.add(int(p))

        vec0 = hp.ang2vec(float(lon[i]), float(lat[i]), lonlat=True)
        disc_pix = hp.query_disc(nside, vec0, pix_margin_rad, inclusive=True, nest=True)
        pix_set.update(int(p) for p in disc_pix.tolist())

        pixels_by_time[key] = pix_set

    # Filter propagation targets by membership in the per-time pixel set.
    target_days = propagation_targets.time.days.to_numpy(zero_copy_only=False)
    target_nanos = propagation_targets.time.nanos.to_numpy(zero_copy_only=False)
    target_pixels = propagation_targets.healpixel.to_numpy(zero_copy_only=False)
    keep = np.zeros(len(propagation_targets), dtype=bool)
    for i in range(len(propagation_targets)):
        key = (int(target_days[i]), int(target_nanos[i]))
        keep[i] = int(target_pixels[i]) in pixels_by_time.get(key, set())

    return propagation_targets.apply_mask(pa.array(keep))


def generate_ephem_for_per_obs_timestamps(
    orbit: Orbits,
    observations: ObservationsTable,
    obscode: str,
    propagator: Propagator,
    *,
    predict_magnitudes: bool = False,
) -> Ephemeris:
    """
    Use 2 body propagation to generate ephemeris for unique time observations
    """
    # We propagate to the mean observation epoch using the provided propagator (typically ASSIST),
    # then use 2-body propagation to generate paired states at each observation timestamp.
    #
    # If the input orbit includes a covariance, we ask the propagator to propagate covariance to
    # the mean epoch as well; the downstream 2-body + ephemeris generation can then produce
    # ephemeris covariances for covariance-gated matching.
    mean_mjd = pc.mean(observations.time.mjd()).as_py()
    mean_time = Timestamp.from_mjd([mean_mjd], scale="utc")

    use_covariance = (
        orbit.coordinates.covariance is not None
        and not orbit.coordinates.covariance.is_all_nan()
    )
    mean_orbit_state = propagator.propagate_orbits(orbit, mean_time, covariance=use_covariance)
    propagated_orbits = propagate_2body(mean_orbit_state, observations.time)
    observers = Observers.from_code(obscode, observations.time)
    ephemeris = generate_ephemeris_2body(
        propagated_orbits, observers, predict_magnitudes=bool(predict_magnitudes)
    )
    return ephemeris


def find_observation_matches(
    observations: ObservationsTable, ephems: Ephemeris, tolerance_deg: float
) -> Tuple[ObservationsTable, Ephemeris]:
    """
    Find the observations that match the ephemeris within the given tolerance

    Parameters
    ----------
    observations : ObservationsTable
        Observations to match with ephemeris (in pairs)
    ephems : Ephemeris
        Ephemeris to match with observations (in pairs)
    tolerance_deg : float
        Tolerance in degrees for matching
    Returns
    -------
    Tuple[ObservationsTable, Ephemeris]
        Observations and ephemeris that match within the given tolerance
    """
    assert len(ephems) == len(
        observations
    ), "Ephemeris must be the same length as observations"
    # Ephemerides may be generated in a different timescale (e.g., TDB) than
    # observations (typically UTC). Normalize to UTC for comparisons.
    assert pc.all(
        ephems.coordinates.time.rescale("utc").equals(
            observations.time.rescale("utc"), precision="ms"
        )
    ).as_py(), "Ephemeris and observations must have matching times"
    # Check for bizarrely large tolerance which might have been
    # sent in as arcseconds instead of degrees
    if tolerance_deg > 2:
        logger.warning(
            "Tolerance is very large, did you pass in arcseconds instead of degrees?"
        )

    distances = haversine_distance_deg(
        observations.ra.to_numpy(),
        ephems.coordinates.lon.to_numpy(),
        observations.dec.to_numpy(),
        ephems.coordinates.lat.to_numpy(),
    )

    mask = pc.less(distances, tolerance_deg)
    matching_observations = observations.apply_mask(mask)
    matching_ephems = ephems.apply_mask(mask)
    return matching_observations, matching_ephems


def _wrap_delta_lon_deg(lon_obs_deg: np.ndarray, lon_pred_deg: np.ndarray) -> np.ndarray:
    """
    Wrap longitude residuals into [-180, 180) degrees.
    """
    return (lon_obs_deg - lon_pred_deg + 180.0) % 360.0 - 180.0


def find_observation_matches_covariance(
    observations: ObservationsTable,
    ephems: Ephemeris,
    n_sigma: float,
) -> Tuple[ObservationsTable, Ephemeris]:
    """
    Covariance-aware matching using a 2D Mahalanobis (chi^2) gate in a local tangent plane.

    Notes
    -----
    - Requires ephemeris spherical coordinate covariances to be present. These are produced
      by `adam_core.dynamics.ephemeris.generate_ephemeris_2body()` when input orbit covariances
      are defined and propagated through `adam_core.dynamics.propagation.propagate_2body()`.
    - Uses a small-angle tangent-plane approximation for the residuals.
    """
    assert len(ephems) == len(
        observations
    ), "Ephemeris must be the same length as observations"
    # Ephemerides may be generated in a different timescale (e.g., TDB) than
    # observations (typically UTC). Normalize to UTC for comparisons.
    assert pc.all(
        ephems.coordinates.time.rescale("utc").equals(
            observations.time.rescale("utc"), precision="ms"
        )
    ).as_py(), "Ephemeris and observations must have matching times"

    if ephems.coordinates.covariance is None or ephems.coordinates.covariance.is_all_nan():
        raise ValueError(
            "Ephemeris has no covariances; cannot use covariance-based matching. "
            "Ensure the input orbit has a covariance and that you are using "
            "propagation/ephemeris generation with covariance propagation enabled."
        )

    # Use `adam_core.coordinates.residuals.Residuals.calculate` so our gating logic matches
    # what THOR uses: wrapped longitude residuals, cosine(latitude) correction, and
    # chi2 = Δᵀ(Σ_obs + Σ_pred)⁻¹Δ in the observed subspace.
    N = len(observations)
    lon_obs = observations.ra.to_numpy(zero_copy_only=False).astype(np.float64)
    lat_obs = observations.dec.to_numpy(zero_copy_only=False).astype(np.float64)

    # Observational 1-sigma uncertainties in degrees.
    ra_sig = observations.ra_sigma.to_numpy(zero_copy_only=False).astype(np.float64)
    dec_sig = observations.dec_sigma.to_numpy(zero_copy_only=False).astype(np.float64)
    ra_sig = np.where(np.isfinite(ra_sig), ra_sig, 0.0)
    dec_sig = np.where(np.isfinite(dec_sig), dec_sig, 0.0)

    # Build a full 6x6 spherical covariance for observations, with only lon/lat populated.
    # The other dimensions are unused (set to NaN in the values), but their diagonal entries
    # must be finite so we can construct a valid covariance matrix container.
    cov = np.zeros((N, 6, 6), dtype=np.float64)
    cov[:, 0, 0] = 1.0
    cov[:, 3, 3] = 1.0
    cov[:, 4, 4] = 1.0
    cov[:, 5, 5] = 1.0
    cov[:, 1, 1] = ra_sig**2
    cov[:, 2, 2] = dec_sig**2

    # Tiny diagonal jitter to avoid pathological singular inversions.
    eps_deg = 1e-12
    cov[:, 1, 1] += eps_deg**2
    cov[:, 2, 2] += eps_deg**2

    obs_coords = SphericalCoordinates.from_kwargs(
        rho=np.full(N, np.nan, dtype=np.float64),
        lon=lon_obs,
        lat=lat_obs,
        vrho=np.full(N, np.nan, dtype=np.float64),
        vlon=np.full(N, np.nan, dtype=np.float64),
        vlat=np.full(N, np.nan, dtype=np.float64),
        time=observations.time,
        covariance=CoordinateCovariances.from_matrix(cov),
        origin=ephems.coordinates.origin,
        frame=ephems.coordinates.frame,
    )

    residuals = Residuals.calculate(
        observed=obs_coords,
        predicted=ephems.coordinates,
        use_predicted_covariance=True,
    )
    chi2 = residuals.chi2.to_numpy(zero_copy_only=False).astype(np.float64)
    chi2 = np.where(np.isfinite(chi2), chi2, np.inf)

    gate = chi2 <= float(n_sigma) ** 2
    mask = pa.array(gate)
    return observations.apply_mask(mask), ephems.apply_mask(mask)


def check_window(
    db_dir: str,
    window: WindowCenters,
    orbit: Orbits,
    tolerance: float,
    propagator_class: Type[Propagator],
    datasets: Optional[set[str]] = None,
    match_method: str = "circle",
    n_sigma: float = 3.0,
    covariance_polygon_vertices: int = 32,
    covariance_mc_num_samples: int = 64,
    covariance_mc_seed: int = 0,
    config_overrides: dict[str, Any] | None = None,
) -> Tuple[PrecoveryCandidates, FrameCandidates]:
    """
    Check a single window for precovery candidates

    Parameters
    ----------
    db_dir : str
        Directory of the database
    window : WindowCenters
        Window to check
    orbit : Orbits
        Orbit to propagate
    tolerance : float
        Tolerance in degrees for matching
    propagator_class : Type[Propagator]
        Propagator class to use for propagation
    datasets : Optional[set[str]], optional
        Datasets to consider, by default None

    Returns
    -------
    Tuple[PrecoveryCandidates, FrameCandidates]
        Precovery candidates and frame candidates
    """
    assert len(window) == 1, "Use _check_windows for multiple windows"
    assert len(orbit) == 1, "_check_window only support one orbit for now"
    logger.info(
        f"check_window orbit: {orbit.orbit_id[0].as_py()} obscode: {window.obscode[0].as_py()} window: {window.window_start().mjd()[0].as_py()} to {window.window_end().mjd()[0].as_py()}"
    )
    db = PrecoveryDatabase.from_dir(db_dir, mode="r", allow_version_mismatch=True)
    if config_overrides:
        # Apply runtime overrides (e.g. magnitude cutoffs) without mutating on-disk config.json.
        # This is critical because `check_window` re-opens the DB from disk even when the
        # caller already has a `PrecoveryDatabase` instance with modified `config`.
        for k, v in config_overrides.items():
            if hasattr(db.config, k):
                setattr(db.config, k, v)
    obscode = window.obscode[0].as_py()
    propagation_targets = db.frames.idx.propagation_targets(
        window,
        datasets,
    )
    if len(propagation_targets) == 0:
        logger.debug(
            f"No propagation targets found for window {window.window_start().mjd()[0].as_py()} to {window.window_end().mjd()[0].as_py()}"
        )
        return PrecoveryCandidates.empty(), FrameCandidates.empty()

    logger.debug(f"Found {len(propagation_targets)} propagation targets")
    # NOTE: `propagation_targets` is (time, healpixel) pairs. For a given exposure time
    # there are typically many healpixels, so `times` contains many duplicates.
    # We must avoid generating an ephemeris for *each* duplicated row.
    times = propagation_targets.time

    # Ensure orbit is propagated to the window center
    time_utc = orbit.coordinates.time.rescale("utc")
    if not (pc.all(time_utc.equals(window.time, precision="ms")).as_py()):
        propagator = propagator_class()
        use_covariance = (
            orbit.coordinates.covariance is not None
            and not orbit.coordinates.covariance.is_all_nan()
        )
        orbit = propagator.propagate_orbits(orbit, window.time, covariance=use_covariance)

    # Deduplicate times (ms precision) so we only do 2-body propagation/ephemeris once per
    # exposure time, then broadcast the predicted healpixel back over all target rows.
    target_times = times.rescale("utc").rounded("ms")
    unique_times = target_times.unique().sort_by(["days", "nanos"])

    # Build mapping from each target row -> index in unique_times using an int64 composite key.
    all_keys = PrecoveryDatabase._timestamp_key(target_times.days, target_times.nanos)
    unique_keys = PrecoveryDatabase._timestamp_key(unique_times.days, unique_times.nanos)
    idx = pc.fill_null(pc.index_in(all_keys, value_set=unique_keys), -1)
    assert pc.all(pc.greater_equal(idx, 0)).as_py(), "Missing time mapping for targets"
    idx64 = pc.cast(idx, pa.int64())

    # create our observers from the individual (deduplicated) times
    observers = Observers.from_code(obscode, unique_times)
    ## first propagate with 2_body
    propagated_orbits = propagate_2body(orbit, unique_times)

    # generate ephemeris (covariances will be present if input orbit covariances exist)
    # Note: we do not need to predict magnitudes here; magnitudes are attached downstream,
    # and predicting them during this step is expensive.
    ephems = generate_ephemeris_2body(
        propagated_orbits, observers, predict_magnitudes=False
    )
    if match_method in ("covariance", "covariance_mc", "covariance_polygon"):
        # For covariance-aware healpixel selection, we need an ephemeris aligned 1:1 with
        # `propagation_targets`. We compute ephemerides at unique times, then broadcast the
        # ephemeris rows back over all targets (by time) to avoid redundant propagation.
        ephems_per_row = Ephemeris.from_pyarrow(ephems.table.take(idx64))
        if match_method == "covariance":
            frames_to_check = find_healpixel_matches_covariance(
                propagation_targets,
                ephems_per_row,
                db.frames.healpix_nside,
                n_sigma=n_sigma,
            )
        elif match_method == "covariance_mc":
            frames_to_check = find_healpixel_matches_covariance_mc(
                propagation_targets,
                ephems_per_row,
                db.frames.healpix_nside,
                n_sigma=n_sigma,
                num_samples=covariance_mc_num_samples,
                seed=covariance_mc_seed,
            )
        else:
            frames_to_check = find_healpixel_matches_covariance_polygon(
                propagation_targets,
                ephems_per_row,
                db.frames.healpix_nside,
                n_sigma=n_sigma,
                num_vertices=covariance_polygon_vertices,
            )
    else:
        # Fast circle mode: compute predicted healpixel per unique time, then broadcast to each
        # target row and filter, avoiding per-row ephemeris work for repeated timestamps.
        ephem_healpixels = pa.array(
            radec_to_healpixel(
                ephems.coordinates.lon.to_numpy(),
                ephems.coordinates.lat.to_numpy(),
                nside=db.frames.healpix_nside,
            ),
            type=pa.int64(),
        )
        predicted_hp_per_row = pc.take(ephem_healpixels, idx64)
        mask = pc.equal(propagation_targets.healpixel, predicted_hp_per_row)
        frames_to_check = propagation_targets.apply_mask(mask)
    logger.debug(f"Found {len(frames_to_check)} healpixel matches")
    candidates = PrecoveryCandidates.empty()
    frame_candidates = FrameCandidates.empty()
    for frame in frames_to_check:
        candidates_healpixel, frame_candidates_healpixel = db._check_frames(
            orbit=orbit,
            generic_frames=frame,
            obscode=obscode,
            tolerance=tolerance,
            datasets=datasets,
            propagator_class=propagator_class,
            match_method=match_method,
            n_sigma=n_sigma,
        )
        candidates = qv.concatenate([candidates, candidates_healpixel])
        frame_candidates = qv.concatenate(
            [frame_candidates, frame_candidates_healpixel]
        )
    return candidates, frame_candidates


check_window_remote = ray.remote(check_window)


class PrecoveryDatabase:
    def __init__(self, frames: FrameDB, directory: str, config: Config = DefaultConfig):
        self.frames = frames
        self._exposures_by_obscode: dict = {}
        self.config = config
        self.directory: str = directory
        # Loaded once from a generated Parquet cache file. We keep a Python dict for
        # readability/debuggability and also pre-materialize Arrow arrays for fast,
        # vectorized lookups in the faint-frame skipping hot path (no per-row Python).
        self._limit_by_code_filter: dict[str, float] = {}
        self._limit_codefid_keys: pa.Array = pa.array([], type=pa.large_string())
        self._limit_codefid_vals: pa.Array = pa.array([], type=pa.float64())

    @staticmethod
    def _timestamp_key(days: pa.Array, nanos: pa.Array) -> pa.Array:
        """
        Build an int64 composite key for Timestamp alignment: key = days*NANOS_IN_DAY + nanos.

        We use this because (at least) pyarrow 22 does not support `index_in()` over struct keys.
        """
        nanos_in_day = pa.scalar(86_400_000_000_000, type=pa.int64())  # 86400 * 1e9
        return pc.add_checked(pc.multiply_checked(days, nanos_in_day), nanos)

    @staticmethod
    def _dict_to_kv_arrays(d: dict[str, float]) -> tuple[pa.Array, pa.Array]:
        """
        Convert a python dict[str, float] into Arrow key/value arrays.
        """
        if not d:
            return (
                pa.array([], type=pa.large_string()),
                pa.array([], type=pa.float64()),
            )
        return (
            pa.array(list(d.keys()), type=pa.large_string()),
            pa.array(list(d.values()), type=pa.float64()),
        )

    @staticmethod
    def _arrow_lookup(needles: pa.Array, keys: pa.Array, values: pa.Array) -> pa.Array:
        """
        Vectorized lookup: for each element in `needles`, return the corresponding `values`
        where `needles[i]` is present in `keys`, else null.
        """
        if len(keys) == 0:
            return pa.nulls(len(needles), type=pa.float64())
        idx = pc.fill_null(pc.index_in(needles, value_set=keys), -1)
        valid = pc.greater_equal(idx, 0)
        idx_safe = pc.cast(pc.if_else(valid, idx, 0), pa.int64())
        out = pc.take(values, idx_safe)
        return pc.if_else(valid, out, None)

    def _load_limiting_magnitudes_cache(self) -> None:
        """
        Load a generated-once limiting magnitude cache file from the DB directory.

        This avoids any subsequent DB lookups during precovery search.
        """
        parq_name = getattr(self.config, "limiting_magnitudes_parquet_file", None)
        if parq_name:
            parq_path = os.path.join(self.directory, parq_name)
            if os.path.exists(parq_path):
                try:
                    table = FilterLimitingMagnitudes.from_parquet(parq_path)
                    self._limit_by_code_filter = table.static_code_filter_map()
                    if len(self._limit_by_code_filter) > 0:
                        self._limit_codefid_keys = pa.array(
                            list(self._limit_by_code_filter.keys()),
                            type=pa.large_string(),
                        )
                        self._limit_codefid_vals = pa.array(
                            list(self._limit_by_code_filter.values()), type=pa.float64()
                        )
                    return
                except Exception as e:
                    logger.warning(
                        f"Failed to read limiting magnitudes parquet cache file: {e}"
                    )

    @classmethod
    def from_dir(
        cls,
        directory: str,
        create: bool = False,
        mode: str = "r",
        allow_version_mismatch: bool = False,
    ):
        if not os.path.exists(directory):
            if create:
                return cls.create(directory)

        try:
            config = Config.from_json(os.path.join(directory, "config.json"))
        except FileNotFoundError:
            if not create:
                raise Exception("No config file found and create=False")
            config = DefaultConfig
            config.to_json(os.path.join(directory, "config.json"))

        if config.build_version != __version__:
            if not allow_version_mismatch:
                raise Exception(
                    f"Version mismatch: \nRunning version: {__version__}\nDatabase"
                    f" version: {config.build_version}\nUse allow_version_mismatch=True"
                    " to ignore this error."
                )

        frame_idx_db = "sqlite:///" + os.path.join(directory, "index.db")
        frame_idx = FrameIndex(frame_idx_db, mode=mode)

        data_path = os.path.join(directory, "data")
        frame_db = FrameDB(
            frame_idx, data_path, config.data_file_max_size, config.nside, mode=mode
        )
        db = cls(frame_db, directory, config)
        db._load_limiting_magnitudes_cache()
        return db

    @classmethod
    def create(
        cls,
        directory: str,
        nside: int = DefaultConfig.nside,
        data_file_max_size: int = DefaultConfig.data_file_max_size,
    ):
        """
        Create a new database on disk in the given directory.
        """
        os.makedirs(directory, exist_ok=True)

        frame_idx_db = "sqlite:///" + os.path.join(directory, "index.db")
        frame_idx = FrameIndex(frame_idx_db, mode="w")

        config = Config(nside=nside, data_file_max_size=data_file_max_size)
        config.to_json(os.path.join(directory, "config.json"))

        data_path = os.path.join(directory, "data")
        os.makedirs(data_path, exist_ok=True)

        frame_db = FrameDB(frame_idx, data_path, data_file_max_size, nside)

        db = cls(frame_db, directory, config)
        db._load_limiting_magnitudes_cache()
        return db

    def precover(
        self,
        orbit: Orbits,
        tolerance: float = 30 * ARCSEC,
        start_mjd: Optional[float] = None,
        end_mjd: Optional[float] = None,
        window_size: int = 7,
        datasets: Optional[set[str]] = None,
        propagator_class: Optional[Type[Propagator]] = None,
        max_processes: Optional[int] = None,
        match_method: str = "circle",
        n_sigma: float = 3.0,
        covariance_polygon_vertices: int = 32,
        covariance_mc_num_samples: int = 64,
        covariance_mc_seed: int = 0,
    ) -> Tuple[PrecoveryCandidates, FrameCandidates]:
        """
        Find observations which match orbit in the database. Observations are
        searched in descending order by mjd.

        orbit: The orbit to match.

        start_mjd: Only consider observations from after this epoch
        (inclusive). If None, find all.

        end_mjd: Only consider observations from before this epoch (inclusive).
        If None, find all.

        datasets: Only consider observations from the indicated datasets.

        Returns
        -------
        Tuple[PrecoveryCandidates, FrameCandidates]
            Precovery candidate observations and frame candidates.
        """
        # basically:
        """
        find all windows between start and end of given size
        for each window:
            propagate to window center
            for each unique epoch,obscode in window:
                propagate to epoch
                find frames which match healpix of propagation
                for each matching frame
                    find matching observations
                    for each matching observation
                        yield match
        """

        if propagator_class is None:
            raise ValueError("A propagator must be provided to run precovery")

        assert len(orbit) == 1, "Use precovery_many for multiple orbits"

        orbit_id = orbit.orbit_id[0].as_py()

        # Normalize the orbit timescale to utc for comparisons
        orbit = orbit.set_column(
            "coordinates.time", orbit.coordinates.time.rescale("utc")
        )

        if datasets is not None:
            self._warn_for_missing_datasets(datasets)

        if start_mjd is None or end_mjd is None:
            first, last = self.frames.idx.mjd_bounds(datasets=datasets)
            if start_mjd is None:
                start_mjd = first
            if end_mjd is None:
                end_mjd = last

        logger.info(
            f"precovering orbit {orbit_id} from {start_mjd} to {end_mjd}, window={window_size}, datasets={datasets or 'all'}"
        )

        windows = self.frames.idx.window_centers(
            start_mjd, end_mjd, window_size, datasets=datasets
        )
        logger.info(f"Searching {len(windows)} windows")
        if len(windows) == 0:
            return PrecoveryCandidates.empty(), FrameCandidates.empty()

        # Runtime configuration overrides that must be respected inside `check_window`, which
        # re-opens the DB from disk.
        #
        # Only pass overrides that are explicitly set on this instance; otherwise we can
        # clobber values coming from config.json (e.g., tests that edit config.json after
        # creating the DB).
        config_overrides: dict[str, Any] = {}

        faint_margin = getattr(self.config, "faint_frame_skip_margin_mag", 0.0)
        try:
            faint_margin_f = float(faint_margin) if faint_margin is not None else 0.0
        except Exception:
            faint_margin_f = 0.0
        if faint_margin_f != 0.0:
            config_overrides["faint_frame_skip_margin_mag"] = faint_margin_f
        max_faint = getattr(self.config, "max_mag_residual_fainter_mag", None)
        if max_faint is not None:
            config_overrides["max_mag_residual_fainter_mag"] = float(max_faint)

        max_bright = getattr(self.config, "max_mag_residual_brighter_mag", None)
        if max_bright is not None:
            config_overrides["max_mag_residual_brighter_mag"] = float(max_bright)

        # Search all windows across all observatory codes in one pass so parallelism can
        # be applied across obscodes (not serialized per-obscode).
        candidates, frame_candidates = self._check_windows(
            windows,
            orbit,
            tolerance,
            propagator_class,
            datasets=datasets,
            max_processes=max_processes,
            match_method=match_method,
            n_sigma=n_sigma,
            covariance_polygon_vertices=covariance_polygon_vertices,
            covariance_mc_num_samples=covariance_mc_num_samples,
            covariance_mc_seed=covariance_mc_seed,
            config_overrides=config_overrides or None,
        )

        # convert these to our new output formats
        # Predicted magnitudes / mag_residual are computed during the search (in `_check_frames()`)
        # to avoid recomputation and to support rejection labeling.

        # Null out the temporary aberrated_coordinates column before returning
        if len(candidates) > 0:
            candidates = candidates.set_column(
                "aberrated_coordinates",
                CartesianCoordinates.from_kwargs(
                    x=pa.nulls(len(candidates), type=pa.float64()),
                    y=pa.nulls(len(candidates), type=pa.float64()),
                    z=pa.nulls(len(candidates), type=pa.float64()),
                    vx=pa.nulls(len(candidates), type=pa.float64()),
                    vy=pa.nulls(len(candidates), type=pa.float64()),
                    vz=pa.nulls(len(candidates), type=pa.float64()),
                    time=candidates.time,
                    origin=Origin.from_kwargs(code=pa.repeat("SUN", len(candidates))),
                    frame="ecliptic",
                ),
            )
        if len(frame_candidates) > 0:
            frame_candidates = frame_candidates.set_column(
                "aberrated_coordinates",
                CartesianCoordinates.from_kwargs(
                    x=pa.nulls(len(frame_candidates), type=pa.float64()),
                    y=pa.nulls(len(frame_candidates), type=pa.float64()),
                    z=pa.nulls(len(frame_candidates), type=pa.float64()),
                    vx=pa.nulls(len(frame_candidates), type=pa.float64()),
                    vy=pa.nulls(len(frame_candidates), type=pa.float64()),
                    vz=pa.nulls(len(frame_candidates), type=pa.float64()),
                    time=frame_candidates.exposure_time_mid,
                    origin=Origin.from_kwargs(
                        code=pa.repeat("SUN", len(frame_candidates))
                    ),
                    frame="ecliptic",
                ),
            )

        return candidates, frame_candidates

    # mypy helper: `_attach_magnitudes()` is implemented once, but it preserves the concrete
    # table type (PrecoveryCandidates in -> PrecoveryCandidates out; FrameCandidates in ->
    # FrameCandidates out). These overloads are for static typing only; no runtime effect.
    @overload
    def _attach_magnitudes(
        self, table: PrecoveryCandidates, orbit: Orbits
    ) -> PrecoveryCandidates:
        ...

    @overload
    def _attach_magnitudes(
        self, table: FrameCandidates, orbit: Orbits
    ) -> FrameCandidates:
        ...

    def _attach_magnitudes(
        self, table: Union[PrecoveryCandidates, FrameCandidates], orbit: Orbits
    ) -> Union[PrecoveryCandidates, FrameCandidates]:
        """
        Calculate and attach predicted magnitudes and residuals to the given table.
        """
        if len(table) == 0:
            return table

        if orbit.physical_parameters is None:
            return table

        # We assume len(orbit) == 1 as enforced in precover()
        H = orbit.physical_parameters.H_v[0].as_py()
        G = orbit.physical_parameters.G[0].as_py()

        if H is None:
            return table

        if G is None:
            G = 0.15

        # Need emission-time cartesian state for photometry.
        # We require this to compute H-G geometry. If it's missing, fail loudly so callers
        # don't accidentally interpret missing magnitudes as valid results.
        if pc.all(pc.is_null(table.aberrated_coordinates.x)).as_py():
            raise ValueError(
                "Cannot compute predicted magnitudes: aberrated_coordinates are missing/null. "
                "This likely indicates ephemeris generation did not attach emission-time cartesian "
                "states, or `_attach_magnitudes()` was called after aberrated_coordinates were stripped."
            )

        # Photometry requires heliocentric coordinates. The stored aberrated_coordinates
        # are in the barycentric ecliptic frame.
        obj_helio = transform_coordinates(
            table.aberrated_coordinates,
            CartesianCoordinates,
            frame_out="ecliptic",
            origin_out=OriginCodes.SUN,
        )

        # Build per-row exposures aligned with `obj_helio` time sampling.
        # For hits, center the exposure midpoint on the observation time to keep
        # observer/object epochs consistent in photometry.
        exposures = table.as_exposures()
        if isinstance(table, PrecoveryCandidates):
            exposures = exposures.set_column(
                "start_time",
                table.time.add_seconds(pc.multiply(table.exposure_duration, -0.5)),
            )

        # Canonicalize reported bands -> canonical filter_id strings expected by
        # bandpass photometry (e.g. 'g' @ W84 -> 'DECam_g').
        canonical = map_to_canonical_filter_bands(
            exposures.observatory_code,
            exposures.filter,
            allow_fallback_filters=True,
        )
        exposures = exposures.set_column(
            "filter", pa.array(canonical, type=pa.large_string())
        )

        # predict_magnitudes expects object_coords and exposures to be the same length.
        pred_mag = predict_magnitudes(
            H=H,
            object_coords=obj_helio,
            exposures=exposures,
            G=G,
            composition="C",
        )

        if isinstance(table, PrecoveryCandidates):
            out_candidates = PrecoveryCandidates.from_pyarrow(
                table.set_column("pred_mag", pred_mag).table
            )
            mag_residual = pc.subtract(out_candidates.mag, pred_mag)
            out_candidates = PrecoveryCandidates.from_pyarrow(
                out_candidates.set_column("mag_residual", mag_residual).table
            )
            return out_candidates

        out_frames = FrameCandidates.from_pyarrow(
            table.set_column("pred_mag", pred_mag).table
        )
        return out_frames

    def _check_windows(
        self,
        windows: WindowCenters,
        orbit: Orbits,
        tolerance: float,
        propagator_class: Type[Propagator],
        datasets: Optional[set[str]] = None,
        max_processes: Optional[int] = None,
        match_method: str = "circle",
        n_sigma: float = 3.0,
        covariance_polygon_vertices: int = 32,
        covariance_mc_num_samples: int = 64,
        covariance_mc_seed: int = 0,
        config_overrides: dict[str, Any] | None = None,
    ) -> Tuple[PrecoveryCandidates, FrameCandidates]:
        """
        Find all observations that match orbit within a list of windows
        """
        assert len(orbit) == 1, "_check_windows only support one orbit for now"
        windows = windows.sort_by(
            [("time.days", "descending"), ("time.nanos", "descending")]
        )
        logger.info(
            f"_check_windows orbit: {orbit.orbit_id[0].as_py()} windows: {len(windows)} obscode: {windows.obscode.unique().to_pylist()}"
        )

        precovery_candidates = PrecoveryCandidates.empty()
        frame_candidates = FrameCandidates.empty()

        use_ray = False
        if max_processes is not None and max_processes > 1:
            try:
                use_ray = initialize_use_ray(num_cpus=max_processes)
            except Exception as exc:
                logger.warning(
                    "Failed to initialize Ray; falling back to single-process window search. "
                    f"Error: {exc}"
                )
                use_ray = False

        if use_ray:
            futures = []
            for window in windows:
                futures.append(
                    check_window_remote.remote(
                        db_dir=self.directory,
                        window=window,
                        # Note: There is no speed benefit to pre-propagating
                        # the orbit to the window center here, since we do
                        # the n-body propagation inside the worker and the
                        # delay to start the job offsets any advantage
                        # from prepropagation.
                        orbit=orbit,
                        tolerance=tolerance,
                        propagator_class=propagator_class,
                        datasets=datasets,
                        match_method=match_method,
                        n_sigma=n_sigma,
                        covariance_polygon_vertices=covariance_polygon_vertices,
                        covariance_mc_num_samples=covariance_mc_num_samples,
                        covariance_mc_seed=covariance_mc_seed,
                        config_overrides=config_overrides,
                    )
                )

                if len(futures) >= max_processes * 2:
                    finished, futures = ray.wait(futures, num_returns=1)
                    precovery_candidates_window, frame_candidates_window = ray.get(
                        finished[0]
                    )
                    precovery_candidates = qv.concatenate(
                        [precovery_candidates, precovery_candidates_window]
                    )
                    frame_candidates = qv.concatenate(
                        [frame_candidates, frame_candidates_window]
                    )

            while len(futures) > 0:
                finished, futures = ray.wait(futures, num_returns=1)
                precovery_candidates_window, frame_candidates_window = ray.get(
                    finished[0]
                )
                precovery_candidates = qv.concatenate(
                    [precovery_candidates, precovery_candidates_window]
                )
                frame_candidates = qv.concatenate(
                    [frame_candidates, frame_candidates_window]
                )
        else:
            propagator = propagator_class()
            for window in windows:
                # For single process, we propagate the orbit
                # to the window center in a loop to avoid
                # duplicating the n-body propagation inside
                # check_window
                use_covariance = (
                    orbit.coordinates.covariance is not None
                    and not orbit.coordinates.covariance.is_all_nan()
                )
                orbit = propagator.propagate_orbits(orbit, window.time, covariance=use_covariance)
                candidates_window, frame_candidates_window = check_window(
                    self.directory,
                    window,
                    orbit=orbit,
                    tolerance=tolerance,
                    propagator_class=propagator_class,
                    datasets=datasets,
                    match_method=match_method,
                    n_sigma=n_sigma,
                    covariance_polygon_vertices=covariance_polygon_vertices,
                    covariance_mc_num_samples=covariance_mc_num_samples,
                    covariance_mc_seed=covariance_mc_seed,
                    config_overrides=config_overrides,
                )
                precovery_candidates = qv.concatenate(
                    [precovery_candidates, candidates_window]
                )
                frame_candidates = qv.concatenate(
                    [frame_candidates, frame_candidates_window]
                )
        return precovery_candidates, frame_candidates

    def _check_frames(
        self,
        orbit: Orbits,
        generic_frames: GenericFrame,
        obscode: str,
        tolerance: float,
        datasets: Optional[set[str]],
        propagator_class: Type[Propagator],
        match_method: str = "circle",
        n_sigma: float = 3.0,
    ) -> Tuple[PrecoveryCandidates, FrameCandidates]:
        """
        Deeply inspect all frames that match the given obscode, mjd, and healpix to
        see if they contain observations which match the ephemeris.
        """
        frames = HealpixFrame.empty()
        for generic_frame in generic_frames:
            frames = qv.concatenate(
                [
                    frames,
                    self.frames.idx.get_frames(
                        obscode,
                        generic_frame.time.mjd()[0].as_py(),
                        generic_frame.healpixel[0].as_py(),
                        datasets,
                    ),
                ]
            )
        if len(frames) == 0:
            return PrecoveryCandidates.empty(), FrameCandidates.empty()

        unique_frame_times = frames.exposure_mid_timestamp().unique()
        observers = Observers.from_code(obscode, unique_frame_times)
        # Compute the position of the ephem carefully.
        propagator = propagator_class()

        # Optional performance feature: if we have configured limiting magnitudes,
        # we can skip deep inspection of frames where the object is predicted to be
        # fainter than the instrument limit.
        # Prefer the generated cache file (loaded once at DB open), falling back to config dicts.
        limit_codefid_keys = self._limit_codefid_keys
        limit_codefid_vals = self._limit_codefid_vals
        faint_margin = float(
            getattr(self.config, "faint_frame_skip_margin_mag", 0.0) or 0.0
        )
        # Magnitudes are only computable if the orbit has H (and a defaultable G).
        try:
            H = (
                orbit.physical_parameters.H_v[0].as_py()
                if orbit.physical_parameters is not None
                else None
            )
            G = (
                orbit.physical_parameters.G[0].as_py()
                if orbit.physical_parameters is not None
                else None
            )
        except Exception:
            H = None
            G = None
        compute_pred_mags = H is not None
        if G is None:
            G = 0.15

        enable_faint_skip = (
            compute_pred_mags
            and limit_codefid_keys is not None
            and len(limit_codefid_keys) > 0
        )
        ephemeris = propagator.generate_ephemeris(
            orbit, observers, predict_magnitudes=compute_pred_mags
        )

        # Align ephemeris rows to frames by exposure-midpoint time (vectorized),
        # then iterate row-wise without repeated time masking.
        frame_mid = frames.exposure_mid_timestamp().rounded("us")
        eph_mid = ephemeris.coordinates.time.rounded("us")

        frame_key = self._timestamp_key(frame_mid.days, frame_mid.nanos)
        eph_key = self._timestamp_key(eph_mid.days, eph_mid.nanos)
        idx = pc.fill_null(pc.index_in(frame_key, value_set=eph_key), -1)
        assert pc.all(pc.greater_equal(idx, 0)).as_py(), "No matching ephemeris found"
        idx64 = pc.cast(idx, pa.int64())
        ephem_for_frames = Ephemeris.from_pyarrow(ephemeris.table.take(idx64))

        precovery_candidates = PrecoveryCandidates.empty()
        frame_candidates = FrameCandidates.empty()

        # Compute per-frame predicted magnitudes (in the frame's canonical filter),
        # and optionally flag frames as "too faint" based on limiting magnitudes.
        canon_arr = pa.nulls(len(frames), type=pa.large_string())
        deltas = pa.nulls(len(frames), type=pa.float64())
        pred_mag_band = pa.nulls(len(frames), type=pa.float64())
        too_faint = pa.repeat(False, len(frames))
        if (
            compute_pred_mags
            and hasattr(ephem_for_frames, "predicted_magnitude_v")
            and not pc.all(pc.is_null(ephem_for_frames.predicted_magnitude_v)).as_py()
        ):
            try:
                canonical = map_to_canonical_filter_bands(
                    frames.obscode,
                    frames.filter,
                    allow_fallback_filters=True,
                )
                canon_arr = pa.array(canonical, type=pa.large_string())

                # Convert ephemeris predicted V magnitudes into each frame's canonical band.
                uniq = sorted(set(map(str, canonical)))
                delta_map = {fid: bandpass_delta_mag("C", "V", fid) for fid in uniq}
                delta_keys, delta_vals = self._dict_to_kv_arrays(delta_map)
                deltas = self._arrow_lookup(canon_arr, delta_keys, delta_vals)
                pred_mag_band = pc.add(ephem_for_frames.predicted_magnitude_v, deltas)

                if enable_faint_skip:
                    # Build per-frame limiting magnitude (require obscode|filter_id cache).
                    sep = pa.scalar("|", type=pa.large_string())
                    codefid = pc.binary_join_element_wise(
                        frames.obscode, canon_arr, sep
                    )
                    limit = self._arrow_lookup(
                        codefid, limit_codefid_keys, limit_codefid_vals
                    )

                    # Too faint if predicted magnitude is greater (numerically) than limit+margin.
                    limit_with_margin = pc.add(limit, faint_margin)
                    too_faint = pc.fill_null(
                        pc.and_(
                            pc.is_valid(limit),
                            pc.greater(pred_mag_band, limit_with_margin),
                        ),
                        False,
                    )
            except Exception as e:
                logger.warning(f"Unable to compute per-frame magnitudes: {e}")

        for i, (f, matching_ephem) in enumerate(zip(frames, ephem_for_frames)):
            is_too_faint = bool(too_faint[i].as_py())
            pred_mag_frame_py = pred_mag_band[i].as_py()
            pred_mag_frame = (
                None if pred_mag_frame_py is None else float(pred_mag_frame_py)
            )

            if is_too_faint:
                frame_candidates = qv.concatenate(
                    [
                        frame_candidates,
                        frame_candidates_from_frame(
                            f,
                            matching_ephem,
                            pred_mag=pred_mag_frame,
                            rejected=True,
                            rejected_reason=REJECT_REASON_LIMITING_MAGNITUDE,
                        ),
                    ]
                )
                continue

            delta_py = deltas[i].as_py()
            delta_i = None if delta_py is None else float(delta_py)
            matches = self.find_matches_in_frame(
                f,
                orbit,
                matching_ephem,
                tolerance,
                propagator,
                match_method=match_method,
                n_sigma=n_sigma,
                pred_mag_delta=delta_i,
                compute_pred_mags=compute_pred_mags,
                max_mag_residual_fainter_mag=getattr(
                    self.config, "max_mag_residual_fainter_mag", None
                ),
                max_mag_residual_brighter_mag=getattr(
                    self.config, "max_mag_residual_brighter_mag", None
                ),
            )
            # If no observations were found in this frame then we
            # return frame candidates
            # Note that for the frame candidate we report the predicted
            # ephemeris at the exposure midpoint not at the observation
            # times which may differ from the exposure midpoint time
            if len(matches) == 0:
                frame_candidates = qv.concatenate(
                    [
                        frame_candidates,
                        frame_candidates_from_frame(
                            f,
                            matching_ephem,
                            pred_mag=pred_mag_frame,
                            rejected=False,
                            rejected_reason=None,
                        ),
                    ]
                )
            else:
                precovery_candidates = qv.concatenate([precovery_candidates, matches])
        return precovery_candidates, frame_candidates

    def find_matches_in_frame(
        self,
        frame: HealpixFrame,
        orbit: Orbits,
        frame_ephem: Ephemeris,
        tolerance: float,
        propagator: Propagator,
        match_method: str = "circle",
        n_sigma: float = 3.0,
        *,
        pred_mag_delta: float | None = None,
        compute_pred_mags: bool = False,
        max_mag_residual_fainter_mag: float | None = None,
        max_mag_residual_brighter_mag: float | None = None,
    ) -> PrecoveryCandidates:
        """
        Find all sources in a single frame which match ephem.
        """
        assert len(frame_ephem) == 1, "ephem should have only one entry"

        # Gather all observations.
        observations: ObservationsTable = self.frames.get_observations(frame)

        # For covariance-based matching, always generate a per-observation ephemeris using
        # the 2-body pipeline so that ephemeris covariances are available (when orbit covariances exist).
        obscode = frame.obscode[0].as_py()

        if match_method in ("covariance", "covariance_mc", "covariance_polygon"):
            per_obs_ephem = generate_ephem_for_per_obs_timestamps(
                orbit,
                observations,
                obscode,
                propagator,
                predict_magnitudes=compute_pred_mags,
            )
            matching_observations, matching_ephem = find_observation_matches_covariance(
                observations, per_obs_ephem, n_sigma=n_sigma
            )

        else:
            # Check if the observations have per-observation MJDs.
            # If so we use 2 body to generate unique ephemeris for each
            if len(observations.time.unique()) > 1:
                per_obs_ephem = generate_ephem_for_per_obs_timestamps(
                    orbit, observations, obscode, propagator
                )
                matching_observations, matching_ephem = find_observation_matches(
                    observations, per_obs_ephem, tolerance
                )
            # Otherwise the default state is to use the same ephemeris
            # for each observation in the frame
            else:
                repeated_ephem = Ephemeris.from_pyarrow(
                    frame_ephem.table.take(np.zeros(len(observations), dtype=int))
                )
                matching_observations, matching_ephem = find_observation_matches(
                    observations, repeated_ephem, tolerance
                )

        if len(matching_observations) == 0:
            return PrecoveryCandidates.empty()

        pred_mag = pa.nulls(len(matching_observations), type=pa.float64())
        mag_residual = pa.nulls(len(matching_observations), type=pa.float64())
        rejected = pa.repeat(False, len(matching_observations))
        rejected_reason = pa.array(
            [None] * len(matching_observations), type=pa.large_string()
        )

        if (
            compute_pred_mags
            and pred_mag_delta is not None
            and hasattr(matching_ephem, "predicted_magnitude_v")
            and not pc.all(pc.is_null(matching_ephem.predicted_magnitude_v)).as_py()
        ):
            delta_s = pa.scalar(float(pred_mag_delta), type=pa.float64())
            pred_mag = pc.add(matching_ephem.predicted_magnitude_v, delta_s)
            mag_residual = pc.subtract(matching_observations.mag, pred_mag)

            # Asymmetric outlier rejection (preferred).
            outlier = None
            if max_mag_residual_fainter_mag is not None:
                too_faint = pc.greater(
                    mag_residual,
                    pa.scalar(float(max_mag_residual_fainter_mag), type=pa.float64()),
                )
                outlier = too_faint if outlier is None else pc.or_(outlier, too_faint)
            if max_mag_residual_brighter_mag is not None:
                too_bright = pc.less(
                    mag_residual,
                    pa.scalar(-float(max_mag_residual_brighter_mag), type=pa.float64()),
                )
                outlier = too_bright if outlier is None else pc.or_(outlier, too_bright)

            if outlier is not None:
                outlier = pc.fill_null(outlier, False)
                rejected = _as_bool_array(outlier)
                rejected_reason = pc.if_else(
                    outlier,
                    pa.scalar(REJECT_REASON_MAG_RESIDUAL, type=pa.large_string()),
                    pa.scalar(None, type=pa.large_string()),
                )

        candidates = candidates_from_ephem(
            matching_observations,
            matching_ephem,
            frame,
            pred_mag=pred_mag,
            mag_residual=mag_residual,
            rejected=rejected,
            rejected_reason=rejected_reason,
        )
        return candidates

    def find_observations_in_region(
        self, ra: float, dec: float, obscode: str
    ) -> ObservationsTable:
        """Gets all the Observations within the same Healpixel as a
        given RA, Dec for a particular observatory (specified as an obscode).

        """
        frames = self.frames.get_frames_for_ra_dec(ra, dec, obscode)
        observations = ObservationsTable.empty()
        for f in frames:
            observations = qv.concatenate(
                [observations, self.frames.get_observations(f)]
            )
        return observations

    def find_observations_in_radius(
        self, ra: float, dec: float, tolerance: float, obscode: str
    ) -> ObservationsTable:
        """Gets all the Observations within a radius (in degrees) of
        a particular RA and Dec at a specific observatory.

        This method is approximate, and does not correctly find
        Observations that are within the radius, but on a different
        healpixel.

        """
        obs_in_region = self.find_observations_in_region(ra, dec, obscode)
        mask = pc.less_equal(
            haversine_distance_deg(
                obs_in_region.ra.to_numpy(), ra, obs_in_region.dec.to_numpy(), dec
            ),
            tolerance,
        )
        return obs_in_region.apply_mask(mask)

    def all_datasets(self) -> set[str]:
        """
        Returns the set of all dataset ID strings loaded in the database.
        """
        return set(self.frames.idx.get_dataset_ids())

    def _warn_for_missing_datasets(self, datasets: set[str]):
        """Log some warning messages if the given set includes
        dataset IDs which are not present in the database.

        """
        any_missing = False
        for ds in datasets:
            if not self.frames.has_dataset(ds):
                any_missing = True
                logger.warn(f'dataset "{ds}" is not in the database')
        if any_missing:
            logger.warn(f"datasets in the databse: {self.all_datasets()}")
