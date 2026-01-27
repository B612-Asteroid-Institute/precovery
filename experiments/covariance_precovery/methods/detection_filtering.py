from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pyarrow as pa
import quivr as qv

from precovery.observation import ObservationsTable
from precovery.precovery_db import find_observation_matches_covariance
from precovery.spherical_geom import haversine_distance_deg

from .footprints import Footprint


@dataclass(frozen=True)
class FilterMetrics:
    n_in: int
    n_after_footprint: int | None
    n_after_chi2: int


class DetectionFilter(Protocol):
    name: str

    def run(
        self,
        *,
        observations: ObservationsTable,
        ephem,  # Ephemeris
        footprint: Footprint | None,
        n_sigma: float,
    ) -> tuple[ObservationsTable, object, FilterMetrics]: ...


class Chi2OnlyFilter:
    name = "chi2_only"

    def run(
        self,
        *,
        observations: ObservationsTable,
        ephem,
        footprint: Footprint | None,
        n_sigma: float,
    ):
        obs2, eph2 = find_observation_matches_covariance(observations, ephem, n_sigma=float(n_sigma))
        return obs2, eph2, FilterMetrics(n_in=len(observations), n_after_footprint=None, n_after_chi2=len(obs2))


class FootprintThenChi2Filter:
    name = "footprint_then_chi2"

    def run(
        self,
        *,
        observations: ObservationsTable,
        ephem,
        footprint: Footprint | None,
        n_sigma: float,
    ):
        if footprint is None:
            return Chi2OnlyFilter().run(observations=observations, ephem=ephem, footprint=None, n_sigma=n_sigma)

        lon = observations.ra.to_numpy(zero_copy_only=False).astype(np.float64)
        lat = observations.dec.to_numpy(zero_copy_only=False).astype(np.float64)
        keep = footprint.contains(lon, lat)
        obs_fp = observations.apply_mask(pa.array(keep))
        eph_fp = ephem.apply_mask(pa.array(keep))

        if len(obs_fp) == 0:
            return ObservationsTable.empty(), ephem.take(np.array([], dtype=np.int64)), FilterMetrics(
                n_in=len(observations),
                n_after_footprint=0,
                n_after_chi2=0,
            )

        obs2, eph2 = find_observation_matches_covariance(obs_fp, eph_fp, n_sigma=float(n_sigma))
        return obs2, eph2, FilterMetrics(
            n_in=len(observations), n_after_footprint=len(obs_fp), n_after_chi2=len(obs2)
        )


def score_mog_tangent_plane(
    *,
    obs_ra_deg: np.ndarray,
    obs_dec_deg: np.ndarray,
    sample_ra_deg: np.ndarray,
    sample_dec_deg: np.ndarray,
    sigma_arcsec: float,
) -> np.ndarray:
    """
    Cheap mixture-of-Gaussians score: treat the sample cloud as a KDE with fixed spherical Gaussian kernel.

    Returns unnormalized log-likelihood for each observation point.
    """
    if len(sample_ra_deg) == 0:
        return np.full(len(obs_ra_deg), -np.inf, dtype=np.float64)
    sig_deg = float(sigma_arcsec) / 3600.0
    sig2 = sig_deg * sig_deg
    # Compute great-circle distances to each sample and do log-sum-exp of -0.5 d^2/sig^2.
    # This is O(N_obs * N_samples) and intended for small sample counts (sigma points / small MC).
    ll = np.empty(len(obs_ra_deg), dtype=np.float64)
    for i in range(len(obs_ra_deg)):
        d = haversine_distance_deg(sample_ra_deg, obs_ra_deg[i], sample_dec_deg, obs_dec_deg[i])
        w = -0.5 * (d * d) / sig2
        m = float(np.max(w))
        ll[i] = m + float(np.log(np.mean(np.exp(w - m))))
    return ll

