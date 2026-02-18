from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ..types import AcceptedCounts, BenchTargets, CandidateDetections, PredictedTargets, PredictedTriples, SubsetPaths


@dataclass(frozen=True)
class GateParams:
    n_sigma: float = 3.0
    det_sigma_floor_arcsec: float = 0.10
    # Asymmetric magnitude residual thresholds (in magnitudes).
    # mag_residual = observed_mag - predicted_mag
    # - If mag_residual > max_mag_residual_fainter_mag => too faint => reject
    # - If mag_residual < -max_mag_residual_brighter_mag => too bright => reject
    max_mag_residual_fainter_mag: float | None = None
    max_mag_residual_brighter_mag: float | None = None


@dataclass(frozen=True)
class BackendCapabilities:
    """
    Declare optional features so the harness can choose the cheapest path.
    """

    supports_enumerate_targets: bool = True
    supports_sql_gate_counts: bool = False
    supports_sql_gate_rows: bool = False


class BenchBackend(Protocol):
    """
    Minimal backend contract for precovery benchmarking.

    Backends are expected to be *batch oriented* and return Arrow/Quivr-friendly tables.
    """

    name: str
    capabilities: BackendCapabilities

    def enumerate_targets(
        self,
        *,
        subset: SubsetPaths,
        start_mjd_utc: float,
        end_mjd_utc: float,
        obscodes: tuple[str, ...],
    ) -> BenchTargets:
        """
        Return distinct (obscode, exposure_mjd_mid) targets in [start,end).
        """

    def fetch_candidates(
        self,
        *,
        subset: SubsetPaths,
        triples: PredictedTriples,
        limit: int | None = None,
    ) -> CandidateDetections:
        """
        Fetch candidate detections matching the (obscode, time_key, healpix) triples.

        `orbit_id` and `target_idx` should be preserved by joining through the `triples`
        input (the backend does not infer them).
        """

    def count_accepted(
        self,
        *,
        subset: SubsetPaths,
        triples: PredictedTriples,
        preds: PredictedTargets,
        gate: GateParams,
    ) -> AcceptedCounts:
        """
        Return accepted counts per (orbit_id, target_idx).

        Backends may implement this in SQL (preferred when inexpensive) or by
        fetching candidate rows and delegating to Python gating.
        """

