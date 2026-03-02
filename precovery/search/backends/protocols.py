from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from ..pipeline_types import (
    AcceptedCounts,
    BenchTargets,
    CandidateDetections,
    PredictedTargets,
    PredictedTriples,
    SubsetPaths,
)


@dataclass(frozen=True)
class GateParams:
    innovation_gate_n_sigma: float = 3.0
    invalid_sigma_fill_floor_arcsec_global: float = 0.10
    # Optional per-obscode sigma floor overrides (arcsec).
    #
    # When provided, the effective floor for each candidate row is:
    #   floor_arcsec = invalid_sigma_fill_floor_arcsec_by_obscode.get(
    #       obscode,
    #       invalid_sigma_fill_floor_arcsec_global,
    #   )
    invalid_sigma_fill_floor_arcsec_by_obscode: dict[str, float] | None = None
    # Optional per-obscode systematic sigma inflation (arcsec), applied in quadrature:
    #   sigma_eff = sqrt(sigma^2 + sigma_sys^2)
    #
    # To avoid globally loosening the gate, this can be applied only when the reported
    # per-detection sigma RMS is below a per-obscode trigger threshold:
    #   sqrt(sigma_lon^2 + sigma_lat^2) * 3600 < trigger_rms_arcsec
    sigma_systematic_arcsec_by_obscode: dict[str, float] | None = None
    apply_systematic_if_reported_rms_lt_arcsec_by_obscode: dict[str, float] | None = None
    # Asymmetric magnitude residual thresholds (in magnitudes).
    # mag_residual = observed_mag - predicted_mag
    # - If mag_residual > max_mag_residual_fainter_mag => too faint => reject
    # - If mag_residual < -max_mag_residual_brighter_mag => too bright => reject
    max_mag_residual_fainter_mag: float | None = None
    max_mag_residual_brighter_mag: float | None = None

    @property
    def n_sigma(self) -> float:
        return float(self.innovation_gate_n_sigma)

    @property
    def det_sigma_floor_arcsec(self) -> float:
        return float(self.invalid_sigma_fill_floor_arcsec_global)

    @property
    def det_sigma_floor_arcsec_by_obscode(self) -> dict[str, float] | None:
        return self.invalid_sigma_fill_floor_arcsec_by_obscode

    @property
    def det_sigma_sys_arcsec_by_obscode(self) -> dict[str, float] | None:
        return self.sigma_systematic_arcsec_by_obscode

    @property
    def det_sigma_sys_apply_rms_lt_arcsec_by_obscode(self) -> dict[str, float] | None:
        return self.apply_systematic_if_reported_rms_lt_arcsec_by_obscode


@dataclass(frozen=True)
class BackendCapabilities:
    """
    Declare optional features so callers can choose the cheapest path.
    """

    supports_enumerate_targets: bool = True
    supports_sql_gate_counts: bool = False
    supports_sql_gate_rows: bool = False
    supports_filter_triples_to_existing_frames: bool = False
    supports_frame_pixels_by_target: bool = False
    supports_fetch_candidates_from_triples_parquet: bool = False
    supports_count_accepted_from_triples_parquet: bool = False
    supports_detection_key_match_totals_from_triples_parquet: bool = False


class SearchBackend(Protocol):
    """
    Minimal backend contract for precovery search + benchmarking.

    Backends are expected to be batch-oriented and return Arrow/Quivr-friendly tables.
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

        If `obscodes` is empty, implementations should treat it as "no station filter".
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
        """

    def filter_triples_to_existing_frames(
        self,
        *,
        subset: SubsetPaths,
        triples: PredictedTriples,
    ) -> PredictedTriples:
        """
        Optionally reduce `triples` to keys that exist in the detections dataset.

        Backends that do not support this efficiently should return `triples` unchanged.
        """

    def frame_pixels_by_target(
        self,
        *,
        subset: SubsetPaths,
        targets: BenchTargets,
    ) -> dict[tuple[str, int], np.ndarray]:
        """
        Return a mapping:
          (obscode, exposure_mjd_mid_key_us) -> np.ndarray[healpixel]

        This is the production “restrict footprints to dataset frames” primitive. Implementations
        that cannot support it cheaply should return an empty dict.
        """

    def fetch_candidates_from_triples_parquet(
        self,
        *,
        subset: SubsetPaths,
        triples_parquet: str,
        limit: int | None = None,
    ) -> CandidateDetections:
        """
        Optional fast path for file-first execution: fetch candidates by joining against
        a parquet file containing PredictedTriples-compatible columns.
        """

    def count_accepted_from_triples_parquet(
        self,
        *,
        subset: SubsetPaths,
        triples_parquet: str,
        preds: PredictedTargets,
        gate: GateParams,
    ) -> AcceptedCounts:
        """
        Optional fast path for file-first execution: return accepted counts by reading
        triples from parquet (without requiring in-memory triple materialization).
        """

    def detection_key_match_totals_from_triples_parquet(
        self,
        *,
        subset: SubsetPaths,
        triples_parquet: str,
    ) -> tuple[int | None, int | None, int | None]:
        """
        Optional backend-provided totals for selected-key accounting:
          (n_detections_selected_exposure_keys,
           n_detections_selected_frame_keys,
           n_detections_healpixel_nonmatch)

        Implementations that cannot provide this cheaply should return
        (None, None, None).
        """
