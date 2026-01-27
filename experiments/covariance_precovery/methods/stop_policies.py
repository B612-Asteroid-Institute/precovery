from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class BatchStats:
    batch_id: int
    n_times: int
    n_pixels: int
    n_frames: int
    n_detections_retrieved: int
    n_after_footprint: int
    n_after_chi2: int
    n_truth_matched: int | None = None
    n_truth_total: int | None = None

    @property
    def efficiency(self) -> float | None:
        if self.n_detections_retrieved <= 0:
            return None
        if self.n_truth_matched is None:
            return None
        return float(self.n_truth_matched) / float(self.n_detections_retrieved)

    @property
    def recall(self) -> float | None:
        if self.n_truth_total is None or self.n_truth_total <= 0:
            return None
        if self.n_truth_matched is None:
            return None
        return float(self.n_truth_matched) / float(self.n_truth_total)


@dataclass(frozen=True)
class StopDecision:
    stop: bool
    reason: str
    # Optional control outputs for adaptive strategies.
    suggested_n_sigma: float | None = None


class StopPolicy(Protocol):
    name: str

    def update(self, stats: BatchStats) -> StopDecision: ...


class GuardrailStopPolicy:
    """
    Hard caps to prevent runaway compute.
    """

    name = "guardrails"

    def __init__(
        self,
        *,
        max_pixels: int | None = None,
        max_frames: int | None = None,
        max_detections_retrieved: int | None = None,
    ):
        self.max_pixels = max_pixels
        self.max_frames = max_frames
        self.max_detections_retrieved = max_detections_retrieved

    def update(self, stats: BatchStats) -> StopDecision:
        if self.max_pixels is not None and stats.n_pixels > self.max_pixels:
            return StopDecision(True, f"n_pixels {stats.n_pixels} > max_pixels {self.max_pixels}")
        if self.max_frames is not None and stats.n_frames > self.max_frames:
            return StopDecision(True, f"n_frames {stats.n_frames} > max_frames {self.max_frames}")
        if self.max_detections_retrieved is not None and stats.n_detections_retrieved > self.max_detections_retrieved:
            return StopDecision(True, f"n_detections_retrieved {stats.n_detections_retrieved} > max {self.max_detections_retrieved}")
        return StopDecision(False, "ok")


class HysteresisThresholdStopPolicy:
    """
    Stop when a scalar stays above a threshold for K consecutive batches.

    This is meant to handle the \"inflate then shrink\" behavior: we don't stop on a single spike.
    """

    name = "hysteresis_threshold"

    def __init__(self, *, field: str, threshold: float, consecutive: int = 3):
        self.field = field
        self.threshold = float(threshold)
        self.consecutive = int(consecutive)
        self._count = 0

    def update(self, stats: BatchStats) -> StopDecision:
        value = getattr(stats, self.field)
        if float(value) > self.threshold:
            self._count += 1
        else:
            self._count = 0
        if self._count >= self.consecutive:
            return StopDecision(True, f"{self.field} above threshold for {self.consecutive} batches")
        return StopDecision(False, "ok")


class EfficiencyStopPolicy:
    """
    Stop when truth efficiency stays below a threshold for K consecutive batches.
    """

    name = "efficiency_stop"

    def __init__(self, *, min_efficiency: float, consecutive: int = 3):
        self.min_efficiency = float(min_efficiency)
        self.consecutive = int(consecutive)
        self._count = 0

    def update(self, stats: BatchStats) -> StopDecision:
        eff = stats.efficiency
        if eff is None:
            return StopDecision(False, "no_truth_efficiency")
        if eff < self.min_efficiency:
            self._count += 1
        else:
            self._count = 0
        if self._count >= self.consecutive:
            return StopDecision(True, f"efficiency below {self.min_efficiency} for {self.consecutive} batches")
        return StopDecision(False, "ok")


class AdaptiveSigmaPolicy:
    """
    Suggest adjusting n_sigma to preserve a recall floor.

    This does NOT stop by itself; it only emits `suggested_n_sigma`.
    """

    name = "adaptive_sigma"

    def __init__(self, *, recall_floor: float = 0.99, step: float = 0.25, min_sigma: float = 2.0, max_sigma: float = 5.0):
        self.recall_floor = float(recall_floor)
        self.step = float(step)
        self.min_sigma = float(min_sigma)
        self.max_sigma = float(max_sigma)
        self._current = 3.0

    @property
    def current(self) -> float:
        return self._current

    def update(self, stats: BatchStats) -> StopDecision:
        rec = stats.recall
        if rec is None:
            return StopDecision(False, "no_recall", suggested_n_sigma=self._current)
        if rec < self.recall_floor:
            self._current = min(self.max_sigma, self._current + self.step)
            return StopDecision(False, f"increase_sigma_for_recall({rec:.3f})", suggested_n_sigma=self._current)
        # If comfortably above floor, allow tightening slowly.
        if rec > min(0.999, self.recall_floor + 0.01):
            self._current = max(self.min_sigma, self._current - self.step)
            return StopDecision(False, f"decrease_sigma_for_cost({rec:.3f})", suggested_n_sigma=self._current)
        return StopDecision(False, "sigma_ok", suggested_n_sigma=self._current)


class CompositeStopPolicy:
    name = "composite"

    def __init__(self, policies: list[StopPolicy]):
        self.policies = policies

    def update(self, stats: BatchStats) -> StopDecision:
        suggested: float | None = None
        reasons: list[str] = []
        for p in self.policies:
            d = p.update(stats)
            reasons.append(f"{p.name}:{d.reason}")
            if d.suggested_n_sigma is not None:
                suggested = d.suggested_n_sigma
            if d.stop:
                return StopDecision(True, "; ".join(reasons), suggested_n_sigma=suggested)
        return StopDecision(False, "; ".join(reasons), suggested_n_sigma=suggested)

