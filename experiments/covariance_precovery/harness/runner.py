from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pyarrow as pa
import quivr as qv

from precovery.precovery_db import PrecoveryCandidates, PrecoveryDatabase


@dataclass(frozen=True)
class RunConfig:
    run_id: str
    match_method: str = "circle"
    n_sigma: float = 3.0
    tolerance_deg: float = 1.0 / 3600.0
    window_size_days: int = 7
    start_mjd: float | None = None
    end_mjd: float | None = None
    datasets: set[str] | None = None
    max_processes: int | None = None
    covariance_polygon_vertices: int = 32
    covariance_mc_num_samples: int = 64
    covariance_mc_seed: int = 0


class RunMetrics(qv.Table):
    run_id = qv.LargeStringColumn()
    match_method = qv.LargeStringColumn()
    n_sigma = qv.Float64Column()
    tolerance_deg = qv.Float64Column()
    window_size_days = qv.Int64Column()
    start_mjd = qv.Float64Column(nullable=True)
    end_mjd = qv.Float64Column(nullable=True)
    max_processes = qv.Int64Column(nullable=True)
    runtime_sec = qv.Float64Column()
    n_candidates = qv.Int64Column()
    n_frame_candidates = qv.Int64Column()


@dataclass(frozen=True)
class RunOutput:
    metrics: RunMetrics
    candidates: PrecoveryCandidates
    frame_candidates: qv.Table


def run_precovery_builtin(
    *,
    db: PrecoveryDatabase,
    orbit,
    cfg: RunConfig,
    propagator_class,
):
    t0 = time.time()
    candidates, frame_candidates = db.precover(
        orbit,
        tolerance=float(cfg.tolerance_deg),
        window_size=int(cfg.window_size_days),
        start_mjd=cfg.start_mjd,
        end_mjd=cfg.end_mjd,
        datasets=cfg.datasets,
        propagator_class=propagator_class,
        max_processes=cfg.max_processes,
        match_method=str(cfg.match_method),
        n_sigma=float(cfg.n_sigma),
        covariance_polygon_vertices=int(cfg.covariance_polygon_vertices),
        covariance_mc_num_samples=int(cfg.covariance_mc_num_samples),
        covariance_mc_seed=int(cfg.covariance_mc_seed),
    )
    dt = time.time() - t0

    metrics = RunMetrics.from_kwargs(
        run_id=[cfg.run_id],
        match_method=[str(cfg.match_method)],
        n_sigma=[float(cfg.n_sigma)],
        tolerance_deg=[float(cfg.tolerance_deg)],
        window_size_days=[int(cfg.window_size_days)],
        start_mjd=[cfg.start_mjd],
        end_mjd=[cfg.end_mjd],
        max_processes=[cfg.max_processes],
        runtime_sec=[float(dt)],
        n_candidates=[int(len(candidates))],
        n_frame_candidates=[int(len(frame_candidates))],
    )
    return RunOutput(metrics=metrics, candidates=candidates, frame_candidates=frame_candidates)


def run_matrix(
    *,
    db: PrecoveryDatabase,
    orbit,
    configs: Sequence[RunConfig],
    propagator_class,
    out_parquet: Path | None = None,
) -> RunMetrics:
    """
    Execute a list of run configurations and return concatenated metrics.

    This is the initial (builtin) harness: it runs `PrecoveryDatabase.precover` using
    existing match methods (`circle`, `covariance`, `covariance_polygon`, `covariance_mc`).
    """
    all_metrics = RunMetrics.empty()
    for cfg in configs:
        out = run_precovery_builtin(db=db, orbit=orbit, cfg=cfg, propagator_class=propagator_class)
        all_metrics = qv.concatenate([all_metrics, out.metrics])

    if out_parquet is not None:
        out_parquet.parent.mkdir(parents=True, exist_ok=True)
        all_metrics.to_parquet(str(out_parquet))
    return all_metrics

