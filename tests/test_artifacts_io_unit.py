from __future__ import annotations

from pathlib import Path

import pyarrow as pa

from precovery.search.artifacts import read_stage23_artifacts, write_stage23_artifacts
from precovery.search.pipeline_types import (
    BenchTargets,
    PredictedTargets,
    PredictedTriples,
    Stage3OrbitMetrics,
)


def test_stage23_artifacts_round_trip(tmp_path: Path) -> None:
    targets = BenchTargets.from_kwargs(
        obscode=["I41"],
        exposure_mjd_mid_utc=[60000.0],
        exposure_mjd_mid_key_us=[int(round(60000.0 * 86400 * 1e6))],
        filter=["r"],
    )
    preds = PredictedTargets.from_kwargs(
        orbit_id=["o1"],
        target_idx=[0],
        obscode=["I41"],
        exposure_mjd_mid_utc=[60000.0],
        exposure_mjd_mid_key_us=[int(round(60000.0 * 86400 * 1e6))],
        canonical_filter_id=[None],
        pred_lon_deg=[10.0],
        pred_lat_deg=[20.0],
        cov_ll_00=[1.0],
        cov_ll_01=[0.0],
        cov_ll_11=[1.0],
        pred_mag=[None],
    )
    triples = PredictedTriples.from_kwargs(
        orbit_id=["o1"],
        target_idx=[0],
        obscode=["I41"],
        exposure_mjd_mid_utc=[60000.0],
        exposure_mjd_mid_key_us=[int(round(60000.0 * 86400 * 1e6))],
        healpixel=[123],
    )
    m = Stage3OrbitMetrics.from_kwargs(
        orbit_id=pa.array(["o1"], type=pa.large_string()),
        n_frames_geometry_matched=pa.array([1], type=pa.int64()),
        n_frames_lim_mag_rejected=pa.array([0], type=pa.int64()),
        n_frames_truth_geometry_matched=pa.array([0], type=pa.int64()),
        n_frames_lim_mag_truth_rejected=pa.array([0], type=pa.int64()),
        n_detections_truth_frame_candidates=pa.array([0], type=pa.int64()),
    )

    write_stage23_artifacts(
        run_dir=tmp_path,
        targets=targets,
        preds=preds,
        triples=triples,
        stage3_orbit_metrics=m,
        micro_timings={"x": 1.0},
    )
    art = read_stage23_artifacts(run_dir=tmp_path)
    assert len(art.targets) == 1
    assert len(art.preds) == 1
    assert len(art.triples) == 1
    assert len(art.stage3_orbit_metrics) == 1

