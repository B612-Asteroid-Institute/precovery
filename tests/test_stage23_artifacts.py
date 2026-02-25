from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from adam_core.orbits import Orbits

import pytest

from precovery.config import Config
from precovery.healpix_geom import radec_to_healpixel
from precovery.search.backend_pipeline import run_stage1_to_stage4_rows_python
from precovery.search.backends.duckdb_parquet import DuckDbParquetBackend
from precovery.search.backends.protocols import GateParams
from precovery.search.pipeline_types import SubsetPaths
from precovery.search.footprints import CovPolygonReconstructedMoc

from .testutils import make_sourceobs_of_orbit


pytestmark = [pytest.mark.integration, pytest.mark.slow]


def _write_subset_one_obs(*, subset_dir: Path, orbit: Orbits) -> Path:
    subset_dir.mkdir(parents=True, exist_ok=True)
    mjd = 50000.0
    o = make_sourceobs_of_orbit(orbit, "I41", mjd)

    obscode = np.asarray([str(o.obscode)], dtype=object)
    mjd_mid = np.asarray([float(o.exposure_mjd_mid)], dtype=np.float64)
    key_us = np.rint(mjd_mid * 86400.0 * 1e6).astype(np.int64)
    filt = np.asarray([str(o.filter)], dtype=object)
    obs_id = np.asarray([o.id.decode("utf8") if isinstance(o.id, (bytes, bytearray)) else str(o.id)], dtype=object)
    t_obs = np.asarray([float(o.mjd)], dtype=np.float64)
    ra = np.asarray([float(o.ra)], dtype=np.float64)
    dec = np.asarray([float(o.dec)], dtype=np.float64)
    hpix = radec_to_healpixel(ra, dec, 32).astype(np.int64)

    tbl = pa.table(
        {
            "obscode": pa.array(obscode, type=pa.large_string()),
            "exposure_mjd_mid_utc": pa.array(mjd_mid, type=pa.float64()),
            "exposure_mjd_mid_key_us": pa.array(key_us, type=pa.int64()),
            "filter": pa.array(filt, type=pa.large_string()),
            "healpixel": pa.array(hpix, type=pa.int64()),
            "observation_id": pa.array(obs_id, type=pa.large_string()),
            "obstime_mjd_utc": pa.array(t_obs, type=pa.float64()),
            "ra_deg": pa.array(ra, type=pa.float64()),
            "dec_deg": pa.array(dec, type=pa.float64()),
            "ra_sigma_deg": pa.array([float(o.ra_sigma)], type=pa.float64()),
            "dec_sigma_deg": pa.array([float(o.dec_sigma)], type=pa.float64()),
            "mag": pa.array([float(o.mag)], type=pa.float64()),
            "mag_sigma": pa.array([float(o.mag_sigma)], type=pa.float64()),
        }
    )
    pq.write_table(tbl, str(subset_dir / "detections.parquet"))
    Config(nside=32, backend="duckdb_parquet", detections_parquet="detections.parquet").to_json(
        str(subset_dir / "config.json")
    )
    return subset_dir


def test_stage23_artifacts_can_be_reused(tmp_path: Path, sample_orbits: Orbits) -> None:
    orbit = sample_orbits[0]
    subset_dir = _write_subset_one_obs(subset_dir=tmp_path / "subset", orbit=orbit)
    backend = DuckDbParquetBackend(parquet_path=subset_dir / "detections.parquet")

    gate = GateParams(innovation_gate_n_sigma=3.0, invalid_sigma_fill_floor_arcsec_global=0.10)
    fp = CovPolygonReconstructedMoc(n_sigma=3.0, polygon_vertices=32)

    run_dir = tmp_path / "run"
    out1 = run_stage1_to_stage4_rows_python(
        backend=backend,
        subset=SubsetPaths(subset_dir=subset_dir),
        orbits=orbit,
        start_mjd_utc=49999.0,
        end_mjd_utc=50001.0,
        obscodes=("I41",),
        window_size_days=7,
        stage2_strategy="assist_window_then_2body_variants:sigma_points",
        healpix_nside=32,
        footprint=fp,
        max_processes=1,
        limit_codefid_keys=pa.array([], type=pa.large_string()),
        limit_codefid_vals=pa.array([], type=pa.float64()),
        faint_margin_mag=0.0,
        gate=gate,
        detailed_timings=False,
        timings=None,
        stage23_run_dir=run_dir,
        reuse_stage23_artifacts=False,
        write_stage23_artifacts_flag=True,
    )

    def _boom(*args, **kwargs):
        raise AssertionError("enumerate_targets should not be called when reusing stage23 artifacts")

    backend.enumerate_targets = _boom  # type: ignore[method-assign]

    out2 = run_stage1_to_stage4_rows_python(
        backend=backend,
        subset=SubsetPaths(subset_dir=subset_dir),
        orbits=orbit,
        start_mjd_utc=49999.0,
        end_mjd_utc=50001.0,
        obscodes=("I41",),
        window_size_days=7,
        stage2_strategy="assist_window_then_2body_variants:sigma_points",
        healpix_nside=32,
        footprint=fp,
        max_processes=1,
        limit_codefid_keys=pa.array([], type=pa.large_string()),
        limit_codefid_vals=pa.array([], type=pa.float64()),
        faint_margin_mag=0.0,
        gate=gate,
        detailed_timings=False,
        timings=None,
        stage23_run_dir=run_dir,
        reuse_stage23_artifacts=True,
        write_stage23_artifacts_flag=False,
    )

    assert len(out1.build.preds) == len(out2.build.preds)
    assert len(out1.build.triples) == len(out2.build.triples)


def test_stage23_can_be_built_from_injected_targets(tmp_path: Path, sample_orbits: Orbits) -> None:
    orbit = sample_orbits[0]
    subset_dir = _write_subset_one_obs(subset_dir=tmp_path / "subset", orbit=orbit)
    backend = DuckDbParquetBackend(parquet_path=subset_dir / "detections.parquet")

    gate = GateParams(innovation_gate_n_sigma=3.0, invalid_sigma_fill_floor_arcsec_global=0.10)
    fp = CovPolygonReconstructedMoc(n_sigma=3.0, polygon_vertices=32)

    # Build injected targets/pixels once, then ensure the pipeline does not re-enumerate.
    targets = backend.enumerate_targets(
        subset=SubsetPaths(subset_dir=subset_dir),
        start_mjd_utc=49999.0,
        end_mjd_utc=50001.0,
        obscodes=("I41",),
    )
    pixels = backend.frame_pixels_by_target(subset=SubsetPaths(subset_dir=subset_dir), targets=targets)

    def _boom(*args, **kwargs):
        raise AssertionError("backend enumeration helpers should not be called when injected inputs are provided")

    backend.enumerate_targets = _boom  # type: ignore[method-assign]
    backend.frame_pixels_by_target = _boom  # type: ignore[method-assign]

    run_dir = tmp_path / "run_injected"
    out1 = run_stage1_to_stage4_rows_python(
        backend=backend,
        subset=SubsetPaths(subset_dir=subset_dir),
        orbits=orbit,
        start_mjd_utc=49999.0,
        end_mjd_utc=50001.0,
        obscodes=("I41",),
        window_size_days=7,
        stage2_strategy="assist_window_then_2body_variants:sigma_points",
        healpix_nside=32,
        footprint=fp,
        max_processes=1,
        limit_codefid_keys=pa.array([], type=pa.large_string()),
        limit_codefid_vals=pa.array([], type=pa.float64()),
        faint_margin_mag=0.0,
        gate=gate,
        detailed_timings=False,
        timings=None,
        stage23_run_dir=run_dir,
        reuse_stage23_artifacts=False,
        write_stage23_artifacts_flag=True,
        targets=targets,
        dataset_pixels_by_target=pixels,
    )

    # Second run should reuse Stage23 from disk (and still not enumerate).
    out2 = run_stage1_to_stage4_rows_python(
        backend=backend,
        subset=SubsetPaths(subset_dir=subset_dir),
        orbits=orbit,
        start_mjd_utc=49999.0,
        end_mjd_utc=50001.0,
        obscodes=("I41",),
        window_size_days=7,
        stage2_strategy="assist_window_then_2body_variants:sigma_points",
        healpix_nside=32,
        footprint=fp,
        max_processes=1,
        limit_codefid_keys=pa.array([], type=pa.large_string()),
        limit_codefid_vals=pa.array([], type=pa.float64()),
        faint_margin_mag=0.0,
        gate=gate,
        detailed_timings=False,
        timings=None,
        stage23_run_dir=run_dir,
        reuse_stage23_artifacts=True,
        write_stage23_artifacts_flag=False,
        targets=targets,
        dataset_pixels_by_target=pixels,
    )

    assert len(out1.build.preds) == len(out2.build.preds)
    assert len(out1.build.triples) == len(out2.build.triples)


def test_windowed_stage2_multi_center_has_unique_orbit_target_keys(
    tmp_path: Path, sample_orbits: Orbits
) -> None:
    # Regression test: when window_size_days yields >1 window center, Stage 2 must still
    # produce exactly one prediction per (orbit_id, target_idx). A prior bug in the
    # window-center slicing logic could duplicate or drop centers, leading to duplicate
    # keys and downstream join explosions in Stage 4.
    orbits = sample_orbits.take([0, 1])

    subset_dir = tmp_path / "subset_multi_center"
    subset_dir.mkdir(parents=True, exist_ok=True)

    # Two exposures separated by > window_size_days to force multiple centers.
    mjds = [50000.0, 50010.0]
    rows = []
    for mjd in mjds:
        for orbit in orbits:
            o = make_sourceobs_of_orbit(orbit, "I41", mjd)
            rows.append(o)

    obscode = np.asarray([str(r.obscode) for r in rows], dtype=object)
    mjd_mid = np.asarray([float(r.exposure_mjd_mid) for r in rows], dtype=np.float64)
    key_us = np.rint(mjd_mid * 86400.0 * 1e6).astype(np.int64)
    filt = np.asarray([str(r.filter) for r in rows], dtype=object)
    obs_id = np.asarray(
        [r.id.decode("utf8") if isinstance(r.id, (bytes, bytearray)) else str(r.id) for r in rows],
        dtype=object,
    )
    t_obs = np.asarray([float(r.mjd) for r in rows], dtype=np.float64)
    ra = np.asarray([float(r.ra) for r in rows], dtype=np.float64)
    dec = np.asarray([float(r.dec) for r in rows], dtype=np.float64)
    hpix = radec_to_healpixel(ra, dec, 32).astype(np.int64)

    tbl = pa.table(
        {
            "obscode": pa.array(obscode, type=pa.large_string()),
            "exposure_mjd_mid_utc": pa.array(mjd_mid, type=pa.float64()),
            "exposure_mjd_mid_key_us": pa.array(key_us, type=pa.int64()),
            "filter": pa.array(filt, type=pa.large_string()),
            "healpixel": pa.array(hpix, type=pa.int64()),
            "observation_id": pa.array(obs_id, type=pa.large_string()),
            "obstime_mjd_utc": pa.array(t_obs, type=pa.float64()),
            "ra_deg": pa.array(ra, type=pa.float64()),
            "dec_deg": pa.array(dec, type=pa.float64()),
            "ra_sigma_deg": pa.array([float(r.ra_sigma) for r in rows], type=pa.float64()),
            "dec_sigma_deg": pa.array([float(r.dec_sigma) for r in rows], type=pa.float64()),
            "mag": pa.array([float(r.mag) for r in rows], type=pa.float64()),
            "mag_sigma": pa.array([float(r.mag_sigma) for r in rows], type=pa.float64()),
        }
    )
    pq.write_table(tbl, str(subset_dir / "detections.parquet"))
    Config(nside=32, backend="duckdb_parquet", detections_parquet="detections.parquet").to_json(
        str(subset_dir / "config.json")
    )
    backend = DuckDbParquetBackend(parquet_path=subset_dir / "detections.parquet")

    gate = GateParams(innovation_gate_n_sigma=3.0, invalid_sigma_fill_floor_arcsec_global=0.10)
    fp = CovPolygonReconstructedMoc(n_sigma=3.0, polygon_vertices=32)

    out = run_stage1_to_stage4_rows_python(
        backend=backend,
        subset=SubsetPaths(subset_dir=subset_dir),
        orbits=orbits,
        start_mjd_utc=49999.0,
        end_mjd_utc=50011.0,
        obscodes=("I41",),
        window_size_days=7,
        stage2_strategy="assist_window_then_2body_variants:sigma_points",
        healpix_nside=32,
        footprint=fp,
        max_processes=1,
        limit_codefid_keys=pa.array([], type=pa.large_string()),
        limit_codefid_vals=pa.array([], type=pa.float64()),
        faint_margin_mag=0.0,
        gate=gate,
        detailed_timings=False,
        timings=None,
        stage23_run_dir=tmp_path / "run_multi_center",
        reuse_stage23_artifacts=False,
        write_stage23_artifacts_flag=False,
    )

    preds = out.build.preds.table.select(["orbit_id", "target_idx"])
    gb = preds.group_by(["orbit_id", "target_idx"]).aggregate([("target_idx", "count")])
    assert gb.num_rows == preds.num_rows
