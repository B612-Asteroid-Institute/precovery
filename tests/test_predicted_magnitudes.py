from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from adam_core.orbits.orbits import Orbits, PhysicalParameters

import pytest

from precovery.config import Config
from precovery.healpix_geom import radec_to_healpixel
from precovery.search.backend_pipeline import run_stage1_to_stage4_rows_python
from precovery.search.backends.duckdb_parquet import DuckDbParquetBackend
from precovery.search.backends.protocols import GateParams
from precovery.search.pipeline_types import SubsetPaths
from precovery.search.footprints import CovPolygonReconstructedMoc
from precovery.search.runtime_config import load_mag_gate_config
from precovery.sourcecatalog import SourceObservation

from .testutils import make_sourceobs_of_orbit


pytestmark = [pytest.mark.integration, pytest.mark.slow]


def _with_hg(orbit: Orbits, *, H_v: float, G: float) -> Orbits:
    pp = PhysicalParameters.from_kwargs(H_v=[float(H_v)], G=[float(G)])
    return orbit.set_column("physical_parameters", pp)


def _write_subset(
    *, subset_dir: Path, observations: list[SourceObservation], cfg: Config
) -> Path:
    subset_dir = Path(subset_dir)
    subset_dir.mkdir(parents=True, exist_ok=True)

    obscode = np.asarray([str(o.obscode) for o in observations], dtype=object)
    mjd_mid = np.asarray([float(o.exposure_mjd_mid) for o in observations], dtype=np.float64)
    key_us = np.rint(mjd_mid * 86400.0 * 1e6).astype(np.int64)
    filt = np.asarray([str(o.filter) for o in observations], dtype=object)
    obs_id = np.asarray([o.id.decode("utf8") for o in observations], dtype=object)
    t_obs = np.asarray([float(o.mjd) for o in observations], dtype=np.float64)
    ra = np.asarray([float(o.ra) for o in observations], dtype=np.float64)
    dec = np.asarray([float(o.dec) for o in observations], dtype=np.float64)
    hpix = radec_to_healpixel(ra, dec, int(cfg.nside)).astype(np.int64)

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
            "ra_sigma_deg": pa.array([float(o.ra_sigma) for o in observations], type=pa.float64()),
            "dec_sigma_deg": pa.array([float(o.dec_sigma) for o in observations], type=pa.float64()),
            "mag": pa.array([float(o.mag) for o in observations], type=pa.float64()),
            "mag_sigma": pa.array([float(o.mag_sigma) for o in observations], type=pa.float64()),
        }
    )
    pq.write_table(tbl, str(subset_dir / "detections.parquet"))
    cfg.to_json(str(subset_dir / "config.json"))
    return subset_dir


def test_pred_mag_populated_in_stage23(tmp_path: Path, sample_orbits: Orbits) -> None:
    orbit = _with_hg(sample_orbits[0], H_v=15.0, G=0.15)
    times = [50000.0, 50001.0, 50002.0]
    obs = [make_sourceobs_of_orbit(orbit, "I41", mjd) for mjd in times]

    cfg = Config(
        nside=32,
        backend="duckdb_parquet",
        detections_parquet="detections.parquet",
        # Enable mag residual gate so Stage 2/3 computes predicted magnitudes.
        max_mag_residual_fainter_mag=1.0,
        max_mag_residual_brighter_mag=1.0,
    )
    subset_dir = _write_subset(subset_dir=tmp_path, observations=obs, cfg=cfg)

    backend = DuckDbParquetBackend(parquet_path=subset_dir / "detections.parquet")
    limit_keys, limit_vals, faint_margin, max_faint, max_bright = load_mag_gate_config(
        subset_dir=subset_dir, obscodes={"I41"}, config=cfg
    )
    gate = GateParams(
        innovation_gate_n_sigma=3.0,
        invalid_sigma_fill_floor_arcsec_global=0.10,
        max_mag_residual_fainter_mag=max_faint,
        max_mag_residual_brighter_mag=max_bright,
    )
    fp = CovPolygonReconstructedMoc(n_sigma=3.0, polygon_vertices=32)

    out = run_stage1_to_stage4_rows_python(
        backend=backend,
        subset=SubsetPaths(subset_dir=subset_dir),
        orbits=orbit,
        start_mjd_utc=min(times) - 1.0,
        end_mjd_utc=max(times) + 1.0,
        obscodes=("I41",),
        window_size_days=7,
        stage2_strategy="assist_window_then_2body_variants:sigma_points",
        healpix_nside=32,
        footprint=fp,
        max_processes=1,
        limit_codefid_keys=limit_keys,
        limit_codefid_vals=limit_vals,
        faint_margin_mag=float(faint_margin),
        gate=gate,
        detailed_timings=False,
        timings=None,
    )

    assert len(out.build.preds) > 0
    assert pc.all(pc.is_finite(out.build.preds.pred_mag)).as_py()


def test_faint_frame_skip_can_avoid_backend_fetch(tmp_path: Path, sample_orbits: Orbits) -> None:
    orbit = _with_hg(sample_orbits[0], H_v=30.0, G=0.15)
    mjd = 50000.0
    # Use a canonical ZTF band so limiting-mag presets apply (I41|ZTF_r).
    obs = [make_sourceobs_of_orbit(orbit, "I41", mjd, filter="ZTF_r")]

    cfg = Config(
        nside=32,
        backend="duckdb_parquet",
        detections_parquet="detections.parquet",
        faint_frame_skip_margin_mag=0.0,
    )
    subset_dir = _write_subset(subset_dir=tmp_path, observations=obs, cfg=cfg)

    # The centralized presets include I41|ZTF_r ~ 20.6, so with pred_mag ~ 30 this frame is skipped
    # and we should never hit the backend fetch.

    backend = DuckDbParquetBackend(parquet_path=subset_dir / "detections.parquet")

    def _boom(*args, **kwargs):
        raise AssertionError("fetch_candidates should not be called when no triples exist")

    backend.fetch_candidates = _boom  # type: ignore[method-assign]

    limit_keys, limit_vals, faint_margin, max_faint, max_bright = load_mag_gate_config(
        subset_dir=subset_dir, obscodes={"I41"}, config=cfg
    )
    gate = GateParams(
        innovation_gate_n_sigma=3.0,
        invalid_sigma_fill_floor_arcsec_global=0.10,
        max_mag_residual_fainter_mag=max_faint,
        max_mag_residual_brighter_mag=max_bright,
    )
    fp = CovPolygonReconstructedMoc(n_sigma=3.0, polygon_vertices=32)

    out = run_stage1_to_stage4_rows_python(
        backend=backend,
        subset=SubsetPaths(subset_dir=subset_dir),
        orbits=orbit,
        start_mjd_utc=mjd - 1.0,
        end_mjd_utc=mjd + 1.0,
        obscodes=("I41",),
        window_size_days=7,
        stage2_strategy="assist_window_then_2body_variants:sigma_points",
        healpix_nside=32,
        footprint=fp,
        max_processes=1,
        limit_codefid_keys=limit_keys,
        limit_codefid_vals=limit_vals,
        faint_margin_mag=float(faint_margin),
        gate=gate,
        detailed_timings=False,
        timings=None,
    )
    assert len(out.gate.accepted_detections) == 0
