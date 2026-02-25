from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from adam_core.orbits.orbits import Orbits, PhysicalParameters

import pytest

from precovery.config import Config
from precovery.healpix_geom import radec_to_healpixel
from precovery.main import precover
from precovery.sourcecatalog import SourceObservation

from .testutils import make_sourceobs_of_orbit


pytestmark = [pytest.mark.integration, pytest.mark.slow]


def _with_hg(orbit: Orbits, *, H_v: float, G: float) -> Orbits:
    pp = PhysicalParameters.from_kwargs(H_v=[float(H_v)], G=[float(G)])
    return orbit.set_column("physical_parameters", pp)


def _write_subset(
    *, subset_dir: Path, observations: list[SourceObservation], cfg: Config
) -> None:
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


def test_detection_mag_outlier_rejected_brighter(tmp_path: Path, sample_orbits: Orbits) -> None:
    orbit = _with_hg(sample_orbits[0], H_v=15.0, G=0.15)
    mjd = 50000.0
    o = make_sourceobs_of_orbit(orbit, "I41", mjd)
    o.mag = 0.0  # extremely bright => negative residual

    cfg = Config(
        nside=32,
        backend="duckdb_parquet",
        detections_parquet="detections.parquet",
        max_mag_residual_fainter_mag=100.0,
        max_mag_residual_brighter_mag=0.1,
    )
    _write_subset(subset_dir=tmp_path, observations=[o], cfg=cfg)

    accepted, counts = precover(
        orbits=orbit,
        database_directory=str(tmp_path),
        start_mjd=mjd - 1.0,
        end_mjd=mjd + 1.0,
        max_processes=1,
    )
    assert len(counts) >= 1
    assert len(accepted) == 0


def test_detection_mag_outlier_rejected_fainter(tmp_path: Path, sample_orbits: Orbits) -> None:
    orbit = _with_hg(sample_orbits[0], H_v=15.0, G=0.15)
    mjd = 50000.0
    o = make_sourceobs_of_orbit(orbit, "I41", mjd)
    o.mag = 100.0  # extremely faint => positive residual

    cfg = Config(
        nside=32,
        backend="duckdb_parquet",
        detections_parquet="detections.parquet",
        max_mag_residual_fainter_mag=0.1,
        max_mag_residual_brighter_mag=100.0,
    )
    _write_subset(subset_dir=tmp_path, observations=[o], cfg=cfg)

    accepted, counts = precover(
        orbits=orbit,
        database_directory=str(tmp_path),
        start_mjd=mjd - 1.0,
        end_mjd=mjd + 1.0,
        max_processes=1,
    )
    assert len(counts) >= 1
    assert len(accepted) == 0

