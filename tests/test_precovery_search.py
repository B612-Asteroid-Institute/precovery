from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from adam_core.orbits import Orbits

import pytest

from precovery.config import Config
from precovery.healpix_geom import radec_to_healpixel
from precovery.main import precover
from precovery.sourcecatalog import SourceObservation

from .testutils import make_sourceobs, make_sourceobs_of_orbit


pytestmark = [pytest.mark.integration, pytest.mark.slow]


def _write_subset_from_observations(
    *, subset_dir: Path, observations: list[SourceObservation], nside: int = 32
) -> None:
    if not observations:
        raise ValueError("observations cannot be empty")

    obscode = np.asarray([str(o.obscode) for o in observations], dtype=object)
    mjd_mid = np.asarray([float(o.exposure_mjd_mid) for o in observations], dtype=np.float64)
    key_us = np.rint(mjd_mid * 86400.0 * 1e6).astype(np.int64)
    filt = np.asarray([str(o.filter) for o in observations], dtype=object)
    obs_id = np.asarray(
        [
            (o.id.decode("utf8") if isinstance(o.id, (bytes, bytearray)) else str(o.id))
            for o in observations
        ],
        dtype=object,
    )
    t_obs = np.asarray([float(o.mjd) for o in observations], dtype=np.float64)
    ra = np.asarray([float(o.ra) for o in observations], dtype=np.float64)
    dec = np.asarray([float(o.dec) for o in observations], dtype=np.float64)
    hpix = radec_to_healpixel(ra, dec, int(nside)).astype(np.int64)

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
            "ra_sigma_deg": pa.array(
                [float(o.ra_sigma) for o in observations], type=pa.float64()
            ),
            "dec_sigma_deg": pa.array(
                [float(o.dec_sigma) for o in observations], type=pa.float64()
            ),
            "mag": pa.array([float(o.mag) for o in observations], type=pa.float64()),
            "mag_sigma": pa.array(
                [float(o.mag_sigma) for o in observations], type=pa.float64()
            ),
        }
    )
    pq.write_table(tbl, str(Path(subset_dir) / "detections.parquet"))

    cfg = Config(nside=int(nside), backend="duckdb_parquet", detections_parquet="detections.parquet")
    cfg.to_json(str(Path(subset_dir) / "config.json"))


def test_precover_finds_inserted_observations(tmp_path: Path, sample_orbits: Orbits) -> None:
    orbit = sample_orbits[0]
    timestamps = [50000.0, 50001.0, 50002.0]

    want = [make_sourceobs_of_orbit(orbit, "I41", mjd) for mjd in timestamps]
    # Add some unrelated detections in a different healpixel so they don't join.
    extra = [make_sourceobs(obscode="I41", mjd=mjd, exposure_duration=30, healpixel=1234) for mjd in timestamps]

    subset_dir = tmp_path
    _write_subset_from_observations(subset_dir=subset_dir, observations=want + extra, nside=32)

    accepted, _counts = precover(
        orbits=orbit,
        database_directory=str(subset_dir),
        start_mjd=min(timestamps) - 1.0,
        end_mjd=max(timestamps) + 1.0,
        max_processes=1,
    )

    have_ids = set(accepted.observation_id.to_pylist())
    want_ids = set(
        o.id.decode("utf8") if isinstance(o.id, (bytes, bytearray)) else str(o.id) for o in want
    )
    assert want_ids.issubset(have_ids)


def test_precover_multiple_orbits_smoke(tmp_path: Path, sample_orbits: Orbits) -> None:
    orbits = sample_orbits[:2]
    t = [50000.0, 50001.0, 50002.0]
    obs: list[SourceObservation] = []
    for o in orbits:
        obs.extend([make_sourceobs_of_orbit(o, "I41", mjd) for mjd in t])

    subset_dir = tmp_path
    _write_subset_from_observations(subset_dir=subset_dir, observations=obs, nside=32)

    accepted, _counts = precover(
        orbits=orbits,
        database_directory=str(subset_dir),
        start_mjd=min(t) - 1.0,
        end_mjd=max(t) + 1.0,
        max_processes=1,
    )
    assert len(accepted) >= 1

