"""
Test runtime knobs.

These settings are intentionally scoped to pytest runs and aim to keep the suite fast
and stable on developer machines/CI while exercising the real code paths.
"""

from __future__ import annotations

import os


# Avoid aggressive XLA memory preallocation in test runs.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import glob

import numpy as np
import pandas as pd
import pytest
import pyarrow as pa
import pyarrow.parquet as pq
from adam_core.orbits import Orbits

from precovery.config import Config
from precovery.healpix_geom import radec_to_healpixel

SAMPLE_ORBITS_FILE = os.path.join(
    os.path.dirname(__file__), "data", "sample_orbits.parquet"
)
TEST_OBSERVATIONS_DIR = os.path.join(os.path.dirname(__file__), "data/index")


@pytest.fixture(scope="session")
def detections_subset_dir(tmp_path_factory) -> str:
    """
    Build a small DuckDB-parquet-backed subset once per pytest session (per xdist worker).
    """
    worker = os.environ.get("PYTEST_XDIST_WORKER", "master")
    root = tmp_path_factory.mktemp(f"detections_subset_{worker}")

    parq_path = os.path.join(root, "detections.parquet")
    cfg_path = os.path.join(root, "config.json")

    # Reuse if already built (helpful for local reruns).
    if os.path.exists(parq_path) and os.path.exists(cfg_path):
        return str(root)

    observation_files = glob.glob(
        os.path.join(TEST_OBSERVATIONS_DIR, "dataset_*", "*.csv")
    )
    dfs: list[pd.DataFrame] = []
    for observation_file in observation_files:
        df = pd.read_csv(
            observation_file,
            float_precision="round_trip",
            dtype={
                "dataset_id": str,
                "observatory_code": str,
                "filter": str,
                "exposure_duration": np.float64,
                "obs_id": str,
                "exposure_id": str,
            },
        )
        dfs.append(df)

    if not dfs:
        raise RuntimeError("No test observation CSVs found")
    df_all = pd.concat(dfs, axis=0, ignore_index=True)

    # Canonical columns expected by DuckDB backend adapters.
    obscode = df_all["observatory_code"].astype(str).to_numpy()
    exposure_mjd_mid_utc = df_all["exposure_mjd_mid"].astype(np.float64).to_numpy()
    exposure_mjd_mid_key_us = np.rint(exposure_mjd_mid_utc * 86400.0 * 1e6).astype(
        np.int64
    )
    ra_deg = df_all["ra"].astype(np.float64).to_numpy()
    dec_deg = df_all["dec"].astype(np.float64).to_numpy()
    healpixel = radec_to_healpixel(ra_deg, dec_deg, 32)

    tbl = pa.table(
        {
            "obscode": pa.array(obscode, type=pa.large_string()),
            "exposure_mjd_mid_utc": pa.array(exposure_mjd_mid_utc, type=pa.float64()),
            "exposure_mjd_mid_key_us": pa.array(
                exposure_mjd_mid_key_us, type=pa.int64()
            ),
            "filter": pa.array(df_all["filter"].astype(str).to_numpy(), type=pa.large_string()),
            "healpixel": pa.array(healpixel.astype(np.int64), type=pa.int64()),
            "observation_id": pa.array(
                df_all["obs_id"].astype(str).to_numpy(), type=pa.large_string()
            ),
            "obstime_mjd_utc": pa.array(
                df_all["mjd"].astype(np.float64).to_numpy(), type=pa.float64()
            ),
            "ra_deg": pa.array(ra_deg, type=pa.float64()),
            "dec_deg": pa.array(dec_deg, type=pa.float64()),
            "ra_sigma_deg": pa.array(
                df_all["ra_sigma"].astype(np.float64).to_numpy(), type=pa.float64()
            ),
            "dec_sigma_deg": pa.array(
                df_all["dec_sigma"].astype(np.float64).to_numpy(), type=pa.float64()
            ),
            "mag": pa.array(
                df_all["mag"].astype(np.float64).to_numpy(), type=pa.float64()
            ),
            "mag_sigma": pa.array(
                df_all["mag_sigma"].astype(np.float64).to_numpy(), type=pa.float64()
            ),
        }
    )
    pq.write_table(tbl, parq_path)

    cfg = Config(nside=32, backend="duckdb_parquet", detections_parquet="detections.parquet")
    cfg.to_json(cfg_path)

    return str(root)


@pytest.fixture
def detections_subset_dir_path(detections_subset_dir: str) -> str:
    return str(detections_subset_dir)


@pytest.fixture(scope="session")
def sample_orbits():
    sample_orbits_file = os.path.join(
        os.path.dirname(__file__), "data", "sample_orbits.parquet"
    )
    orbits = Orbits.from_parquet(sample_orbits_file)

    return orbits
