import json
import sqlite3
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from adam_core.coordinates.covariances import CoordinateCovariances
from adam_core.coordinates.origin import Origin
from adam_core.coordinates.spherical import SphericalCoordinates
from adam_core.orbits.ephemeris import Ephemeris
from adam_core.time import Timestamp

from experiments.covariance_precovery.harness.stage3_healpixel_bench import FrameTimeTargets
from experiments.covariance_precovery.harness.stage4_detection_filter_bench import (
    _query_frames_for_pixels,
    run_stage4_detection_filter_bench,
)
from precovery.frame_db import HealpixFrame


def test_stage4_query_frames_chunks_in_clause() -> None:
    # Build an in-memory frames table with 5 healpixels.
    conn = sqlite3.connect(":memory:")
    try:
        conn.executescript(
            """
            CREATE TABLE frames (
              dataset_id TEXT NOT NULL,
              obscode TEXT NOT NULL,
              exposure_id TEXT NOT NULL,
              filter TEXT,
              exposure_mjd_start REAL NOT NULL,
              exposure_mjd_mid REAL NOT NULL,
              exposure_duration REAL NOT NULL,
              healpixel INTEGER NOT NULL,
              data_uri TEXT NOT NULL,
              data_offset INTEGER NOT NULL,
              data_length INTEGER NOT NULL
            );
            """
        )
        rows = [
            (
                "ds",
                "I41",
                "exp",
                "r",
                60000.0,
                60000.5,
                30.0,
                int(hp),
                "ds/2000-01/frames_00000000.data",
                0,
                0,
            )
            for hp in range(1, 6)
        ]
        conn.executemany(
            """
            INSERT INTO frames (
              dataset_id, obscode, exposure_id, filter,
              exposure_mjd_start, exposure_mjd_mid, exposure_duration,
              healpixel, data_uri, data_offset, data_length
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        out = _query_frames_for_pixels(
            conn=conn,
            obscode="I41",
            exposure_mjd_mid=60000.5,
            healpixels=np.array([1, 2, 3, 4, 5], dtype=np.int64),
            max_in_clause=2,
        )
        assert len(out) == 5
        assert sorted(int(r["healpixel"]) for r in out) == [1, 2, 3, 4, 5]
    finally:
        conn.close()


def test_stage4_smoke_writes_parquets(tmp_path: Path, precovery_db_with_data) -> None:
    """
    End-to-end smoke: create minimal Stage 2 artifacts + truth crossmatch, then run Stage 4.
    """
    subset_dir = tmp_path
    db = precovery_db_with_data

    # Pick one real frame from the subset index.
    index_db = subset_dir / "index.db"
    with sqlite3.connect(str(index_db)) as conn:
        row = conn.execute(
            """
            SELECT
              dataset_id, obscode, exposure_id, filter,
              exposure_mjd_start, exposure_mjd_mid, exposure_duration,
              healpixel, data_uri, data_offset, data_length
            FROM frames
            LIMIT 1
            """
        ).fetchone()
    assert row is not None
    (
        dataset_id,
        obscode,
        exposure_id,
        filt,
        mjd_start,
        mjd_mid,
        mjd_dur,
        healpixel,
        data_uri,
        data_offset,
        data_length,
    ) = row

    hf = HealpixFrame.from_kwargs(
        dataset_id=[str(dataset_id)],
        obscode=[str(obscode)],
        exposure_id=[str(exposure_id)],
        filter=[str(filt)],
        exposure_mjd_start=[float(mjd_start)],
        exposure_mjd_mid=[float(mjd_mid)],
        exposure_duration=[float(mjd_dur)],
        healpixel=[int(healpixel)],
        data_uri=[str(data_uri)],
        data_offset=[int(data_offset)],
        data_length=[int(data_length)],
    )
    obs = db.frames.get_observations(hf)
    assert len(obs) > 0
    obsid = obs.id[0].as_py()
    obsid_s = obsid.decode("utf8") if isinstance(obsid, (bytes, bytearray)) else str(obsid)

    ra0 = float(obs.ra[0].as_py())
    dec0 = float(obs.dec[0].as_py())

    # Minimal artifacts required by Stage 4 truth mapping.
    artifacts = subset_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)

    designation = "TEST"
    pq.write_table(
        pa.table(
            {
                "orbit_id": pa.array([designation], pa.large_string()),
                "object_id": pa.array([designation], pa.large_string()),
            }
        ),
        artifacts / "orbits_selected_sbdb.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "matched": pa.array([True], pa.bool_()),
                "designation": pa.array([designation], pa.large_string()),
                "obscode": pa.array([str(obscode)], pa.large_string()),
                    # Frame-key fields used by Stage 4 optional truth-frame speedups.
                    "match_dataset_id": pa.array([str(dataset_id)], pa.large_string()),
                    "match_exposure_id": pa.array([str(exposure_id)], pa.large_string()),
                    "healpixel": pa.array([int(healpixel)], pa.int64()),
                    # Stage 4 harness expects truth_* columns (we evaluate recall by time/sky position,
                    # not by observation_id).
                    "truth_obsid": pa.array([obsid_s], pa.large_string()),
                    "truth_time_mjd_utc": pa.array([float(mjd_mid)], pa.float64()),
                    "truth_ra_deg": pa.array([float(ra0)], pa.float64()),
                    "truth_dec_deg": pa.array([float(dec0)], pa.float64()),
            }
        ),
        artifacts / "truth_precovery_crossmatch.parquet",
    )

    # Minimal Stage 2 run directory: one target, one orbit, one ephemeris row with covariance.
    stage2_run_dir = subset_dir / "stage2_run"
    (stage2_run_dir / "inputs").mkdir(parents=True, exist_ok=True)
    (stage2_run_dir / "strategies" / "2body_with_covariance" / "mean_ephemeris").mkdir(
        parents=True, exist_ok=True
    )

    FrameTimeTargets.from_kwargs(
        obscode=[str(obscode)],
        time=Timestamp.from_mjd([float(mjd_mid)], scale="utc"),
    ).to_parquet(str(stage2_run_dir / "inputs" / "frame_time_targets.parquet"))

    # Strategy meta for target_idx mapping.
    (stage2_run_dir / "strategies" / "2body_with_covariance" / "meta.json").write_text(
        json.dumps(
            {"n_orbits": 1, "time_chunk_size": 1, "n_time_targets": 1},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    # One-row ephemeris with covariance, aligned to the frame mid time and the chosen observation sky position.
    cov = np.zeros((1, 6, 6), dtype=np.float64)
    cov[:, 0, 0] = 1.0
    cov[:, 3, 3] = 1.0
    cov[:, 4, 4] = 1.0
    cov[:, 5, 5] = 1.0
    cov[:, 1, 1] = (1.0 / 3600.0) ** 2
    cov[:, 2, 2] = (1.0 / 3600.0) ** 2

    eph = Ephemeris.from_kwargs(
        orbit_id=[designation],
        coordinates=SphericalCoordinates.from_kwargs(
            rho=[np.nan],
            lon=[ra0],
            lat=[dec0],
            vrho=[np.nan],
            vlon=[np.nan],
            vlat=[np.nan],
            time=Timestamp.from_mjd([float(mjd_mid)], scale="utc"),
            covariance=CoordinateCovariances.from_matrix(cov),
            origin=Origin.from_kwargs(code=[str(obscode)]),
            frame="equatorial",
        ),
    )
    eph.to_parquet(
        str(
            stage2_run_dir
            / "strategies"
            / "2body_with_covariance"
            / "mean_ephemeris"
            / "part-000000.parquet"
        )
    )

    # Minimal Stage 3 selected_keys artifacts (required by Stage 4 harness).
    stage3_run_dir = subset_dir / "artifacts" / "stage3" / stage2_run_dir.name
    keys_tbl = pa.table(
        {
            "orbit_id": pa.array([designation], pa.large_string()),
            "target_idx": pa.array([0], pa.int64()),
            "healpixel": pa.array([int(healpixel)], pa.int64()),
        }
    )
    for footprint in ["point", "cov_disc", "cov_mc", "cov_polygon_moc"]:
        p = stage3_run_dir / "selected_keys" / "2body_with_covariance" / footprint
        p.mkdir(parents=True, exist_ok=True)
        pq.write_table(keys_tbl, p / "selected_keys_unique.parquet")

    run_dir = run_stage4_detection_filter_bench(
        subset_dir=subset_dir,
        stage2_run_dir=stage2_run_dir,
        healpix_nside=32,
        n_sigma=3.0,
        strategies=["2body_with_covariance"],
        only_truth=True,
    )

    m_path = run_dir / "metrics.parquet"
    c_path = run_dir / "coverage.parquet"
    assert m_path.exists()
    assert c_path.exists()

    m = pq.read_table(str(m_path))
    c = pq.read_table(str(c_path))
    assert "strategy" in m.column_names
    assert "detection_filter" in m.column_names
    assert "runtime_total_sec" in m.column_names
    assert "recall" in c.column_names

    c_df = c.to_pandas()
    # At least one combination should recover the single truth detection.
    assert (c_df["n_truth_matched"] == 1).any()
    assert (c_df["n_recovered"] == 1).any()
    assert (c_df["recall"] == 1.0).any()

