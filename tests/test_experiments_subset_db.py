import sqlite3
from pathlib import Path

import pytest

from experiments.covariance_precovery.data.subset_db import (
    SubsetSpec,
    _month_bounds_mjd_utc,
    build_trimmed_index_db,
)


def _create_source_index_db(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE frames (
              id INTEGER PRIMARY KEY,
              dataset_id VARCHAR NOT NULL,
              obscode VARCHAR NOT NULL,
              exposure_id VARCHAR NOT NULL,
              filter VARCHAR,
              exposure_mjd_start FLOAT NOT NULL,
              exposure_mjd_mid FLOAT NOT NULL,
              exposure_duration FLOAT NOT NULL,
              healpixel INTEGER NOT NULL,
              data_uri VARCHAR NOT NULL,
              data_offset INTEGER NOT NULL,
              data_length INTEGER NOT NULL
            );

            CREATE TABLE datasets (
              id VARCHAR PRIMARY KEY,
              name VARCHAR,
              reference_doi VARCHAR,
              documentation_url VARCHAR,
              sia_url VARCHAR
            );
            """
        )
        conn.executemany(
            "INSERT INTO datasets (id, name) VALUES (?, ?)",
            [
                ("ztf", "ZTF"),
                ("atlas", "ATLAS"),
            ],
        )

        # Insert frames spanning two months, two datasets, two obscodes.
        aug_start, aug_end = _month_bounds_mjd_utc("2019-08")
        sep_start, _ = _month_bounds_mjd_utc("2019-09")
        assert aug_start < aug_end
        assert aug_end == sep_start

        rows = [
            # In-scope: ztf + I41 + August
            (
                1,
                "ztf",
                "I41",
                "exp1",
                "r",
                aug_start + 0.1,
                aug_start + 0.2,
                30.0,
                123,
                "ztf/2019-08/frames_00000001.data",
                0,
                100,
            ),
            # Out-of-scope obscode
            (
                2,
                "ztf",
                "T05",
                "exp2",
                "r",
                aug_start + 0.3,
                aug_start + 0.4,
                30.0,
                123,
                "ztf/2019-08/frames_00000002.data",
                0,
                100,
            ),
            # Out-of-scope dataset
            (
                3,
                "atlas",
                "I41",
                "exp3",
                "o",
                aug_start + 0.5,
                aug_start + 0.6,
                30.0,
                123,
                "atlas/2019-08/frames_00000003.data",
                0,
                100,
            ),
            # Out-of-scope month (September)
            (
                4,
                "ztf",
                "I41",
                "exp4",
                "r",
                sep_start + 0.1,
                sep_start + 0.2,
                30.0,
                123,
                "ztf/2019-09/frames_00000004.data",
                0,
                100,
            ),
        ]
        conn.executemany(
            """
            INSERT INTO frames (
              id, dataset_id, obscode, exposure_id, filter,
              exposure_mjd_start, exposure_mjd_mid, exposure_duration,
              healpixel, data_uri, data_offset, data_length
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )


def test_build_trimmed_index_db_filters_frames(tmp_path: Path) -> None:
    dest = tmp_path / "subset"
    dest.mkdir()
    src = dest / "index_full.db"
    _create_source_index_db(src)

    spec = SubsetSpec(dataset_ids=("ztf",), year_months=("2019-08",), obscodes=("I41",))
    out = build_trimmed_index_db(dest_db_dir=dest, spec=spec)
    assert out.exists()

    with sqlite3.connect(out) as conn:
        rows = conn.execute(
            "SELECT id, dataset_id, obscode, exposure_id FROM frames ORDER BY id"
        ).fetchall()
        assert rows == [(1, "ztf", "I41", "exp1")]

        # Ensure key indices exist (as expected by FrameIndex).
        idx_names = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            ).fetchall()
        }
        assert "fast_query" in idx_names
        assert "window_centers_idx" in idx_names


@pytest.mark.parametrize(
    "ym,expected_start_mjd",
    [
        ("2000-01", 51544.0),  # 2000-01-01 00:00 UTC
        ("2019-08", 58696.0),  # 2019-08-01 00:00 UTC
    ],
)
def test_month_bounds_sanity(ym: str, expected_start_mjd: float) -> None:
    start, end = _month_bounds_mjd_utc(ym)
    assert start < end
    # Month boundary conversion should be exact at day boundaries for these examples.
    assert abs(start - expected_start_mjd) < 1e-6

