from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from math import floor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from .gcs_copy import CopyTool, default_copy_tool


GCS_ROOT = "gs://adam-dataset-dev/production/dagster/complete_precovery_db"


@dataclass(frozen=True)
class SubsetSpec:
    dataset_ids: tuple[str, ...]
    year_months: tuple[str, ...]  # "YYYY-MM"
    obscodes: tuple[str, ...]


def _mjd_utc_from_ymd(year: int, month: int, day: int) -> float:
    """
    Convert a UTC Gregorian calendar date at 00:00:00 to MJD.

    Notes
    -----
    - Uses the standard astronomical JD algorithm for the proleptic Gregorian calendar.
    - Good enough for month boundary filtering (the precovery index stores MJDs in UTC).
    """
    y = int(year)
    m = int(month)
    d = float(day)

    if m <= 2:
        y -= 1
        m += 12

    A = floor(y / 100)
    B = 2 - A + floor(A / 4)
    jd = floor(365.25 * (y + 4716)) + floor(30.6001 * (m + 1)) + d + B - 1524.5
    mjd = jd - 2400000.5
    return float(mjd)


def _month_bounds_mjd_utc(year_month: str) -> tuple[float, float]:
    year_s, month_s = year_month.split("-", 1)
    year = int(year_s)
    month = int(month_s)
    if not (1 <= month <= 12):
        raise ValueError(f"Invalid month in {year_month!r}")

    if month == 12:
        end_year, end_month = year + 1, 1
    else:
        end_year, end_month = year, month + 1

    start_mjd = _mjd_utc_from_ymd(year, month, 1)
    end_mjd = _mjd_utc_from_ymd(end_year, end_month, 1)
    return start_mjd, end_mjd


def _time_predicate_sql(year_months: Sequence[str], col: str = "exposure_mjd_mid") -> str:
    # OR-of-intervals: (col >= a AND col < b) OR ...
    clauses: list[str] = []
    for ym in year_months:
        a, b = _month_bounds_mjd_utc(ym)
        clauses.append(f"({col} >= {a:.10f} AND {col} < {b:.10f})")
    if not clauses:
        raise ValueError("year_months cannot be empty")
    return "(" + " OR ".join(clauses) + ")"


def download_partition_data(
    *,
    dest_db_dir: Path,
    spec: SubsetSpec,
    copy_tool: CopyTool | None = None,
    gcs_root: str = GCS_ROOT,
) -> None:
    """
    Download only the selected dataset/month partitions and metadata needed to run a local DB.

    Layout produced (compatible with `PrecoveryDatabase.from_dir`):
      <dest_db_dir>/
        config.json
        index_full.db          (downloaded full index; used as source for trimming)
        index.db               (trimmed; produced by `build_trimmed_index_db`)
        data/<dataset>/<YYYY-MM>/frames_*.data
    """
    dest_db_dir.mkdir(parents=True, exist_ok=True)
    copy_tool = copy_tool or default_copy_tool()

    # Always download config.json.
    copy_tool.cp(f"{gcs_root}/config.json", dest_db_dir / "config.json")

    # Download the full index.db once (we trim it locally).
    copy_tool.cp(f"{gcs_root}/index.db", dest_db_dir / "index_full.db")

    # Download the month partitions.
    for ds in spec.dataset_ids:
        for ym in spec.year_months:
            # Copy only the partition's frame blobs (avoid directory-recursion semantics
            # differences across copy tools).
            src_glob = f"{gcs_root}/data/{ds}/{ym}/frames_*.data"
            dst_prefix = dest_db_dir / "data" / ds / ym
            dst_prefix.mkdir(parents=True, exist_ok=True)
            copy_tool.cp(src_glob, dst_prefix)


def download_full_index_only(
    *,
    dest_db_dir: Path,
    copy_tool: CopyTool | None = None,
    gcs_root: str = GCS_ROOT,
) -> None:
    """
    Download only the *full* production index + config (no data blobs).

    This produces a local DB directory compatible with `PrecoveryDatabase.from_dir(...)`
    when paired with an experiments-only lazy blob downloader.
    """
    dest_db_dir.mkdir(parents=True, exist_ok=True)
    copy_tool = copy_tool or default_copy_tool()
    root = str(gcs_root).rstrip("/")
    copy_tool.cp(f"{root}/config.json", dest_db_dir / "config.json")
    copy_tool.cp(f"{root}/index.db", dest_db_dir / "index.db")


def build_trimmed_index_db(
    *,
    dest_db_dir: Path,
    spec: SubsetSpec,
    gcs_root: str = GCS_ROOT,
    src_index_db: Path | None = None,
) -> Path:
    """
    Create <dest_db_dir>/index.db containing only the selected datasets/months/obscodes.
    Uses <dest_db_dir>/index_full.db as the source.
    """
    src = src_index_db if src_index_db is not None else (dest_db_dir / "index_full.db")
    if not src.exists():
        raise FileNotFoundError(f"Missing source index db: {src}")

    out = dest_db_dir / "index.db"
    out_tmp = dest_db_dir / "index.trimmed.db"
    if out_tmp.exists():
        out_tmp.unlink()

    time_pred = _time_predicate_sql(spec.year_months, col="exposure_mjd_mid")
    dataset_list = ",".join("?" for _ in spec.dataset_ids)
    use_obscode_filter = len(spec.obscodes) > 0
    obscode_list = ",".join("?" for _ in spec.obscodes) if use_obscode_filter else ""

    with sqlite3.connect(out_tmp) as conn:
        # Use DELETE journaling for a temporary DB that will be renamed.
        # WAL uses sidecar files (<db>-wal/<db>-shm) which would not be moved by `Path.replace()`.
        conn.execute("PRAGMA journal_mode=DELETE;")
        conn.execute("PRAGMA synchronous=NORMAL;")

        # Create tables + indexes matching FrameIndex._create_tables
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

            CREATE INDEX fast_query ON frames (exposure_mjd_mid, healpixel, obscode);
            CREATE INDEX window_centers_idx ON frames (exposure_mjd_mid, dataset_id, obscode);

            CREATE TABLE datasets (
              id VARCHAR PRIMARY KEY,
              name VARCHAR,
              reference_doi VARCHAR,
              documentation_url VARCHAR,
              sia_url VARCHAR
            );
            """
        )

        conn.execute("ATTACH DATABASE ? AS src", (str(src),))

        # Copy dataset metadata for included datasets (if present).
        conn.execute(
            f"""
            INSERT INTO datasets (id, name, reference_doi, documentation_url, sia_url)
            SELECT id, name, reference_doi, documentation_url, sia_url
            FROM src.datasets
            WHERE id IN ({dataset_list})
            """,
            tuple(spec.dataset_ids),
        )

        # Copy frames filtered by dataset, obscode, and month-range predicate.
        if use_obscode_filter:
            conn.execute(
                f"""
                INSERT INTO frames (
                  id, dataset_id, obscode, exposure_id, filter,
                  exposure_mjd_start, exposure_mjd_mid, exposure_duration,
                  healpixel, data_uri, data_offset, data_length
                )
                SELECT
                  id, dataset_id, obscode, exposure_id, filter,
                  exposure_mjd_start, exposure_mjd_mid, exposure_duration,
                  healpixel, data_uri, data_offset, data_length
                FROM src.frames
                WHERE dataset_id IN ({dataset_list})
                  AND obscode IN ({obscode_list})
                  AND {time_pred}
                """,
                tuple(spec.dataset_ids) + tuple(spec.obscodes),
            )
        else:
            conn.execute(
                f"""
                INSERT INTO frames (
                  id, dataset_id, obscode, exposure_id, filter,
                  exposure_mjd_start, exposure_mjd_mid, exposure_duration,
                  healpixel, data_uri, data_offset, data_length
                )
                SELECT
                  id, dataset_id, obscode, exposure_id, filter,
                  exposure_mjd_start, exposure_mjd_mid, exposure_duration,
                  healpixel, data_uri, data_offset, data_length
                FROM src.frames
                WHERE dataset_id IN ({dataset_list})
                  AND {time_pred}
                """,
                tuple(spec.dataset_ids),
            )

        # Ensure no active transaction before DETACH/ANALYZE.
        conn.commit()
        conn.execute("DETACH DATABASE src")
        conn.execute("ANALYZE;")

    # If we are overwriting an existing index.db (common when a full index was copied locally),
    # preserve it as index_full.db for reproducibility.
    if out.exists():
        backup = dest_db_dir / "index_full.db"
        if not backup.exists():
            out.replace(backup)
        else:
            out.unlink()
    out_tmp.replace(out)
    return out


def write_manifest(dest_db_dir: Path, spec: SubsetSpec) -> Path:
    manifest = {
        "gcs_root": GCS_ROOT,
        "dataset_ids": list(spec.dataset_ids),
        "year_months": list(spec.year_months),
        "obscodes": list(spec.obscodes),
    }
    path = dest_db_dir / "subset_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path


def write_index_only_manifest(*, dest_db_dir: Path, gcs_root: str = GCS_ROOT) -> Path:
    manifest = {
        "mode": "full_index_only",
        "gcs_root": str(gcs_root),
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    path = Path(dest_db_dir) / "subset_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path


def _parse_csv_list(arg: str) -> tuple[str, ...]:
    a = str(arg).strip()
    if a.lower() in {"all", "*"}:
        return tuple()
    items = [x.strip() for x in a.split(",") if x.strip()]
    if not items:
        raise ValueError(f"Empty list: {arg!r}")
    return tuple(items)


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download and build a trimmed precovery DB subset.")
    parser.add_argument(
        "--dest",
        required=True,
        help="Destination directory for the local subset DB (will be created).",
    )
    parser.add_argument(
        "--index-only",
        action="store_true",
        help=(
            "Download only config.json + the full index.db (no data blobs, no trimming). "
            "Use this with experiments-only lazy blob downloading."
        ),
    )
    parser.add_argument(
        "--gcs-root",
        type=str,
        default=GCS_ROOT,
        help="GCS root for the production precovery DB (default: complete_precovery_db).",
    )
    parser.add_argument(
        "--datasets",
        required=False,
        help="Comma-separated dataset IDs (e.g., atlas,ztf,nsc,skymapper).",
    )
    parser.add_argument(
        "--months",
        required=False,
        help="Comma-separated YYYY-MM values to include (e.g., 2019-08,2019-09).",
    )
    parser.add_argument(
        "--obscodes",
        required=False,
        help="Comma-separated observatory codes to include (e.g., I41,T05,T08,W84,Q05).",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip GCS download step and only (re)build a trimmed index.db in-place.",
    )
    parser.add_argument(
        "--index-src",
        default=None,
        help=(
            "Optional path to a source index sqlite DB. If omitted, defaults to <dest>/index_full.db. "
            "When --skip-download is used, you typically want this to be <dest>/index.db (a full index)."
        ),
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    dest = Path(args.dest).expanduser().resolve()

    if bool(args.index_only):
        download_full_index_only(dest_db_dir=dest, gcs_root=str(args.gcs_root))
        write_index_only_manifest(dest_db_dir=dest, gcs_root=str(args.gcs_root))
        return

    if args.datasets is None or args.months is None or args.obscodes is None:
        raise ValueError("--datasets, --months, and --obscodes are required unless --index-only is set.")

    spec = SubsetSpec(
        dataset_ids=_parse_csv_list(args.datasets),
        year_months=_parse_csv_list(args.months),
        obscodes=_parse_csv_list(args.obscodes),
    )

    if not bool(args.skip_download):
        download_partition_data(dest_db_dir=dest, spec=spec)
        src_index_db = None
    else:
        src_index_db = Path(args.index_src).expanduser().resolve() if args.index_src else (dest / "index.db")
    build_trimmed_index_db(dest_db_dir=dest, spec=spec, src_index_db=src_index_db)
    write_manifest(dest, spec)


if __name__ == "__main__":
    main()

