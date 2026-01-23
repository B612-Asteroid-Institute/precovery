from __future__ import annotations

import argparse
import os

from .config import Config
from .filter_limiting_magnitudes import FilterLimitingMagnitudes
from .frame_db import FrameIndex


def generate_limiting_magnitudes_parquet_cache(
    *, db_dir: str, out_file: str = "limiting_magnitudes.parquet"
) -> str:
    """
    Read limiting magnitudes from `index.db` and write a generated-once Parquet cache file.
    """
    idx = FrameIndex("sqlite:///" + os.path.join(db_dir, "index.db"), mode="r")
    try:
        rows = idx.limiting_magnitude_rows()
    finally:
        idx.close()

    if len(rows) == 0:
        table = FilterLimitingMagnitudes.empty()
    else:
        obscodes, filter_ids, limiting_mags, mag_systems = zip(*rows)
        table = FilterLimitingMagnitudes.from_kwargs(
            obscode=list(obscodes),
            filter_id=list(filter_ids),
            limiting_mag=list(limiting_mags),
            mag_system=list(mag_systems),
        )

    out_path = os.path.join(db_dir, out_file)
    table.to_parquet(out_path)
    return out_path


def import_limiting_magnitudes(
    *,
    db_dir: str,
    table: FilterLimitingMagnitudes,
) -> None:
    """
    Upsert limiting magnitude rows into the DB.
    """
    idx = FrameIndex("sqlite:///" + os.path.join(db_dir, "index.db"), mode="w")
    try:
        idx.migrate()
        obscodes = table.obscode.to_pylist()
        filter_ids = table.filter_id.to_pylist()
        limiting_mags = table.limiting_mag.to_pylist()
        mag_systems = table.mag_system.to_pylist()

        for obscode, filter_id, limiting_mag, mag_system in zip(
            obscodes, filter_ids, limiting_mags, mag_systems
        ):
            idx.upsert_limiting_magnitude(
                obscode=str(obscode),
                filter_id=str(filter_id),
                limiting_mag=float(limiting_mag),
                mag_system=None if mag_system is None else str(mag_system),
            )
    finally:
        idx.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import limiting magnitudes into a precovery DB and generate a fast cache file."
    )
    parser.add_argument("--db-dir", required=True, help="Precovery DB directory")
    parser.add_argument(
        "--in-file",
        required=True,
        help="Parquet file containing a FilterLimitingMagnitudes table",
    )
    parser.add_argument(
        "--cache-file",
        default="limiting_magnitudes.parquet",
        help="Cache parquet filename to write in the DB directory",
    )
    parser.add_argument(
        "--margin-mag",
        type=float,
        default=0.0,
        help="Optional faint-frame skip margin (added to limiting mag)",
    )

    args = parser.parse_args()

    db_dir = args.db_dir
    in_file = args.in_file
    if not os.path.exists(os.path.join(db_dir, "index.db")):
        raise SystemExit(f"{db_dir} does not look like a precovery DB (missing index.db)")

    if not in_file.lower().endswith(".parquet"):
        raise SystemExit("--in-file must be .parquet")
    table = FilterLimitingMagnitudes.from_parquet(in_file)

    import_limiting_magnitudes(db_dir=db_dir, table=table)
    out_path = generate_limiting_magnitudes_parquet_cache(db_dir=db_dir, out_file=args.cache_file)

    # Update the DB's config.json so precovery loads the cache file automatically.
    cfg_path = os.path.join(db_dir, "config.json")
    cfg = Config.from_json(cfg_path)
    cfg.limiting_magnitudes_parquet_file = args.cache_file
    cfg.faint_frame_skip_margin_mag = float(args.margin_mag)
    cfg.to_json(cfg_path)

    print(f"Wrote cache: {out_path}")
