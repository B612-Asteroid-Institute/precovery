from __future__ import annotations

import argparse
import os

from .frame_db import FrameIndex


def migrate_db(db_dir: str) -> None:
    """
    Run additive schema migrations on an existing precovery DB.
    """
    index_path = os.path.join(db_dir, "index.db")
    if not os.path.exists(index_path):
        raise SystemExit(
            f"{db_dir} does not look like a precovery DB (missing index.db)"
        )

    idx = FrameIndex("sqlite:///" + index_path, mode="w")
    try:
        idx.migrate()
    finally:
        idx.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate a precovery DB schema in-place."
    )
    parser.add_argument("--db-dir", required=True, help="Precovery DB directory")
    args = parser.parse_args()
    migrate_db(args.db_dir)
    print("Migration complete.")
