import argparse
import logging
import time

import sqlalchemy as sq
from sqlalchemy import and_, or_

from precovery import precovery_db

logger = logging.getLogger("root")

DATABASE_DIR = (
    "/epyc/ssd/users/moeyensj/precovery/precovery_data/nsc/precovery_defrag_db"
)
ORBITS_FILE = "test_orbits.csv"
OUTPUT_FILE = "test_orbits_matches.csv"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run precovery using test_orbits.csv in this directory"
    )
    parser.add_argument("--database_dir", default=DATABASE_DIR, type=str)
    parser.add_argument("--num_days", default=100, type=int)
    parser.add_argument("--sequential", action="store_true")
    args = parser.parse_args()

    db = precovery_db.PrecoveryDatabase.from_dir(args.database_dir, create=True)

    mjds = [57028 + i for i in range(args.num_days)]

    _rows = []
    n_queries = 0
    if args.sequential:
        t = time.time()
        for i in range(len(mjds) - 1):
            stmt = sq.select(
                db.frames.idx.frames.c.mjd, db.frames.idx.frames.c.obscode
            ).where(
                db.frames.idx.frames.c.mjd <= mjds[i + 1],
                db.frames.idx.frames.c.mjd >= mjds[i],
            )
            rows = db.frames.idx.dbconn.execute(stmt)
            for row in rows:
                _rows.append(row)
            n_queries += 1
    else:
        t = time.time()
        stmt = sq.select(
            db.frames.idx.frames.c.mjd, db.frames.idx.frames.c.obscode
        ).where(
            or_(
                *[
                    and_(
                        db.frames.idx.frames.c.mjd <= mjds[i + 1],
                        db.frames.idx.frames.c.mjd >= mjds[i],
                    )
                    for i in range(len(mjds) - 1)
                ]
            )
        )
        rows = db.frames.idx.dbconn.execute(stmt).fetchall()
        _rows += rows
        n_queries += 1

    print(f"{n_queries} queries got {len(_rows)} rows in {time.time() - t}s")
