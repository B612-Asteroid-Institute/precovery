import argparse
import logging
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import quivr as qv
from adam_core.orbits import Orbits
from adam_core.propagator.adam_assist import ASSISTPropagator

from precovery import precovery_db

from .precovery_db import PrecoveryCandidates

logger = logging.getLogger("root")
# logger.setLevel(logging.DEBUG)

DATABASE_DIR = "/mnt/data/projects/precovery/precovery_data/nsc/precovery_month_db"
ORBITS_FILE = "test_orbits.csv"
OUTPUT_FILE = "test_orbits_matches.csv"

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run precovery using test_orbits.csv in this directory"
    )
    parser.add_argument("--database_dir", default=DATABASE_DIR, type=str)
    parser.add_argument("--orbits_file", default=ORBITS_FILE, type=str)
    parser.add_argument("--out_file", default=OUTPUT_FILE, type=str)
    parser.add_argument(
        "--tolerance",
        default=1 / 3600,
        type=float,
        help=(
            "Astrometric tolerance to within which observations are considered"
            " precoveries."
        ),
    )
    parser.add_argument(
        "--num_orbits",
        default=1,
        type=int,
        help="Number of orbits from orbits_file to run through precovery database.",
    )
    args = parser.parse_args()

    db = precovery_db.PrecoveryDatabase.from_dir(args.database_dir, create=True)

    orbits_file = Path(args.orbits_file).resolve()
    orbits = Orbits.from_parquet(orbits_file)

    all_matches: List[PrecoveryCandidates] = []
    for i in range(np.minimum(len(orbits), args.num_orbits)):
        # Select a single orbit
        orbit = orbits[i]

        candidates, frame_candidates = db.precover(
            orbit, tolerance=1 / 3600, propagator_class=ASSISTPropagator
        )

        print(
            f"Found {len(candidates)} potential matches for orbit ID: {orbit.object_id[0].as_py()}"
        )

    precovery_candidates = qv.concatenate(all_matches)
    precovery_candidates.to_parquet(args.out_file)
