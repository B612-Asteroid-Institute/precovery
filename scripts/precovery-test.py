import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.time import Time

from precovery import precovery_db
from precovery.orbit import EpochTimescale, Orbit

logger = logging.getLogger("root")
# logger.setLevel(logging.DEBUG)

DATABASE_DIR = "/mnt/data/projects/precovery/precovery_data/nsc/precovery_month_db"
ORBITS_FILE = "test_orbits.csv"
OUTPUT_FILE = "test_orbits_matches.csv"


def matches_to_dict(matches):
    data = {}
    for i, m in enumerate(matches):
        for k, v in m.__dict__.items():
            if i == 0:
                data[k] = []

            data[k].append(v)

    return data


def matches_to_df(matches):
    data = matches_to_dict(matches)
    df = pd.DataFrame(data)
    # Organize columns
    df = df[
        [
            "mjd_utc",
            "ra_deg",
            "dec_deg",
            "ra_sigma_arcsec",
            "dec_sigma_arcsec",
            "mag",
            "mag_sigma",
            "filter",
            "obscode",
            "exposure_id",
            "observation_id",
            "healpix_id",
            "pred_ra_deg",
            "pred_dec_deg",
            "pred_vra_degpday",
            "pred_vdec_degpday",
            "delta_ra_arcsec",
            "delta_dec_arcsec",
            "distance_arcsec",
        ]
    ]
    # Sort rows
    df.sort_values(
        by=["mjd_utc", "obscode"], inplace=True, ignore_index=True, ascending=True
    )
    return df


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
    orbits = pd.read_csv(orbits_file)

    matches_dfs = []
    for i in range(np.minimum(len(orbits), args.num_orbits)):
        # Select a single orbit
        orbit_i = orbits.iloc[i : i + 1]

        orbit_id = orbit_i["orbit_id"].values[0]
        # Define precovery orbit
        orbit = Orbit.cartesian(
            orbit_id,
            orbit_i["x"].values[0],
            orbit_i["y"].values[0],
            orbit_i["z"].values[0],
            orbit_i["vx"].values[0],
            orbit_i["vy"].values[0],
            orbit_i["vz"].values[0],
            Time(orbit_i["mjd_tdb"].values[0], scale="tdb", format="mjd").utc.mjd,
            EpochTimescale.UTC,
            20,
            0.15,
        )

        matches = [m for m in db.precover(orbit, tolerance=1 / 3600)[0].to_dataclass()]
        matches_df = matches_to_df(matches)
        matches_df.insert(0, "orbit_id", orbit_id)
        matches_dfs.append(matches_df)

        print(f"Found {len(matches)} potential matches for orbit ID: {orbit_id}")
        print(matches_df)

    matches_df = pd.concat(matches_dfs, ignore_index=True)
    matches_df.to_csv(args.out_file, index=False)
