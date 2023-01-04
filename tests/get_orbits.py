import argparse
import os

import pandas as pd
from astropy.time import Time
from astroquery.jplsbdb import SBDB

SAMPLE_ORBITS_FILE = os.path.join(os.path.dirname(__file__), "data", "sample_orbits.csv")

TARGETS = [
    # Atira
    "2020 AV2",
    "163693",

    # Aten
    "2010 TK7",
    "3753",

    # Apollo
    "54509",
    "2063",

    # Amor
    "1221",
    "433",
    "3908",

    # IMB
    "434",
    "1876",
    "2001",

    # MBA
    "2",
    "6",
    "6522",
    "202930",

    # Jupiter Trojans
    "911",
    "1143",
    "1172",
    "3317",

    # Centaur
    "5145",
    "5335",
    "49036",

    # Trans-Neptunian Objects
    "15760",
    "15788",
    "15789",

    # ISOs
    # "A/2017 U1" # Remove ISO since pyoorb can't invert a negative semi-major axis
]

def get_sample_orbits(targets: list[str]) -> pd.DataFrame:
    """
    Query JPL Small-Body Database for orbits of targets.

    Parameters
    ----------
    targets : list[str]
        List of target names.

    Returns
    -------
    orbits: `~pandas.DataFrame`
        DataFrame of containing Keplerian and Cometary elements of each target.
    """
    orbit_dfs = []
    for i, target in enumerate(targets):
        result = SBDB.query(target, full_precision=True, phys=True)

        # Extract the epoch at which the elements are defined
        # and convert it to MJD in TT time scale
        tdb_jd = Time(result["orbit"]["epoch"], scale="tdb", format="jd")
        epoch_df = pd.DataFrame({"mjd_tt": tdb_jd.tt.mjd}, index=[i])

        # Extract the orbital elements and their errors
        elements_df = pd.DataFrame(result["orbit"]["elements"], index=[i])

        # Extract the physical parameters and their errors
        if "G" not in result["phys_par"].keys():
            G = 0.15
        else:
            G = result["phys_par"]["G"]
        phys_df = pd.DataFrame({
                "H": result["phys_par"]["H"],
                "G": G,
            }, 
            index=[i]
        )

        # Combine into a single DataFrame and insert orbit ID
        orbit_i_df = epoch_df.join(elements_df).join(phys_df)
        orbit_i_df.insert(0, "orbit_id", result["object"]["des"])
        orbit_i_df.insert(1, "orbit_name", result["object"]["fullname"])

        orbit_dfs.append(orbit_i_df)

    orbits_df = pd.concat(orbit_dfs)
    return orbits_df

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Get orbits of sample targets from JPL's Small-Body Database."
    )
    parser.add_argument(
        "--out_file",
        type=str, 
        default=SAMPLE_ORBITS_FILE,
        help="Path to output file saved as a CSV.",
    )

    args = parser.parse_args()

    orbits = get_sample_orbits(TARGETS)
    orbits.to_csv(args.out_file, index=False)