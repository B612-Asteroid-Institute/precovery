import argparse
import os

from adam_core.orbits import Orbits
from adam_core.orbits.query import query_sbdb

SAMPLE_ORBITS_FILE = os.path.join(
    os.path.dirname(__file__), "data", "sample_orbits.parquet"
)

TARGETS = [
    # Atira
    "2020 AV2",
    "163693",
    # Aten
    "2010 TK7",
    "3753",
    # Apollo
    # Excluding YORP due to nongravs
    # "54509",
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


def get_sample_orbits(targets: list[str]) -> Orbits:
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
    orbits = query_sbdb(targets)
    return orbits


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
    orbits.to_parquet(args.out_file)
