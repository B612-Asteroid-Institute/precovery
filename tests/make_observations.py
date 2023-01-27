import argparse
import os

import numpy as np
import pandas as pd
from astropy.time import Time

from precovery.orbit import EpochTimescale, Orbit, PropagationIntegrator

SAMPLE_ORBITS_FILE = os.path.join(
    os.path.dirname(__file__), "data", "sample_orbits.csv"
)
TEST_OBSERVATIONS_DIR = os.path.join(os.path.dirname(__file__), "data/index")


def dataframe_to_orbit(
    orbits_df: pd.DataFrame,
    orbit_type: str = "keplerian",
) -> list[Orbit]:
    """
    Initialize list of Orbit objects from a pandas DataFrame.

    TODO: Add support for cartesian orbits.

    Parameters
    ----------
    orbits_df : `~pd.DataFrame`
        DataFrame containing orbital elements for each orbit.
    orbit_type : str, optional
        Type of orbit to initialize. Must be either "keplerian" or "cometary".

    Returns
    -------
    orbits : list[Orbit]
        List of Orbit objects.
    """
    orbits = []
    for i in range(len(orbits_df)):
        if orbit_type == "keplerian":
            orbit_i = Orbit.keplerian(
                i,
                orbits_df["a"].values[i],
                orbits_df["e"].values[i],
                orbits_df["i"].values[i],
                orbits_df["om"].values[i],
                orbits_df["w"].values[i],
                orbits_df["ma"].values[i],
                orbits_df["mjd_tt"].values[i],
                EpochTimescale.TT,
                orbits_df["H"].values[i],
                orbits_df["G"].values[i],
            )

        elif orbit_type == "cometary":
            # Extract time of perihelion passage
            tp = Time(
                orbits_df["tp"].values[i],
                format="jd",
                scale="tdb",
            )
            orbit_i = Orbit.cometary(
                i,
                orbits_df["q"].values[i],
                orbits_df["e"].values[i],
                orbits_df["i"].values[i],
                orbits_df["om"].values[i],
                orbits_df["w"].values[i],
                tp.tt.mjd,
                orbits_df["mjd_tt"].values[i],
                EpochTimescale.TT,
                orbits_df["H"].values[i],
                orbits_df["G"].values[i],
            )

        else:
            raise ValueError("orbit_type must be either 'keplerian' or 'cometary'")

        orbits.append(orbit_i)

    return orbits


def make_observations(
    orbits_df: pd.DataFrame,
    orbit_type: str = "keplerian",
) -> pd.DataFrame:
    """
    Makes 3 synthetic observations files to use for testing. Orbits are read from the input dataframe
    into the Orbit class. Setting orbit_type will determine which representation of the orbit
    should be used to generate the observations.

    Observations are created with the following cadence: 4 observations per day for 2 weeks per observatory.
    Each observatory will be offset by 10 days from the previous observatory so that we have overlapping
    and non-overlapping coverage between observatories. Each nightly observation is seperated by 30 minutes.
    The epoch of each orbit is used as the start time of the 2-week observation period. The exposure duration
    is 30, 60, 90, and 120 seconds for each nightly observation quadruplet. The last two observatories are
    combined into one dataset to test multi-observatory functionality.

    Parameters
    ----------
    orbits_df : `~pd.DataFrame`
        DataFrame containing orbital elements for each orbit.
    orbit_type : str, optional
        Type of orbit to initialize. Must be either "keplerian" or "cometary".

    Returns
    -------
    observations : `~pandas.DataFrame`
        DataFrame containing observations.
    """
    # Extract orbit names and read orbits into Orbit class
    orbit_ids = orbits_df["orbit_name"].values
    orbits = dataframe_to_orbit(orbits_df, orbit_type=orbit_type)

    # Create list of observatory codes, each observatory code will be placed into
    # its own dataset, with the exception of the last two which will be combined into one
    observatory_codes = ["500", "I11", "I41", "F51"]

    # Each observatory is offset by a fixed amount from the start of the observation period ( approximate
    # location on Earth relative to GMT)
    observatory_nightly_offsets = [0, -5 / 24, -7 / 24, -10 / 24]

    # Each observatory's observation window is offset so we create overlapping and non-overlapping coverage
    observatory_window_offsets = [0, 10, 20, 30]

    # Four observations daily for 2 weeks
    dts = np.linspace(0, 14, 14 * 2)
    dts = np.concatenate(
        [dts, dts + 1 / 24 / 2, dts + 1 / 24, dts + 1 / 24 + 1 / 24 / 2]
    )
    dts.sort()

    # Create exposure quads
    unique_exposure_durations = np.array([30.0, 60.0, 90.0, 120.0])
    num_obs = int(len(dts) / len(unique_exposure_durations))
    exposure_duration = np.hstack([unique_exposure_durations for i in range(num_obs)])

    # Set random seed
    rng = np.random.default_rng(seed=2023)

    ephemeris_dfs = []
    for i, orbit in enumerate(orbits):
        initial_epoch = Time(orbit._epoch, scale="tt", format="mjd")

        ephemeris_list = []
        for (
            observatory_code,
            observatory_nightly_offset,
            observatory_window_offset,
        ) in zip(
            observatory_codes, observatory_nightly_offsets, observatory_window_offsets
        ):
            # Calculate a random offset from the start of the exposure
            # to give each observation a unique obervation time
            exposure_start_times = (
                initial_epoch.utc.mjd
                + dts
                + observatory_nightly_offset
                + observatory_window_offset
            )
            exposure_mid_times = exposure_start_times + exposure_duration / 2 / 86400
            observation_times = exposure_start_times + rng.uniform(
                0, exposure_duration / 86400
            )

            ephemeris_list = orbit.compute_ephemeris(
                observatory_code,
                observation_times,
                method=PropagationIntegrator.N_BODY,
                time_scale=EpochTimescale.UTC,
            )

            ephemeris_dict = {
                "mjd": [],
                "ra": [],
                "dec": [],
                "mag": [],
            }
            for eph_i in ephemeris_list:
                # PYOORB basic ephemeris
                # modified julian date
                # right ascension (deg)
                # declination (deg)
                # dra/dt sky-motion (deg/day, including cos(dec) factor)
                # ddec/dt sky-motion (deg/day)
                # solar phasae angle (deg)
                # solar elongation angle (deg)
                # heliocentric distance (au)
                # geocentric distance (au)
                # predicted apparent V-band magnitude
                # true anomaly (deg)
                ephemeris_dict["mjd"].append(eph_i._raw_data[0])
                ephemeris_dict["ra"].append(eph_i._raw_data[1])
                ephemeris_dict["dec"].append(eph_i._raw_data[2])
                ephemeris_dict["mag"].append(eph_i._raw_data[9])

            ephemeris_df = pd.DataFrame(ephemeris_dict)
            ephemeris_df.insert(0, "object_id", orbit_ids[orbit.orbit_id])
            ephemeris_df.insert(4, "ra_sigma", 0.0)
            ephemeris_df.insert(4, "dec_sigma", 0.0)
            ephemeris_df.insert(6, "mag_sigma", 0.0)
            ephemeris_df.insert(7, "filter", "V")

            ephemeris_df["observatory_code"] = observatory_code
            ephemeris_df["exposure_mjd_start"] = exposure_start_times
            ephemeris_df["exposure_mjd_mid"] = exposure_mid_times
            ephemeris_df["exposure_duration"] = exposure_duration
            ephemeris_dfs.append(ephemeris_df)

    observations = pd.concat(ephemeris_dfs, ignore_index=True)
    observations.insert(1, "obs_id", [f"obs_{i:08d}" for i in range(len(observations))])

    for observatory_code in ["500", "I11"]:
        observations.loc[
            observations["observatory_code"] == observatory_code, "dataset_id"
        ] = f"dataset_{observatory_code}"

    # Combine I41 and F51 into one dataset
    observations.loc[
        observations["observatory_code"].isin(["I41", "F51"]), "dataset_id"
    ] = "dataset_I41+F51"

    observations["exposure_id"] = observations[
        ["observatory_code", "exposure_mjd_mid"]
    ].apply(lambda x: f"{x[0]}_{x[1]:.5f}", axis=1)
    observations.sort_values(
        by=["dataset_id", "exposure_mjd_mid", "exposure_id"], inplace=True
    )

    return observations


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Get orbits of sample targets from JPL's Small-Body Database."
    )
    parser.add_argument(
        "--in_file",
        type=str,
        default=SAMPLE_ORBITS_FILE,
        help="Path to input orbits file saved as a CSV",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=TEST_OBSERVATIONS_DIR,
        help=(
            "Directory in which to save new observations. "
            "Each dataset is saved into a unique directory within the passed directory."
        ),
    )
    parser.add_argument(
        "--orbit_type",
        type=str,
        default="keplerian",
        help="Type of orbit to initialize. Must be either 'keplerian' or 'cometary'.",
    )

    args = parser.parse_args()
    orbits_df = pd.read_csv(args.in_file)
    observations = make_observations(orbits_df, orbit_type=args.orbit_type)
    for dataset_id in observations["dataset_id"].unique():
        dataset_observations = observations[observations["dataset_id"] == dataset_id]
        dataset_dir = os.path.join(args.out_dir, dataset_id)
        os.makedirs(dataset_dir, exist_ok=True)
        dataset_observations.to_csv(
            os.path.join(dataset_dir, f"{dataset_id}_observations.csv"),
            index=False,
            float_format="%.16f",
        )
