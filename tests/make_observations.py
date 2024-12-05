import argparse
import os

import numpy as np
import pandas as pd
from adam_assist import ASSISTPropagator
from adam_core.observers import Observers
from adam_core.orbits import Orbits
from adam_core.time import Timestamp

SAMPLE_ORBITS_FILE = os.path.join(
    os.path.dirname(__file__), "data", "sample_orbits.parquet"
)
TEST_OBSERVATIONS_DIR = os.path.join(os.path.dirname(__file__), "data/index")


def make_observations(
    orbits: Orbits,
) -> pd.DataFrame:
    """
    Makes 3 synthetic observations files to use for testing. Orbits are an instance
    of the adam_core Orbits class

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
    # Initialize an assist propagator for ephemeris generation
    propagator = ASSISTPropagator()

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

    # Define different uncertainties for different observatories
    # Set astrometric uncertainties for I41 to nan to test that they are interpreted correctly
    # Set photometric uncertainties for 500 to nan to test that they are interpreted correctly
    astrometric_uncertainties = {
        "500": 1.0 / 3600.0,
        "I11": 0.01 / 3600.0,
        "I41": np.nan,
        "F51": 0.1 / 3600.0,
    }
    photometric_uncertainties = {
        "500": np.nan,
        "I11": 0.001,
        "I41": 0.1,
        "F51": 0.01,
    }

    # Create exposure quads
    unique_exposure_durations = np.array([30.0, 60.0, 90.0, 120.0])
    num_obs = int(len(dts) / len(unique_exposure_durations))
    exposure_duration = np.hstack([unique_exposure_durations for i in range(num_obs)])

    # Set random seed
    rng = np.random.default_rng(seed=2023)

    ephemeris_dfs = []
    for i in range(len(orbits)):
        # initial_epoch = Time(orbit._epoch, scale="tt", format="mjd")
        print(f"Generating observations for object {orbits[i].object_id[0].as_py()}")
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
                orbits[i].coordinates.time.rescale("utc").mjd()[0].as_py()
                + dts
                + observatory_nightly_offset
                + observatory_window_offset
            )
            exposure_mid_times = exposure_start_times + exposure_duration / 2 / 86400
            # observation_times = exposure_start_times + rng.uniform(
            #     0, exposure_duration / 86400
            # )
            observation_times = exposure_mid_times

            times = Timestamp.from_mjd(observation_times, scale="utc")
            # create observers
            observers = Observers.from_code(observatory_code, times)
            ephemeris = propagator.generate_ephemeris(orbits[i], observers)

            ephemeris_df = pd.DataFrame(
                {
                    "mjd": ephemeris.coordinates.time.mjd().to_pylist(),
                    "ra": ephemeris.coordinates.lon.to_pylist(),
                    "dec": ephemeris.coordinates.lat.to_pylist(),
                }
            )

            ephemeris_df.insert(0, "object_id", orbits[i].object_id[0].as_py())
            # We don't use magnitudes anywhere in our tests right now
            ephemeris_df.insert(3, "mag", 18.0)
            ephemeris_df.insert(
                4, "ra_sigma", astrometric_uncertainties[observatory_code]
            )
            ephemeris_df.insert(
                5, "dec_sigma", astrometric_uncertainties[observatory_code]
            )
            ephemeris_df.insert(
                6, "mag_sigma", photometric_uncertainties[observatory_code]
            )
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
    args = parser.parse_args()
    orbits = Orbits.from_parquet(args.in_file)
    observations = make_observations(orbits)
    for dataset_id in observations["dataset_id"].unique():
        dataset_observations = observations[observations["dataset_id"] == dataset_id]
        dataset_dir = os.path.join(args.out_dir, dataset_id)
        os.makedirs(dataset_dir, exist_ok=True)
        dataset_observations.to_csv(
            os.path.join(dataset_dir, f"{dataset_id}_observations.csv"),
            index=False,
            float_format="%.16f",
        )
