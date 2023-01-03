import argparse
from astropy.time import Time
import numpy as np
import os
import pandas as pd

from precovery.orbit import Orbit
from precovery.orbit import EpochTimescale
from precovery.orbit import PropagationIntegrator

SAMPLE_ORBITS_FILE = os.path.join(os.path.dirname(__file__), "data", "sample_orbits.csv")
TEST_OBSERVATION_FILE = os.path.join(os.path.dirname(__file__), "data", "observations.h5")

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

        elif orbit_i == "cometary":
            
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
        orbit_type: str = "keplerian"
    ) -> pd.DataFrame:
    """
    Make a synthetic observations file to use for testing. Orbits are read from the input dataframe 
    into the Orbit class. Setting orbit_type will determine which representation of the orbit
    should be used to generate the observations.

    Observations are created with the following cadence: 4 observations per day for 2 weeks. 
    Each observations is seperated by 30 minutes. The epoch of each orbit is used as the start time
    of the 2-week observation period. The exposure duration is 30, 60, 90, and 120 seconds for each 
    nightly observaion quadruplet. 

    Parameters
    ----------
`   orbits_df : `~pd.DataFrame`
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

    # Four observations daily for two weeks
    dts = np.linspace(0, 14, 15)
    dts = np.concatenate([dts, dts + 1/24/2, dts + 1/24, dts + 1/24 + 1/24/2])
    dts.sort()

    # Create exposure triplets 
    unique_exposure_durations = np.array([30., 60., 90., 120.])
    num_obs = int(len(dts) / len(unique_exposure_durations))
    exposure_duration = np.hstack([unique_exposure_durations for i in range(num_obs)])

    # Create list of observatory codes
    OBSERVATORY_CODES = ["500", "I11", "I41", "F51"]
    observatory_codes = [OBSERVATORY_CODES[j] for j in range(len(unique_exposure_durations)) for i in range(num_obs)]

    ephemeris_dfs = []
    for i, orbit in enumerate(orbits):
        initial_epoch = Time(orbit._epoch, scale="tt", format="mjd")
        # Observation times are defined at the center of the exposure (for now)
        observation_times = initial_epoch.utc.mjd + dts + exposure_duration / 86400 / 2.
        exposure_ids = [f"{obs_i}_{k:06d}" for k, obs_i in enumerate(observatory_codes)]

        ephemeris_list = []
        for obs_i, time_i in zip(observatory_codes, observation_times):
            ephemeris_list.append(orbit.compute_ephemeris(obs_i, [time_i])[0])
            
        ephemeris_dict = {
            "mjd_utc" : [],
            "ra" : [],
            "dec" : [],
            "mag" : [],
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
            ephemeris_dict["mjd_utc"].append(eph_i._raw_data[0])
            ephemeris_dict["ra"].append(eph_i._raw_data[1])
            ephemeris_dict["dec"].append(eph_i._raw_data[2])
            ephemeris_dict["mag"].append(eph_i._raw_data[9])

        ephemeris_df = pd.DataFrame(ephemeris_dict)
        ephemeris_df.insert(0, "object_id", orbit_ids[i])
        ephemeris_df.insert(4, "ra_sigma", 0.)
        ephemeris_df.insert(4, "dec_sigma", 0.)
        ephemeris_df.insert(6, "mag_sigma", 0.)
        ephemeris_df.insert(7, "filter", "V")
        ephemeris_df.insert(8, "exposure_id", exposure_ids)
        ephemeris_df["observatory_code"] = observatory_codes
        ephemeris_df["mjd_start_utc"] = initial_epoch.utc.mjd + dts
        ephemeris_df["mjd_mid_utc"] = initial_epoch.utc.mjd + dts + exposure_duration / 86400 / 2.
        ephemeris_df["exposure_duration"] = exposure_duration
        ephemeris_dfs.append(ephemeris_df)

    observations = pd.concat(ephemeris_dfs, ignore_index=True)
    observations.sort_values(by=["mjd_mid_utc", "observatory_code"], inplace=True)
    observations.insert(1, "obs_id", [f"obs_{i:08d}" for i in range(len(observations))])

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
        "--out_file",
        type=str, 
        default=TEST_OBSERVATION_FILE,
        help="Path to output observations file saved as a HDF5.",
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
    observations.to_hdf(args.out_file, key="data", mode="w", format="table")
