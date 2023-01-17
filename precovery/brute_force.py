import logging
import multiprocessing as mp
import os
import time
from functools import partial
from typing import Iterable

import numpy as np
import pandas as pd
from astropy.time import Time
from sklearn.neighbors import BallTree

from precovery.precovery_db import PrecoveryDatabase

from .healpix_geom import radec_to_healpixel
# replace this usage with Orbit.compute_ephemeris
from .orbit import Orbit
from .residuals import calc_residuals
from .utils import calcChunkSize, yieldChunks

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


logger = logging.getLogger(__name__)

__all__ = [
    "get_frame_times_by_obscode",
    "ephemerides_from_orbits",
    "intersecting_frames",
    "attribution_worker",
    "attribute_observations",
]


def get_frame_times_by_obscode(
    mjd_start: float,
    mjd_end: float,
    precovery_db: PrecoveryDatabase,
):
    all_frame_mjd = precovery_db.frames.idx.unique_frame_times()
    frame_mjd_within_range = [
        x for x in all_frame_mjd if (x[0] > mjd_start and x[0] < mjd_end)
    ]

    frame_mjds_by_obscode = dict()
    for mjd, obscode in frame_mjd_within_range:
        frame_mjds_by_obscode.setdefault(obscode, []).append(mjd)

    return frame_mjds_by_obscode


def ephemerides_from_orbits(
    orbits: Iterable[Orbit],
    epochs: Iterable[tuple],
):

    ephemeris_dfs = []
    for orbit in orbits:
        for obscode in epochs.keys():
            eph = orbit.compute_ephemeris(obscode=obscode, epochs=epochs[obscode])
            mjd = [w.mjd for w in eph]
            ra = [w.ra for w in eph]
            dec = [w.dec for w in eph]
            ephemeris_df = pd.DataFrame.from_dict(
                {
                    "mjd_utc": mjd,
                    "RA_deg": ra,
                    "Dec_deg": dec,
                }
            )
            ephemeris_df["orbit_id"] = orbit.orbit_id
            ephemeris_df["observatory_code"] = obscode
            ephemeris_dfs.append(ephemeris_df)

    ephemerides = pd.concat(ephemeris_dfs, ignore_index=True)
    return ephemerides


def intersecting_frames(
    ephemerides,
    precovery_db: PrecoveryDatabase,
    neighbors=False,
):

    frame_identifier_dfs = []
    healpixel = radec_to_healpixel(
        list(ephemerides["RA_deg"]),
        list(ephemerides["Dec_deg"]),
        precovery_db.frames.healpix_nside,
        include_neighbors=neighbors
    )

    # we are reformatting this dataframe to account for the 9 healpixels returned from
    # radec_to_healpixel (central plus eight neighbors)
    frame_identifier_df = pd.concat(
        [
            pd.DataFrame.from_dict(
                {
                    "mjd_utc": [x[0] for y in range(9)],
                    "obscode": [x[1] for y in range(9)],
                    "healpixel": list(x[2]),
                }
            )
            for x in zip(
                ephemerides["mjd_utc"], ephemerides["observatory_code"], list(healpixel.transpose())
            )
        ],
        ignore_index=True,
    )
    frame_identifier_dfs.append(frame_identifier_df)

    unique_frame_identifier_df = pd.concat(frame_identifier_dfs, ignore_index=True).drop_duplicates()

    filtered_frames = []
    for fi in unique_frame_identifier_df.itertuples():
        for f in precovery_db.frames.idx.frames_by_obscode_mjd_hp(fi.obscode, fi.mjd_utc, fi.healpixel):
            filtered_frames.append(f)
   
    return filtered_frames


def attribution_worker(
    ephemeris,
    observations,
    eps=1 / 3600,
    include_probabilistic=True,
):

    """
    gather attributions for a df of ephemerides, observations

    First filters ephemerides to match the chunked observations

    """

    # Create observer's dictionary from observations
    observers = {}
    for observatory_code in observations["observatory_code"].unique():
        observers[observatory_code] = Time(
            observations[observations["observatory_code"].isin([observatory_code])][
                "mjd_utc"
            ].unique(),
            scale="utc",
            format="mjd",
        )

    # Group the predicted ephemerides and observations by visit / exposure
    observations_grouped = observations.groupby(by=["observatory_code", "mjd_utc"])
    observations_visits = [
        observations_grouped.get_group(g) for g in observations_grouped.groups
    ]

    # We pre-computed the ephemerides. Now we filter the ephemeris for only visits
    # that have observations in the obs group passed to the worker

    ephemeris_pre_grouped = ephemeris.groupby(by=["observatory_code", "mjd_utc"])
    obs_group_keys = list(observations_grouped.groups.keys())
    indices_to_drop = pd.Int64Index([])
    for g_key in list(ephemeris_pre_grouped.groups.keys()):
        if g_key not in obs_group_keys:
            indices_to_drop = indices_to_drop.union(
                ephemeris_pre_grouped.get_group(g_key).index
            )

    ephemeris_filtered = ephemeris.drop(indices_to_drop)

    # Group the now-filtered ephemerides. There should only be visits for the observation set
    ephemeris_grouped = ephemeris_filtered.groupby(by=["observatory_code", "mjd_utc"])
    ephemeris_visits = [
        ephemeris_grouped.get_group(g) for g in ephemeris_grouped.groups
    ]

    # Loop through each unique exposure and visit, find the nearest observations within
    # eps (haversine metric)
    distances = []
    orbit_ids_associated = []
    obs_ids_associated = []
    obs_times_associated = []
    eps_rad = np.radians(eps)
    residuals = []
    stats = []
    for ephemeris_visit, observations_visit in zip(
        ephemeris_visits, observations_visits
    ):

        assert len(ephemeris_visit["mjd_utc"].unique()) == 1
        assert len(observations_visit["mjd_utc"].unique()) == 1
        assert (
            observations_visit["mjd_utc"].unique()[0]
            == ephemeris_visit["mjd_utc"].unique()[0]
        )

        obs_ids = observations_visit[["obs_id"]].values
        obs_times = observations_visit[["mjd_utc"]].values
        orbit_ids = ephemeris_visit[["orbit_id"]].values
        coords = observations_visit[["RA_deg", "Dec_deg"]].values
        coords_predicted = ephemeris_visit[["RA_deg", "Dec_deg"]].values
        coords_sigma = observations_visit[["RA_sigma_deg", "Dec_sigma_deg"]].values

        # Haversine metric requires latitude first then longitude...
        coords_latlon = np.radians(observations_visit[["Dec_deg", "RA_deg"]].values)
        coords_predicted_latlon = np.radians(
            ephemeris_visit[["Dec_deg", "RA_deg"]].values
        )

        num_obs = len(coords_predicted)
        k = np.minimum(3, num_obs)

        # Build BallTree with a haversine metric on predicted ephemeris
        tree = BallTree(coords_predicted_latlon, metric="haversine")
        # Query tree using observed RA, Dec
        d, i = tree.query(
            coords_latlon,
            k=k,
            return_distance=True,
            dualtree=True,
            breadth_first=False,
            sort_results=False,
        )

        # Select all observations with distance smaller or equal
        # to the maximum given distance
        mask = np.where(d <= eps_rad)

        if len(d[mask]) > 0:
            orbit_ids_associated.append(orbit_ids[i[mask]])
            obs_ids_associated.append(obs_ids[mask[0]])
            obs_times_associated.append(obs_times[mask[0]])
            distances.append(d[mask].reshape(-1, 1))

            residuals_visit, stats_visit = calc_residuals(
                coords[mask[0]],
                coords_predicted[i[mask]],
                sigmas_actual=coords_sigma[mask[0]],
                include_probabilistic=True,
            )
            residuals.append(residuals_visit)
            stats.append(np.vstack(stats_visit).T)

    if len(distances) > 0:
        distances = np.degrees(np.vstack(distances))
        orbit_ids_associated = np.vstack(orbit_ids_associated)
        obs_ids_associated = np.vstack(obs_ids_associated)
        obs_times_associated = np.vstack(obs_times_associated)
        residuals = np.vstack(residuals)
        stats = np.vstack(stats)

        attributions = {
            "orbit_id": orbit_ids_associated[:, 0],
            "obs_id": obs_ids_associated[:, 0],
            "mjd_utc": obs_times_associated[:, 0],
            "distance": distances[:, 0],
            "residual_ra_arcsec": residuals[:, 0] * 3600,
            "residual_dec_arcsec": residuals[:, 1] * 3600,
            "chi2": stats[:, 0],
        }
        if include_probabilistic:
            attributions["probability"] = stats[:, 1]
            attributions["mahalanobis_distance"] = stats[:, 2]

        attributions = pd.DataFrame(attributions)

    else:
        columns = [
            "orbit_id",
            "obs_id",
            "mjd_utc",
            "distance",
            "residual_ra_arcsec",
            "residual_dec_arcsec",
            "chi2",
        ]
        if include_probabilistic:
            columns += ["probability", "mahalanobis_distance"]

        attributions = pd.DataFrame(columns=columns)

    return attributions


def attribute_observations(
    orbits,
    mjd_start: float,
    mjd_end: float,
    precovery_db: PrecoveryDatabase,
    eps=5 / 3600,
    include_probabilistic=True,
    backend="PYOORB",
    backend_kwargs={},
    orbits_chunk_size=10,
    observations_chunk_size=100000,
    num_jobs: int = 1,
):
    logger.info("Running observation attribution...")
    time_start = time.time()

    num_orbits = len(orbits)

    attribution_dfs = []

    # prepare ephemeris and observation dictionaries
    # ephemeris, observations = get_observations_and_ephemerides_for_orbits(
    #     orbits, mjd_start, mjd_end, precovery_db
    # )

    epochs = get_frame_times_by_obscode(mjd_start, mjd_end, precovery_db=precovery_db)
    ephemerides = ephemerides_from_orbits(orbits, epochs)
    frames_to_search = intersecting_frames(ephemerides, precovery_db=precovery_db, neighbors=True)
    observations = precovery_db.extract_observations_by_frames(frames_to_search)
    num_workers = min(num_jobs, mp.cpu_count() + 1)
    if num_workers > 1:

        p = mp.Pool(processes=num_workers)

        # Send up to orbits_chunk_size orbits to each OD worker for processing
        chunk_size_ = calcChunkSize(
            num_orbits, num_workers, orbits_chunk_size, min_chunk_size=1
        )

        orbits_split = [
            orbits[i : i + chunk_size_] for i in range(0, len(orbits), chunk_size_)
        ]

        eph_split = []
        for orbit_c in orbits.split(orbits_chunk_size):
            eph_split.append(
                ephemerides[
                    ephemerides["orbit_id"].isin([orbit.orbit_id for orbit in orbit_c])
                ]
            )
        for observations_c in yieldChunks(observations, observations_chunk_size):

            obs = [observations_c for i in range(len(orbits_split))]
            attribution_dfs_i = p.starmap(
                partial(
                    attribution_worker,
                    eps=eps,
                    include_probabilistic=include_probabilistic,
                    backend=backend,
                    backend_kwargs=backend_kwargs,
                ),
                zip(
                    eph_split,
                    obs,
                ),
            )
            attribution_dfs += attribution_dfs_i

        p.close()

    else:
        for observations_c in yieldChunks(observations, observations_chunk_size):
            for orbit_c in [
                orbits[i : i + orbits_chunk_size]
                for i in range(0, len(orbits), orbits_chunk_size)
            ]:

                eph_c = ephemerides[
                    ephemerides["orbit_id"].isin([orbit.orbit_id for orbit in orbit_c])
                ]
                attribution_df_i = attribution_worker(
                    eph_c,
                    observations_c,
                    eps=eps,
                    include_probabilistic=include_probabilistic,
                )
                attribution_dfs.append(attribution_df_i)

    attributions = pd.concat(attribution_dfs)
    attributions.sort_values(
        by=["orbit_id", "mjd_utc", "distance"], inplace=True, ignore_index=True
    )

    time_end = time.time()
    logger.info(
        "Attributed {} observations to {} orbits.".format(
            attributions["obs_id"].nunique(), attributions["orbit_id"].nunique()
        )
    )
    logger.info(
        "Attribution completed in {:.3f} seconds.".format(time_end - time_start)
    )
    return attributions
