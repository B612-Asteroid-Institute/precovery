import dataclasses
import logging
import multiprocessing as mp
import os
import time
import warnings
from functools import partial
from typing import Iterable

import numpy as np
import pandas as pd
from astropy.time import Time
from sklearn.neighbors import BallTree

from precovery.precovery_db import PrecoveryCandidate, PrecoveryDatabase

from .healpix_geom import radec_to_healpixel
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


@dataclasses.dataclass
class BruteForceAttribution(PrecoveryCandidate):
    chi2: float
    orbit_id: str
    probability: float
    mahalanobis_distance: float


def get_frame_times_by_obscode(
    mjd_start: float,
    mjd_end: float,
    precovery_db: PrecoveryDatabase,
    obscodes_specified: Iterable[str] = [],
):
    all_frame_mjd = precovery_db.frames.idx.unique_frame_times()
    frame_mjd_within_range = [
        x for x in all_frame_mjd if (x[0] > mjd_start and x[0] < mjd_end)
    ]
    frame_mjds_by_obscode = dict()
    for exposure_mjd_mid, obscode in frame_mjd_within_range:
        if len(obscodes_specified) == 0 or obscode in obscodes_specified:
            frame_mjds_by_obscode.setdefault(obscode, []).append(exposure_mjd_mid)

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
            vra = [w.ra_velocity for w in eph]
            vdec = [w.dec_velocity for w in eph]
            ephemeris_df = pd.DataFrame.from_dict(
                {
                    "mjd_utc": mjd,
                    "ra_deg": ra,
                    "dec_deg": dec,
                    "vra_degperday": vra,
                    "vdec_degperday": vdec,
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
    healpix_id = radec_to_healpixel(
        list(ephemerides["ra_deg"]),
        list(ephemerides["dec_deg"]),
        precovery_db.frames.healpix_nside,
        include_neighbors=neighbors,
    )

    # we are reformatting this dataframe to account for the 9 healpixels returned from
    # radec_to_healpixel (central plus eight neighbors)
    frame_identifier_df = pd.concat(
        [
            pd.DataFrame.from_dict(
                {
                    "mjd_utc": [x[0] for y in range(9)],
                    "obscode": [x[1] for y in range(9)],
                    "healpix_id": list(x[2]),
                }
            )
            for x in zip(
                ephemerides["mjd_utc"],
                ephemerides["observatory_code"],
                list(healpix_id.transpose()),
            )
        ],
        ignore_index=True,
    )
    frame_identifier_dfs.append(frame_identifier_df)

    unique_frame_identifier_df = pd.concat(
        frame_identifier_dfs, ignore_index=True
    ).drop_duplicates()

    filtered_frames = []
    for fi in unique_frame_identifier_df.itertuples():
        for f in precovery_db.frames.idx.frames_by_obscode_mjd_hp(
            fi.obscode, fi.mjd_utc, fi.healpix_id
        ):
            filtered_frames.append(f)

    return filtered_frames


def attribution_worker(
    ephemeris,
    observations,
    tolerance=1 / 3600,
    include_probabilistic=True,
):
    """
    gather attributions for a df of ephemerides, observations

    First filters ephemerides to match the chunked observations

    """

    attributions = []
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
    indices_to_drop = pd.Index([], dtype="int64")
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
    # tolerance (haversine metric)
    distances = []
    orbit_ids_associated = []
    obs_ids_associated = []
    mag_associated = []
    mag_sigma_associated = []
    obs_times_associated = []
    coords_associated = []
    coords_pred_associated = []
    velocities_associated = []
    coords_sigma_associated = []
    frame_metadata_associated = []
    tolerance_rad = np.radians(tolerance)
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
        mag = observations_visit[["mag"]].values
        mag_sigma = observations_visit[["mag_sigma"]].values
        orbit_ids = ephemeris_visit[["orbit_id"]].values
        coords = observations_visit[["ra_deg", "dec_deg"]].values
        coords_predicted = ephemeris_visit[["ra_deg", "dec_deg"]].values
        velocities = ephemeris_visit[["vra_degperday", "vdec_degperday"]].values
        coords_sigma = observations_visit[["ra_sigma_deg", "dec_sigma_deg"]].values
        frame_metadata = observations_visit[
            [
                "exposure_mjd_start",
                "exposure_mjd_mid",
                "filter",
                "observatory_code",
                "exposure_id",
                "exposure_duration",
                "healpix_id",
                "dataset_id",
            ]
        ].values
        # Haversine metric requires latitude first then longitude...
        coords_latlon = np.radians(observations_visit[["dec_deg", "ra_deg"]].values)
        coords_predicted_latlon = np.radians(
            ephemeris_visit[["dec_deg", "ra_deg"]].values
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
        mask = np.where(d <= tolerance_rad)

        if len(d[mask]) > 0:
            orbit_ids_associated.append(orbit_ids[i[mask]])
            obs_ids_associated.append(obs_ids[mask[0]])
            obs_times_associated.append(obs_times[mask[0]])
            mag_associated.append(mag[mask[0]])
            mag_sigma_associated.append(mag_sigma[mask[0]])
            coords_associated.append(coords[mask[0]])
            coords_pred_associated.append(coords_predicted[i[mask]])
            velocities_associated.append(velocities[i[mask]])
            coords_sigma_associated.append(coords_sigma[mask[0]])
            frame_metadata_associated.append(frame_metadata[mask[0]])
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
        mag_associated = np.vstack(mag_associated)
        mag_sigma_associated = np.vstack(mag_sigma_associated)
        coords_associated = np.vstack(coords_associated)
        coords_pred_associated = np.vstack(coords_pred_associated)
        velocities_associated = np.vstack(velocities_associated)
        coords_sigma_associated = np.vstack(coords_sigma_associated)
        frame_metadata_associated = np.vstack(frame_metadata_associated)
        residuals = np.vstack(residuals)
        stats = np.vstack(stats)

        attributions = [
            BruteForceAttribution(
                mjd=mjd_utc[0],
                ra_deg=coords[0],
                dec_deg=coords[1],
                ra_sigma_arcsec=coords_sigma[0] * 3600.0,
                dec_sigma_arcsec=coords_sigma[1] * 3600.0,
                mag=mag[0],
                mag_sigma=mag_sigma[0],
                observation_id=obs_id[0],
                pred_ra_deg=coords_pred[0],
                pred_dec_deg=coords_pred[1],
                pred_vra_degpday=velocities[0],
                pred_vdec_degpday=velocities[1],
                delta_ra_arcsec=residual[0] * 3600.0,
                delta_dec_arcsec=residual[1] * 3600.0,
                distance_arcsec=distance[0] * 3600.0,
                exposure_mjd_start=frame_metadata[0],
                exposure_mjd_mid=frame_metadata[1],
                filter=frame_metadata[2],
                obscode=frame_metadata[3],
                exposure_id=frame_metadata[4],
                exposure_duration=frame_metadata[5],
                healpix_id=frame_metadata[6],
                dataset_id=frame_metadata[7],
                chi2=stats[0],
                orbit_id=orbit_id[0],
                probability=stats[1],
                mahalanobis_distance=stats[2],
            )
            for (
                orbit_id,
                obs_id,
                mjd_utc,
                mag,
                mag_sigma,
                distance,
                residual,
                coords,
                coords_pred,
                velocities,
                coords_sigma,
                stats,
                frame_metadata,
            ) in zip(
                orbit_ids_associated,
                obs_ids_associated,
                obs_times_associated,
                mag_associated,
                mag_sigma_associated,
                distances,
                residuals,
                coords_associated,
                coords_pred_associated,
                velocities_associated,
                coords_sigma_associated,
                stats,
                frame_metadata_associated,
            )
        ]
    return attributions


def attribute_observations(
    orbits,
    mjd_start: float,
    mjd_end: float,
    precovery_db: PrecoveryDatabase,
    tolerance=5 / 3600,
    include_probabilistic=True,
    orbits_chunk_size: int = 10,
    observations_chunk_size: int = 100000,
    num_jobs: int = 1,
    obscodes_specified: Iterable[str] = [],
):
    """


    Parameters
    ----------
    orbits : list
        List of Orbit objects to attribute observations to.
    mjd_start : float
        Start time of observations to attribute.
    mjd_end : float
        End time of observations to attribute.
    precovery_db : PrecoveryDatabase
        Precovery database to attribute observations from.
    tolerance : float, optional
        Tolerance in degrees to attribute observations to orbits, by default 5 / 3600
    include_probabilistic : bool, optional
        Whether to include probabilistic statistics in the attribution, by default True
    orbits_chunk_size : int, optional
        Number of orbits to attribute in each chunk, by default 10
    observations_chunk_size : int, optional
        Number of observations to attribute in each chunk, by default 100000
    num_jobs : int, optional
        Number of jobs to run in parallel, by default 1
        Note that there is a very rare edge case when multiple versions of the same orbit
        are provided - in this case the ballTree used will fail to consider all orbits
        when computing nearest neighbors.
    obscodes_specified : Iterable[str], optional
        List of obscodes to attribute observations from, an empty list means no filter
        is applied on the obscodes returned (i.e. every dataset is searched)
    """

    logger.info("Running observation attribution...")
    time_start = time.time()

    num_orbits = len(orbits)

    attribution_lists = []
    observations = []
    epochs = get_frame_times_by_obscode(
        mjd_start,
        mjd_end,
        precovery_db=precovery_db,
        obscodes_specified=obscodes_specified,
    )
    ephemerides = ephemerides_from_orbits(orbits, epochs)
    frames_to_search = intersecting_frames(
        ephemerides, precovery_db=precovery_db, neighbors=True
    )
    if len(frames_to_search) != 0:
        observations = precovery_db.extract_observations_by_frames(
            frames_to_search, include_frame_metadata=True
        )
    else:
        warning = (
            "No intersecting frames were found. This is unlikely unless"
            " a sparsely populated database or small O(1) orbit set is used."
        )
        warnings.warn(warning, UserWarning)
        return []
    num_workers = min(num_jobs, mp.cpu_count() + 1)

    logger.info(
        "Splitting {} observations from {} frames over {} workers.".format(
            len(observations),
            len(frames_to_search),
            num_workers,
        )
    )
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
        for orbit_c in orbits_split:
            eph_split.append(
                ephemerides[
                    ephemerides["orbit_id"].isin([orbit.orbit_id for orbit in orbit_c])
                ]
            )
        for observations_c in yieldChunks(observations, observations_chunk_size):
            obs = [observations_c for i in range(len(orbits_split))]
            attribution_lists_i = p.starmap(
                partial(
                    attribution_worker,
                    tolerance=tolerance,
                    include_probabilistic=include_probabilistic,
                ),
                zip(
                    eph_split,
                    obs,
                ),
            )
            attribution_lists += attribution_lists_i

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
                attribution_list_i = attribution_worker(
                    eph_c,
                    observations_c,
                    tolerance=tolerance,
                    include_probabilistic=include_probabilistic,
                )
                attribution_lists.append(attribution_list_i)

    # Flatten the list of attribution lists
    attributions = [
        attribution
        for attribution_list in attribution_lists
        for attribution in attribution_list
    ]
    attribution_df = pd.DataFrame(attributions)
    attribution_df.sort_values(
        by=["orbit_id", "mjd", "distance_arcsec"], inplace=True, ignore_index=True
    )
    time_end = time.time()
    logger.info(
        "Attributed {} observations ({} unique) to {} orbits.".format(
            attribution_df["observation_id"].count(),
            attribution_df["observation_id"].nunique(),
            attribution_df["orbit_id"].nunique(),
        )
    )
    logger.info(
        "Attribution completed in {:.3f} seconds.".format(time_end - time_start)
    )
    return attributions, attribution_df
