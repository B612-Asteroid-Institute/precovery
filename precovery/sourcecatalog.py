import dataclasses
from typing import Dict, Iterator, List, Optional

import pandas as pd

from . import healpix_geom


@dataclasses.dataclass
class SourceObservation:
    exposure_id: str
    obscode: str
    id: bytes
    ra: float
    dec: float
    ra_sigma: float
    dec_sigma: float
    mag: float
    mag_sigma: float
    filter: str
    mjd_start: float
    mjd_mid: float
    exposure_duration: float


@dataclasses.dataclass
class SourceExposure:
    exposure_id: str
    obscode: str
    filter: str
    mjd_start: float
    mjd_mid: float
    exposure_duration: float
    observations: List[SourceObservation]


@dataclasses.dataclass
class SourceFrame:
    exposure_id: str
    obscode: str
    filter: str
    mjd_start: float
    mjd_mid: float
    exposure_duration: float
    healpixel: int
    observations: List[SourceObservation]


def iterate_frames(
    filename: str,
    limit: Optional[int] = None,
    nside: int = 32,
    skip: int = 0,
    key: str = "data",
    chunksize: int = 100000,
) -> Iterator[SourceFrame]:
    for exp in iterate_exposures(filename, limit, skip, key, chunksize):
        for frame in source_exposure_to_frames(exp, nside):
            yield frame


def source_exposure_to_frames(
    src_exp: SourceExposure, nside: int = 32
) -> List[SourceFrame]:
    """ """
    by_pixel: Dict[int, SourceFrame] = {}
    for obs in src_exp.observations:
        pixel = healpix_geom.radec_to_healpixel(obs.ra, obs.dec, nside)
        frame = by_pixel.get(pixel)
        if frame is None:
            frame = SourceFrame(
                exposure_id=src_exp.exposure_id,
                obscode=src_exp.obscode,
                filter=src_exp.filter,
                mjd_start=src_exp.mjd_start,
                mjd_mid=src_exp.mjd_mid,
                exposure_duration=src_exp.exposure_duration,
                healpixel=pixel,
                observations=[],
            )
            by_pixel[pixel] = frame
        frame.observations.append(obs)
    return list(by_pixel.values())


def iterate_exposures(
    filename,
    limit: Optional[int] = None,
    skip: int = 0,
    key: str = "data",
    chunksize: int = 100000,
):
    """
    Yields unique exposures from observations in a file
    """
    current_exposure: Optional[SourceExposure] = None
    n = 0
    for obs in iterate_observations(filename, key=key, chunksize=chunksize):
        if current_exposure is None:
            # first iteration
            current_exposure = SourceExposure(
                exposure_id=obs.exposure_id,
                obscode=obs.obscode,
                filter=obs.filter,
                mjd_start=obs.mjd_start,
                mjd_mid=obs.mjd_mid,
                exposure_duration=obs.exposure_duration,
                observations=[obs],
            )
        elif obs.exposure_id == current_exposure.exposure_id:
            # continuing an existing exposure
            current_exposure.observations.append(obs)
        else:
            # New exposure
            if skip > 0:
                skip -= 1
            else:
                yield current_exposure
                n += 1
            if limit is not None and n >= limit:
                return
            current_exposure = SourceExposure(
                exposure_id=obs.exposure_id,
                obscode=obs.obscode,
                filter=obs.filter,
                mjd_start=obs.mjd_start,
                mjd_mid=obs.mjd_mid,
                exposure_duration=obs.exposure_duration,
                observations=[obs],
            )
    yield current_exposure


def iterate_observations(
    filename: str, key: str = "data", chunksize: int = 100000
) -> Iterator[SourceObservation]:
    with pd.HDFStore(filename, key=key, mode="r") as store:
        for chunk in store.select(
            key=key,
            iterator=True,
            chunksize=chunksize,
            columns=[
                "obs_id",
                "exposure_id",
                "ra",
                "dec",
                "ra_sigma",
                "dec_sigma",
                "mag",
                "mag_sigma",
                "filter",
                "mjd_start_utc",
                "mjd_mid_utc",
                "exposure_duration",
                "observatory_code",
            ],
        ):
            exposure_ids = chunk["exposure_id"].values
            obscodes = chunk["observatory_code"].values
            ids = chunk["obs_id"].values
            ras = chunk["ra"].values.astype(float)
            decs = chunk["dec"].values.astype(float)
            ra_sigmas = chunk["ra_sigma"].values.astype(float)
            dec_sigmas = chunk["dec_sigma"].values.astype(float)
            mags = chunk["mag"].values.astype(float)
            mag_sigmas = chunk["mag_sigma"].values.astype(float)
            filters = chunk["filter"].values
            mjds_start = chunk["mjd_start_utc"].values.astype(float)
            mjds_mid = chunk["mjd_mid_utc"].values.astype(float)
            exposure_durations = chunk["exposure_duration"].values.astype(float)

            for (
                exposure_id,
                obscode,
                id,
                ra,
                dec,
                ra_sigma,
                dec_sigma,
                mag,
                mag_sigma,
                filter,
                mjd_start,
                mjd_mid,
                exposure_duration,
            ) in zip(
                exposure_ids,
                obscodes,
                ids,
                ras,
                decs,
                ra_sigmas,
                dec_sigmas,
                mags,
                mag_sigmas,
                filters,
                mjds_start,
                mjds_mid,
                exposure_durations,
            ):
                obs = SourceObservation(
                    exposure_id,
                    obscode,
                    id.encode(),
                    ra,
                    dec,
                    ra_sigma,
                    dec_sigma,
                    mag,
                    mag_sigma,
                    filter,
                    mjd_start,
                    mjd_mid,
                    exposure_duration,
                )
                yield (obs)
