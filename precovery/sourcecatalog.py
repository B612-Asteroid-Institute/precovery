import dataclasses
from typing import Dict, Iterator, List, Optional

import pandas as pd
from rich.progress import BarColumn, Progress, TimeElapsedColumn, TimeRemainingColumn

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
    epoch: float


@dataclasses.dataclass
class SourceExposure:
    exposure_id: str
    obscode: str
    filter: str
    mjd: float
    observations: List[SourceObservation]


@dataclasses.dataclass
class SourceFrame:
    exposure_id: str
    obscode: str
    filter: str
    mjd: float
    healpixel: int
    observations: List[SourceObservation]


def iterate_frames(
    filename: str,
    limit: Optional[int] = None,
    nside: int = 32,
    skip: int = 0,
    key: str = "data",
    chunksize: int = 100000
) -> Iterator[SourceFrame]:
    for exp in iterate_exposures(filename, limit, skip, key, chunksize):
        for frame in source_exposure_to_frames(exp, nside):
            yield frame


def source_exposure_to_frames(
    src_exp: SourceExposure, nside: int = 32
) -> List[SourceFrame]:
    by_pixel: Dict[int, SourceFrame] = {}
    for obs in src_exp.observations:
        pixel = healpix_geom.radec_to_healpixel(obs.ra, obs.dec, nside)
        frame = by_pixel.get(pixel)
        if frame is None:
            frame = SourceFrame(
                exposure_id=src_exp.exposure_id,
                obscode=src_exp.obscode,
                filter=src_exp.filter,
                mjd=src_exp.mjd,
                healpixel=pixel,
                observations=[],
            )
            by_pixel[pixel] = frame
        frame.observations.append(obs)
    return list(by_pixel.values())


def iterate_exposures(
        filename,
        limit:
        Optional[int] = None,
        skip: int = 0,
        key: str = "data",
        chunksize: int = 100000
    ):
    current_exposure: Optional[SourceExposure] = None
    n = 0
    for obs in iterate_observations(filename, key=key, chunksize=chunksize):
        if current_exposure is None:
            # first iteration
            current_exposure = SourceExposure(
                exposure_id=obs.exposure_id,
                obscode=obs.obscode,
                filter=obs.filter,
                mjd=obs.epoch,
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
                mjd=obs.epoch,
                observations=[obs],
            )
    yield current_exposure


def iterate_observations(
        filename: str,
        key: str = "data",
        chunksize: int = 100000
    ) -> Iterator[SourceObservation]:
    with pd.HDFStore(filename, key=key, mode="r") as store:

        n_rows = store.get_storer(key).nrows

        with Progress(
            "[progress.description]{task.description}",
            BarColumn(),
            "[progress.percentage]{task.completed} / {task.total}  {task.percentage:>3.0f}%",
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ) as progress:
            read_observations = progress.add_task(
                "loading observations...", total=n_rows
            )
            for chunk in store.select(
                key=key,
                iterator=True,
                chunksize=chunksize,
                columns=[
                    "obs_id", "exposure_id", "mjd_utc",
                    "ra", "dec", "ra_sigma", "dec_sigma",
                    "mag", "mag_sigma", "filter",
                    "observatory_code"
                ]
            ):
                exposure_ids = chunk["exposure_id"].values
                obscodes = chunk["observatory_code"].values
                ids = chunk["obs_id"].values
                ras = chunk["ra"].values
                decs = chunk["dec"].values
                ra_sigmas = chunk["ra_sigma"].values
                dec_sigmas = chunk["dec_sigma"].values
                mags = chunk["mag"].values
                mag_sigmas = chunk["mag_sigma"].values
                filters = chunk["filter"].values
                epochs = chunk["mjd_utc"].values

                for exposure_id, obscode, id, ra, dec, ra_sigma, dec_sigma, mag, mag_sigma, filter, epoch in zip(
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
                    epochs
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
                        epoch
                    )
                    yield (obs)
                    progress.update(read_observations, advance=1)
