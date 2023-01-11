import csv
import dataclasses
from typing import Dict, Iterator, List, Optional

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
) -> Iterator[SourceFrame]:
    for exp in iterate_exposures(filename, limit, skip):
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
):
    """
    Yields unique exposures from observations in a file
    """
    current_exposure: Optional[SourceExposure] = None
    n = 0
    for obs in iterate_observations(filename):
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


def iterate_observations(filename: str) -> Iterator[SourceObservation]:
    with open(filename) as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            obs = SourceObservation(
                exposure_id=row["exposure_id"],
                obscode=row["observatory_code"],
                id=row["obs_id"].encode(),
                ra=float(row["ra"]),
                dec=float(row["dec"]),
                ra_sigma=float(row["ra_sigma"]),
                dec_sigma=float(row["dec_sigma"]),
                mag=float(row["mag"]),
                mag_sigma=float(row["mag_sigma"]),
                filter=row["filter"],
                mjd_start=float(row["mjd_start_utc"]),
                mjd_mid=float(row["mjd_mid_utc"]),
                exposure_duration=float(row["exposure_duration"]),
            )
            yield (obs)
