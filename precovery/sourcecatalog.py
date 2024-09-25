import csv
import dataclasses
from collections import defaultdict
from typing import DefaultDict, Iterator, List, Optional

import numpy as np
import pandas as pd

from . import healpix_geom


def allow_nan(x: str) -> float:
    """
    NaNs are often serialized as empty strings. This function converts them back to NaNs.

    Parameters
    ----------
    x : str
    """
    if x == "":
        return np.nan
    else:
        return float(x)


@dataclasses.dataclass
class SourceObservation:
    exposure_id: str
    obscode: str
    id: bytes
    mjd: float
    ra: float
    dec: float
    ra_sigma: float
    dec_sigma: float
    mag: float
    mag_sigma: float
    filter: str
    exposure_mjd_start: float
    exposure_mjd_mid: float
    exposure_duration: float


@dataclasses.dataclass
class SourceFrame:
    exposure_id: str
    obscode: str
    filter: str
    exposure_mjd_start: float
    exposure_mjd_mid: float
    exposure_duration: float
    healpixel: int
    observations: List[SourceObservation]

    @classmethod
    def from_observation(cls, obs: SourceObservation, healpixel: int) -> "SourceFrame":
        """
        Constructs a SourceFrame, sourcing fields from a SourceObservation.

        The observations list is left empty.
        """
        return SourceFrame(
            exposure_id=obs.exposure_id,
            obscode=obs.obscode,
            filter=obs.filter,
            exposure_mjd_start=obs.exposure_mjd_start,
            exposure_mjd_mid=obs.exposure_mjd_mid,
            exposure_duration=obs.exposure_duration,
            healpixel=healpixel,
            observations=[],
        )


def bundle_into_frames(
    observations: Iterator[SourceObservation], nside: int = 32
) -> Iterator[SourceFrame]:
    """Groups SourceObservations into SourceFrames, suitable for
    loading into the database. The observations iterator should be
    sorted by exposure_id so that all SourceObservations for a frame
    are linked together.

    nside is the healpix nside parameter.
    """

    # Chomp through the observations iterator for an entire
    # exposure. When exposure_id changes, do the work of partitioning
    # the exposure's data into frames, and yield each frame out
    # one-by-one.
    cur_exposure_id: Optional[str] = None
    observations_by_healpixel: DefaultDict[int, List[SourceObservation]] = defaultdict(
        list
    )
    for obs in observations:
        if cur_exposure_id is None:
            # first iteration
            cur_exposure_id = obs.exposure_id

        if obs.exposure_id != cur_exposure_id:
            # change of exposure: emit frames
            for pixel, pixel_observations in observations_by_healpixel.items():
                frame = SourceFrame.from_observation(pixel_observations[0], pixel)
                frame.observations = pixel_observations
                yield frame

            # Reset observations dictionary
            observations_by_healpixel.clear()
            cur_exposure_id = obs.exposure_id

        healpixel = healpix_geom.radec_to_healpixel(obs.ra, obs.dec, nside)
        observations_by_healpixel[healpixel].append(obs)

    # Yield the final iteration
    for pixel, pixel_observations in observations_by_healpixel.items():
        frame = SourceFrame.from_observation(pixel_observations[0], pixel)
        frame.observations = pixel_observations
        yield frame


def observations_from_csv_file(
    filename: str, limit: Optional[int] = None, skip: int = 0
) -> Iterator[SourceObservation]:
    """
    Reads a stream of SourceObservations out of a CSV data file.
    """
    with open(filename) as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            obs = SourceObservation(
                exposure_id=str(row["exposure_id"]),
                obscode=str(row["observatory_code"]),
                id=str(row["obs_id"]).encode(),
                mjd=float(row["mjd"]),
                ra=float(row["ra"]),
                dec=float(row["dec"]),
                ra_sigma=allow_nan(row["ra_sigma"]),
                dec_sigma=allow_nan(row["dec_sigma"]),
                mag=float(row["mag"]),
                mag_sigma=allow_nan(row["mag_sigma"]),
                filter=str(row["filter"]),
                exposure_mjd_start=float(row["exposure_mjd_start"]),
                exposure_mjd_mid=float(row["exposure_mjd_mid"]),
                exposure_duration=float(row["exposure_duration"]),
            )
            yield (obs)


def observations_from_dataframe(df: pd.DataFrame) -> Iterator[SourceObservation]:
    raise NotImplementedError("not implemented yet")


def frames_from_csv_file(
    filename: str,
    limit: Optional[int] = None,
    nside: int = 32,
    skip: int = 0,
) -> Iterator[SourceFrame]:
    """Reads out SourceFrames from a CSV file, suitable for loading
    into the database. The rows of the source CSV file are expected to
    be sorted by exposure_id.

    """

    observations = observations_from_csv_file(filename, limit, skip)
    frames = bundle_into_frames(observations, nside)
    for f in frames:
        yield f
