import dataclasses
from typing import Dict, Iterator, List, Optional

import tables
from rich.progress import BarColumn, Progress, TimeElapsedColumn, TimeRemainingColumn

from . import healpix_geom


@dataclasses.dataclass
class SourceObservation:
    exposure_id: str
    obscode: str
    id: bytes
    ra: float
    dec: float
    epoch: float


@dataclasses.dataclass
class SourceExposure:
    exposure_id: str
    obscode: str
    mjd: float
    observations: List[SourceObservation]


@dataclasses.dataclass
class SourceFrame:
    exposure_id: str
    obscode: str
    mjd: float
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
    by_pixel: Dict[int, SourceFrame] = {}
    for obs in src_exp.observations:
        pixel = healpix_geom.radec_to_healpixel(obs.ra, obs.dec, nside)
        frame = by_pixel.get(pixel)
        if frame is None:
            frame = SourceFrame(
                exposure_id=src_exp.exposure_id,
                obscode=src_exp.obscode,
                mjd=src_exp.mjd,
                healpixel=pixel,
                observations=[],
            )
            by_pixel[pixel] = frame
        frame.observations.append(obs)
    return list(by_pixel.values())


def iterate_exposures(filename, limit: Optional[int] = None, skip: int = 0):
    current_exposure: Optional[SourceExposure] = None
    n = 0
    for obs in iterate_observations(filename):
        if current_exposure is None:
            # first iteration
            current_exposure = SourceExposure(
                exposure_id=obs.exposure_id,
                obscode=obs.obscode,
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
                mjd=obs.epoch,
                observations=[obs],
            )
    yield current_exposure


def iterate_observations(filename: str) -> Iterator[SourceObservation]:
    with tables.open_file(filename, mode="r") as f:
        table = f.get_node("/data/table")

        expected_format = [
            "class_star",
            "dec",
            "decerr",
            "deltamjd",
            "mag_auto",
            "magerr_auto",
            "mean_dec",
            "mean_mjd",
            "mean_ra",
            "mjd",
            "ra",
            "raerr",
        ]
        assert table.attrs["values_block_0_kind"] == expected_format

        n_rows = table.nrows
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
            for row in table.iterrows():
                exposure_id = row["exposure"]
                obscode = _obscode_from_exposure_id(exposure_id)
                id = row["id"]
                dec = row["values_block_0"][1]
                ra = row["values_block_0"][10]

                epoch = row["values_block_0"][9]

                obs = SourceObservation(exposure_id, obscode, id, ra, dec, epoch)
                yield (obs)
                progress.update(read_observations, advance=1)


def _obscode_from_exposure_id(exposure_id: bytes) -> str:
    # The table has no explicit information on which instrument sourced the
    # exposure. We have to glean it out of the exposure ID.
    exp_prefix = exposure_id[:3].decode()
    if exp_prefix == "c4d":
        # The CTIO-4m with DECam.
        return "807"
    elif exp_prefix == "ksb":
        # The Bok 2.3m with 90Prime
        return "V00"
    elif exp_prefix == "k4m":
        # The KPNO 4pm with Mosaic3
        return "695"
    else:
        raise ValueError(
            f"can't determine instrument for exposure {exposure_id.decode()}"
        )
