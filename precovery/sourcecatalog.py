from typing import Optional

import tables
from rich.progress import BarColumn, Progress, TimeElapsedColumn, TimeRemainingColumn


def iterate_exposures(filename, limit: Optional[int] = None):
    exposure_data = []
    exposure_meta = {
        "exposure_id": None,
    }
    n = 0
    for obs in iterate_observations(filename):
        (exposure, exp_prefix, obscode, id, dec, ra, epoch) = obs
        if exposure == exposure_meta["exposure_id"]:
            # Continuing an existing exposure
            exposure_data.append((ra, dec, id))
            if ra > exposure_meta["ra_max"]:
                exposure_meta["ra_max"] = ra
            elif ra < exposure_meta["ra_min"]:
                exposure_meta["ra_min"] = ra
            if dec > exposure_meta["dec_max"]:
                exposure_meta["dec_max"] = dec
            elif dec < exposure_meta["dec_min"]:
                exposure_meta["dec_min"] = dec
        else:
            # New exposure
            if len(exposure_data) > 0:
                yield exposure_meta, exposure_data
                n += 1
                if limit is not None and n >= limit:
                    return
            exposure_meta = {
                "exposure_id": exposure,
                "obscode": obscode,
                "ra_min": ra,
                "ra_max": ra,
                "dec_min": dec,
                "dec_max": dec,
                "mjd": epoch,
            }
            exposure_data = [(ra, dec, id)]
    if len(exposure_data) > 0:
        yield exposure_meta, exposure_data


def iterate_observations(filename):
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
                exposure = row["exposure"]

                # The table has no explicit information on which instrument sourced the
                # exposure. We have to glean it out of the exposure ID.
                exp_prefix = exposure[:3].decode()
                if exp_prefix == "c4d":
                    # The CTIO-4m with DECam.
                    obscode = "807"
                elif exp_prefix == "ksb":
                    # The Bok 2.3m with 90Prime
                    obscode = "V00"
                elif exp_prefix == "k4m":
                    # The KPNO 4pm with Mosaic3
                    obscode = "695"
                else:
                    raise ValueError(
                        f"can't determine instrument for exposure {exposure}"
                    )

                id = row["id"]
                dec = row["values_block_0"][1]
                ra = row["values_block_0"][10]
                epoch = row["values_block_0"][9]

                yield (exposure, exp_prefix, obscode, id, dec, ra, epoch)
                progress.update(read_observations, advance=1)
