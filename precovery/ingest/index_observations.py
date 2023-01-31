import argparse
import datetime
import glob
import logging
import multiprocessing
import os

from precovery.config import DefaultConfig
from precovery.precovery_db import PrecoveryDatabase

logger = multiprocessing.get_logger()
logger.setLevel(logging.INFO)


def arg_wrap_individual_file(args):
    """
    Wrapper to expand arguments for use with imap_unordered.

    This keeps the original function signature the same but allows us to run it in parallel.
    """
    return index_individual_file(*args)


def index_individual_file(
    db_dir,
    filename,
    dataset_id,
    dataset_name,
    reference_doi,
    documentation_url,
    sia_url,
):
    """
    Loads a single calendar month dataset file into the database and filesystem.

    Note:
        This function only works because the files are pre-divided by dataset/calendar month.
        This means that there should be no chance of collision writing to the binaries files on disk.
    """

    logger = multiprocessing.get_logger()
    logger.info(f"Indexing {filename}")

    # Initialize the database client within the worker.
    db = PrecoveryDatabase.from_dir(db_dir, mode="w")
    db.frames.add_dataset(
        dataset_id=dataset_id,
        name=dataset_name,
        reference_doi=reference_doi,
        documentation_url=documentation_url,
        sia_url=sia_url,
    )
    db.frames.load_csv(
        filename,
        dataset_id,
    )
    logger.info(f"Indexed {filename}")
    db.frames.close()


def index(
    data_dir="data",
    out_dir="database",
    cpu_count=None,
    dataset_id=None,
    nside=DefaultConfig.nside,
    data_file_max_size=DefaultConfig.data_file_max_size,
    dataset_name=None,
    reference_doi=None,
    documentation_url=None,
    sia_url=None,
):
    logger = multiprocessing.get_logger()
    logger.setLevel(logging.INFO)

    files = glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True)
    files = sorted(set(files))
    logger.info(f"Found {len(files)} observation files in {data_dir}:")
    for f in files:
        logger.info(f"\t{os.path.basename(f)}")

    # Ensure database is initialized in main thread before starting
    PrecoveryDatabase.create(
        out_dir, nside=nside, data_file_max_size=data_file_max_size
    )

    time_start = datetime.datetime.now()

    completed = 0
    cpu_count = cpu_count or multiprocessing.cpu_count() + 1

    pool = multiprocessing.Pool(cpu_count)

    for result in pool.imap_unordered(
        arg_wrap_individual_file,
        [
            (
                out_dir,
                filename,
                dataset_id,
                dataset_name,
                reference_doi,
                documentation_url,
                sia_url,
            )
            for filename in files
        ],
    ):
        completed += 1
        logger.info(f"{completed}/{len(files)} files completed.")
    pool.close()
    pool.join()

    time_end = datetime.datetime.now()
    duration = time_end - time_start
    logger.info(f"All files indexed in {duration}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index observations for precovery.")
    parser.add_argument(
        "data_dir",
        help="Directory containing csv files to be indexed.",
        type=str,
    )
    parser.add_argument(
        "out_dir",
        help="Directory where the indexed observations and database should be saved.",
        type=str,
    )
    parser.add_argument("dataset_id", help="Dataset ID for this file.", type=str)
    parser.add_argument(
        "--nside", default=DefaultConfig.nside, type=int, help="Healpix nside parameter"
    )
    parser.add_argument(
        "--cpu_count", default=None, type=int, help="Number of CPUs to use"
    )
    parser.add_argument(
        "--data_file_max_size",
        default=DefaultConfig.data_file_max_size,
        type=int,
        help="Maximum size in bytes of the binary indexed observation files.",
    )
    parser.add_argument(
        "--dataset_name", help="Dataset name for this file.", type=str, default=None
    )
    parser.add_argument(
        "--reference_doi",
        help="DOI of the reference paper for this dataset.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--documentation_url",
        help="URL of any documentation available for this dataset.",
        type=str,
    )
    parser.add_argument(
        "--sia_url", help="Simple Image Access URL for this dataset.", type=str
    )

    args = parser.parse_args()

    index(**args)  # type: ignore
