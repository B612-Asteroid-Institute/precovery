import argparse

from precovery.precovery_db import PrecoveryDatabase


def parse_args():
    parser = argparse.ArgumentParser(
        "precoverydb-load-hdf", "populate a precoverydb with data from NSC hdf5 files"
    )
    parser.add_argument("db-dir", help="Directory holding the database")
    parser.add_argument("data-file", help="Data file holding data to load")
    parser.add_argument(
        "--skip",
        type=int,
        help=(
            "Number of source frames to skip in the source dataset, "
            + "for resuming a load operation"
        ),
        default=0,
    )
    parser.add_argument(
        "--max", type=int, help="Maximum number of frames to store", default=-1
    )
    parser.add_argument(
        "--create", type=bool, help="Create database if it doesn't exist", default=False
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    db = PrecoveryDatabase.from_dir(args.db_dir, args.create)
    if args.max == -1:
        limit = None
    else:
        limit == args.max
    db.frames.load_hdf5(args.data_file, skip=args.skip, limit=limit)


if __name__ == "__main__":
    main()
