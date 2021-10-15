import argparse

from precovery.precovery_db import PrecoveryDatabase


def parse_args():
    parser = argparse.ArgumentParser("precoverydb-defrag", "defragment a precoverydb")
    parser.add_argument(
        "src_dir", help="Directory holding the existing, fragmented database"
    )
    parser.add_argument("dst_dir", help="Directory to hold the new, defragged database")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    src_db = PrecoveryDatabase.from_dir(args.src_dir, create=False)
    dst_db = PrecoveryDatabase.from_dir(args.dst_dir, create=True)

    src_db.frames.defragment(dst_db.frames.idx, dst_db.frames)


if __name__ == "__main__":
    main()
