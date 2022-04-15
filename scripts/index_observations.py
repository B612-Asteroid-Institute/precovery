import os
import glob
import argparse
import datetime
import numpy as np

from precovery.config import DefaultConfig
from precovery.precovery_db import PrecoveryDatabase

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Index observations for precovery.")
    parser.add_argument("data_dir",
        help="Directory containing hdf files to be indexed.",
        type=str,
    )
    parser.add_argument("out_dir",
        help="Directory where the indexed observations and database should be saved.",
        type=str
    )
    parser.add_argument("--nside",
        default=DefaultConfig.nside,
        type=int,
        help="Healpix nside parameter"
    )
    parser.add_argument(
        "--data_file_max_size",
        default=DefaultConfig.data_file_max_size,
        type=int,
        help="Maximum size in bytes of the binary indexed observation files."
    )

    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.data_dir, "*.h5")))
    print(f"Found {len(files)} observation files in {args.data_dir}:")
    for f in files:
        print(f"\t{os.path.basename(f)}")

    db = PrecoveryDatabase.create(
        args.out_dir,
        nside=args.nside,
        data_file_max_size=args.data_file_max_size
    )

    status_file = os.path.join(args.out_dir, "files_indexed.txt")
    if not os.path.exists(status_file):
        read_files = []
        np.savetxt(status_file, read_files, fmt="%s", delimiter='\n')
    else:
        read_files = np.loadtxt(status_file, dtype=str, delimiter='\n', ndmin=1).tolist()

    time_start = datetime.datetime.now()
    for i, observations_file in enumerate(files):
        print(f"Processing ({i + 1}/{len(files)}): {observations_file}")
        if observations_file not in set(read_files):
            db.frames.load_hdf5(observations_file)
            read_files.append(observations_file)
            np.savetxt(status_file, read_files, fmt="%s", delimiter='\n')
        else:
            print(f"File has been indexed previously.")

    time_end = datetime.datetime.now()
    duration = (time_end - time_start)
    print(f"All files indexed in {duration}.")