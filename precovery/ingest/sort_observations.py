"""
Take any number of adam dataset csv files and reorganize them into calendar month files

We want to do this so that we can run multiple indexing jobs in parallel.
Each worker can be assigned one or more calendar files without fear of collisions in writing to
output files (sorted by dataset/yyyy-mm/*.dat)
"""

import argparse
import glob
import os
import shutil

import pandas as pd


def run(input_dir, output_dir):
    shutil.rmtree(args.output_dir, ignore_errors=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # Get a list of all the files in the input directory
    files = sorted(glob.glob(os.path.join(args.input_dir, "*.h5")))
    # print(f"Found {len(files)} observation files in {args.input_dir}:")
    for observation_file in files:
        # Iterate through observation file
        # extract calendar month from mjd
        # create calendar month file if it doesn't exist
        # append observation to calendar month file
        # print(f"Processing {os.path.basename(observation_file)}")
        iterator = pd.read_csv(observation_file, iterator=True)
        for chunk in iterator:
            print(chunk.columns)
            # Note that we need to handle exposures that run across utc midnight between
            # calendar months. We will need to place observation in both months and then
            # handle possible duplicates on the query side.
            chunk["cal_month"] = (
                pd.to_datetime(chunk["mjd"] + 2400000.5, origin="julian", unit="D")
                .dt.to_period("M")
                .astype(str)
            )

            unique_months = chunk["cal_month"].unique()

            for month in unique_months:
                write_header = False
                mode = "a"
                month_file = (os.path.join(args.output_dir, f"{month}.csv"),)
                if not os.path.exists(month_file):
                    write_header = True
                    mode = "w"
                chunk[chunk["cal_month"] == month].to_csv(
                    os.path.join(args.output_dir, f"{month}.csv"),
                    mode=mode,
                    header=write_header,
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rearrange adam csv files by calendar month."
    )
    parser.add_argument(
        "--input_dir",
        help="Directory containing csv files to be reorganized.",
        type=str,
    )
    parser.add_argument(
        "--output_dir",
        default="out",
        help="Directory where the reorganized files should be saved.",
        type=str,
    )

    args = parser.parse_args()
    run(args.input_dir, args.output_dir)
