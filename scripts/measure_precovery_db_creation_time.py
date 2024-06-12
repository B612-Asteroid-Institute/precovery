import argparse
import json
import shutil
import time
from datetime import datetime

import numpy as np

from precovery.precovery_db import PrecoveryDatabase


def count_lines(file_path):
    with open(file_path) as f:
        return sum(1 for _ in f)


def measure_db_creation_time(database_directory, csv_file_path, dataset_id, nside):
    start_time = time.time()

    db = PrecoveryDatabase.create(database_directory, nside)
    db.frames.add_dataset(dataset_id)
    db.frames.load_csv(csv_file_path, dataset_id)

    elapsed_time = time.time() - start_time
    db.frames.close()

    return elapsed_time


def print_statistics(elapsed_times):
    elapsed_times_array = np.array(elapsed_times)
    print("Summary statistics:")
    print(f"  Mean time: {np.mean(elapsed_times_array):.2f} seconds")
    print(f"  Median time: {np.median(elapsed_times_array):.2f} seconds")
    print(f"  P10 time: {np.percentile(elapsed_times_array, 10):.2f} seconds")
    print(f"  P90 time: {np.percentile(elapsed_times_array, 90):.2f} seconds")
    print(f"  Standard deviation: {np.std(elapsed_times_array):.2f} seconds")


def save_results_to_json(
    db, output_json_file, elapsed_times, num_lines, input_data_filename, run_timestamp
):
    elapsed_times_array = np.array(elapsed_times)
    results = {
        "input_data_filename": input_data_filename,
        "number_of_rows": num_lines,
        "healpix_nside": db.frames.healpix_nside,
        "run_timestamp": run_timestamp,
        "database_statistics": {
            "number_of_frames": db.frames.idx.n_frames(),
            "database_size_bytes": db.frames.idx.n_bytes(),
            "unique_datasets": list(db.frames.idx.get_dataset_ids()),
        },
        "summary_statistics": {
            "mean_time": np.mean(elapsed_times_array),
            "median_time": np.median(elapsed_times_array),
            "p10_time": np.percentile(elapsed_times_array, 10),
            "p90_time": np.percentile(elapsed_times_array, 90),
            "stdev_time": np.std(elapsed_times_array),
        },
        "execution_times": elapsed_times,
    }

    with open(output_json_file, "w") as json_file:
        json.dump(results, json_file, indent=4)


def main(args):
    elapsed_times = []
    num_lines = count_lines(args.csv_file_path)

    for i in range(args.num_iterations):
        print(f"Running iteration {i + 1}/{args.num_iterations}")
        shutil.rmtree(args.database_directory, ignore_errors=True)
        elapsed_time = measure_db_creation_time(
            args.database_directory,
            args.csv_file_path,
            args.dataset_id,
            2**args.healpixel_order,
        )
        elapsed_times.append(elapsed_time)

    run_timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    if args.output_json_file is not None:
        output_json_file = args.output_json_file
    else:
        output_json_file = "precovery_create_benchmark.json"

    # Load the existing database for gathering statistics
    db = PrecoveryDatabase.from_dir(args.database_directory)

    save_results_to_json(
        db,
        output_json_file,
        elapsed_times,
        num_lines,
        args.csv_file_path,
        run_timestamp,
    )
    print(f"Results saved to {output_json_file}")

    print_statistics(elapsed_times)

    # Close the database instance
    db.frames.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Measure Precovery database creation time."
    )
    parser.add_argument(
        "database_directory", help="Directory path for the Precovery database."
    )
    parser.add_argument("csv_file_path", help="Path to the input CSV file.")
    parser.add_argument("dataset_id", help="Dataset ID for the input data.")
    parser.add_argument(
        "-p", "--healpixel_order", type=int, default=12, help="Healpixel order to use."
    )
    parser.add_argument(
        "-n",
        "--num_iterations",
        type=int,
        default=20,
        help="Number of times to create the database (default: 20).",
    )
    parser.add_argument(
        "-o",
        "--output_json_file",
        help="Output JSON file (default: results_TIMESTAMP.json).",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Display verbose output."
    )
    args = parser.parse_args()

    if args.verbose:
        print(f"Running with the following arguments: {args}")

    main(args)
