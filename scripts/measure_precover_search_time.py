import argparse
import json
import os
import time
from datetime import datetime
from typing import List

import pandas as pd

from precovery.orbit import EpochTimescale, Orbit
from precovery.precovery_db import FrameCandidate, PrecoveryCandidate, PrecoveryDatabase


def sample_orbits():
    sample_orbits_file = os.path.join(
        os.path.dirname(__file__), "..", "tests", "data", "sample_orbits.csv"
    )
    df = pd.read_csv(sample_orbits_file)
    orbits = []
    for i in range(len(df)):
        orbit = Orbit.keplerian(
            i,
            df["a"].values[i],
            df["e"].values[i],
            df["i"].values[i],
            df["om"].values[i],
            df["w"].values[i],
            df["ma"].values[i],
            df["mjd_tt"].values[i],
            EpochTimescale.TT,
            df["H"].values[i],
            df["G"].values[i],
        )
        orbits.append(orbit)
    return orbits


def measure_precover_performance(
    database_directory: str, orbits: List[Orbit]
) -> tuple[PrecoveryDatabase, List[dict]]:
    db = PrecoveryDatabase.from_dir(database_directory)
    results = []

    for orbit in orbits:
        start_time = time.time()
        precover_results = db.precover(orbit)
        elapsed_time = time.time() - start_time

        precovery_count = sum(
            isinstance(res, PrecoveryCandidate) for res in precover_results
        )
        frame_count = sum(isinstance(res, FrameCandidate) for res in precover_results)

        results.append(
            {
                "elapsed_time": elapsed_time,
                "precovery_count": precovery_count,
                "frame_count": frame_count,
            }
        )

    return db, results


def calculate_statistics(results: List[dict]):
    import numpy as np

    elapsed_times = [res["elapsed_time"] for res in results]

    mean_time = np.mean(elapsed_times)
    median_time = np.median(elapsed_times)
    p10 = np.percentile(elapsed_times, 10)
    p90 = np.percentile(elapsed_times, 90)
    stdev = np.std(elapsed_times)

    return {
        "mean_time": mean_time,
        "median_time": median_time,
        "p10": p10,
        "p90": p90,
        "stdev": stdev,
    }


def save_results_to_json(
    db: PrecoveryDatabase,
    output_json_file: str,
    results: List[dict],
    database_directory: str,
    run_timestamp: str,
):
    statistics = calculate_statistics(results)

    report = {
        "results": results,
        "statistics": statistics,
        "database_directory": database_directory,
        "run_timestamp": run_timestamp,
        "database": {
            "size_bytes": db.frames.idx.n_bytes(),
            "n_frames": db.frames.idx.n_frames(),
            "dataset_ids": list(db.frames.idx.get_dataset_ids()),
        },
    }

    with open(output_json_file, "w") as f:
        json.dump(report, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Measure the performance of the precover method on a PrecoveryDatabase."
    )
    parser.add_argument(
        "database_directory", help="Path to the PrecoveryDatabase directory."
    )
    parser.add_argument(
        "-o",
        "--output_json_file",
        default="precover_bench.json",
        help="Path to the output JSON file.",
    )
    args = parser.parse_args()

    orbits_to_test = sample_orbits()[:10]  # Test with 10 sample orbits

    db, results = measure_precover_performance(args.database_directory, orbits_to_test)

    print("\nResults for precover method:")
    for i, result in enumerate(results, start=1):
        print(
            f"Orbit {i}: {result['elapsed_time']:.2f} seconds, "
            f"{result['precovery_count']} PrecoveryCandidates, {result['frame_count']} FrameCandidates"
        )

    run_timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    save_results_to_json(
        db, args.output_json_file, results, args.database_directory, run_timestamp
    )
    print(f"\nResults saved to JSON file: {args.output_json_file}")
