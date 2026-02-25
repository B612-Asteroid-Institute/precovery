import cProfile
import pstats
from pstats import SortKey

import pytest
import pyarrow.compute as pc
import pyarrow.parquet as pq
from pathlib import Path

from precovery.main import precover


pytestmark = [pytest.mark.profile, pytest.mark.integration, pytest.mark.slow]


@pytest.mark.profile
def test_precovery_profile(sample_orbits, detections_subset_dir, tmp_path):
    """
    Detailed profiling of precovery performance.
    
    This test runs a precovery search with profiling enabled to help identify
    performance bottlenecks and optimization opportunities.
    """
    profiler = cProfile.Profile(subcalls=True, builtins=True)
    profiler.enable()
    profiler.bias = 0
    
    subset_dir = detections_subset_dir
    det = pq.read_table(str(Path(subset_dir) / "detections.parquet"), columns=["exposure_mjd_mid_utc"])
    if det.num_rows == 0:
        raise RuntimeError("empty test detections parquet")
    m = pc.min_max(det["exposure_mjd_mid_utc"])
    start_mjd = float(m["min"].as_py())
    end_mjd = start_mjd + 1.0

    accepted, _accepted_counts = precover(
        orbits=sample_orbits[0],
        database_directory=str(subset_dir),
        tolerance=1 / 3600,
        start_mjd=start_mjd,
        end_mjd=end_mjd,
        max_processes=1,
    )
    
    profiler.disable()
    
    # Save results to the temporary test directory
    stats_file = tmp_path / "precovery_profile.prof"
    profiler.dump_stats(stats_file)
    
    # Print summary to console during test run
    stats = pstats.Stats(profiler).sort_stats(SortKey.CUMULATIVE)
    stats.print_stats(50)
    
    # Optional: Print location of profile file for later analysis
    print(f"\nProfile data saved to: {stats_file}")
    print("To visualize: snakeviz", str(stats_file))
    
    # Make sure we got some results (or at least a deterministic empty output).
    assert accepted is not None