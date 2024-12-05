import cProfile
import pstats
from pstats import SortKey

import pytest
from adam_assist import ASSISTPropagator

from precovery.main import precover


@pytest.mark.profile
def test_precovery_profile(sample_orbits, precovery_db_with_data, tmp_path):
    """
    Detailed profiling of precovery performance.
    
    This test runs a precovery search with profiling enabled to help identify
    performance bottlenecks and optimization opportunities.
    """
    profiler = cProfile.Profile(subcalls=True, builtins=True)
    profiler.enable()
    profiler.bias = 0
    
    # Run precovery
    candidates, frame_candidates = precover(
        orbits=sample_orbits[0],
        database_directory=precovery_db_with_data.directory,
        tolerance=1/3600,
        propagator_class=ASSISTPropagator,
        max_processes=1
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
    
    # Make sure we got some results
    assert len(candidates) > 0 or len(frame_candidates) > 0 