import pytest
from adam_assist import ASSISTPropagator
from adam_core.dynamics.propagation import propagate_2body
from adam_core.time import Timestamp
import pyarrow.compute as pc
import pyarrow.parquet as pq
from pathlib import Path

from precovery.observation import ObservationsTable
from precovery.main import precover

from .testutils import make_sourceframe_with_observations


pytestmark = [pytest.mark.benchmark, pytest.mark.slow]


@pytest.mark.benchmark(group="framedb_binary_internals")
@pytest.mark.parametrize("nobs", [1, 10, 100, 1000])
def test_benchmark_iterate_frame_observations(benchmark, frame_db, nobs):
    # Build a frame with nobs observations in a single frame.
    frame_db.add_dataset(
        dataset_id="test_dataset",
    )
    src_frame = make_sourceframe_with_observations(nobs, healpixel=1, obscode="obs")
    frame_db.add_frames("test_dataset", [src_frame])

    # Get the frame from the database.
    hp_frames = list(frame_db.idx.frames_for_healpixel(1, "obs"))
    assert len(hp_frames) == 1
    frame = hp_frames[0]

    # Benchmark iterating over the observations in the frame.
    def benchmark_case():
        frame_db.get_observations(frame)

    benchmark(benchmark_case)


@pytest.mark.benchmark(group="framedb_binary_internals")
@pytest.mark.parametrize("nobs", [1, 10, 100, 1000])
def test_benchmark_store_observations(benchmark, frame_db, nobs):
    # Build a frame with nobs observations in a single frame.
    frame_db.add_dataset(
        dataset_id="test_dataset",
    )
    src_frame = make_sourceframe_with_observations(nobs, healpixel=1, obscode="obs")
    yearmonth = frame_db._compute_year_month_str(src_frame)
    observations: ObservationsTable = ObservationsTable.from_srcobs(
        src_frame.observations
    )

    def benchmark_case():
        frame_db.store_observations(observations, "test_dataset", yearmonth)

    benchmark(benchmark_case)


@pytest.mark.benchmark(group="orbits")
@pytest.mark.parametrize("propagate_distance", [1, 10, 100, 1000])
def test_benchmark_propagate_orbit_nbody(benchmark, sample_orbits, propagate_distance):

    propagator = ASSISTPropagator()
    orbit = sample_orbits[0]

    def benchmark_case():
        times = Timestamp.from_mjd(
            [orbit.coordinates.time.mjd()[0].as_py() + propagate_distance], scale="utc"
        )
        propagator.propagate_orbits(orbit, times)

    benchmark(benchmark_case)


@pytest.mark.benchmark(group="orbits")
@pytest.mark.parametrize("propagate_distance", [1, 10, 100, 1000])
def test_benchmark_propagate_orbit_2body(benchmark, sample_orbits, propagate_distance):

    orbit = sample_orbits[0]

    def benchmark_case():
        times = Timestamp.from_mjd(
            [orbit.coordinates.time.mjd()[0].as_py() + propagate_distance], scale="utc"
        )
        propagate_2body(orbit, times)

    benchmark(benchmark_case)


@pytest.mark.benchmark(group="precovery")
@pytest.mark.parametrize("max_processes", [1])
def test_benchmark_precovery_search(benchmark, detections_subset_dir, sample_orbits, max_processes):

    orbit = sample_orbits[0]
    # Keep the benchmark bounded. The refactored pipeline is designed for large-scale runs,
    # but the test suite should not spend minutes scanning multi-day ranges.
    det = pq.read_table(str(Path(detections_subset_dir) / "detections.parquet"), columns=["exposure_mjd_mid_utc"])
    m = pc.min_max(det["exposure_mjd_mid_utc"])
    start_mjd = float(m["min"].as_py())
    end_mjd = start_mjd + 1.0

    def benchmark_case():
        precover(
            orbits=orbit,
            database_directory=str(detections_subset_dir),
            max_processes=max_processes,
            start_mjd=start_mjd,
            end_mjd=end_mjd,
        )

    benchmark.pedantic(benchmark_case, iterations=1, rounds=1)
