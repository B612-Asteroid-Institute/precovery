from adam_assist import ASSISTPropagator

from experiments.covariance_precovery.harness.runner import RunConfig, run_matrix
from precovery.sourcecatalog import bundle_into_frames

from .testutils import make_sourceobs, make_sourceobs_of_orbit


def test_runner_matrix_smoke(precovery_db, sample_orbits, tmp_path) -> None:
    # Build a tiny DB so the runner stays fast.
    orbit = sample_orbits[0]
    timestamps = [50000.0, 50001.0, 50002.0]
    object_observations = [make_sourceobs_of_orbit(orbit, "I41", mjd) for mjd in timestamps]
    extra_observations = [make_sourceobs(obscode="I41", mjd=mjd, exposure_duration=30) for mjd in timestamps]
    frames = list(bundle_into_frames(object_observations + extra_observations))

    ds_id = "test_dataset_1"
    precovery_db.frames.add_dataset(ds_id)
    precovery_db.frames.add_frames(ds_id, frames)

    metrics = run_matrix(
        db=precovery_db,
        orbit=orbit,
        configs=[
            RunConfig(
                run_id="circle",
                match_method="circle",
                window_size_days=7,
                start_mjd=49999.0,
                end_mjd=50003.0,
                datasets={ds_id},
            ),
            RunConfig(
                run_id="cov",
                match_method="covariance",
                window_size_days=7,
                n_sigma=3.0,
                start_mjd=49999.0,
                end_mjd=50003.0,
                datasets={ds_id},
            ),
        ],
        propagator_class=ASSISTPropagator,
        out_parquet=tmp_path / "metrics.parquet",
    )

    assert len(metrics) == 2
    assert (tmp_path / "metrics.parquet").exists()

