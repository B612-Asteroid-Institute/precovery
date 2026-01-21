import pyarrow.compute as pc
from adam_assist import ASSISTPropagator
from precovery.config import Config
from precovery.precovery_db import PrecoveryDatabase
from precovery.sourcecatalog import bundle_into_frames

from .test_predicted_magnitudes import _with_hg
from .testutils import make_sourceobs_of_orbit


def test_detection_mag_outlier_rejected(tmp_path, sample_orbits):
    # Build a tiny on-disk DB (so check_window re-opens it from_dir) with a single frame+obs.
    db = PrecoveryDatabase.create(str(tmp_path), nside=32)

    # Enable magnitude residual rejection via config.json (read by from_dir).
    cfg_path = tmp_path / "config.json"
    cfg = Config.from_json(str(cfg_path))
    # Asymmetric cut: allow much fainter but not much brighter.
    cfg.max_mag_residual_fainter_mag = 100.0
    cfg.max_mag_residual_brighter_mag = 0.1
    cfg.to_json(str(cfg_path))

    orbit = _with_hg(sample_orbits[0], H_v=15.0, G=0.15)
    mjd = 50000.0
    o = make_sourceobs_of_orbit(orbit, "I41", mjd)
    # Make the observation magnitude wildly inconsistent with the predicted magnitude.
    o.mag = 0.0

    db.frames.add_dataset("ds")
    db.frames.add_frames("ds", bundle_into_frames([o]))

    matches, misses = db.precover(
        orbit,
        propagator_class=ASSISTPropagator,
        start_mjd=mjd - 1,
        end_mjd=mjd + 1,
    )
    assert len(misses) == 0
    assert len(matches) == 1
    assert pc.is_finite(matches.pred_mag)[0].as_py()
    assert pc.is_finite(matches.mag_residual)[0].as_py()
    assert bool(matches.rejected[0].as_py()) is True
    assert matches.rejected_reason[0].as_py() == "mag_residual"


def test_detection_mag_outlier_rejected_fainter(tmp_path, sample_orbits):
    db = PrecoveryDatabase.create(str(tmp_path), nside=32)

    cfg_path = tmp_path / "config.json"
    cfg = Config.from_json(str(cfg_path))
    # Asymmetric cut: allow slightly brighter but not much fainter.
    cfg.max_mag_residual_fainter_mag = 0.1
    cfg.max_mag_residual_brighter_mag = 100.0
    cfg.to_json(str(cfg_path))

    orbit = _with_hg(sample_orbits[0], H_v=15.0, G=0.15)
    mjd = 50000.0
    o = make_sourceobs_of_orbit(orbit, "I41", mjd)
    # Force a very large positive residual (much fainter than predicted).
    o.mag = 100.0

    db.frames.add_dataset("ds")
    db.frames.add_frames("ds", bundle_into_frames([o]))

    matches, misses = db.precover(
        orbit,
        propagator_class=ASSISTPropagator,
        start_mjd=mjd - 1,
        end_mjd=mjd + 1,
    )
    assert len(misses) == 0
    assert len(matches) == 1
    assert bool(matches.rejected[0].as_py()) is True
    assert matches.rejected_reason[0].as_py() == "mag_residual"

