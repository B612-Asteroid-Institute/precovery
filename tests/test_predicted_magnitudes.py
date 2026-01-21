import pyarrow.compute as pc
from adam_assist import ASSISTPropagator
from adam_core.orbits.orbits import PhysicalParameters

from precovery.healpix_geom import radec_to_healpixel
from precovery.precovery_db import ARCSEC, PrecoveryDatabase
from precovery.sourcecatalog import bundle_into_frames

from .testutils import make_sourceobs, make_sourceobs_of_orbit


def _with_hg(orbit, *, H_v: float, G: float):
    pp = PhysicalParameters.from_kwargs(H_v=[H_v], G=[G])
    return orbit.set_column("physical_parameters", pp)


def test_pred_mag_populated_for_hits(precovery_db, sample_orbits):
    orbit = _with_hg(sample_orbits[0], H_v=15.0, G=0.15)
    times = [50000.0, 50001.0, 50002.0]

    object_observations = [make_sourceobs_of_orbit(orbit, "I41", mjd) for mjd in times]
    extra_observations = [
        make_sourceobs(obscode="I41", mjd=mjd, exposure_duration=30) for mjd in times
    ]

    precovery_db.frames.add_dataset("ds")
    precovery_db.frames.add_frames(
        "ds", bundle_into_frames(object_observations + extra_observations)
    )

    matches, misses = precovery_db.precover(orbit, propagator_class=ASSISTPropagator)
    assert len(matches) == 3
    assert len(misses) == 0

    assert pc.all(pc.is_finite(matches.pred_mag)).as_py()
    assert pc.all(pc.is_finite(matches.mag_residual)).as_py()
    assert pc.all(pc.equal(matches.rejected, False)).as_py()


def test_pred_mag_populated_for_misses(precovery_db, sample_orbits):
    # Create frames in the same healpixel as the predicted position but outside a tight tolerance,
    # so we get FrameCandidates (misses) with predicted magnitudes.
    orbit = _with_hg(sample_orbits[0], H_v=15.0, G=0.15)
    times = [50000.0, 50001.0, 50002.0]

    obs = []
    for mjd in times:
        o = make_sourceobs_of_orbit(orbit, "I41", mjd)
        # Offset by 5 arcsec; keep tolerance at 1 arcsec.
        ra_off = o.ra + (5.0 / 3600.0)
        # Ensure we stay in the same healpixel at the DB nside (32) to force the frame to be checked.
        hp0 = int(radec_to_healpixel(o.ra, o.dec, nside=32))
        hp1 = int(radec_to_healpixel(ra_off, o.dec, nside=32))
        if hp0 != hp1:
            # If we crossed a boundary, use a smaller offset.
            ra_off = o.ra + (1.5 / 3600.0)
        obs.append(
            make_sourceobs(
                exposure_id=o.exposure_id,
                id=o.id,
                obscode="I41",
                ra=ra_off,
                dec=o.dec,
                mjd=mjd,
                exposure_duration=o.exposure_duration,
                filter=o.filter,
            )
        )

    precovery_db.frames.add_dataset("ds")
    precovery_db.frames.add_frames("ds", bundle_into_frames(obs))

    matches, misses = precovery_db.precover(
        orbit, tolerance=1.0 * ARCSEC, propagator_class=ASSISTPropagator
    )
    assert len(matches) == 0
    assert len(misses) > 0
    assert pc.all(pc.is_finite(misses.pred_mag)).as_py()
    assert pc.all(pc.equal(misses.rejected, False)).as_py()


def test_faint_frame_skip_avoids_observation_fetch(
    tmp_path, sample_orbits, monkeypatch
):
    # Build a tiny on-disk DB (so check_window re-opens it from_dir) with a single frame.
    db = PrecoveryDatabase.create(str(tmp_path), nside=32)
    orbit = _with_hg(sample_orbits[0], H_v=30.0, G=0.15)

    mjd = 50000.0
    o = make_sourceobs_of_orbit(orbit, "I41", mjd)
    db.frames.add_dataset("ds")
    db.frames.add_frames("ds", bundle_into_frames([o]))

    # Write limiting magnitudes parquet cache to the DB dir (config defaults to this filename).
    from precovery.filter_limiting_magnitudes import FilterLimitingMagnitudes

    limit = FilterLimitingMagnitudes.from_kwargs(
        obscode=["I41"],
        filter_id=["V"],
        limiting_mag=[10.0],
        mag_system=[None],
    )
    limit.to_parquet(tmp_path / "limiting_magnitudes.parquet")

    # Ensure any observation fetch would fail the test.
    def _boom(*args, **kwargs):
        raise AssertionError(
            "get_observations should not be called for faint-skipped frames"
        )

    monkeypatch.setattr("precovery.frame_db.FrameDB.get_observations", _boom)

    matches, misses = db.precover(
        orbit,
        propagator_class=ASSISTPropagator,
        start_mjd=mjd - 1,
        end_mjd=mjd + 1,
    )
    # The frame is skipped as "too faint", but we keep a rejected FrameCandidate.
    assert len(matches) == 0
    assert len(misses) == 1
    assert bool(misses.rejected[0].as_py()) is True
    assert misses.rejected_reason[0].as_py() == "limiting_magnitude"
