from precovery.precovery_db import (
    FrameCandidate,
    PrecoveryCandidate,
    PrecoveryDatabase,
    sift_candidates,
)
from precovery.sourcecatalog import bundle_into_frames

from .testutils import make_sourceobs


def test_find_observations_in_regions(tmp_path):
    db = PrecoveryDatabase.create(str(tmp_path), nside=32)

    db.frames.add_dataset("test_dataset")
    # Make 3 observations. Two are within 2arcsec of each other, the
    # other is far away.
    observations = [
        make_sourceobs(
            exposure_id=b"exp0",
            id=b"obs1",
            ra=1,
            dec=2,
            obscode="testobs",
        ),
        make_sourceobs(
            exposure_id=b"exp0",
            id=b"obs2",
            ra=1.0 + 2 / 3600.0,
            dec=2,
            obscode="testobs",
        ),
        make_sourceobs(
            exposure_id=b"exp0",
            id=b"obs3",
            ra=40,
            dec=41,
            obscode="testobs",
        ),
    ]
    frames = list(bundle_into_frames(observations))
    assert len(frames) == 2
    dataset_id = "test_dataset"
    db.frames.add_dataset(dataset_id)
    db.frames.add_frames(dataset_id, frames)

    # Should get two results within 10 arcsec
    results = list(db.find_observations_in_radius(1, 2, 10 / 3600.0, "testobs"))
    assert len(results) == 2

    # Should get one result within 1 arcsec
    results = db.find_observations_in_radius(1, 2, 1 / 3600.0, "testobs")
    assert len(list(results)) == 1


def test_sift_candidates():
    cand1 = PrecoveryCandidate(
        mjd=1,
        ra_deg=1,
        dec_deg=1,
        ra_sigma_arcsec=1,
        dec_sigma_arcsec=1,
        mag=1,
        mag_sigma=1,
        filter="r",
        obscode="I41",
        exposure_id="exp1",
        exposure_mjd_start=1,
        exposure_mjd_mid=1,
        exposure_duration=1,
        observation_id="obs1",
        healpix_id=1,
        pred_ra_deg=1,
        pred_dec_deg=1,
        pred_vra_degpday=1,
        pred_vdec_degpday=1,
        delta_ra_arcsec=1,
        delta_dec_arcsec=1,
        distance_arcsec=1,
        dataset_id="dataset1",
    )
    # Same time as cand1 but different obs
    cand2 = PrecoveryCandidate(
        mjd=1,
        ra_deg=2,
        dec_deg=2,
        ra_sigma_arcsec=2,
        dec_sigma_arcsec=2,
        mag=2,
        mag_sigma=2,
        filter="r",
        obscode="I41",
        exposure_id=2,
        exposure_mjd_start=2,
        exposure_mjd_mid=2,
        exposure_duration=2,
        observation_id="obs2",
        healpix_id=2,
        pred_ra_deg=2,
        pred_dec_deg=2,
        pred_vra_degpday=2,
        pred_vdec_degpday=2,
        delta_ra_arcsec=2,
        delta_dec_arcsec=2,
        distance_arcsec=2,
        dataset_id="dataset2",
    )

    cand3 = PrecoveryCandidate(
        mjd=2,
        ra_deg=2,
        dec_deg=2,
        ra_sigma_arcsec=2,
        dec_sigma_arcsec=2,
        mag=2,
        mag_sigma=2,
        filter="r",
        obscode="I41",
        exposure_id=2,
        exposure_mjd_start=2,
        exposure_mjd_mid=2,
        exposure_duration=2,
        observation_id="obs4",
        healpix_id=2,
        pred_ra_deg=2,
        pred_dec_deg=2,
        pred_vra_degpday=2,
        pred_vdec_degpday=2,
        delta_ra_arcsec=2,
        delta_dec_arcsec=2,
        distance_arcsec=2,
        dataset_id="dataset2",
    )

    cand4 = FrameCandidate(
        filter="r",
        obscode="I41",
        exposure_id=3,
        exposure_mjd_start=3,
        exposure_mjd_mid=3,
        exposure_duration=3,
        healpix_id=3,
        pred_ra_deg=3,
        pred_dec_deg=3,
        pred_vra_degpday=3,
        pred_vdec_degpday=3,
        dataset_id="dataset3",
    )

    cand5 = FrameCandidate(
        filter="r",
        obscode="I41",
        exposure_id=3,
        exposure_mjd_start=4,
        exposure_mjd_mid=4,
        exposure_duration=4,
        healpix_id=3,
        pred_ra_deg=3,
        pred_dec_deg=3,
        pred_vra_degpday=3,
        pred_vdec_degpday=3,
        dataset_id="dataset3",
    )

    cands = [cand5, cand3, cand2, cand1, cand4]
    cands_sorted = sift_candidates(cands)
    assert cands_sorted == ([cand1, cand2, cand3], [cand4, cand5])
