from precovery.precovery_db import (
    FrameCandidate,
    FrameCandidatesQv,
    PrecoveryCandidate,
    PrecoveryCandidatesQv,
)


def test_precovery_candidate_table():
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
        exposure_id="2",
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
        exposure_id="2",
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

    cand = PrecoveryCandidatesQv.from_dataclass([cand1, cand2, cand3])
    assert len(cand) == 3
    assert len(cand.exposures().id) == 2
    assert cand.point_source_detections()
    assert cand.predicted_ephemeris()
    assert cand.point_source_detections().link_to_exposures(cand.exposures())

    roundtrip = cand.to_dataclass()
    assert len(roundtrip) == 3
    assert roundtrip[0] == cand1
    assert roundtrip[1] == cand2
    assert roundtrip[2] == cand3


def test_precovery_frame_candidate_table():
    cand4 = FrameCandidate(
        filter="r",
        obscode="I41",
        exposure_id="2",
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
        exposure_id="2",
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

    f_cand = FrameCandidatesQv.from_frame_candidates([cand4, cand5])
    assert len(f_cand) == 2
    assert len(f_cand.exposures().id) == 1
    assert f_cand.predicted_ephemeris()

    roundtrip = f_cand.to_frame_candidates()
    assert len(roundtrip) == 2
    assert roundtrip[0] == cand4
    assert roundtrip[1] == cand5
