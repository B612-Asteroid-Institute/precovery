from precovery.precovery_db import FrameCandidate, PrecoveryCandidate
from precovery.utils import candidates_to_dataframe, frames_to_dataframe


def test_candidates_to_dataframe():
    """Test conversion of a list of precovery candidates to a pandas dataframe."""
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
        mjd=3,
        ra_deg=3,
        dec_deg=3,
        ra_sigma_arcsec=3,
        dec_sigma_arcsec=3,
        mag=3,
        mag_sigma=3,
        filter="r",
        obscode="I41",
        exposure_id=3,
        exposure_mjd_start=3,
        exposure_mjd_mid=3,
        exposure_duration=3,
        observation_id="obs3",
        healpix_id=3,
        pred_ra_deg=3,
        pred_dec_deg=3,
        pred_vra_degpday=3,
        pred_vdec_degpday=3,
        delta_ra_arcsec=3,
        delta_dec_arcsec=3,
        distance_arcsec=3,
        dataset_id="dataset3",
    )

    as_df = candidates_to_dataframe([cand1, cand2, cand3])
    assert len(as_df) == 3
    assert as_df["mjd"].iloc[0] == 1
    assert as_df["mjd"].iloc[1] == 1
    assert as_df["mjd"].iloc[2] == 3
    assert as_df["ra_deg"].iloc[0] == 1
    assert as_df["ra_deg"].iloc[1] == 2
    assert as_df["ra_deg"].iloc[2] == 3
    assert as_df["dec_deg"].iloc[0] == 1
    assert as_df["dec_deg"].iloc[1] == 2
    assert as_df["dec_deg"].iloc[2] == 3
    assert as_df["ra_sigma_arcsec"].iloc[0] == 1
    assert as_df["ra_sigma_arcsec"].iloc[1] == 2
    assert as_df["ra_sigma_arcsec"].iloc[2] == 3
    assert as_df["dec_sigma_arcsec"].iloc[0] == 1
    assert as_df["dec_sigma_arcsec"].iloc[1] == 2
    assert as_df["dec_sigma_arcsec"].iloc[2] == 3
    assert as_df["mag"].iloc[0] == 1
    assert as_df["mag"].iloc[1] == 2
    assert as_df["mag"].iloc[2] == 3
    assert as_df["mag_sigma"].iloc[0] == 1
    assert as_df["mag_sigma"].iloc[1] == 2
    assert as_df["mag_sigma"].iloc[2] == 3
    assert as_df["filter"].iloc[0] == "r"
    assert as_df["filter"].iloc[1] == "r"
    assert as_df["filter"].iloc[2] == "r"
    assert as_df["obscode"].iloc[0] == "I41"
    assert as_df["obscode"].iloc[1] == "I41"
    assert as_df["obscode"].iloc[2] == "I41"
    assert as_df["exposure_id"].iloc[0] == "exp1"
    assert as_df["exposure_id"].iloc[1] == 2
    assert as_df["exposure_id"].iloc[2] == 3
    assert as_df["exposure_mjd_start"].iloc[0] == 1
    assert as_df["exposure_mjd_start"].iloc[1] == 2
    assert as_df["exposure_mjd_start"].iloc[2] == 3
    assert as_df["exposure_mjd_mid"].iloc[0] == 1
    assert as_df["exposure_mjd_mid"].iloc[1] == 2
    assert as_df["exposure_mjd_mid"].iloc[2] == 3
    assert as_df["exposure_duration"].iloc[0] == 1
    assert as_df["exposure_duration"].iloc[1] == 2
    assert as_df["exposure_duration"].iloc[2] == 3
    assert as_df["observation_id"].iloc[0] == "obs1"
    assert as_df["observation_id"].iloc[1] == "obs2"
    assert as_df["observation_id"].iloc[2] == "obs3"
    assert as_df["healpix_id"].iloc[0] == 1
    assert as_df["healpix_id"].iloc[1] == 2
    assert as_df["healpix_id"].iloc[2] == 3
    assert as_df["pred_ra_deg"].iloc[0] == 1
    assert as_df["pred_ra_deg"].iloc[1] == 2
    assert as_df["pred_ra_deg"].iloc[2] == 3
    assert as_df["pred_dec_deg"].iloc[0] == 1
    assert as_df["pred_dec_deg"].iloc[1] == 2
    assert as_df["pred_dec_deg"].iloc[2] == 3
    assert as_df["pred_vra_degpday"].iloc[0] == 1
    assert as_df["pred_vra_degpday"].iloc[1] == 2
    assert as_df["pred_vra_degpday"].iloc[2] == 3
    assert as_df["pred_vdec_degpday"].iloc[0] == 1
    assert as_df["pred_vdec_degpday"].iloc[1] == 2
    assert as_df["pred_vdec_degpday"].iloc[2] == 3
    assert as_df["delta_ra_arcsec"].iloc[0] == 1
    assert as_df["delta_ra_arcsec"].iloc[1] == 2
    assert as_df["delta_ra_arcsec"].iloc[2] == 3
    assert as_df["delta_dec_arcsec"].iloc[0] == 1
    assert as_df["delta_dec_arcsec"].iloc[1] == 2
    assert as_df["delta_dec_arcsec"].iloc[2] == 3
    assert as_df["distance_arcsec"].iloc[0] == 1
    assert as_df["distance_arcsec"].iloc[1] == 2
    assert as_df["distance_arcsec"].iloc[2] == 3
    assert as_df["dataset_id"].iloc[0] == "dataset1"
    assert as_df["dataset_id"].iloc[1] == "dataset2"
    assert as_df["dataset_id"].iloc[2] == "dataset3"


def test_frame_candidates_to_dataframe():
    """Test that we can convert a list of FrameCandidate objects to a pandas DataFrame."""
    cand1 = FrameCandidate(
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

    cand2 = FrameCandidate(
        filter="r",
        obscode="I41",
        exposure_id="str_exposure_id",
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

    as_df = frames_to_dataframe([cand1, cand2])

    assert as_df["filter"].iloc[0] == "r"
    assert as_df["obscode"].iloc[0] == "I41"
    assert as_df["exposure_id"].iloc[0] == 3
    assert as_df["exposure_mjd_start"].iloc[0] == 3
    assert as_df["exposure_mjd_mid"].iloc[0] == 3
    assert as_df["exposure_duration"].iloc[0] == 3
    assert as_df["healpix_id"].iloc[0] == 3
    assert as_df["pred_ra_deg"].iloc[0] == 3
    assert as_df["pred_dec_deg"].iloc[0] == 3
    assert as_df["pred_vra_degpday"].iloc[0] == 3
    assert as_df["pred_vdec_degpday"].iloc[0] == 3
    assert as_df["dataset_id"].iloc[0] == "dataset3"

    assert as_df["filter"].iloc[1] == "r"
    assert as_df["obscode"].iloc[1] == "I41"
    assert as_df["exposure_id"].iloc[1] == "str_exposure_id"
    assert as_df["exposure_mjd_start"].iloc[1] == 4
    assert as_df["exposure_mjd_mid"].iloc[1] == 4
    assert as_df["exposure_duration"].iloc[1] == 4
    assert as_df["healpix_id"].iloc[1] == 3
    assert as_df["pred_ra_deg"].iloc[1] == 3
    assert as_df["pred_dec_deg"].iloc[1] == 3
    assert as_df["pred_vra_degpday"].iloc[1] == 3
    assert as_df["pred_vdec_degpday"].iloc[1] == 3
    assert as_df["dataset_id"].iloc[1] == "dataset3"
