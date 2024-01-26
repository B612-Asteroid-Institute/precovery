import os

import numpy as np
import pandas as pd

from precovery.orbit import EpochTimescale, Orbit, OrbitElementType
from precovery.precovery_db import (
    FrameCandidate,
    FrameCandidatesQv,
    PrecoveryCandidate,
    PrecoveryCandidatesQv,
)

SAMPLE_ORBITS_FILE = os.path.join(
    os.path.dirname(__file__), "data", "sample_orbits.csv"
)


def test_orbit():
    orbits_df = pd.read_csv(SAMPLE_ORBITS_FILE)
    orbit_name_mapping = {}
    orbits_keplerian = []
    adam_orbits = []
    for i in range(len(orbits_df)):
        orbit_name_mapping[i] = orbits_df["orbit_name"].values[i]
        orbit = Orbit.keplerian(
            i,
            orbits_df["a"].values[i],
            orbits_df["e"].values[i],
            orbits_df["i"].values[i],
            orbits_df["om"].values[i],
            orbits_df["w"].values[i],
            orbits_df["ma"].values[i],
            orbits_df["mjd_tt"].values[i],
            EpochTimescale.TT,
            orbits_df["H"].values[i],
            orbits_df["G"].values[i],
        )
        orbits_keplerian.append(orbit)
        adam_orbits.append(orbit.to_adam_core())

    for orbit, adam_orbit, i in zip(
        orbits_keplerian, adam_orbits, range(len(orbits_keplerian))
    ):
        rt_orbit = Orbit.from_adam_core(
            i,
            adam_orbit,
            timescale=EpochTimescale.TT,
            orbit_type=OrbitElementType.KEPLERIAN,
            absolute_magnitude=orbit._state_vector[0][-2],
            photometric_slope_parameter=orbit._state_vector[0][-1],
        )
        assert np.allclose(
            orbit._state_vector[0], rt_orbit._state_vector[0], rtol=0, atol=1e-12
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

    cand = PrecoveryCandidatesQv.from_dataclass(
        [cand1, cand2, cand3], source_orbit_id="1"
    )
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

    f_cand = FrameCandidatesQv.from_dataclass([cand4, cand5], source_orbit_id="1")
    assert len(f_cand) == 2
    assert len(f_cand.exposures().id) == 1
    assert f_cand.predicted_ephemeris()

    roundtrip = f_cand.to_dataclass()
    assert len(roundtrip) == 2
    assert roundtrip[0] == cand4
    assert roundtrip[1] == cand5
