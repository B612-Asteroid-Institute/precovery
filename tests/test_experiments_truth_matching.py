import numpy as np
import quivr as qv
from adam_core.time import Timestamp

from experiments.covariance_precovery.truth.mpc_truth import (
    TruthObservation,
    match_candidates_to_truth,
)


class FakeCandidates(qv.Table):
    obscode = qv.LargeStringColumn()
    time = Timestamp.as_column()
    ra_deg = qv.Float64Column()
    dec_deg = qv.Float64Column()
    observation_id = qv.LargeStringColumn()


def test_match_candidates_to_truth_by_time_and_sky() -> None:
    # Truth: one obs at I41 at MJD 60000.0
    truth = TruthObservation.from_kwargs(
        mpc_id=[1],
        obscode=["I41"],
        obsid=["truth_obs_1"],
        time_mjd_utc=[60000.0],
        ra_deg=[10.0],
        dec_deg=[20.0],
    )

    # Candidate: matches within 1s and 1"
    cand = FakeCandidates.from_kwargs(
        obscode=["I41", "I41"],
        time=Timestamp.from_mjd([60000.0 + (0.5 / 86400.0), 60000.1], scale="utc"),
        ra_deg=[10.0 + (0.5 / 3600.0), 11.0],
        dec_deg=[20.0, 20.0],
        observation_id=["cand_match", "cand_far"],
    )

    matches, summary = match_candidates_to_truth(
        truth=truth, candidates=cand, time_tol_sec=2.0, dist_tol_arcsec=2.0
    )

    assert summary.n_truth == 1
    assert summary.n_truth_matched == 1
    assert summary.n_candidates == 2
    assert summary.n_candidates_matched == 1

    assert len(matches) == 1
    assert matches.truth_obsid[0].as_py() == "truth_obs_1"
    assert matches.candidate_observation_id[0].as_py() == "cand_match"
    assert matches.distance_arcsec[0].as_py() <= 2.0

