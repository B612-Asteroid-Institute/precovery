from __future__ import annotations

import numpy as np

from precovery.search.backends.protocols import GateParams
import precovery.search.gate_counts as gate_counts
from precovery.search.gate_counts import (
    _inflate_candidate_sigmas_for_gate,
    gate_keep_and_reject_masks_python,
)
from precovery.search.pipeline_types import CandidateDetections, PredictedTargets


def _one_candidate(
    *,
    obscode: str = "I41",
    ra_sigma_deg: float = 1.0e-4,
    dec_sigma_deg: float = 1.0e-4,
    mag: float = 20.0,
) -> CandidateDetections:
    return CandidateDetections.from_kwargs(
        orbit_id=["o1"],
        target_idx=[0],
        obscode=[obscode],
        exposure_mjd_mid_key_us=[1],
        healpixel=[1],
        filter=["r"],
        observation_id=["obs-1"],
        obstime_mjd_utc=[60000.0],
        ra_deg=[10.0],
        dec_deg=[5.0],
        ra_sigma_deg=[float(ra_sigma_deg)],
        dec_sigma_deg=[float(dec_sigma_deg)],
        mag=[float(mag)],
        mag_sigma=[0.1],
    )


def _one_pred(*, pred_mag: float | None = 20.0) -> PredictedTargets:
    return PredictedTargets.from_kwargs(
        orbit_id=["o1"],
        target_idx=[0],
        obscode=["I41"],
        exposure_mjd_mid_utc=[60000.0],
        exposure_mjd_mid_key_us=[1],
        canonical_filter_id=["r"],
        pred_lon_deg=[10.0],
        pred_lat_deg=[5.0],
        cov_ll_00=[1.0e-8],
        cov_ll_01=[0.0],
        cov_ll_11=[1.0e-8],
        pred_mag=[pred_mag],
    )


def test_gate_uses_preinflated_sigmas_without_second_floor_fill(monkeypatch) -> None:
    cands = _one_candidate(obscode="T05", ra_sigma_deg=0.0, dec_sigma_deg=np.nan)
    preds = _one_pred()
    gate = GateParams(
        innovation_gate_n_sigma=3.0,
        invalid_sigma_fill_floor_arcsec_global=0.1,
        invalid_sigma_fill_floor_arcsec_by_obscode={"T05": 0.3},
    )

    seen: dict[str, object] = {}

    def _fake_innov(**kwargs):
        seen.update(kwargs)
        return np.array([True], dtype=bool)

    monkeypatch.setattr("precovery.search.gate_counts.innov_ellipse_keep_mask", _fake_innov)

    keep_innov, rejected_innov, rejected_mag, keep_final = gate_keep_and_reject_masks_python(
        candidates=cands, preds=preds, gate=gate
    )

    expected = 0.3 / 3600.0
    got_ra = np.asarray(seen["obs_lon_sigma_deg"], dtype=np.float64)
    got_dec = np.asarray(seen["obs_lat_sigma_deg"], dtype=np.float64)
    assert np.allclose(got_ra, [expected])
    assert np.allclose(got_dec, [expected])
    assert "invalid_sigma_fill_floor_arcsec" not in seen
    assert keep_innov.tolist() == [True]
    assert rejected_innov.tolist() == [False]
    assert rejected_mag.tolist() == [False]
    assert keep_final.tolist() == [True]


def test_gate_skips_mag_residual_computation_when_disabled(monkeypatch) -> None:
    cands = _one_candidate(mag=22.0)
    preds = _one_pred(pred_mag=20.0)
    gate = GateParams(
        innovation_gate_n_sigma=3.0,
        invalid_sigma_fill_floor_arcsec_global=0.1,
        max_mag_residual_fainter_mag=None,
        max_mag_residual_brighter_mag=None,
    )

    monkeypatch.setattr(
        "precovery.search.gate_counts.innov_ellipse_keep_mask",
        lambda **kwargs: np.array([True], dtype=bool),
    )
    calls = {"fill_null": 0}
    original_fill_null = gate_counts.pc.fill_null

    def _count_fill_null(*args, **kwargs):
        calls["fill_null"] += 1
        return original_fill_null(*args, **kwargs)

    monkeypatch.setattr("precovery.search.gate_counts.pc.fill_null", _count_fill_null)

    keep_innov, rejected_innov, rejected_mag, keep_final = gate_keep_and_reject_masks_python(
        candidates=cands, preds=preds, gate=gate
    )
    assert calls["fill_null"] == 0
    assert keep_innov.tolist() == [True]
    assert rejected_innov.tolist() == [False]
    assert rejected_mag.tolist() == [False]
    assert keep_final.tolist() == [True]


def test_gate_computes_mag_residual_when_enabled(monkeypatch) -> None:
    cands = _one_candidate(mag=22.0)
    preds = _one_pred(pred_mag=20.0)
    gate = GateParams(
        innovation_gate_n_sigma=3.0,
        invalid_sigma_fill_floor_arcsec_global=0.1,
        max_mag_residual_fainter_mag=1.0,
    )

    monkeypatch.setattr(
        "precovery.search.gate_counts.innov_ellipse_keep_mask",
        lambda **kwargs: np.array([True], dtype=bool),
    )
    calls = {"fill_null": 0}
    original_fill_null = gate_counts.pc.fill_null

    def _count_fill_null(*args, **kwargs):
        calls["fill_null"] += 1
        return original_fill_null(*args, **kwargs)

    monkeypatch.setattr("precovery.search.gate_counts.pc.fill_null", _count_fill_null)

    keep_innov, rejected_innov, rejected_mag, keep_final = gate_keep_and_reject_masks_python(
        candidates=cands, preds=preds, gate=gate
    )
    assert calls["fill_null"] >= 2
    assert keep_innov.tolist() == [True]
    assert rejected_innov.tolist() == [False]
    assert rejected_mag.tolist() == [True]
    assert keep_final.tolist() == [False]


def test_systematic_inflation_applies_only_below_trigger() -> None:
    cands = CandidateDetections.from_kwargs(
        orbit_id=["o1", "o1", "o1"],
        target_idx=[0, 0, 0],
        obscode=["T05", "T05", "I41"],
        exposure_mjd_mid_key_us=[1, 1, 1],
        healpixel=[1, 1, 1],
        filter=["r", "r", "r"],
        observation_id=["a", "b", "c"],
        obstime_mjd_utc=[60000.0, 60000.0, 60000.0],
        ra_deg=[10.0, 10.0, 10.0],
        dec_deg=[5.0, 5.0, 5.0],
        ra_sigma_deg=[5.0e-5, 2.0e-4, 5.0e-5],
        dec_sigma_deg=[5.0e-5, 2.0e-4, 5.0e-5],
        mag=[20.0, 20.0, 20.0],
        mag_sigma=[0.1, 0.1, 0.1],
    )
    gate = GateParams(
        innovation_gate_n_sigma=3.0,
        invalid_sigma_fill_floor_arcsec_global=0.1,
        sigma_systematic_arcsec_by_obscode={"T05": 0.3},
        apply_systematic_if_reported_rms_lt_arcsec_by_obscode={"T05": 0.4},
    )
    ra_eff, dec_eff = _inflate_candidate_sigmas_for_gate(candidates=cands, gate=gate)

    sys_deg = 0.3 / 3600.0
    expected_inflated = np.sqrt((5.0e-5**2) + (sys_deg**2))
    assert np.isclose(ra_eff[0], expected_inflated)
    assert np.isclose(dec_eff[0], expected_inflated)
    assert np.isclose(ra_eff[1], 2.0e-4)
    assert np.isclose(dec_eff[1], 2.0e-4)
    assert np.isclose(ra_eff[2], 5.0e-5)
    assert np.isclose(dec_eff[2], 5.0e-5)
