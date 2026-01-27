from experiments.covariance_precovery.methods.stop_policies import (
    AdaptiveSigmaPolicy,
    BatchStats,
    EfficiencyStopPolicy,
    GuardrailStopPolicy,
    HysteresisThresholdStopPolicy,
)


def test_guardrail_triggers() -> None:
    pol = GuardrailStopPolicy(max_pixels=10)
    d = pol.update(
        BatchStats(
            batch_id=0,
            n_times=1,
            n_pixels=11,
            n_frames=0,
            n_detections_retrieved=0,
            n_after_footprint=0,
            n_after_chi2=0,
        )
    )
    assert d.stop is True


def test_hysteresis_requires_consecutive() -> None:
    pol = HysteresisThresholdStopPolicy(field="n_pixels", threshold=10, consecutive=2)
    s1 = BatchStats(0, 1, 11, 0, 0, 0, 0)
    s2 = BatchStats(1, 1, 11, 0, 0, 0, 0)
    assert pol.update(s1).stop is False
    assert pol.update(s2).stop is True


def test_efficiency_stop_requires_truth() -> None:
    pol = EfficiencyStopPolicy(min_efficiency=0.1, consecutive=2)
    s = BatchStats(0, 1, 0, 0, 100, 0, 0, n_truth_matched=None, n_truth_total=None)
    assert pol.update(s).stop is False


def test_adaptive_sigma_increases_when_recall_low() -> None:
    pol = AdaptiveSigmaPolicy(recall_floor=0.9, step=0.5, min_sigma=2.0, max_sigma=5.0)
    s = BatchStats(0, 1, 0, 0, 10, 0, 0, n_truth_matched=5, n_truth_total=10)
    d = pol.update(s)
    assert d.suggested_n_sigma is not None
    assert d.suggested_n_sigma > 3.0

