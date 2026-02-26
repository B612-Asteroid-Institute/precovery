from __future__ import annotations

import numpy as np
import pyarrow as pa

from bench.data.neomod_synthetic_covariance import (
    _fetch_real_neo_candidate_table_from_sbdb,
    _stratified_select_neo_ids,
    build_synthetic_covariance_model,
    generate_covariance_matrices_for_batch,
)
from bench.data.prepare_neomod_benchmark_bundle import _normalize_detection_batch


def _sample_sbdb_table() -> pa.Table:
    return pa.table(
        {
            "pdes": pa.array(["A", "B", "C", "D"], type=pa.large_string()),
            "full_name": pa.array(["A", "B", "C", "D"], type=pa.large_string()),
            "neo": pa.array(["Y", "Y", "Y", "Y"], type=pa.large_string()),
            "condition_code": pa.array(["0", "3", "5", "8"], type=pa.large_string()),
            "data_arc": pa.array(["5000", "900", "120", "10"], type=pa.large_string()),
            "rms": pa.array(["0.2", "0.4", "0.8", "1.4"], type=pa.large_string()),
            "sigma_a": pa.array(["1e-11", "2e-9", "5e-7", "3e-4"], type=pa.large_string()),
            "sigma_e": pa.array(["1e-11", "2e-8", "5e-6", "3e-3"], type=pa.large_string()),
            "sigma_i": pa.array(["1e-8", "4e-7", "2e-4", "8e-2"], type=pa.large_string()),
            "sigma_om": pa.array(["1e-8", "4e-7", "2e-4", "8e-2"], type=pa.large_string()),
            "sigma_w": pa.array(["1e-8", "4e-7", "2e-4", "8e-2"], type=pa.large_string()),
            "sigma_ma": pa.array(["1e-8", "4e-7", "2e-4", "8e-2"], type=pa.large_string()),
            "n_obs_used": pa.array(["5000", "900", "100", "12"], type=pa.large_string()),
            "source": pa.array(["ORB", "ORB", "ORB", "ORB"], type=pa.large_string()),
        }
    )


def _sample_sbdb_cov_scale_table() -> pa.Table:
    return pa.table(
        {
            "query_id": pa.array(["A", "B", "C", "D"], type=pa.large_string()),
            "sbdb_object_id": pa.array(["A", "B", "C", "D"], type=pa.large_string()),
            "pos_sigma_rms_au": pa.array([1e-8, 5e-8, 2e-6, 3e-5], type=pa.float64()),
            "vel_sigma_rms_au_per_d": pa.array([1e-10, 5e-10, 2e-8, 3e-7], type=pa.float64()),
        }
    )


def test_build_synthetic_covariance_model_has_valid_weights() -> None:
    model = build_synthetic_covariance_model(sbdb_table=_sample_sbdb_table(), seed=7)
    w = np.asarray([s.weight for s in model.strata], dtype=np.float64)
    assert len(model.strata) == 4
    assert np.all(w > 0.0)
    assert np.isclose(np.sum(w), 1.0)


def test_build_synthetic_covariance_model_from_cov_scales_has_valid_weights() -> None:
    model = build_synthetic_covariance_model(sbdb_table=_sample_sbdb_cov_scale_table(), seed=17)
    w = np.asarray([s.weight for s in model.strata], dtype=np.float64)
    assert len(model.strata) == 4
    assert np.all(w > 0.0)
    assert np.isclose(np.sum(w), 1.0)


def test_generate_covariance_matrices_are_finite_psd() -> None:
    model = build_synthetic_covariance_model(sbdb_table=_sample_sbdb_table(), seed=11)
    rng = np.random.default_rng(11)
    cov = generate_covariance_matrices_for_batch(n_rows=256, model=model, rng=rng)

    assert cov.shape == (256, 6, 6)
    assert np.isfinite(cov).all()

    # Symmetry + PSD (within tiny numerical tolerance).
    assert np.allclose(cov, np.transpose(cov, (0, 2, 1)))
    evals = np.linalg.eigvalsh(cov)
    assert np.min(evals) >= -1e-20


def test_fetch_real_neo_candidate_table_from_sbdb_pages_dedup(monkeypatch) -> None:
    class _Resp:
        def __init__(self, payload: dict[str, object]) -> None:
            self._payload = payload

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return self._payload

    calls: list[dict[str, str]] = []
    payloads = [
        {"data": [["100"], ["101"]]},
        {"data": [["101"], ["102"]]},
        {"data": []},
    ]

    def _fake_get(url: str, *, params: dict[str, str], timeout: float) -> _Resp:
        assert "sbdb_query.api" in url
        assert float(timeout) == 10.0
        calls.append(dict(params))
        idx = len(calls) - 1
        return _Resp(payloads[idx] if idx < len(payloads) else {"data": []})

    import bench.data.neomod_synthetic_covariance as mod

    monkeypatch.setattr(mod.requests, "get", _fake_get)
    out = _fetch_real_neo_candidate_table_from_sbdb(max_rows=4, page_size=2, timeout_sec=10.0)

    assert out["pdes"].to_pylist() == ["100", "101", "102"]
    assert calls[0]["limit-from"] == "0"
    assert calls[1]["limit-from"] == "2"
    assert calls[0]["sb-group"] == "neo"
    assert calls[0]["sb-kind"] == "a"


def test_stratified_select_neo_ids_is_deterministic_and_varied() -> None:
    pdes = [f"X{i:03d}" for i in range(80)]
    cc = [0, 1, 2, 3, 4, 5, 6, 9] * 10
    arc = [5.0, 20.0, 120.0, 300.0, 900.0, 2000.0, 7000.0, 15000.0] * 10
    rms = [0.1, 0.15, 0.25, 0.35, 0.7, 1.0, 1.8, 3.0] * 10
    nobs = [20, 40, 60, 120, 200, 400, 800, 2000] * 10
    cands = pa.table(
        {
            "pdes": pa.array(pdes, type=pa.large_string()),
            "condition_code": pa.array(cc, type=pa.int64()),
            "data_arc_days": pa.array(arc, type=pa.float64()),
            "rms": pa.array(rms, type=pa.float64()),
            "n_obs_used": pa.array(nobs, type=pa.int64()),
        }
    )

    s1 = _stratified_select_neo_ids(candidates=cands, max_rows=24, seed=123)
    s2 = _stratified_select_neo_ids(candidates=cands, max_rows=24, seed=123)
    s3 = _stratified_select_neo_ids(candidates=cands, max_rows=24, seed=456)

    ids1 = s1["pdes"].to_pylist()
    ids2 = s2["pdes"].to_pylist()
    ids3 = s3["pdes"].to_pylist()
    assert ids1 == ids2
    assert ids1 != ids3

    arc_sel = np.asarray(s1["data_arc_days"].to_numpy(zero_copy_only=False), dtype=np.float64)
    dec_sel = np.asarray(s1["uncertainty_decile"].to_numpy(zero_copy_only=False), dtype=np.int64)

    assert np.any(arc_sel < 30.0)
    assert np.any(arc_sel >= 3650.0)
    assert np.any(dec_sel == 0)
    assert np.any(dec_sel == 9)


def test_normalize_detection_batch_matches_backend_contract_columns() -> None:
    time = pa.StructArray.from_arrays(
        [pa.array([61000, 61000], type=pa.int64()), pa.array([1, 2], type=pa.int64())],
        names=["days", "nanos"],
    )
    t = pa.table(
        {
            "id": pa.array(["o1", "o2"], type=pa.large_string()),
            "time": time,
            "ra": pa.array([10.0, 20.0], type=pa.float64()),
            "dec": pa.array([-10.0, 5.0], type=pa.float64()),
            "ra_sigma": pa.array([1e-4, 2e-4], type=pa.float64()),
            "dec_sigma": pa.array([1e-4, 2e-4], type=pa.float64()),
            "mag": pa.array([22.1, 23.2], type=pa.float64()),
            "mag_sigma": pa.array([0.1, 0.2], type=pa.float64()),
            "filter": pa.array(["r", "r"], type=pa.large_string()),
            "observatory_code": pa.array(["X05", "X05"], type=pa.large_string()),
        }
    )

    out = _normalize_detection_batch(
        batch=t,
        start_mjd=60999.0,
        end_mjd=61001.0,
        obscode="X05",
        healpix_nside=32,
    )
    assert out.num_rows == 2

    required = {
        "obscode",
        "exposure_mjd_mid_utc",
        "exposure_mjd_mid_key_us",
        "filter",
        "healpixel",
        "observation_id",
        "obstime_mjd_utc",
        "ra_deg",
        "dec_deg",
        "ra_sigma_deg",
        "dec_sigma_deg",
        "mag",
        "mag_sigma",
    }
    assert required.issubset(set(out.column_names))
