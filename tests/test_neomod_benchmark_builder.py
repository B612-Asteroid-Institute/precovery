from __future__ import annotations

import io
from pathlib import Path
import types
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from bench.data.neomod_synthetic_covariance import (
    _fetch_real_neo_candidate_table_from_sbdb,
    _stratified_select_neo_ids,
    build_synthetic_covariance_model,
    generate_covariance_matrices_for_batch,
)
from bench.data.prepare_neomod_benchmark_bundle import (
    InputUris,
    _materialize_input_uris_local,
    _orbit_photometry_bucket_index,
    _write_orbit_photometry_bucket_inputs,
    _normalize_detection_batch,
)
from bench.benchmarks.workload import month_bounds_mjd_utc


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
            # Input sigmas are in arcsec; normalization should emit degrees.
            "ra_sigma": pa.array([0.36, 0.72], type=pa.float64()),
            "dec_sigma": pa.array([0.36, 0.72], type=pa.float64()),
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
    np.testing.assert_allclose(
        np.asarray(out["ra_sigma_deg"].to_numpy(zero_copy_only=False), dtype=np.float64),
        np.asarray([1e-4, 2e-4], dtype=np.float64),
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(out["dec_sigma_deg"].to_numpy(zero_copy_only=False), dtype=np.float64),
        np.asarray([1e-4, 2e-4], dtype=np.float64),
        rtol=0.0,
        atol=1e-12,
    )


def test_normalize_detection_batch_sigma_floor_and_cap_mas() -> None:
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
            # arcsec => [5 mas, 50 mas] after conversion
            "ra_sigma": pa.array([0.005, 0.05], type=pa.float64()),
            "dec_sigma": pa.array([0.005, 0.05], type=pa.float64()),
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
        detection_sigma_floor_mas=10.0,
        detection_sigma_cap_mas=30.0,
    )
    expect_deg = np.asarray([10.0 / 3_600_000.0, 30.0 / 3_600_000.0], dtype=np.float64)
    np.testing.assert_allclose(
        np.asarray(out["ra_sigma_deg"].to_numpy(zero_copy_only=False), dtype=np.float64),
        expect_deg,
        rtol=0.0,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        np.asarray(out["dec_sigma_deg"].to_numpy(zero_copy_only=False), dtype=np.float64),
        expect_deg,
        rtol=0.0,
        atol=1e-14,
    )


def test_materialize_input_uris_local_prefers_existing_local_files(tmp_path) -> None:
    q = tmp_path / "quad.parquet"
    n = tmp_path / "noise.parquet"
    o = tmp_path / "orbits.parquet"
    q.write_bytes(b"q")
    n.write_bytes(b"n")
    o.write_bytes(b"o")

    out = _materialize_input_uris_local(
        inputs=InputUris(quad_truth=str(q), noise_100=str(n), neomod_orbits=str(o)),
        cache_dir=tmp_path / "cache",
    )
    assert out.quad_truth == str(q.resolve())
    assert out.noise_100 == str(n.resolve())
    assert out.neomod_orbits == str(o.resolve())
    assert not (tmp_path / "cache").exists()


def test_materialize_input_uris_local_downloads_and_reuses_cache(tmp_path, monkeypatch) -> None:
    payloads: dict[str, bytes] = {
        "bucket/quad.parquet": b"quad",
        "bucket/noise.parquet": b"noise",
        "bucket/orbits.parquet": b"orbits",
    }

    class _FakeFs:
        def open_input_file(self, path: str):
            return io.BytesIO(payloads[path])

    calls: list[str] = []

    def _fake_from_uri(uri: str):
        calls.append(str(uri))
        return _FakeFs(), str(uri).split("://", 1)[1]

    import bench.data.prepare_neomod_benchmark_bundle as mod

    fake_pafs = types.SimpleNamespace(
        FileSystem=types.SimpleNamespace(from_uri=staticmethod(_fake_from_uri))
    )
    monkeypatch.setattr(mod, "pafs", fake_pafs)

    inputs = InputUris(
        quad_truth="gs://bucket/quad.parquet",
        noise_100="gs://bucket/noise.parquet",
        neomod_orbits="gs://bucket/orbits.parquet",
    )
    cache_dir = tmp_path / "cache"

    out1 = _materialize_input_uris_local(inputs=inputs, cache_dir=cache_dir)
    assert len(calls) == 3
    assert (Path(out1.quad_truth)).read_bytes() == b"quad"
    assert (Path(out1.noise_100)).read_bytes() == b"noise"
    assert (Path(out1.neomod_orbits)).read_bytes() == b"orbits"

    out2 = _materialize_input_uris_local(inputs=inputs, cache_dir=cache_dir)
    assert len(calls) == 3
    assert out2 == out1


def test_orbit_photometry_bucket_index_is_stable_and_bounded() -> None:
    n = 64
    i1 = _orbit_photometry_bucket_index(orbit_id="abc123", n_buckets=n)
    i2 = _orbit_photometry_bucket_index(orbit_id="abc123", n_buckets=n)
    i3 = _orbit_photometry_bucket_index(orbit_id="xyz999", n_buckets=n)
    assert 0 <= i1 < n
    assert 0 <= i3 < n
    assert i1 == i2


def test_write_orbit_photometry_bucket_inputs_maps_object_to_orbit(tmp_path) -> None:
    start_mjd, _ = month_bounds_mjd_utc("2026-01")
    day = int(np.floor(start_mjd)) + 1
    time = pa.StructArray.from_arrays(
        [
            pa.array([day, day, day, day], type=pa.int64()),
            pa.array([0, 1, 2, 3], type=pa.int64()),
        ],
        names=["days", "nanos"],
    )
    truth = pa.table(
        {
            "id": pa.array(["d1", "d2", "d3", "d4"], type=pa.large_string()),
            "object_id": pa.array(["A", "B", "C", "A"], type=pa.large_string()),
            "observatory_code": pa.array(["X05", "X05", "X05", "X05"], type=pa.large_string()),
            "time": time,
            "ra": pa.array([10.0, 20.0, 30.0, 40.0], type=pa.float64()),
            "dec": pa.array([1.0, 2.0, 3.0, 4.0], type=pa.float64()),
            "mag": pa.array([21.0, 22.0, 23.0, None], type=pa.float64()),
            "mag_sigma": pa.array([0.1, 0.1, 0.1, 0.1], type=pa.float64()),
            "filter": pa.array(["r", "r", "r", "r"], type=pa.large_string()),
        }
    )
    truth_path = tmp_path / "truth.parquet"
    pq.write_table(truth, truth_path)

    mapping = pa.table(
        {
            "object_id": pa.array(["A", "B"], type=pa.large_string()),
            "orbit_id": pa.array(["oA", "oB"], type=pa.large_string()),
        }
    )

    bucket_paths = _write_orbit_photometry_bucket_inputs(
        truth_source=str(truth_path),
        object_orbit_mapping=mapping,
        month="2026-01",
        obscode="X05",
        bucket_dir=tmp_path / "buckets",
        n_buckets=8,
        batch_size=2,
    )
    assert len(bucket_paths) >= 1

    rows = []
    for p in bucket_paths:
        t = pq.read_table(p)
        rows.extend(t.to_pylist())
    assert len(rows) == 2
    orbit_ids = sorted([str(r["orbit_id"]) for r in rows])
    assert orbit_ids == ["oA", "oB"]
