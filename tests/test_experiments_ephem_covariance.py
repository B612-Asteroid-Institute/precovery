import numpy as np
import pyarrow as pa

from adam_core.coordinates.covariances import CoordinateCovariances
from adam_core.coordinates.origin import Origin
from adam_core.coordinates.spherical import SphericalCoordinates
from adam_core.orbits.ephemeris import Ephemeris
from adam_core.time import Timestamp

from precovery.search.covariance import cov_ll_from_ephemeris
from experiments.covariance_precovery.harness.stage3_healpixel_bench import (
    FrameTimeTargets,
    _map_times_to_target_idx_by_obscode,
)
from experiments.covariance_precovery.harness.stage4_detection_filter_bench import (
    _collapsed_ephemeris_map_for_part,
)


def test_cov_ll_from_ephemeris_extracts_lonlat_block() -> None:
    # Build a covariance where only the (lon,lat) block is populated.
    cov = np.zeros((1, 6, 6), dtype=np.float64)
    cov[0, 1, 1] = 11.0
    cov[0, 1, 2] = 12.0
    cov[0, 2, 1] = 12.0
    cov[0, 2, 2] = 22.0

    t = Timestamp.from_kwargs(days=[60000], nanos=[1_234_567], scale="utc")
    coords = SphericalCoordinates.from_kwargs(
        rho=[1.0],
        lon=[10.0],
        lat=[20.0],
        vrho=[0.0],
        vlon=[0.0],
        vlat=[0.0],
        time=t,
        covariance=CoordinateCovariances.from_matrix(cov),
        origin=Origin.from_kwargs(code=pa.array(["500"], pa.large_string())),
        frame="equatorial",
    )
    eph = Ephemeris.from_kwargs(orbit_id=["o1"], coordinates=coords)

    cov_ll = cov_ll_from_ephemeris(eph)
    assert cov_ll.shape == (1, 2, 2)
    assert np.allclose(cov_ll[0], np.array([[11.0, 12.0], [12.0, 22.0]]))


def test_cov_ll_from_ephemeris_missing_covariance_is_nan() -> None:
    t = Timestamp.from_kwargs(days=[60000], nanos=[0], scale="utc")
    coords = SphericalCoordinates.from_kwargs(
        rho=[1.0],
        lon=[10.0],
        lat=[20.0],
        vrho=[0.0],
        vlon=[0.0],
        vlat=[0.0],
        time=t,
        covariance=CoordinateCovariances.nulls(1),
        origin=Origin.from_kwargs(code=pa.array(["500"], pa.large_string())),
        frame="equatorial",
    )
    eph = Ephemeris.from_kwargs(orbit_id=["o1"], coordinates=coords)

    cov_ll = cov_ll_from_ephemeris(eph)
    assert np.isnan(cov_ll).all()


def _targets_table(*, obscode: list[str], time: Timestamp) -> pa.Table:
    ft = FrameTimeTargets.from_kwargs(obscode=obscode, time=time)
    idx = np.arange(len(ft), dtype=np.int64)
    return pa.Table.from_arrays(
        [
            pa.array(idx, type=pa.int64()),
            pa.array(ft.obscode.to_pylist(), type=pa.large_string()),
            ft.table.column("time").combine_chunks(),
        ],
        names=["target_idx", "obscode", "time"],
    )


def test_map_times_to_target_idx_by_obscode_matches_after_rounding() -> None:
    # target time and query time differ by < 1 us, so precision="us" should match.
    targets = _targets_table(
        obscode=["500"],
        time=Timestamp.from_kwargs(days=[60000], nanos=[1_234_567], scale="utc"),
    )
    out = _map_times_to_target_idx_by_obscode(
        obscode=np.asarray(["500"], dtype=object),
        time_utc=Timestamp.from_kwargs(days=[60000], nanos=[1_234_999], scale="utc"),
        targets=targets,
        dt_sec=60.0,
        precision="us",
    )
    assert out.tolist() == [0]


def test_map_times_to_target_idx_by_obscode_fallback_nearest_within_tolerance() -> None:
    # Query time differs by seconds, so rounded join won't match; fallback nearest should.
    targets = _targets_table(
        obscode=["500"],
        time=Timestamp.from_kwargs(days=[60000], nanos=[0], scale="utc"),
    )
    # +5 seconds
    out = _map_times_to_target_idx_by_obscode(
        obscode=np.asarray(["500"], dtype=object),
        time_utc=Timestamp.from_kwargs(days=[60000], nanos=[5_000_000_000], scale="utc"),
        targets=targets,
        dt_sec=10.0,
        precision="us",
    )
    assert out.tolist() == [0]


def test_stage4_reuses_stage3_collapsed_ephemeris_part(tmp_path) -> None:
    stage3_run_dir = tmp_path / "stage3"
    strategy = "assist_window_then_2body_variants"
    variant_kind = "sigma_points"
    part_stem = "part-000000"
    out_dir = (
        stage3_run_dir
        / "collapsed_ephemeris"
        / f"{strategy}:{variant_kind}"
        / "parts"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{part_stem}.parquet"

    cov = np.zeros((1, 6, 6), dtype=np.float64)
    cov[0, 1, 1] = 1.1
    cov[0, 1, 2] = 0.2
    cov[0, 2, 1] = 0.2
    cov[0, 2, 2] = 2.2

    t = Timestamp.from_kwargs(days=[60000], nanos=[123_456_789], scale="utc")
    coords = SphericalCoordinates.from_kwargs(
        rho=[1.0],
        lon=[10.0],
        lat=[20.0],
        vrho=[0.0],
        vlon=[0.0],
        vlat=[0.0],
        time=t,
        covariance=CoordinateCovariances.from_matrix(cov),
        origin=Origin.from_kwargs(code=pa.array(["500"], pa.large_string())),
        frame="equatorial",
    )
    ep = Ephemeris.from_kwargs(orbit_id=["o1"], coordinates=coords)
    ep.to_parquet(str(out_path))

    targets = _targets_table(obscode=["500"], time=t)
    out = _collapsed_ephemeris_map_for_part(
        stage3_run_dir=stage3_run_dir,
        strategy=strategy,
        variant_kind=variant_kind,
        part_stem=part_stem,
        targets_tbl=targets,
        dt_days=60.0 / 86400.0,
    )

    assert ("o1", 0) in out
    lon0, lat0, cov_ll = out[("o1", 0)]
    assert lon0 == 10.0
    assert lat0 == 20.0
    assert cov_ll is not None
    assert np.allclose(cov_ll, np.array([[1.1, 0.2], [0.2, 2.2]], dtype=np.float64))

