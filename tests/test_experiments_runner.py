import healpy as hp
import numpy as np
import pyarrow as pa

from experiments.covariance_precovery.harness.stage3_healpixel_bench import (
    _designation_from_object_id,
    _collapse_variant_ephemeris_group,
    _predicted_pixels_from_mean_row,
)
from adam_core.coordinates.origin import Origin
from adam_core.coordinates.spherical import SphericalCoordinates
from adam_core.orbits.variants import VariantEphemeris
from adam_core.time import Timestamp


def test_stage3_designation_normalization() -> None:
    assert _designation_from_object_id("191305 (2003 HQ20)") == "191305"
    assert _designation_from_object_id("(2019 NY21)") == "2019 NY21"
    assert _designation_from_object_id("6928 Lanna (1994 TM3)") == "6928"


def test_stage3_point_pixels_matches_healpy() -> None:
    lon, lat = 10.0, 20.0
    nside = 64
    pix = _predicted_pixels_from_mean_row(
        lon_deg=lon,
        lat_deg=lat,
        cov_ll_deg2=None,
        nside=nside,
        footprint="point",
        n_sigma=3.0,
        polygon_vertices=32,
        mc_num_samples=64,
        mc_seed=0,
    )
    assert pix.tolist() == [int(hp.ang2pix(nside, lon, lat, lonlat=True, nest=True))]


def test_stage3_cov_disc_contains_point_pixel() -> None:
    lon, lat = 10.0, 20.0
    nside = 64
    cov_ll = np.array([[1e-6, 0.0], [0.0, 1e-6]], dtype=np.float64)
    pix_point = int(hp.ang2pix(nside, lon, lat, lonlat=True, nest=True))
    pix_disc = _predicted_pixels_from_mean_row(
        lon_deg=lon,
        lat_deg=lat,
        cov_ll_deg2=cov_ll,
        nside=nside,
        footprint="cov_disc",
        n_sigma=3.0,
        polygon_vertices=32,
        mc_num_samples=64,
        mc_seed=0,
    )
    assert pix_point in set(pix_disc.tolist())


def test_stage3_variant_ephemeris_collapse_smoke() -> None:
    frame = "equatorial"

    variants = VariantEphemeris.from_kwargs(
        orbit_id=pa.array(["o1", "o1"], pa.large_string()),
        object_id=pa.array(["o1", "o1"], pa.large_string()),
        variant_id=pa.array(["0", "1"], pa.large_string()),
        weights=[0.5, 0.5],
        weights_cov=[0.5, 0.5],
        coordinates=SphericalCoordinates.from_kwargs(
            rho=[1.0, 1.0],
            lon=[10.001, 9.999],
            lat=[20.002, 19.998],
            vrho=[0.0, 0.0],
            vlon=[0.0, 0.0],
            vlat=[0.0, 0.0],
            time=Timestamp.from_kwargs(days=[60000, 60000], nanos=[0, 0], scale="utc"),
            origin=Origin.from_kwargs(code=pa.array(["500", "500"], pa.large_string())),
            frame=frame,
        ),
    )

    collapsed = variants.collapse_by_object_id()
    cov = collapsed.coordinates.covariance.to_matrix()[0]
    assert cov.shape == (6, 6)
    assert np.isfinite(cov).all()

    # Also exercise our Stage3 helper that wraps collapse and builds the mean ephemeris row.
    collapsed2 = _collapse_variant_ephemeris_group(
        variants=variants,
    )
    cov2 = collapsed2.coordinates.covariance.to_matrix()[0]
    assert np.isfinite(cov2).all()

