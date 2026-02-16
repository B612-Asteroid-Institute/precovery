from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from adam_core.coordinates.covariances import CoordinateCovariances
from adam_core.orbits.ephemeris import Ephemeris


def cov_ll_elements_from_coordinate_covariances(
    cov: CoordinateCovariances,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract lon/lat covariance elements from `CoordinateCovariances` without materializing (N,6,6).

    The covariance is stored as a flattened length-36 row-major list per row.
    For the (lon,lat) 2x2 block (rows/cols 1 and 2):
      - (1,1) -> index 7
      - (1,2) -> index 8
      - (2,2) -> index 14

    Returns
    -------
    cov00, cov01, cov11 : ndarray float64 (N,)
        With NaNs where values are null / missing.
    """
    if len(cov) == 0:
        z = np.zeros(0, dtype=np.float64)
        return z, z, z

    # `values` is a LargeListArray<float64> where each row is length 36 (or null).
    vals = cov.table.column("values").combine_chunks()
    c00 = pc.fill_null(pc.list_element(vals, 7), np.nan)
    c01 = pc.fill_null(pc.list_element(vals, 8), np.nan)
    c11 = pc.fill_null(pc.list_element(vals, 14), np.nan)
    return (
        np.asarray(c00.to_numpy(zero_copy_only=False), dtype=np.float64),
        np.asarray(c01.to_numpy(zero_copy_only=False), dtype=np.float64),
        np.asarray(c11.to_numpy(zero_copy_only=False), dtype=np.float64),
    )


def cov_ll_from_ephemeris(ephem: Ephemeris) -> np.ndarray:
    """
    Return sky covariance in (lon,lat) deg^2 as an array shaped (N,2,2).

    Missing covariances become NaNs.
    """
    n = int(len(ephem))
    out = np.full((n, 2, 2), np.nan, dtype=np.float64)
    if n == 0:
        return out
    cov = ephem.coordinates.covariance
    # IMPORTANT: Avoid `cov.is_all_nan()` here; it materializes full (N,6,6) matrices.
    if cov is None:
        return out
    c00, c01, c11 = cov_ll_elements_from_coordinate_covariances(cov)
    out[:, 0, 0] = c00
    out[:, 0, 1] = c01
    out[:, 1, 0] = c01
    out[:, 1, 1] = c11
    return out


def attach_cov_ll_to_ephemeris(*, ephem: Ephemeris, cov_ll_deg2: np.ndarray) -> Ephemeris:
    """
    Return a copy of `ephem` with a (lon,lat) covariance block attached.

    This is intentionally minimal: we only guarantee correctness for downstream code that
    consumes the sky covariance via `cov_ll_from_ephemeris()`.
    """
    n = int(len(ephem))
    cov_ll = np.asarray(cov_ll_deg2, dtype=np.float64)
    if cov_ll.shape != (n, 2, 2):
        raise ValueError(f"cov_ll_deg2 must be shape (N,2,2) with N={n}, got {cov_ll.shape}")

    # Construct a full 6x6 covariance per row, but populate only the (lon,lat) block.
    #
    # If the ephemeris already has a covariance, preserve it and overwrite just the
    # relevant (lon,lat) block. This keeps other modeled terms (if any) intact.
    cov_existing = ephem.coordinates.covariance
    if cov_existing is not None:
        try:
            cov6 = np.asarray(cov_existing.to_matrix(), dtype=np.float64)
        except Exception:
            cov6 = np.full((n, 6, 6), np.nan, dtype=np.float64)
    else:
        cov6 = np.full((n, 6, 6), np.nan, dtype=np.float64)

    cov6[:, 1, 1] = cov_ll[:, 0, 0]
    cov6[:, 1, 2] = cov_ll[:, 0, 1]
    cov6[:, 2, 1] = cov_ll[:, 1, 0]
    cov6[:, 2, 2] = cov_ll[:, 1, 1]

    cov_col = CoordinateCovariances.from_matrix(cov6)
    # Set nested coordinates covariance column.
    return ephem.set_column("coordinates.covariance", cov_col)

