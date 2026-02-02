from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import quivr as qv


class FootprintGeometry(qv.Table):
    """
    One row per (strategy, variant_kind, footprint, orbit_id, target_idx) describing a reusable
    footprint geometry object.

    This is intended to be persisted by Stage 3 and reused by Stage 4 so geometry construction
    is not duplicated (perfect accounting).
    """

    strategy = qv.LargeStringColumn()
    variant_kind = qv.LargeStringColumn(nullable=True)
    footprint = qv.LargeStringColumn()
    orbit_id = qv.LargeStringColumn()
    target_idx = qv.Int64Column()

    # Geometry kind:
    #   - "point": center only
    #   - "ellipse_cov": covariance-defined ellipse (used for contains() and optionally polygon vertices)
    #   - "polygon_vertices": explicit polygon vertices (lon/lat) in points table
    #   - "corridor_polyline": explicit path polyline in points table
    geometry_kind = qv.LargeStringColumn()

    # Common center (deg)
    lon0_deg = qv.Float64Column(nullable=True)
    lat0_deg = qv.Float64Column(nullable=True)

    # Optional covariance (lon/lat block, deg^2)
    cov_ll_00 = qv.Float64Column(nullable=True)
    cov_ll_01 = qv.Float64Column(nullable=True)
    cov_ll_10 = qv.Float64Column(nullable=True)
    cov_ll_11 = qv.Float64Column(nullable=True)

    # Optional parameters
    n_sigma = qv.Float64Column(nullable=True)
    polygon_vertices = qv.Int64Column(nullable=True)
    polygon_mode = qv.LargeStringColumn(nullable=True)  # e.g. "angle_sort" or "convex_hull"
    corridor_radius_arcsec = qv.Float64Column(nullable=True)
    corridor_step_arcsec = qv.Float64Column(nullable=True)
    buffer_arcsec = qv.Float64Column(nullable=True)

    # Link to points table when applicable
    geom_id = qv.LargeStringColumn(nullable=True)


class FootprintGeometryPoint(qv.Table):
    """
    Long-form points for footprint geometries. The geometry is identified by (geom_id, kind).
    """

    geom_id = qv.LargeStringColumn()
    kind = qv.LargeStringColumn()  # "polygon_vertex" | "corridor_path"
    idx = qv.Int32Column()
    lon_deg = qv.Float64Column()
    lat_deg = qv.Float64Column()


@dataclass(frozen=True)
class GeometryArtifacts:
    geometry_parquet: Path
    points_parquet: Path


def geometry_artifact_paths(*, run_dir: Path, strategy: str, variant_kind: str | None, footprint: str) -> GeometryArtifacts:
    strat_key = str(strategy) if variant_kind is None else f"{strategy}:{variant_kind}"
    base = run_dir / "footprint_geometry" / strat_key / str(footprint)
    return GeometryArtifacts(
        geometry_parquet=base / "geometry.parquet",
        points_parquet=base / "geometry_points.parquet",
    )


def write_geometry_artifacts(
    *,
    run_dir: Path,
    strategy: str,
    variant_kind: str | None,
    footprint: str,
    geometry_rows: list[dict[str, object]],
    points_rows: list[dict[str, object]],
) -> None:
    paths = geometry_artifact_paths(run_dir=run_dir, strategy=strategy, variant_kind=variant_kind, footprint=footprint)
    paths.geometry_parquet.parent.mkdir(parents=True, exist_ok=True)

    geom = FootprintGeometry.from_pyarrow(pa.Table.from_pylist(geometry_rows)) if geometry_rows else FootprintGeometry.empty()
    geom.to_parquet(str(paths.geometry_parquet))

    pts = (
        FootprintGeometryPoint.from_pyarrow(pa.Table.from_pylist(points_rows))
        if points_rows
        else FootprintGeometryPoint.empty()
    )
    pts.to_parquet(str(paths.points_parquet))


def read_geometry_artifacts(
    *,
    run_dir: Path,
    strategy: str,
    variant_kind: str | None,
    footprint: str,
) -> tuple[FootprintGeometry, FootprintGeometryPoint]:
    paths = geometry_artifact_paths(run_dir=run_dir, strategy=strategy, variant_kind=variant_kind, footprint=footprint)
    geom = FootprintGeometry.from_parquet(str(paths.geometry_parquet))
    pts = FootprintGeometryPoint.from_parquet(str(paths.points_parquet))
    return geom, pts


def cov_ll_matrix_from_row(row: dict[str, object]) -> np.ndarray:
    return np.array(
        [
            [float(row["cov_ll_00"]), float(row["cov_ll_01"])],
            [float(row["cov_ll_10"]), float(row["cov_ll_11"])],
        ],
        dtype=np.float64,
    )

