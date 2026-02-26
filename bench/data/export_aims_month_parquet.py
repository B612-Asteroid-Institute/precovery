from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.compute as pc
import pyarrow.parquet as pq

from ..data.gcs_copy import default_copy_tool
from ..selection.bq_select import (
    bq_table_ref,
    estimate_bq_bytes,
    export_bq_table_to_gcs_parquet,
    run_bq_query_to_table,
)

try:
    import healpy as hp  # type: ignore[import-untyped]
except Exception:  # pragma: no cover
    hp = None  # type: ignore[assignment]


@dataclass(frozen=True)
class ExportConfig:
    bq_project_id: str = "moeyens-thor-dev"
    bq_dataset_id: str = "ai_aleck_scratch"
    gcs_export_prefix: str = "gs://ak-scratch/precovery/covariance_precovery/bq_exports"


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _month_bounds_utc(year_month: str) -> tuple[datetime, datetime]:
    ym = str(year_month).strip()
    if len(ym) != 7 or ym[4] != "-":
        raise ValueError(f"Invalid year_month: {year_month!r} (expected YYYY-MM)")
    y = int(ym[:4])
    m = int(ym[5:])
    start = datetime(y, m, 1, tzinfo=timezone.utc)
    if m == 12:
        end = datetime(y + 1, 1, 1, tzinfo=timezone.utc)
    else:
        end = datetime(y, m + 1, 1, tzinfo=timezone.utc)
    return start, end


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Export a month of detections from BigQuery AIMS into a local parquet dataset "
            "matching the DuckDB backend schema (computes healpixel locally at nside=32, nest=True)."
        )
    )
    p.add_argument("--month", action="append", required=True, help="Month window YYYY-MM (repeatable).")
    p.add_argument("--obscode", action="append", required=True, help="Observatory code (repeatable).")
    p.add_argument("--out-dir", type=str, required=True, help="Root output directory.")

    p.add_argument("--source-table", type=str, default="moeyens-thor-dev.production_aims.aims")
    p.add_argument("--bq-project-id", type=str, default=ExportConfig.bq_project_id)
    p.add_argument("--bq-dataset-id", type=str, default=ExportConfig.bq_dataset_id)
    p.add_argument("--gcs-export-prefix", type=str, default=ExportConfig.gcs_export_prefix)
    p.add_argument("--max-bytes-estimate", type=int, default=0, help="Refuse exports above this bytes estimate (0=off).")
    p.add_argument("--dry-run", action="store_true", help="Only estimate bytes; do not export or download.")
    p.add_argument(
        "--delete-shards",
        action="store_true",
        help="Delete downloaded parquet shards after building the final dataset (default: keep).",
    )
    p.add_argument("--healpix-nside", type=int, default=32, help="Healpix nside for computed healpixel (nest=True).")
    p.add_argument("--merge", action="store_true", help="Also merge the final dataset into a single parquet file.")
    return p.parse_args()


def _merge_shards(*, shards_dir: Path, merged_path: Path) -> None:
    parquet_files = sorted(p for p in shards_dir.rglob("*.parquet") if p.is_file())
    if not parquet_files:
        raise RuntimeError(f"No parquet files under {shards_dir}")
    dataset = ds.dataset([str(p) for p in parquet_files], format="parquet")
    schema = dataset.schema
    merged_path.parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(str(merged_path), schema=schema, compression="snappy")
    try:
        for batch in dataset.to_batches(batch_size=65536):
            writer.write_table(pa.Table.from_batches([batch], schema=schema))
    finally:
        writer.close()


def _build_aims_export_query(
    *,
    source_table: str,
    obscodes: list[str],
    start_utc: datetime,
    end_utc: datetime,
) -> str:
    if not obscodes:
        raise ValueError("obscodes cannot be empty")
    stn_list = ", ".join(f"'{str(s)}'" for s in obscodes)
    d0 = start_utc.date().isoformat()
    d1 = end_utc.date().isoformat()
    return f"""
    SELECT
      CAST(observatory_code AS STRING) AS obscode,
      FORMAT_DATE('%Y-%m', partition_date) AS year_month,
      CAST(exposure.mjd_start + 0.5 * exposure.duration / 86400.0 AS FLOAT64) AS exposure_mjd_mid_utc,
      CAST(ROUND((exposure.mjd_start + 0.5 * exposure.duration / 86400.0) * 86400.0 * 1e6) AS INT64) AS exposure_mjd_mid_key_us,
      CAST(exposure.filter AS STRING) AS filter,
      CAST(observation.id AS STRING) AS observation_id,
      CAST(observation.mjd AS FLOAT64) AS obstime_mjd_utc,
      CAST(observation.ra AS FLOAT64) AS ra_deg,
      CAST(observation.dec AS FLOAT64) AS dec_deg,
      CAST(COALESCE(observation.ra_sigma, 0.0) AS FLOAT64) AS ra_sigma_deg,
      CAST(COALESCE(observation.dec_sigma, 0.0) AS FLOAT64) AS dec_sigma_deg,
      CAST(observation.mag AS FLOAT64) AS mag,
      CAST(observation.mag_sigma AS FLOAT64) AS mag_sigma
    FROM `{str(source_table)}`
    WHERE observatory_code IN ({stn_list})
      AND partition_date >= DATE('{d0}')
      AND partition_date <  DATE('{d1}')
    """


def _compute_healpixel_nested(
    *, ra_deg: np.ndarray, dec_deg: np.ndarray, nside: int
) -> np.ndarray:
    if hp is None:
        raise RuntimeError(
            "healpy is not available; cannot compute healpixel. Install healpy or use the precovery-db exporter."
        )
    lon = np.asarray(ra_deg, dtype=np.float64)
    lat = np.asarray(dec_deg, dtype=np.float64)
    return np.asarray(hp.ang2pix(int(nside), lon, lat, lonlat=True, nest=True), dtype=np.int64)


def _write_final_dataset(
    *,
    shards_dir: Path,
    out_dir: Path,
    healpix_nside: int,
) -> None:
    """
    Convert raw exported shards into a parquet dataset matching DuckDB backend expectations
    by computing `healpixel` locally from (ra_deg,dec_deg).
    """
    parquet_files = sorted(p for p in shards_dir.rglob("*.parquet") if p.is_file())
    if not parquet_files:
        raise RuntimeError(f"No parquet shards found under {shards_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    part = 0
    for p in parquet_files:
        pf = pq.ParquetFile(p)
        for batch in pf.iter_batches(batch_size=131072):
            t = pa.Table.from_batches([batch])
            # Ensure numeric dtypes and compute healpixel.
            ra = pc.cast(t["ra_deg"], pa.float64()).to_numpy(zero_copy_only=False)
            dec = pc.cast(t["dec_deg"], pa.float64()).to_numpy(zero_copy_only=False)
            healpixel = _compute_healpixel_nested(ra_deg=ra, dec_deg=dec, nside=int(healpix_nside))
            t2 = t.append_column("healpixel", pa.array(healpixel, type=pa.int64()))
            pq.write_table(t2, out_dir / f"part-{part:06d}.parquet")
            part += 1

def main() -> None:
    args = _parse_args()
    months = [str(m) for m in args.month]
    obscodes = [str(o) for o in args.obscode]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stn_label = "_".join(sorted(obscodes))
    stn_label = "".join(c if (c.isalnum() or c == "_") else "_" for c in stn_label)
    max_bytes = None if int(args.max_bytes_estimate) <= 0 else int(args.max_bytes_estimate)

    for ym in months:
        start_utc, end_utc = _month_bounds_utc(ym)
        table_id = f"precovery_aims_detections__{stn_label}__{ym.replace('-', '')}"
        dest_table = bq_table_ref(
            project_id=str(args.bq_project_id),
            dataset_id=str(args.bq_dataset_id),
            table_id=str(table_id),
        )
        gcs_dir = f"{str(args.gcs_export_prefix).rstrip('/')}/{table_id}"
        export_uri = f"{gcs_dir}/part-*.parquet"

        query = _build_aims_export_query(
            source_table=str(args.source_table),
            obscodes=obscodes,
            start_utc=start_utc,
            end_utc=end_utc,
        )
        bytes_est = int(estimate_bq_bytes(str(query)))
        if max_bytes is not None and int(bytes_est) > int(max_bytes):
            raise RuntimeError(
                f"Refusing export: estimated bytes {bytes_est} exceeds max_bytes_estimate={int(max_bytes)}"
            )

        local_root = out_dir / f"throughput_{ym}__{stn_label}"
        local_root.mkdir(parents=True, exist_ok=True)
        if bool(args.dry_run):
            meta = {
                "generated_at_utc": _now_utc(),
                "months": [ym],
                "obscodes": obscodes,
                "source_table": str(args.source_table),
                "destination_table": str(dest_table),
                "gcs_export_dir": str(gcs_dir),
                "bytes_estimate": int(bytes_est),
                "dry_run": True,
            }
            (local_root / "meta.json").write_text(
                json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            continue

        run_bq_query_to_table(query=str(query), destination_table=str(dest_table), replace=True)
        export_bq_table_to_gcs_parquet(source_table=str(dest_table), destination_uri=str(export_uri))

        shards_dir = local_root / "detections_parquet_shards"
        shards_dir.mkdir(parents=True, exist_ok=True)
        copy_tool = default_copy_tool()
        copy_tool.cp(str(gcs_dir), shards_dir, recursive=True)

        final_dir = local_root / "detections_parquet"
        _write_final_dataset(
            shards_dir=shards_dir,
            out_dir=final_dir,
            healpix_nside=int(args.healpix_nside),
        )
        merged_path = local_root / "detections.parquet"
        if bool(args.merge):
            _merge_shards(shards_dir=final_dir, merged_path=merged_path)
        if bool(args.delete_shards):
            # Best-effort cleanup to save disk.
            for p in sorted(shards_dir.rglob("*.parquet")):
                try:
                    p.unlink()
                except Exception:
                    pass

        meta = {
            "generated_at_utc": _now_utc(),
            "months": [ym],
            "obscodes": obscodes,
            "source_table": str(args.source_table),
            "destination_table": str(dest_table),
            "gcs_export_dir": str(gcs_dir),
            "local_shards_dir": str(shards_dir),
            "local_detections_parquet_dir": str(final_dir),
            "bytes_estimate": int(bytes_est),
            "merged_parquet": str(merged_path) if bool(args.merge) else None,
            "healpix_nside": int(args.healpix_nside),
            "healpix_nest": True,
        }
        (local_root / "meta.json").write_text(
            json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()

