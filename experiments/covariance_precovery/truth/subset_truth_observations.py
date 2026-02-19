from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import quivr as qv

from ..selection.bq_select import BqConfig
from ..selection.subset_designations import read_subset_window
from ..selection.subset_sampling import SelectedDesignations


def _run_bq_json(query: str, *, max_rows: int = 100000) -> list[dict[str, Any]]:
    try:
        proc = subprocess.run(
            [
                "bq",
                "query",
                "--nouse_legacy_sql",
                "--format=json",
                f"--max_rows={int(max_rows)}",
                query,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        msg = (e.stderr or "").strip() or (e.stdout or "").strip() or str(e)
        raise RuntimeError(f"BigQuery query failed: {msg}") from e
    out = proc.stdout.strip()
    return [] if not out else json.loads(out)


def _utc_ts(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt = dt.astimezone(timezone.utc)
    return dt.isoformat().replace("+00:00", "Z")


def _unix_to_mjd(unix_seconds: np.ndarray) -> np.ndarray:
    # 1970-01-01 00:00:00 UTC is MJD 40587.0
    return unix_seconds / 86400.0 + 40587.0


class TruthObservationByDesignation(qv.Table):
    designation = qv.LargeStringColumn()
    obscode = qv.LargeStringColumn()
    obsid = qv.LargeStringColumn()
    time_mjd_utc = qv.Float64Column()
    ra_deg = qv.Float64Column()
    dec_deg = qv.Float64Column()


@dataclass(frozen=True)
class TruthObsPersistResult:
    truth_parquet: Path
    meta_json: Path


def fetch_truth_observations_for_designations(
    *,
    cfg: BqConfig,
    designations: Sequence[str],
    obscodes: Sequence[str],
    start_utc: datetime,
    end_utc: datetime,
    chunk_size: int = 200,
    max_rows: int = 200000,
) -> TruthObservationByDesignation:
    if not designations:
        return TruthObservationByDesignation.empty()
    if not obscodes:
        raise ValueError("obscodes cannot be empty")

    stn_list = ", ".join(f"'{s}'" for s in obscodes)
    t0 = _utc_ts(start_utc)
    t1 = _utc_ts(end_utc)

    out = TruthObservationByDesignation.empty()
    for i0 in range(0, len(designations), int(chunk_size)):
        chunk = list(designations[i0 : i0 + int(chunk_size)])
        # BigQuery SQL string literal escape: single quote is doubled.
        des_list = ", ".join("'" + d.replace("'", "''") + "'" for d in chunk)

        # We use the raw table for designation mapping, but join to the clustered view for numeric RA/Dec.
        query = f"""
        WITH sel AS (
          SELECT designation
          FROM UNNEST([{des_list}]) AS designation
        )
        SELECT
          COALESCE(permid, provid) AS designation,
          stn AS obscode,
          obsid,
          obstime,
          SAFE_CAST(ra AS FLOAT64) AS ra_deg,
          SAFE_CAST(dec AS FLOAT64) AS dec_deg
        FROM `{cfg.obs_sbn_table}`
        WHERE stn IN ({stn_list})
          AND obstime >= TIMESTAMP('{t0}')
          AND obstime <  TIMESTAMP('{t1}')
          AND COALESCE(permid, provid) IN (SELECT designation FROM sel)
          AND SAFE_CAST(ra AS FLOAT64) IS NOT NULL
          AND SAFE_CAST(dec AS FLOAT64) IS NOT NULL
        """

        rows = _run_bq_json(query, max_rows=max_rows)
        if not rows:
            continue

        obstime = np.array([r["obstime"] for r in rows], dtype="datetime64[ns]")
        unix_s = obstime.astype("datetime64[s]").astype(np.int64).astype(np.float64)
        mjd = _unix_to_mjd(unix_s)

        out = qv.concatenate(
            [
                out,
                TruthObservationByDesignation.from_kwargs(
                    designation=[str(r["designation"]) for r in rows],
                    obscode=[str(r["obscode"]) for r in rows],
                    obsid=[str(r["obsid"]) for r in rows],
                    time_mjd_utc=mjd,
                    ra_deg=[float(r["ra_deg"]) for r in rows],
                    dec_deg=[float(r["dec_deg"]) for r in rows],
                ),
            ]
        )

    return out


def fetch_and_persist_truth_observations_for_subset_selection(
    *,
    subset_dir: Path,
    cfg: BqConfig,
    selected_designations_parquet: Path | None = None,
    out_tag: str | None = None,
) -> TruthObsPersistResult:
    win = read_subset_window(subset_dir)
    win.artifacts_dir.mkdir(parents=True, exist_ok=True)

    if selected_designations_parquet is None:
        selected_designations_parquet = win.artifacts_dir / "selected_designations.parquet"
    if not selected_designations_parquet.exists():
        raise FileNotFoundError(f"Missing selected designations parquet at {selected_designations_parquet}")

    selected = SelectedDesignations.from_parquet(str(selected_designations_parquet))
    des = [str(x) for x in selected.designation.to_pylist()]

    truth = fetch_truth_observations_for_designations(
        cfg=cfg,
        designations=des,
        obscodes=list(win.obscodes),
        start_utc=win.start_utc,
        end_utc=win.end_utc_exclusive,
    )

    tag = None if (out_tag is None or not str(out_tag).strip()) else str(out_tag).strip()
    truth_name = "truth_observations_selected.parquet" if tag is None else f"truth_observations_selected_{tag}.parquet"
    out_truth = win.artifacts_dir / truth_name
    truth.to_parquet(str(out_truth))

    meta = {
        "subset_dir": str(win.subset_dir),
        "selected_designations_parquet": str(selected_designations_parquet),
        "n_selected_designations": int(len(selected)),
        "n_truth_observations": int(len(truth)),
        "obscodes": list(win.obscodes),
        "start_utc": win.start_utc.isoformat().replace("+00:00", "Z"),
        "end_utc_exclusive": win.end_utc_exclusive.isoformat().replace("+00:00", "Z"),
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    meta_name = (
        "truth_observations_selected_meta.json"
        if tag is None
        else f"truth_observations_selected_meta_{tag}.json"
    )
    out_meta = win.artifacts_dir / meta_name
    out_meta.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")

    return TruthObsPersistResult(truth_parquet=out_truth, meta_json=out_meta)


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Persist truth observations for selected designations in a subset window.")
    p.add_argument("--subset-dir", type=str, required=True)
    p.add_argument(
        "--selected-designations-parquet",
        type=str,
        default=None,
        help="Optional SelectedDesignations parquet (default: subset artifacts/selected_designations.parquet).",
    )
    p.add_argument(
        "--out-tag",
        type=str,
        default=None,
        help="Optional tag to avoid overwriting prior truth_observations_selected.parquet.",
    )
    args = p.parse_args()

    out = fetch_and_persist_truth_observations_for_subset_selection(
        subset_dir=Path(args.subset_dir),
        cfg=BqConfig(),
        selected_designations_parquet=(
            None if args.selected_designations_parquet is None else Path(args.selected_designations_parquet)
        ),
        out_tag=args.out_tag,
    )
    print(f"truth_parquet={out.truth_parquet}")
    print(f"meta_json={out.meta_json}")


if __name__ == "__main__":
    main()

