from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import quivr as qv

from .bq_select import (
    BqConfig,
    ObservedDesignationWindowSummary,
    count_designations_observed_any_station_in_window,
    find_designations_observed_any_station_in_window,
)


_MJD0_UTC = datetime(1858, 11, 17, tzinfo=timezone.utc)


def datetime_utc_from_mjd(mjd: float) -> datetime:
    return _MJD0_UTC + timedelta(days=float(mjd))


@dataclass(frozen=True)
class SubsetWindow:
    subset_dir: Path
    index_db: Path
    min_mjd: float
    max_mjd: float
    obscodes: tuple[str, ...]

    @property
    def start_utc(self) -> datetime:
        return datetime_utc_from_mjd(self.min_mjd)

    @property
    def end_utc_exclusive(self) -> datetime:
        # BigQuery query uses "< end"; add a small epsilon to include the max-midpoint day.
        return datetime_utc_from_mjd(self.max_mjd) + timedelta(seconds=1)

    @property
    def artifacts_dir(self) -> Path:
        return self.subset_dir / "artifacts"


def read_subset_window(subset_dir: Path) -> SubsetWindow:
    index_db = subset_dir / "index.db"
    if not index_db.exists():
        raise FileNotFoundError(f"Missing subset index.db at {index_db}")

    conn = sqlite3.connect(str(index_db))
    try:
        row = conn.execute("SELECT MIN(exposure_mjd_mid), MAX(exposure_mjd_mid) FROM frames").fetchone()
        if row is None or row[0] is None or row[1] is None:
            raise ValueError(f"No frames found in {index_db}")
        min_mjd = float(row[0])
        max_mjd = float(row[1])

        obscodes = [
            str(r[0])
            for r in conn.execute("SELECT DISTINCT obscode FROM frames ORDER BY obscode").fetchall()
            if r and r[0] is not None
        ]
    finally:
        conn.close()

    return SubsetWindow(
        subset_dir=subset_dir,
        index_db=index_db,
        min_mjd=min_mjd,
        max_mjd=max_mjd,
        obscodes=tuple(obscodes),
    )


def _parse_bq_utc(s: str) -> datetime:
    # BigQuery CAST(TIMESTAMP AS STRING) is typically ISO-ish and `fromisoformat` accepts a space separator.
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _merge_window_summaries(
    summaries: list[ObservedDesignationWindowSummary],
) -> ObservedDesignationWindowSummary:
    # Merge by designation across disjoint time chunks.
    acc: dict[str, dict[str, object]] = {}
    for s in summaries:
        for i in range(len(s)):
            des = str(s.designation[i].as_py())
            t_min = _parse_bq_utc(str(s.t_min_utc[i].as_py()))
            t_max = _parse_bq_utc(str(s.t_max_utc[i].as_py()))
            n_obs = int(s.n_obs[i].as_py())
            stn_csv = str(s.stn_csv[i].as_py() or "")
            stns = {p for p in stn_csv.split(",") if p}

            if des not in acc:
                acc[des] = {
                    "t_min": t_min,
                    "t_max": t_max,
                    "n_obs": n_obs,
                    "stns": stns,
                }
            else:
                a = acc[des]
                a["t_min"] = min(a["t_min"], t_min)  # type: ignore[arg-type]
                a["t_max"] = max(a["t_max"], t_max)  # type: ignore[arg-type]
                a["n_obs"] = int(a["n_obs"]) + n_obs
                a["stns"] = set(a["stns"]) | stns

    # Stable order (largest n_obs first) for easier inspection.
    items = sorted(
        acc.items(),
        key=lambda kv: (int(kv[1]["n_obs"]), kv[0]),
        reverse=True,
    )

    return ObservedDesignationWindowSummary.from_kwargs(
        designation=[k for k, _ in items],
        t_min_utc=[v["t_min"].isoformat().replace("+00:00", "Z") for _, v in items],  # type: ignore[index]
        t_max_utc=[v["t_max"].isoformat().replace("+00:00", "Z") for _, v in items],  # type: ignore[index]
        n_obs=[int(v["n_obs"]) for _, v in items],  # type: ignore[index]
        n_stn=[len(set(v["stns"])) for _, v in items],  # type: ignore[index]
        stn_csv=[",".join(sorted(set(v["stns"]))) for _, v in items],  # type: ignore[index]
    )


class DesignationList(qv.Table):
    designation = qv.LargeStringColumn()


def _fetch_partitioned_window(
    *,
    cfg: BqConfig,
    start_utc: datetime,
    end_utc: datetime,
    obscodes: list[str],
    max_rows: int,
    max_partitions: int = 256,
) -> tuple[ObservedDesignationWindowSummary, int]:
    """
    Fetch a complete designation summary for a time window by partitioning on FARM_FINGERPRINT
    until each partition's row-count is <= `max_rows`.
    """
    partitions = 1
    counts: list[int] = []
    while True:
        counts = [
            count_designations_observed_any_station_in_window(
                cfg=cfg,
                start_utc=start_utc,
                end_utc=end_utc,
                obscodes=obscodes,
                partition_mod=partitions,
                partition_idx=i,
            )
            for i in range(partitions)
        ]
        if all(c <= max_rows for c in counts):
            break
        if partitions >= max_partitions:
            raise RuntimeError(
                f"Exceeded max_partitions={max_partitions} while keeping partition counts <= {max_rows}. "
                f"Max partition count was {max(counts) if counts else 0}."
            )
        partitions *= 2

    parts: list[ObservedDesignationWindowSummary] = []
    for i, c in enumerate(counts):
        if c == 0:
            continue
        parts.append(
            find_designations_observed_any_station_in_window(
                cfg=cfg,
                start_utc=start_utc,
                end_utc=end_utc,
                obscodes=obscodes,
                max_rows=max_rows,
                partition_mod=partitions,
                partition_idx=i,
            )
        )

    return _merge_window_summaries(parts), partitions


def fetch_and_persist_designations_for_subset(
    *,
    subset_dir: Path,
    cfg: BqConfig,
    chunk_days: int = 7,
    max_rows_per_chunk: int = 100000,
    max_partitions_per_chunk: int = 256,
) -> dict[str, Path]:
    """
    For a local subset directory containing a trimmed `index.db`, query BigQuery to find all
    MPC designations observed by the stations present in the subset over the subset time window,
    and persist results as local parquet artifacts.

    Note: this intentionally returns *designations* (COALESCE(permid, provid)), since those map
    cleanly to MPCQ/MPC lookups. We do *not* restrict later MPC history fetches to these stations.
    """
    win = read_subset_window(subset_dir)
    win.artifacts_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[ObservedDesignationWindowSummary] = []
    partitions_used: list[int] = []
    t0 = win.start_utc
    t1 = win.end_utc_exclusive
    step = timedelta(days=int(chunk_days))
    cur = t0
    while cur < t1:
        nxt = min(cur + step, t1)
        s, p = _fetch_partitioned_window(
            cfg=cfg,
            start_utc=cur,
            end_utc=nxt,
            obscodes=list(win.obscodes),
            max_rows=int(max_rows_per_chunk),
            max_partitions=int(max_partitions_per_chunk),
        )
        summaries.append(s)
        partitions_used.append(int(p))
        cur = nxt

    merged = _merge_window_summaries(summaries)
    out_summary = win.artifacts_dir / "bq_designations_window_summary.parquet"
    merged.to_parquet(str(out_summary))

    out_designations = win.artifacts_dir / "bq_designations_unique.parquet"
    DesignationList.from_kwargs(designation=merged.designation.to_pylist()).to_parquet(
        str(out_designations)
    )

    meta = {
        "subset_dir": str(win.subset_dir),
        "index_db": str(win.index_db),
        "min_mjd": win.min_mjd,
        "max_mjd": win.max_mjd,
        "start_utc": win.start_utc.isoformat().replace("+00:00", "Z"),
        "end_utc_exclusive": win.end_utc_exclusive.isoformat().replace("+00:00", "Z"),
        "obscodes": list(win.obscodes),
        "chunk_days": int(chunk_days),
        "max_rows_per_chunk": int(max_rows_per_chunk),
        "max_partitions_per_chunk": int(max_partitions_per_chunk),
        "partitions_used_per_chunk": partitions_used,
        "n_designations": int(len(merged)),
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    out_meta = win.artifacts_dir / "bq_designations_meta.json"
    out_meta.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")

    return {
        "summary_parquet": out_summary,
        "designations_parquet": out_designations,
        "meta_json": out_meta,
    }


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Persist BigQuery designation lists for a local subset.")
    p.add_argument("--subset-dir", type=str, required=True)
    p.add_argument("--chunk-days", type=int, default=7)
    p.add_argument("--max-rows-per-chunk", type=int, default=200000)
    args = p.parse_args()

    out = fetch_and_persist_designations_for_subset(
        subset_dir=Path(args.subset_dir),
        cfg=BqConfig(),
        chunk_days=int(args.chunk_days),
        max_rows_per_chunk=int(args.max_rows_per_chunk),
    )
    for k, v in out.items():
        print(f"{k}={v}")


if __name__ == "__main__":
    main()

