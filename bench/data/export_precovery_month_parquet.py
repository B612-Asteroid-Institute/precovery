from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ..benchmarks.workload import month_bounds_mjd_utc

# NOTE: The original helper that exported a local precovery subset DB into a parquet dataset
# was removed during earlier refactors. We keep this CLI as a placeholder so we don't lose the
# invocation/API, but it currently requires re-porting the underlying export implementation.
export_precovery_subset_to_parquet = None  # type: ignore[assignment]


@dataclass(frozen=True)
class ExportMeta:
    generated_at_utc: str
    subset_dir: str
    months: list[str]
    obscodes: list[str]
    out_dir: str
    start_mjd_utc: float
    end_mjd_utc: float
    max_frames: int | None
    chunk_frames: int
    n_frames: int
    n_detections: int


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export month detections from local precovery subset to parquet for DuckDB backend."
    )
    p.add_argument("--subset-dir", type=str, required=True)
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--month", action="append", required=True, help="Month window YYYY-MM (repeatable).")
    p.add_argument("--obscode", action="append", required=True, help="Observatory code (repeatable).")
    p.add_argument("--max-frames", type=int, default=0, help="Optional cap on number of frames exported (0=off).")
    p.add_argument("--chunk-frames", type=int, default=10_000, help="Frames per export chunk.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if export_precovery_subset_to_parquet is None:
        raise SystemExit(
            "Local subset export helper is not available in this checkout. "
            "If you need this functionality, we can restore/port it from the previous bench harness."
        )
    subset_dir = Path(args.subset_dir)
    out_dir = Path(args.out_dir)
    months = [str(x) for x in args.month]
    obscodes = tuple(str(x) for x in args.obscode)
    if len(months) != 1:
        raise SystemExit("For now export supports exactly one --month.")
    month = months[0]
    start_mjd, end_mjd = month_bounds_mjd_utc(month)

    max_frames = None if int(args.max_frames) <= 0 else int(args.max_frames)
    stats = export_precovery_subset_to_parquet(
        subset_dir=subset_dir,
        out_dir=out_dir,
        start_mjd_utc=float(start_mjd),
        end_mjd_utc=float(end_mjd),
        obscodes=tuple(obscodes),
        year_month=str(month),
        max_frames=max_frames,
        chunk_frames=int(args.chunk_frames),
    )

    meta = ExportMeta(
        generated_at_utc=_now_utc(),
        subset_dir=str(subset_dir),
        months=[month],
        obscodes=list(obscodes),
        out_dir=str(out_dir),
        start_mjd_utc=float(start_mjd),
        end_mjd_utc=float(end_mjd),
        max_frames=max_frames,
        chunk_frames=int(args.chunk_frames),
        n_frames=int(stats.n_frames),
        n_detections=int(stats.n_detections),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "meta.json").write_text(json.dumps(meta.__dict__, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

