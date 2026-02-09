from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import quivr as qv

from .bq_select import (
    BqConfig,
    DesignationOrbitWindowFeatures,
    count_designation_orbit_features_for_window,
    fetch_designation_orbit_features_for_window,
)
from .subset_designations import read_subset_window
from .subset_sampling import SamplingConfig, SelectedDesignations, select_designations_from_features


@dataclass(frozen=True)
class TimeBlock:
    label: str
    start_utc: datetime
    end_utc_exclusive: datetime


def _utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def generate_even_time_blocks(
    *,
    start_utc: datetime,
    end_utc_exclusive: datetime,
    n_blocks: int,
    block_days: int,
) -> list[TimeBlock]:
    """
    Generate N blocks of fixed duration, roughly evenly spaced across [start, end).
    """
    s = _utc(start_utc)
    e = _utc(end_utc_exclusive)
    if e <= s:
        raise ValueError("end_utc_exclusive must be > start_utc")
    if int(n_blocks) <= 0:
        raise ValueError("n_blocks must be > 0")
    if int(block_days) <= 0:
        raise ValueError("block_days must be > 0")

    span_days = (e - s).total_seconds() / 86400.0
    if span_days <= float(block_days):
        mid = s + 0.5 * (e - s)
        b0 = mid - timedelta(days=int(block_days) // 2)
        b1 = b0 + timedelta(days=int(block_days))
        return [TimeBlock(label="block_000", start_utc=b0, end_utc_exclusive=b1)]

    # Pick starts on an evenly spaced grid.
    max_start = e - timedelta(days=int(block_days))
    grid = np.linspace(0.0, 1.0, int(n_blocks), endpoint=True)
    blocks: list[TimeBlock] = []
    for i, f in enumerate(grid.tolist()):
        t0 = s + (max_start - s) * float(f)
        t1 = t0 + timedelta(days=int(block_days))
        blocks.append(
            TimeBlock(
                label=f"block_{i:03d}",
                start_utc=_utc(t0),
                end_utc_exclusive=_utc(t1),
            )
        )
    return blocks


def _fetch_partitioned_features(
    *,
    cfg: BqConfig,
    start_utc: datetime,
    end_utc: datetime,
    obscodes: list[str],
    max_rows_per_partition: int,
    max_partitions: int,
) -> tuple[DesignationOrbitWindowFeatures, int]:
    """
    Fetch `DesignationOrbitWindowFeatures` for a window using deterministic partitioning
    to respect `bq query` row limits.
    """
    partitions = 1
    while True:
        counts = [
            count_designation_orbit_features_for_window(
                cfg=cfg,
                start_utc=start_utc,
                end_utc=end_utc,
                obscodes=obscodes,
                partition_mod=partitions,
                partition_idx=i,
            )
            for i in range(partitions)
        ]
        if all(int(c) <= int(max_rows_per_partition) for c in counts):
            break
        if partitions >= int(max_partitions):
            raise RuntimeError(
                f"Exceeded max_partitions={max_partitions} while keeping partition counts <= {max_rows_per_partition}. "
                f"Max partition count was {max(counts) if counts else 0}."
            )
        partitions *= 2

    parts: list[DesignationOrbitWindowFeatures] = []
    for i, c in enumerate(counts):
        if int(c) == 0:
            continue
        parts.append(
            fetch_designation_orbit_features_for_window(
                cfg=cfg,
                start_utc=start_utc,
                end_utc=end_utc,
                obscodes=obscodes,
                max_rows=int(max_rows_per_partition),
                partition_mod=partitions,
                partition_idx=i,
            )
        )
    out = DesignationOrbitWindowFeatures.empty()
    for p in parts:
        out = qv.concatenate([out, p])
    return out, int(partitions)


def fetch_and_persist_features_for_blocks(
    *,
    subset_dir: Path,
    cfg: BqConfig,
    blocks: list[TimeBlock],
    obscodes: list[str] | None = None,
    max_rows_per_partition: int = 100000,
    max_partitions: int = 256,
) -> list[Path]:
    """
    Persist BQ features for multiple time blocks under:
      <subset_dir>/artifacts/long_span_blocks/<block.label>/
    """
    win = read_subset_window(subset_dir)
    win.artifacts_dir.mkdir(parents=True, exist_ok=True)
    stns = list(obscodes) if obscodes is not None else list(win.obscodes)
    if not stns:
        raise ValueError("obscodes cannot be empty")

    out_paths: list[Path] = []
    root = win.artifacts_dir / "long_span_blocks"
    root.mkdir(parents=True, exist_ok=True)

    for b in blocks:
        bdir = root / str(b.label)
        bdir.mkdir(parents=True, exist_ok=True)
        feats, partitions = _fetch_partitioned_features(
            cfg=cfg,
            start_utc=_utc(b.start_utc),
            end_utc=_utc(b.end_utc_exclusive),
            obscodes=stns,
            max_rows_per_partition=int(max_rows_per_partition),
            max_partitions=int(max_partitions),
        )
        out = bdir / "bq_designation_orbit_window_features.parquet"
        feats.to_parquet(str(out))
        meta = {
            "subset_dir": str(win.subset_dir),
            "block": {
                "label": str(b.label),
                "start_utc": _utc(b.start_utc).isoformat().replace("+00:00", "Z"),
                "end_utc_exclusive": _utc(b.end_utc_exclusive).isoformat().replace("+00:00", "Z"),
            },
            "obscodes": stns,
            "n_rows": int(len(feats)),
            "partitions_used": int(partitions),
            "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        (bdir / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
        out_paths.append(out)

    return out_paths


def select_designations_across_blocks(
    *,
    subset_dir: Path,
    blocks: list[TimeBlock],
    sampling_cfg: SamplingConfig,
    cfg: BqConfig,
    obscodes: list[str] | None = None,
    max_rows_per_partition: int = 100000,
    max_partitions: int = 256,
) -> Path:
    """
    Select designations across multiple blocks by running the existing per-window
    stratified selection independently per block, then de-duplicating designations.
    """
    win = read_subset_window(subset_dir)
    win.artifacts_dir.mkdir(parents=True, exist_ok=True)
    stns = list(obscodes) if obscodes is not None else list(win.obscodes)

    # Roughly split the budget across blocks.
    n_per = max(1, int(int(sampling_cfg.n_total) // max(1, len(blocks))))
    selected_all: list[SelectedDesignations] = []
    used: set[str] = set()

    for b in blocks:
        feats, _ = _fetch_partitioned_features(
            cfg=cfg,
            start_utc=_utc(b.start_utc),
            end_utc=_utc(b.end_utc_exclusive),
            obscodes=stns,
            max_rows_per_partition=int(max_rows_per_partition),
            max_partitions=int(max_partitions),
        )
        # Reuse the existing per-window selection logic.
        mid = 0.5 * (_utc(b.start_utc).timestamp() + _utc(b.end_utc_exclusive).timestamp())
        window_mid_mjd = (mid / 86400.0) + 40587.0  # unix->mjd
        sel_block = select_designations_from_features(
            features=feats,
            window_mid_mjd=float(window_mid_mjd),
            cfg=SamplingConfig(n_total=int(n_per), seed=int(sampling_cfg.seed)),
        )

        # De-dupe across blocks.
        keep_des: list[str] = []
        keep_str: list[str] = []
        keep_h: list[int] = []
        for d, s, h in zip(
            sel_block.designation.to_pylist(),
            sel_block.stratum.to_pylist(),
            sel_block.hash64.to_pylist(),
        ):
            ds = str(d)
            if ds in used:
                continue
            used.add(ds)
            keep_des.append(ds)
            keep_str.append(str(s))
            keep_h.append(int(h))

        selected_all.append(
            SelectedDesignations.from_kwargs(
                designation=keep_des, stratum=keep_str, hash64=keep_h
            )
        )

    out = SelectedDesignations.empty()
    for t in selected_all:
        out = qv.concatenate([out, t])

    out_path = win.artifacts_dir / "selected_designations_long_span.parquet"
    out.to_parquet(str(out_path))

    meta = {
        "subset_dir": str(win.subset_dir),
        "n_selected": int(len(out)),
        "n_blocks": int(len(blocks)),
        "blocks": [
            {
                "label": str(b.label),
                "start_utc": _utc(b.start_utc).isoformat().replace("+00:00", "Z"),
                "end_utc_exclusive": _utc(b.end_utc_exclusive).isoformat().replace("+00:00", "Z"),
            }
            for b in blocks
        ],
        "obscodes": stns,
        "sampling": {
            "n_total": int(sampling_cfg.n_total),
            "seed": int(sampling_cfg.seed),
        },
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    (win.artifacts_dir / "selected_designations_long_span_meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n"
    )

    return out_path

