from __future__ import annotations

import os
import platform
import subprocess
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RamChunkMeta:
    system_ram_bytes: int | None
    ram_budget_frac: float
    max_usage_bytes: int | None
    effective_bytes_per_row: float
    max_rows_fit: int
    resolved_time_chunk_size: int


def system_total_ram_bytes() -> int:
    """
    Best-effort total physical RAM size in bytes.
    """
    try:
        if platform.system() == "Darwin":
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
            return int(out)
        if platform.system() == "Linux":
            with open("/proc/meminfo", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        parts = line.split()
                        return int(parts[1]) * 1024  # kB -> bytes
    except Exception:
        pass
    try:
        if hasattr(os, "sysconf"):
            pages = int(os.sysconf("SC_PHYS_PAGES"))
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            return int(pages * page_size)
    except Exception:
        pass
    raise RuntimeError("Could not determine total system RAM size")


def resolve_time_chunk_size_from_ram_budget(
    *,
    n_time_targets: int,
    rows_per_time_target: int,
    ram_budget_frac: float = 0.10,
    effective_bytes_per_row: float = 200.0,
    min_chunk: int = 1,
) -> tuple[int, RamChunkMeta]:
    """
    Resolve a time-chunk size (targets per chunk) that respects a fixed RAM budget.

    We treat `rows_per_time_target` as the expansion factor (e.g., K variants per target).
    """
    n_time_targets = int(n_time_targets)
    rows_per_time_target = int(rows_per_time_target)
    if n_time_targets <= 0:
        return 0, RamChunkMeta(
            system_ram_bytes=None,
            ram_budget_frac=float(ram_budget_frac),
            max_usage_bytes=None,
            effective_bytes_per_row=float(effective_bytes_per_row),
            max_rows_fit=0,
            resolved_time_chunk_size=0,
        )
    if rows_per_time_target <= 0:
        raise ValueError("rows_per_time_target must be > 0")
    if not np.isfinite(float(ram_budget_frac)) or float(ram_budget_frac) <= 0.0:
        raise ValueError("ram_budget_frac must be finite and > 0")
    bpr = float(effective_bytes_per_row)
    if not np.isfinite(bpr) or bpr <= 0.0:
        raise ValueError("effective_bytes_per_row must be finite and > 0")
    min_chunk = int(min_chunk)
    if min_chunk <= 0:
        min_chunk = 1

    try:
        ram = int(system_total_ram_bytes())
        max_usage = int(float(ram_budget_frac) * float(ram))
        max_rows_fit = int(max_usage // int(bpr))
        chunk = int(max_rows_fit // rows_per_time_target)
        chunk = int(max(chunk, min_chunk))
        chunk = int(min(chunk, n_time_targets))
        return chunk, RamChunkMeta(
            system_ram_bytes=int(ram),
            ram_budget_frac=float(ram_budget_frac),
            max_usage_bytes=int(max_usage),
            effective_bytes_per_row=float(bpr),
            max_rows_fit=int(max_rows_fit),
            resolved_time_chunk_size=int(chunk),
        )
    except Exception:
        # Fallback: conservative minimum.
        return int(min_chunk), RamChunkMeta(
            system_ram_bytes=None,
            ram_budget_frac=float(ram_budget_frac),
            max_usage_bytes=None,
            effective_bytes_per_row=float(bpr),
            max_rows_fit=0,
            resolved_time_chunk_size=int(min_chunk),
        )

