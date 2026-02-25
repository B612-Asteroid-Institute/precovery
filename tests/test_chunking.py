from __future__ import annotations

from precovery.search.chunking import resolve_time_chunk_size_from_ram_budget


def test_resolve_time_chunk_size_from_ram_budget_monkeypatched(monkeypatch) -> None:
    # Fix system RAM so the computation is deterministic.
    monkeypatch.setattr("precovery.search.chunking.system_total_ram_bytes", lambda: 1_000_000_000)

    # 10% budget => 100,000,000 bytes.
    # 200 bytes/row => 500,000 rows fit.
    # rows_per_time_target=100 => 5,000 targets per chunk.
    chunk, meta = resolve_time_chunk_size_from_ram_budget(
        n_time_targets=50_000,
        rows_per_time_target=100,
        ram_budget_frac=0.10,
        effective_bytes_per_row=200.0,
        min_chunk=1,
    )
    assert chunk == 5000
    assert meta.system_ram_bytes == 1_000_000_000
    assert meta.resolved_time_chunk_size == 5000

