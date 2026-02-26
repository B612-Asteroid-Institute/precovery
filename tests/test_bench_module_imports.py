from __future__ import annotations

import importlib
import pkgutil

import bench


def test_import_all_bench_modules() -> None:
    failures: list[str] = []
    for modinfo in pkgutil.walk_packages(bench.__path__, prefix="bench."):
        name = str(modinfo.name)
        try:
            importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
    assert not failures, "bench module import failures:\n" + "\n".join(sorted(failures))
