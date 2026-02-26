from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentPaths:
    root: Path

    @property
    def local_db(self) -> Path:
        return self.root / "local_db"

    @property
    def results(self) -> Path:
        return self.root / "results"

    @property
    def tmp(self) -> Path:
        return self.root / "tmp"

