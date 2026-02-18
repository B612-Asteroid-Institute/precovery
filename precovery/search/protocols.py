from __future__ import annotations

from typing import Protocol, overload

from adam_core.orbits import Orbits

from precovery.config import Config
from precovery.frame_db import FrameDB
from precovery.precovery_db import FrameCandidates, PrecoveryCandidates


class SearchDB(Protocol):
    """
    Minimal interface required by the performance-first `precovery.search` pipeline.

    This intentionally does *not* expose the full `PrecoveryDatabase` surface area;
    stages depend only on what they actually use so profiling/optimization can proceed
    with clear boundaries.
    """

    frames: FrameDB
    config: Config
    directory: str

    def _refresh_limiting_magnitudes_cache_if_needed(self) -> None: ...  # noqa: SLF001

