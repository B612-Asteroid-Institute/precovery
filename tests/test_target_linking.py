from __future__ import annotations

import numpy as np

from adam_core.time import Timestamp

from precovery.search.pipeline_types import BenchTargets
from precovery.search.target_link import map_times_to_target_idx_by_obscode


def test_map_times_to_target_idx_exact_key_us_join() -> None:
    targets = BenchTargets.from_kwargs(
        obscode=["I41", "I41", "T05"],
        exposure_mjd_mid_utc=[60000.0, 60000.1, 60000.2],
        exposure_mjd_mid_key_us=[
            int(round(60000.0 * 86400 * 1e6)),
            int(round(60000.1 * 86400 * 1e6)),
            int(round(60000.2 * 86400 * 1e6)),
        ],
        filter=["r", "r", "o"],
    )

    times = Timestamp.from_mjd([60000.0, 60000.1, 60000.2], scale="utc")
    idx = map_times_to_target_idx_by_obscode(
        obscode=np.array(["I41", "I41", "T05"], dtype=object),
        time_utc=times,
        targets=targets,
        dt_sec=0.0,
    )
    assert idx.tolist() == [0, 1, 2]

