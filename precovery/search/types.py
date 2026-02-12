from __future__ import annotations

import quivr as qv
from adam_core.time import Timestamp


class FrameTimeTargets(qv.Table):
    """
    Distinct per-frame propagation targets.

    We intentionally carry both:
    - `exposure_mjd_mid`: the exact float stored in sqlite (used for equality joins)
    - `time`: a Timestamp view of that float (used for pairing with adam-core APIs)
    """

    obscode = qv.LargeStringColumn()
    exposure_mjd_mid = qv.Float64Column()
    time = Timestamp.as_column()


class TargetPixels(qv.Table):
    """
    Exploded (obscode, exposure_mjd_mid, healpixel) triples used to join against `frames`.
    """

    obscode = qv.LargeStringColumn()
    exposure_mjd_mid = qv.Float64Column()
    healpixel = qv.Int64Column()

