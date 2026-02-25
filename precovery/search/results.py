from __future__ import annotations

import quivr as qv


class AcceptedDetections(qv.Table):
    """
    Accepted detection rows after Stage 4 gating.

    This is intentionally smaller than legacy `PrecoveryCandidates`: it preserves the
    detection identity + astrometry/photometry needed for downstream analysis and truth
    scoring, without tying results to a specific storage backend.
    """

    orbit_id = qv.LargeStringColumn()
    target_idx = qv.Int64Column()

    obscode = qv.LargeStringColumn()
    exposure_mjd_mid_key_us = qv.Int64Column()
    healpixel = qv.Int64Column()
    filter = qv.LargeStringColumn(nullable=True)

    observation_id = qv.LargeStringColumn()
    obstime_mjd_utc = qv.Float64Column()
    ra_deg = qv.Float64Column()
    dec_deg = qv.Float64Column()
    ra_sigma_deg = qv.Float64Column()
    dec_sigma_deg = qv.Float64Column()
    mag = qv.Float64Column(nullable=True)
    mag_sigma = qv.Float64Column(nullable=True)

