from datetime import datetime, timezone

import pytest

from experiments.covariance_precovery.selection.bq_select import (
    BqConfig,
    estimate_bq_bytes,
    find_objects_observed_in_window,
)


@pytest.mark.xfail(reason="Requires BigQuery access and stable dataset; intended for manual runs.")
def test_bq_selection_smoke() -> None:
    cfg = BqConfig()
    start = datetime(2019, 8, 1, tzinfo=timezone.utc)
    end = datetime(2019, 8, 2, tzinfo=timezone.utc)
    obscodes = ["I41"]

    qbytes = estimate_bq_bytes(
        f"SELECT COUNT(1) AS n FROM `{cfg.obs_table}` WHERE stn='I41' LIMIT 1"
    )
    assert qbytes >= 0

    res = find_objects_observed_in_window(cfg=cfg, start_utc=start, end_utc=end, obscodes=obscodes, limit_per_stn=5)
    assert len(res) >= 0

