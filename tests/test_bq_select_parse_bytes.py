from __future__ import annotations

from experiments.covariance_precovery.selection.bq_select import _parse_bq_bytes_processed


def test_parse_bq_bytes_processed_plain() -> None:
    txt = "Query successfully validated. Assuming the tables are not modified, running this query will process 12345 bytes of data."
    assert _parse_bq_bytes_processed(txt) == 12345


def test_parse_bq_bytes_processed_upper_bound() -> None:
    txt = "Query successfully validated. Assuming the tables are not modified, running this query will process upper bound of 2,637,618,380 bytes of data."
    assert _parse_bq_bytes_processed(txt) == 2637618380


def test_parse_bq_bytes_processed_missing() -> None:
    assert _parse_bq_bytes_processed("no bytes here") == 0

