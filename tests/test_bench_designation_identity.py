from __future__ import annotations

import pyarrow as pa

from bench.data.bundles import _map_truth_to_orbit_ids
from bench.selection.fetch_mpcq_handoff import _build_designation_mapping_table


def _designation_resolution_table() -> pa.Table:
    return pa.table(
        {
            "selected_designation": pa.array(["2021", "2021 OU", "A"], type=pa.large_string()),
            "permid_norm": pa.array([None, None, None], type=pa.large_string()),
            "primary_provid": pa.array([None, None, None], type=pa.large_string()),
            "secondary_provids": pa.array(
                [[], ["K21O00U"], ["A-1"]],
                type=pa.list_(pa.large_string()),
            ),
            "mpcq_request_provids": pa.array(
                [["2021"], ["2021 OU", "K21O00U"], ["A", "A-1"]],
                type=pa.list_(pa.large_string()),
            ),
            "resolution_status": pa.array(
                ["resolved", "resolved", "resolved"],
                type=pa.large_string(),
            ),
        }
    )


def test_map_truth_to_orbit_ids_keeps_provisional_and_numbered_distinct() -> None:
    truth = pa.table(
        {
            "designation": pa.array(["2021", "2021 OU", "K21O00U", "MISSING"], type=pa.large_string()),
            "obscode": pa.array(["I41", "I41", "I41", "I41"], type=pa.large_string()),
            "truth_obsid": pa.array(["o1", "o2", "o3", "o4"], type=pa.large_string()),
            "truth_time_mjd_utc": pa.array([60000.1, 60000.2, 60000.3, 60000.4], type=pa.float64()),
            "match_observation_id": pa.array(["m1", "m2", "m3", "m4"], type=pa.large_string()),
            "match_time_mjd_utc": pa.array([60000.1, 60000.2, 60000.3, 60000.4], type=pa.float64()),
            "delta_t_sec": pa.array([0.1, 0.1, 0.1, 0.1], type=pa.float64()),
            "distance_arcsec": pa.array([0.2, 0.2, 0.2, 0.2], type=pa.float64()),
        }
    )
    selected_orbit_mapping = pa.table(
        {
            "selected_designation": pa.array(["2021", "2021 OU", "A"], type=pa.large_string()),
            "orbit_id": pa.array(["2021", "2021 OU", "A"], type=pa.large_string()),
        }
    )

    mapped, unresolved = _map_truth_to_orbit_ids(
        truth_window=truth,
        selected_orbit_mapping=selected_orbit_mapping,
        designation_resolution=_designation_resolution_table(),
    )

    mapped_ids = mapped["orbit_id"].to_pylist()
    assert mapped_ids == ["2021", "2021 OU", "2021 OU"]
    assert unresolved.num_rows == 1
    assert unresolved["designation"][0].as_py() == "MISSING"
    assert unresolved["unresolved_reason"][0].as_py() == "designation_not_in_selected_aliases"
    assert unresolved["n_truth_rows"][0].as_py() == 1


def test_build_designation_mapping_table_expands_aliases_and_keeps_unresolved() -> None:
    mapping = _build_designation_mapping_table(
        designations=["A", "B"],
        strata=["neo|arc|dt|u|i", "mba|arc|dt|u|i"],
        designation_resolution=pa.table(
            {
                "selected_designation": pa.array(["A"], type=pa.string()),
                "resolution_status": pa.array(["resolved"], type=pa.string()),
                "mpcq_request_provids": pa.array([["A", "A-1"]], type=pa.list_(pa.string())),
            }
        ),
        alias_request_mode="all_linked",
    )
    rows = list(
        zip(
            mapping["selected_designation"].to_pylist(),
            mapping["requested_provid"].to_pylist(),
            mapping["resolution_status"].to_pylist(),
        )
    )
    assert ("A", "A", "resolved") in rows
    assert ("A", "A-1", "resolved") in rows
    assert ("B", "B", "unresolved") in rows
    assert len(rows) == 3


def test_build_designation_mapping_table_selected_only_mode() -> None:
    mapping = _build_designation_mapping_table(
        designations=["A"],
        strata=None,
        designation_resolution=pa.table(
            {
                "selected_designation": pa.array(["A"], type=pa.string()),
                "resolution_status": pa.array(["resolved"], type=pa.string()),
                "mpcq_request_provids": pa.array([["A", "A-1"]], type=pa.list_(pa.string())),
            }
        ),
        alias_request_mode="selected_only",
    )
    assert mapping.num_rows == 1
    assert mapping["selected_designation"][0].as_py() == "A"
    assert mapping["requested_provid"][0].as_py() == "A"
