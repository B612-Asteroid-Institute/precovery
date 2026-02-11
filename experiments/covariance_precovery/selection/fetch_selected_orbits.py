from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq

from adam_core.orbits import Orbits
from mpcq.client import BigQueryMPCClient
from mpcq.orbits import MPCOrbits

from .bq_select import BqConfig
from .subset_designations import read_subset_window
from .subset_sampling import SelectedDesignations


@dataclass(frozen=True)
class OrbitFetchResult:
    mpcq_orbits_parquet: Path
    orbits_parquet: Path
    meta_json: Path


def _resolve_mpcq_provids_for_selected_designations(
    *,
    selected_designations_parquet: Path,
    requested_designations: list[str],
) -> tuple[list[str], dict[str, object]]:
    """
    mpcq's query_orbits/query_observations are defined for provisional designations.

    Our selected_designations.parquet often stores designation keys like COALESCE(permid, provid),
    so we map those to the primary provisional designation using a local features parquet if present.
    """
    artifacts_dir = selected_designations_parquet.parent
    candidates = [
        artifacts_dir / "selected_designation_features.parquet",
        artifacts_dir / "bq_designation_orbit_window_features.parquet",
        artifacts_dir / "bq_designation_orbit_features.parquet",
    ]
    features_path = next((p for p in candidates if p.exists()), None)
    if features_path is None:
        return requested_designations, {
            "mpcq_provids_source": None,
            "n_requested": len(requested_designations),
            "n_mapped": 0,
            "n_missing_primary_provid": len(requested_designations),
        }

    feat = pq.read_table(
        features_path,
        columns=["designation", "unpacked_primary_provisional_designation"],
    ).combine_chunks()
    keep = pc.and_(pc.is_valid(feat.column("designation")), pc.is_valid(feat.column("unpacked_primary_provisional_designation")))
    feat_kept = feat.filter(keep)

    mapping: dict[str, str] = {}
    for d, p in zip(
        feat_kept.column("designation").to_pylist(),
        feat_kept.column("unpacked_primary_provisional_designation").to_pylist(),
    ):
        if d is None or p is None:
            continue
        mapping[str(d)] = str(p)

    out: list[str] = []
    n_mapped = 0
    n_missing = 0
    for d in requested_designations:
        mp = mapping.get(d)
        if mp is None:
            out.append(d)
            n_missing += 1
            continue
        out.append(mp)
        if mp != d:
            n_mapped += 1

    return out, {
        "mpcq_provids_source": str(features_path),
        "n_requested": len(requested_designations),
        "n_mapped": int(n_mapped),
        "n_missing_primary_provid": int(n_missing),
    }


def fetch_selected_orbits_for_subset(
    *,
    subset_dir: Path,
    selected_designations_parquet: Path | None = None,
    cfg: BqConfig | None = None,
) -> OrbitFetchResult:
    """
    Fetch orbits for the selected designations using MPCQ's BigQuery client and persist:
    - the raw `mpcq.orbits.MPCOrbits` table
    - the converted `adam_core.orbits.Orbits` table
    """
    if cfg is None:
        cfg = BqConfig()
    win = read_subset_window(subset_dir)
    win.artifacts_dir.mkdir(parents=True, exist_ok=True)

    if selected_designations_parquet is None:
        selected_designations_parquet = win.artifacts_dir / "selected_designations.parquet"
    selected = SelectedDesignations.from_parquet(str(selected_designations_parquet))
    requested_designations = [str(x) for x in selected.designation.to_pylist()]
    provids, mpcq_provids_meta = _resolve_mpcq_provids_for_selected_designations(
        selected_designations_parquet=Path(selected_designations_parquet),
        requested_designations=requested_designations,
    )

    client = BigQueryMPCClient(dataset_id=cfg.dataset_id, views_dataset_id=cfg.views_dataset_id)
    mpc_orbits = client.query_orbits(provids)

    out_mpcq = win.artifacts_dir / "mpcq_orbits_selected.parquet"
    mpc_orbits.to_parquet(str(out_mpcq))

    # Convert exactly as `mpcq` defines it (see `mpcq.orbits.MPCOrbits.orbits()`).
    orbits: Orbits = mpc_orbits.orbits()
    out_orbits = win.artifacts_dir / "orbits_selected.parquet"
    orbits.to_parquet(str(out_orbits))

    meta = {
        "subset_dir": str(win.subset_dir),
        "selected_designations_parquet": str(selected_designations_parquet),
        "n_selected_designations": int(len(selected)),
        "mpcq_provids_meta": mpcq_provids_meta,
        "n_orbits_rows_returned": int(len(mpc_orbits)),
        "n_orbits_converted": int(len(orbits)),
        "dataset_id": cfg.dataset_id,
        "views_dataset_id": cfg.views_dataset_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    out_meta = win.artifacts_dir / "orbits_selected_meta.json"
    out_meta.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")

    return OrbitFetchResult(
        mpcq_orbits_parquet=out_mpcq,
        orbits_parquet=out_orbits,
        meta_json=out_meta,
    )


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Fetch and persist adam_core Orbits for selected designations.")
    p.add_argument("--subset-dir", type=str, required=True)
    args = p.parse_args()

    out = fetch_selected_orbits_for_subset(subset_dir=Path(args.subset_dir))
    print(f"mpcq_orbits_parquet={out.mpcq_orbits_parquet}")
    print(f"orbits_parquet={out.orbits_parquet}")
    print(f"meta_json={out.meta_json}")


if __name__ == "__main__":
    main()

