from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

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
    provids = [str(x) for x in selected.designation.to_pylist()]

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

