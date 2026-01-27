from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import quivr as qv
from adam_core.orbits import Orbits
from adam_core.orbits.query import query_sbdb

from ..selection.subset_designations import read_subset_window
from ..selection.subset_sampling import SelectedDesignations


class OrbitFetchFailures(qv.Table):
    designation = qv.LargeStringColumn()
    error = qv.LargeStringColumn()


@dataclass(frozen=True)
class SbdbOrbitFetchResult:
    orbits_parquet: Path
    failures_parquet: Path
    meta_json: Path


def fetch_selected_orbits_via_sbdb(
    *,
    subset_dir: Path,
    selected_designations_parquet: Path | None = None,
    batch_size: int = 25,
) -> SbdbOrbitFetchResult:
    win = read_subset_window(subset_dir)
    win.artifacts_dir.mkdir(parents=True, exist_ok=True)

    if selected_designations_parquet is None:
        selected_designations_parquet = win.artifacts_dir / "selected_designations.parquet"
    selected = SelectedDesignations.from_parquet(str(selected_designations_parquet))
    designations = [str(x) for x in selected.designation.to_pylist()]

    out_orbits = Orbits.empty()
    failures = OrbitFetchFailures.empty()

    for i0 in range(0, len(designations), int(batch_size)):
        batch = designations[i0 : i0 + int(batch_size)]
        try:
            o = query_sbdb(batch)
            out_orbits = qv.concatenate([out_orbits, o])
        except Exception as e:  # noqa: BLE001
            # Fall back to per-designation so one bad name doesn't poison the batch.
            for d in batch:
                try:
                    o1 = query_sbdb([d])
                    out_orbits = qv.concatenate([out_orbits, o1])
                except Exception as e1:  # noqa: BLE001
                    failures = qv.concatenate(
                        [
                            failures,
                            OrbitFetchFailures.from_kwargs(
                                designation=[d],
                                error=[f"{type(e1).__name__}: {e1}"],
                            ),
                        ]
                    )

    out_orbits_path = win.artifacts_dir / "orbits_selected_sbdb.parquet"
    out_orbits.to_parquet(str(out_orbits_path))

    failures_path = win.artifacts_dir / "orbits_selected_sbdb_failures.parquet"
    failures.to_parquet(str(failures_path))

    meta = {
        "subset_dir": str(subset_dir),
        "selected_designations_parquet": str(selected_designations_parquet),
        "n_selected": int(len(selected)),
        "n_orbits_returned": int(len(out_orbits)),
        "n_failures": int(len(failures)),
        "batch_size": int(batch_size),
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    meta_path = win.artifacts_dir / "orbits_selected_sbdb_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")

    return SbdbOrbitFetchResult(
        orbits_parquet=out_orbits_path,
        failures_parquet=failures_path,
        meta_json=meta_path,
    )


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Fetch adam_core Orbits for selected designations via SBDB.")
    p.add_argument("--subset-dir", type=str, required=True)
    p.add_argument("--batch-size", type=int, default=25)
    args = p.parse_args()

    out = fetch_selected_orbits_via_sbdb(
        subset_dir=Path(args.subset_dir),
        batch_size=int(args.batch_size),
    )
    print(f"orbits_parquet={out.orbits_parquet}")
    print(f"failures_parquet={out.failures_parquet}")
    print(f"meta_json={out.meta_json}")


if __name__ == "__main__":
    main()

