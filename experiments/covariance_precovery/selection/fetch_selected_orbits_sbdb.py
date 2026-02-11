from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import quivr as qv
from adam_core.orbits import Orbits
from adam_core.orbits.query.sbdb import query_sbdb_new

from ..selection.subset_designations import read_subset_window
from ..selection.subset_sampling import SelectedDesignations
from .designation_normalization import normalize_designation


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
    artifacts_dir: Path | None = None,
    batch_size: int = 25,
) -> SbdbOrbitFetchResult:
    win = read_subset_window(subset_dir)
    out_dir = Path(artifacts_dir).expanduser().resolve() if artifacts_dir is not None else win.artifacts_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if selected_designations_parquet is None:
        selected_designations_parquet = win.artifacts_dir / "selected_designations.parquet"
    selected = SelectedDesignations.from_parquet(str(selected_designations_parquet))
    # Canonicalize input IDs so:
    # - they match truth `designation` strings, and
    # - they can be used as stable `orbit_id` values via `orbit_id_from_input=True`.
    designations = [normalize_designation(str(x)) for x in selected.designation.to_pylist()]

    out_orbits = Orbits.empty()
    failures = OrbitFetchFailures.empty()

    for i0 in range(0, len(designations), int(batch_size)):
        batch = designations[i0 : i0 + int(batch_size)]
        try:
            o = query_sbdb_new(batch, allow_missing=True, orbit_id_from_input=True)
            out_orbits = qv.concatenate([out_orbits, o])

            # Record missing IDs as failures (query returned only those SBDB resolved).
            got = set(str(x) for x in o.orbit_id.to_pylist())
            missing = [d for d in batch if d not in got]
            for d in missing:
                failures = qv.concatenate(
                    [
                        failures,
                        OrbitFetchFailures.from_kwargs(
                            designation=[d],
                            error=["NotFoundError: object was not found"],
                        ),
                    ]
                )
        except Exception:  # noqa: BLE001
            # Fall back to per-designation so one bad name doesn't poison the batch.
            for d in batch:
                try:
                    o1 = query_sbdb_new([d], allow_missing=False, orbit_id_from_input=True)
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

    # De-duplicate just in case SBDB returns duplicates across queries.
    if len(out_orbits) > 0:
        out_orbits = out_orbits.drop_duplicates(subset=["orbit_id"])

    out_orbits_path = out_dir / "orbits_selected_sbdb.parquet"
    out_orbits.to_parquet(str(out_orbits_path))

    failures_path = out_dir / "orbits_selected_sbdb_failures.parquet"
    failures.to_parquet(str(failures_path))

    meta = {
        "subset_dir": str(subset_dir),
        "artifacts_dir": str(out_dir),
        "selected_designations_parquet": str(selected_designations_parquet),
        "n_selected": int(len(selected)),
        "n_orbits_returned": int(len(out_orbits)),
        "n_failures": int(len(failures)),
        "batch_size": int(batch_size),
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    meta_path = out_dir / "orbits_selected_sbdb_meta.json"
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
    p.add_argument(
        "--selected-designations-parquet",
        type=str,
        default=None,
        help=(
            "Optional path to selected designations parquet. "
            "Defaults to <subset_dir>/artifacts/selected_designations.parquet."
        ),
    )
    args = p.parse_args()

    out = fetch_selected_orbits_via_sbdb(
        subset_dir=Path(args.subset_dir),
        selected_designations_parquet=(
            None
            if args.selected_designations_parquet is None
            else Path(args.selected_designations_parquet).expanduser().resolve()
        ),
        batch_size=int(args.batch_size),
    )
    print(f"orbits_parquet={out.orbits_parquet}")
    print(f"failures_parquet={out.failures_parquet}")
    print(f"meta_json={out.meta_json}")


if __name__ == "__main__":
    main()

