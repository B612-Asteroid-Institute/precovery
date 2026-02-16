from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import quivr as qv
from adam_core.orbits import Orbits

from precovery.precovery_db import PrecoveryDatabase
from precovery.search import SearchAgg, precover_orbit_with_metrics


@dataclass(frozen=True)
class Inputs:
    db_dir: Path
    orbits_parquet: Path
    bins_parquet: Path


def _read_bins(*, bins_parquet: Path) -> pa.Table:
    # We only need the stratum label + the orbit identifier used downstream.
    t = pq.read_table(
        str(bins_parquet), columns=["selected_designation", "stratum", "regime"]
    ).combine_chunks()
    keep = pc.and_(pc.is_valid(t["selected_designation"]), pc.is_valid(t["stratum"]))
    return t.filter(keep)


def _select_designations_round_robin(*, bins: pa.Table, n_orbits: int) -> list[str]:
    """
    Deterministically sample across `stratum` groups:
    - within each stratum, sort by selected_designation
    - round-robin across strata until we have `n_orbits`
    """
    desig = [str(x) for x in bins["selected_designation"].to_pylist()]
    stratum = [str(x) for x in bins["stratum"].to_pylist()]

    groups: dict[str, list[str]] = {}
    for d, s in zip(desig, stratum, strict=True):
        groups.setdefault(s, []).append(d)
    for s in list(groups.keys()):
        groups[s] = sorted(set(groups[s]))

    strata = sorted(groups.keys(), key=lambda k: (len(groups[k]), k))
    ptr: dict[str, int] = {k: 0 for k in strata}
    out: list[str] = []
    while len(out) < int(n_orbits):
        progressed = False
        for k in strata:
            p = ptr[k]
            if p >= len(groups[k]):
                continue
            out.append(groups[k][p])
            ptr[k] = p + 1
            progressed = True
            if len(out) >= int(n_orbits):
                break
        if not progressed:
            break
    return out


def _select_designations_balanced_by_regime(*, bins: pa.Table, n_orbits: int) -> list[str]:
    """
    Select `n_orbits` designations with broad coverage:
    - First, within each regime, select a long round-robin list across strata.
    - Then, round-robin across regimes to interleave them.
    """
    regimes = sorted({str(x) for x in bins["regime"].to_pylist() if x is not None})
    per_regime: dict[str, list[str]] = {}
    for r in regimes:
        b = bins.filter(pc.equal(bins["regime"], pa.scalar(r, type=pa.string())))
        # Oversample within regime so the outer interleave has enough depth.
        per_regime[r] = _select_designations_round_robin(bins=b, n_orbits=int(n_orbits))

    out: list[str] = []
    ptr = {r: 0 for r in regimes}
    while len(out) < int(n_orbits):
        progressed = False
        for r in regimes:
            p = ptr[r]
            if p >= len(per_regime[r]):
                continue
            d = per_regime[r][p]
            ptr[r] = p + 1
            out.append(d)
            progressed = True
            if len(out) >= int(n_orbits):
                break
        if not progressed:
            break
    return out


def _orbit_index_by_id(orbits: Orbits) -> dict[str, int]:
    m: dict[str, int] = {}
    for i, oid in enumerate(orbits.orbit_id.to_pylist()):
        if oid is None:
            continue
        m[str(oid)] = int(i)
    return m


def main() -> None:
    p = argparse.ArgumentParser(description="Profile production precovery implementation on experiment DB + stratified orbit sample.")
    p.add_argument(
        "--db-dir",
        type=str,
        default=str(
            Path(__file__).resolve().parents[1]
            / "local_db"
            / "full_precovery_n32"
        ),
        help="PrecoveryDatabase directory (contains index.db + data/ + config.json).",
    )
    p.add_argument(
        "--allow-version-mismatch",
        action="store_true",
        help="Allow loading DBs built with a different precovery version string.",
    )
    p.add_argument(
        "--orbits-parquet",
        type=str,
        default=str(
            Path(__file__).resolve().parents[1]
            / "local_db"
            / "full_precovery_n32"
            / "artifacts"
            / "w_20200101_20240101__I41_T05_T08_W84"
            / "orbits_selected_sbdb.parquet"
        ),
        help="adam_core Orbits parquet used for the experiments window.",
    )
    p.add_argument(
        "--bins-parquet",
        type=str,
        default=str(Path.cwd() / "handoff_mpcq_filtered" / "mpcq_orbits_with_bins.parquet"),
        help="MPCQ orbit parquet with computed `stratum` bins (selected_designation + stratum).",
    )
    p.add_argument("--n-orbits", type=int, default=24, help="Number of orbits to run (round-robin by stratum).")
    p.add_argument(
        "--balanced-regimes",
        action="store_true",
        help="Interleave selections across regimes (NEO/MCA/MBA/JupiterPlus/TNO/...).",
    )
    p.add_argument("--start-mjd", type=float, default=None, help="Search start MJD (defaults to DB bounds).")
    p.add_argument("--end-mjd", type=float, default=None, help="Search end MJD (defaults to DB bounds).")
    p.add_argument("--window-size-days", type=int, default=7)
    p.add_argument(
        "--propagation-strategy",
        type=str,
        default="assist_window_then_2body_variants:sigma_points",
    )
    p.add_argument("--n-sigma", type=float, default=3.0)
    p.add_argument("--target-chunk-size", type=int, default=10_000)
    p.add_argument("--max-processes", type=int, default=1)
    args = p.parse_args()

    inputs = Inputs(
        db_dir=Path(args.db_dir),
        orbits_parquet=Path(args.orbits_parquet),
        bins_parquet=Path(args.bins_parquet),
    )

    db = PrecoveryDatabase.from_dir(
        str(inputs.db_dir), mode="r", allow_version_mismatch=bool(args.allow_version_mismatch)
    )
    orbits = Orbits.from_parquet(str(inputs.orbits_parquet))

    bins = _read_bins(bins_parquet=inputs.bins_parquet)
    if bool(args.balanced_regimes):
        want = _select_designations_balanced_by_regime(bins=bins, n_orbits=int(args.n_orbits))
    else:
        want = _select_designations_round_robin(bins=bins, n_orbits=int(args.n_orbits))

    stratum_by_desig = dict(
        zip(
            bins["selected_designation"].to_pylist(),
            bins["stratum"].to_pylist(),
            strict=False,
        )
    )
    regime_by_desig = dict(
        zip(
            bins["selected_designation"].to_pylist(),
            bins["regime"].to_pylist(),
            strict=False,
        )
    )

    idx_by_id = _orbit_index_by_id(orbits)
    keep_idx: list[int] = []
    keep_id: list[str] = []
    keep_stratum: list[str] = []
    keep_regime: list[str] = []
    for d in want:
        i = idx_by_id.get(str(d))
        if i is None:
            continue
        keep_idx.append(int(i))
        keep_id.append(str(d))
        keep_stratum.append(str(stratum_by_desig.get(d, "unknown")))
        keep_regime.append(str(regime_by_desig.get(d, "unknown")))

    sub = orbits.take(keep_idx)
    start_mjd = args.start_mjd
    end_mjd = args.end_mjd

    run_tables: list[pa.Table] = []
    agg_total = SearchAgg()
    for oid, stratum, regime, orbit in zip(keep_id, keep_stratum, keep_regime, sub, strict=True):
        cands, frames, run, agg = precover_orbit_with_metrics(
            db=db,
            orbit=orbit,
            tolerance=None,
            start_mjd=start_mjd,
            end_mjd=end_mjd,
            datasets=None,
            window_size_days=int(args.window_size_days),
            propagation_strategy=str(args.propagation_strategy),
            footprint=None,
            n_sigma=float(args.n_sigma),
            target_chunk_size=int(args.target_chunk_size),
            max_processes=int(args.max_processes),
        )

        # accumulate totals
        agg_total.n_targets += agg.n_targets
        agg_total.n_predicted_rows += agg.n_predicted_rows
        agg_total.n_predicted_pixels += agg.n_predicted_pixels
        agg_total.n_frames_joined += agg.n_frames_joined
        agg_total.n_frames_loaded += agg.n_frames_loaded
        agg_total.n_frames_faint_skipped += agg.n_frames_faint_skipped
        agg_total.n_observations_loaded += agg.n_observations_loaded
        agg_total.n_accepted += agg.n_accepted
        agg_total.n_candidates += agg.n_candidates
        agg_total.enum_sec += agg.enum_sec
        agg_total.propagate_sec += agg.propagate_sec
        agg_total.footprint_sec += agg.footprint_sec
        agg_total.join_frames_sec += agg.join_frames_sec
        agg_total.io_sec += agg.io_sec
        agg_total.filter_sec += agg.filter_sec
        agg_total.photometry_sec += agg.photometry_sec
        agg_total.total_sec += agg.total_sec

        r = run.table
        # add stratum + quick counts for printing
        r = r.append_column("regime", pa.array([str(regime)], type=pa.large_string()))
        r = r.append_column("stratum", pa.array([str(stratum)], type=pa.large_string()))
        r = r.append_column("n_candidates_returned", pa.array([int(len(cands))], type=pa.int64()))
        r = r.append_column("n_frames_returned", pa.array([int(len(frames))], type=pa.int64()))
        run_tables.append(r)

        print(
            f"orbit_id={oid} regime={regime} stratum={stratum} "
            f"targets={agg.n_targets} frames_joined={agg.n_frames_joined} obs_loaded={agg.n_observations_loaded} "
            f"sec(total={agg.total_sec:.3f}, prop={agg.propagate_sec:.3f}, fp={agg.footprint_sec:.3f}, "
            f"join={agg.join_frames_sec:.3f}, io={agg.io_sec:.3f}, filter={agg.filter_sec:.3f}, photo={agg.photometry_sec:.3f})"
        )

    if run_tables:
        out = pa.concat_tables(run_tables).combine_chunks()
        # sort by total_sec desc for quick hotspot identification
        order = pc.sort_indices(out["total_sec"], sort_keys=[("total_sec", "descending")])
        out2 = out.take(order)
        print("\n=== top by total_sec ===")
        for rec in out2.to_pylist()[: min(10, out2.num_rows)]:
            print(
                f"{rec.get('orbit_id')} {rec.get('stratum')} total={rec.get('total_sec'):.3f} "
                f"prop={rec.get('propagate_sec'):.3f} io={rec.get('io_sec'):.3f} filter={rec.get('filter_sec'):.3f}"
            )

    print("\n=== totals ===")
    print(
        f"n_orbits={len(keep_id)} "
        f"targets={agg_total.n_targets} frames_joined={agg_total.n_frames_joined} obs_loaded={agg_total.n_observations_loaded} "
        f"sec(total={agg_total.total_sec:.3f}, prop={agg_total.propagate_sec:.3f}, fp={agg_total.footprint_sec:.3f}, "
        f"join={agg_total.join_frames_sec:.3f}, io={agg_total.io_sec:.3f}, filter={agg_total.filter_sec:.3f}, "
        f"photo={agg_total.photometry_sec:.3f})"
    )


if __name__ == "__main__":
    main()

