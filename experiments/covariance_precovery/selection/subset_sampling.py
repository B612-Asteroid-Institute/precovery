from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import quivr as qv

from .bq_select import (
    BqConfig,
    DesignationOrbitWindowFeatures,
    count_designation_orbit_features_for_window,
    fetch_designation_orbit_features_for_window,
)
from .subset_designations import read_subset_window


@dataclass(frozen=True)
class SamplingConfig:
    n_total: int = 200
    seed: int = 0

    # thresholds chosen to be simple + interpretable; we can tune later from distributions
    arc_bins_days: tuple[float, float, float] = (7.0, 30.0, 180.0)
    dt_bins_days: tuple[float, float, float] = (30.0, 180.0, 1000.0)

    hi_i_deg: float = 30.0


class SelectedDesignations(qv.Table):
    designation = qv.LargeStringColumn()
    stratum = qv.LargeStringColumn()
    hash64 = qv.Int64Column()


def _hash64(designation: str, *, seed: int) -> int:
    # stable across platforms/runs
    h = hashlib.blake2b(digest_size=8, person=str(int(seed)).encode("utf-8"))
    h.update(designation.encode("utf-8"))
    u = int.from_bytes(h.digest(), byteorder="big", signed=False)
    # Store as *signed* int64 for Arrow/Quivr compatibility.
    return u if u < (1 << 63) else u - (1 << 64)


def _bins_label(v: float | None, edges: Iterable[float], *, prefix: str) -> str:
    if v is None or not np.isfinite(float(v)):
        return f"{prefix}_unknown"
    x = float(v)
    e = list(float(a) for a in edges)
    if x < e[0]:
        return f"{prefix}_lt_{e[0]:g}"
    if x < e[1]:
        return f"{prefix}_{e[0]:g}_{e[1]:g}"
    if x < e[2]:
        return f"{prefix}_{e[1]:g}_{e[2]:g}"
    return f"{prefix}_ge_{e[2]:g}"


def _regime_label(*, a: float | None, e: float | None, q: float | None) -> str:
    # prefer q if present; otherwise compute q=a(1-e) when possible
    qv_ = None
    if q is not None and np.isfinite(float(q)):
        qv_ = float(q)
    elif a is not None and e is not None and np.isfinite(float(a)) and np.isfinite(float(e)):
        qv_ = float(a) * (1.0 - float(e))

    av_ = float(a) if a is not None and np.isfinite(float(a)) else None

    if qv_ is not None:
        if qv_ < 1.3:
            return "NEO"
        if qv_ < 1.666:
            return "MCA"
    if av_ is not None:
        if av_ >= 30.0:
            return "TNO"
        if av_ >= 5.0:
            return "JupiterPlus"
        if av_ >= 1.666:
            return "MBA"
    return "Unknown"


def _u_label(u: int | None) -> str:
    if u is None:
        return "u_unknown"
    ui = int(u)
    if ui <= 2:
        return "u_0_2"
    if ui <= 5:
        return "u_3_5"
    return "u_6p"


def _row_stratum(
    *,
    cfg: SamplingConfig,
    window_mid_mjd: float,
    a: float | None,
    e: float | None,
    i: float | None,
    q: float | None,
    epoch_mjd: float | None,
    arc_days: float | None,
    u_param: int | None,
) -> str:
    regime = _regime_label(a=a, e=e, q=q)
    arc = _bins_label(arc_days, cfg.arc_bins_days, prefix="arc")

    dt_days = None
    if epoch_mjd is not None and np.isfinite(float(epoch_mjd)):
        dt_days = abs(float(epoch_mjd) - float(window_mid_mjd))
    dt = _bins_label(dt_days, cfg.dt_bins_days, prefix="dt")

    ui = _u_label(u_param)
    hi = "i_hi" if (i is not None and np.isfinite(float(i)) and float(i) >= cfg.hi_i_deg) else "i_norm"
    return "|".join([regime, arc, dt, ui, hi])


def _round_robin_sample(groups: dict[str, list[tuple[int, int]]], n_total: int) -> list[int]:
    """
    groups[stratum] = list[(hash64, row_idx)] sorted by hash64.
    """
    strata = sorted(groups.keys(), key=lambda k: (len(groups[k]), k))
    selected: list[int] = []
    ptrs = {k: 0 for k in strata}
    while len(selected) < int(n_total):
        progressed = False
        for k in strata:
            p = ptrs[k]
            if p >= len(groups[k]):
                continue
            _, idx = groups[k][p]
            ptrs[k] = p + 1
            selected.append(int(idx))
            progressed = True
            if len(selected) >= int(n_total):
                break
        if not progressed:
            break
    return selected


def _concat_features(parts: list[DesignationOrbitWindowFeatures]) -> DesignationOrbitWindowFeatures:
    out = DesignationOrbitWindowFeatures.empty()
    for p in parts:
        out = qv.concatenate([out, p])
    return out


def fetch_and_persist_window_features_for_subset(
    *,
    subset_dir: Path,
    cfg: BqConfig,
    max_rows_per_partition: int = 100000,
    max_partitions: int = 256,
) -> Path:
    win = read_subset_window(subset_dir)
    win.artifacts_dir.mkdir(parents=True, exist_ok=True)

    partitions = 1
    while True:
        counts = [
            count_designation_orbit_features_for_window(
                cfg=cfg,
                start_utc=win.start_utc,
                end_utc=win.end_utc_exclusive,
                obscodes=list(win.obscodes),
                partition_mod=partitions,
                partition_idx=i,
            )
            for i in range(partitions)
        ]
        if all(c <= int(max_rows_per_partition) for c in counts):
            break
        if partitions >= int(max_partitions):
            raise RuntimeError(
                f"Exceeded max_partitions={max_partitions} while keeping partition counts <= {max_rows_per_partition}. "
                f"Max partition count was {max(counts) if counts else 0}."
            )
        partitions *= 2

    parts: list[DesignationOrbitWindowFeatures] = []
    for i, c in enumerate(counts):
        if c == 0:
            continue
        parts.append(
            fetch_designation_orbit_features_for_window(
                cfg=cfg,
                start_utc=win.start_utc,
                end_utc=win.end_utc_exclusive,
                obscodes=list(win.obscodes),
                max_rows=int(max_rows_per_partition),
                partition_mod=partitions,
                partition_idx=i,
            )
        )

    features = _concat_features(parts)
    out_path = win.artifacts_dir / "bq_designation_orbit_window_features.parquet"
    features.to_parquet(str(out_path))

    meta = {
        "subset_dir": str(win.subset_dir),
        "index_db": str(win.index_db),
        "min_mjd": win.min_mjd,
        "max_mjd": win.max_mjd,
        "start_utc": win.start_utc.isoformat().replace("+00:00", "Z"),
        "end_utc_exclusive": win.end_utc_exclusive.isoformat().replace("+00:00", "Z"),
        "obscodes": list(win.obscodes),
        "max_rows_per_partition": int(max_rows_per_partition),
        "partitions_used": int(partitions),
        "n_rows": int(len(features)),
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    (win.artifacts_dir / "bq_designation_orbit_window_features_meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n"
    )

    return out_path


def select_designations_from_features(
    *,
    features: DesignationOrbitWindowFeatures,
    window_mid_mjd: float,
    cfg: SamplingConfig,
) -> SelectedDesignations:
    designation = [str(x) for x in features.designation.to_pylist()]

    def _col_f64(name: str) -> np.ndarray:
        arr = getattr(features, name).to_numpy(zero_copy_only=False)
        return arr.astype(np.float64, copy=False)

    def _col_i64(name: str) -> np.ndarray:
        arr = getattr(features, name).to_numpy(zero_copy_only=False)
        return arr.astype(np.float64, copy=False)  # keep NaN for nulls

    a = _col_f64("a")
    e = _col_f64("e")
    inc = _col_f64("i")
    qv_ = _col_f64("q")
    epoch = _col_f64("epoch_mjd")
    arc = _col_f64("arc_length_total")
    u_param = _col_i64("u_param")

    groups: dict[str, list[tuple[int, int]]] = {}
    for idx, des in enumerate(designation):
        ai = None if not np.isfinite(a[idx]) else float(a[idx])
        ei = None if not np.isfinite(e[idx]) else float(e[idx])
        ii = None if not np.isfinite(inc[idx]) else float(inc[idx])
        qi = None if not np.isfinite(qv_[idx]) else float(qv_[idx])
        epi = None if not np.isfinite(epoch[idx]) else float(epoch[idx])
        arci = None if not np.isfinite(arc[idx]) else float(arc[idx])
        ui = None if not np.isfinite(u_param[idx]) else int(u_param[idx])

        stratum = _row_stratum(
            cfg=cfg,
            window_mid_mjd=float(window_mid_mjd),
            a=ai,
            e=ei,
            i=ii,
            q=qi,
            epoch_mjd=epi,
            arc_days=arci,
            u_param=ui,
        )
        h = _hash64(des, seed=int(cfg.seed))
        groups.setdefault(stratum, []).append((h, idx))

    for k in groups:
        groups[k].sort(key=lambda t: t[0])

    selected_idx = _round_robin_sample(groups, n_total=int(cfg.n_total))
    selected_idx_set = set(selected_idx)

    # stable output order by hash
    selected_rows: list[tuple[int, str, str]] = []
    for i in selected_idx:
        des = designation[i]
        h = _hash64(des, seed=int(cfg.seed))
        # recompute stratum for output (cheap)
        ai = None if not np.isfinite(a[i]) else float(a[i])
        ei = None if not np.isfinite(e[i]) else float(e[i])
        ii = None if not np.isfinite(inc[i]) else float(inc[i])
        qi = None if not np.isfinite(qv_[i]) else float(qv_[i])
        epi = None if not np.isfinite(epoch[i]) else float(epoch[i])
        arci = None if not np.isfinite(arc[i]) else float(arc[i])
        ui = None if not np.isfinite(u_param[i]) else int(u_param[i])
        stratum = _row_stratum(
            cfg=cfg,
            window_mid_mjd=float(window_mid_mjd),
            a=ai,
            e=ei,
            i=ii,
            q=qi,
            epoch_mjd=epi,
            arc_days=arci,
            u_param=ui,
        )
        selected_rows.append((h, des, stratum))

    selected_rows.sort(key=lambda t: t[0])
    return SelectedDesignations.from_kwargs(
        designation=[d for _, d, _ in selected_rows],
        stratum=[s for _, _, s in selected_rows],
        hash64=[int(h) for h, _, _ in selected_rows],
    )


def persist_selection_for_subset(
    *,
    subset_dir: Path,
    cfg: BqConfig,
    sampling_cfg: SamplingConfig,
) -> dict[str, Path]:
    win = read_subset_window(subset_dir)
    win.artifacts_dir.mkdir(parents=True, exist_ok=True)

    features_path = win.artifacts_dir / "bq_designation_orbit_window_features.parquet"
    if not features_path.exists():
        features_path = fetch_and_persist_window_features_for_subset(subset_dir=subset_dir, cfg=cfg)

    features = DesignationOrbitWindowFeatures.from_parquet(str(features_path))
    window_mid_mjd = 0.5 * (float(win.min_mjd) + float(win.max_mjd))
    selected = select_designations_from_features(
        features=features, window_mid_mjd=window_mid_mjd, cfg=sampling_cfg
    )

    out_sel = win.artifacts_dir / "selected_designations.parquet"
    selected.to_parquet(str(out_sel))

    # Persist selected features for convenience/debugging.
    sel_set = set(selected.designation.to_pylist())
    keep = [d in sel_set for d in features.designation.to_pylist()]
    mask = np.array(keep, dtype=bool)
    # boolean mask for pyarrow table filtering
    sel_features = DesignationOrbitWindowFeatures.from_pyarrow(
        features.table.filter(mask)  # type: ignore[arg-type]
    )
    out_sel_features = win.artifacts_dir / "selected_designation_features.parquet"
    sel_features.to_parquet(str(out_sel_features))

    meta = {
        "subset_dir": str(win.subset_dir),
        "window_mid_mjd": window_mid_mjd,
        "n_selected": int(len(selected)),
        "sampling": {
            "n_total": int(sampling_cfg.n_total),
            "seed": int(sampling_cfg.seed),
            "arc_bins_days": list(sampling_cfg.arc_bins_days),
            "dt_bins_days": list(sampling_cfg.dt_bins_days),
            "hi_i_deg": float(sampling_cfg.hi_i_deg),
        },
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    out_meta = win.artifacts_dir / "selected_designations_meta.json"
    out_meta.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")

    # Stratum counts are useful and cheap to store.
    counts: dict[str, int] = {}
    for s in selected.stratum.to_pylist():
        counts[str(s)] = counts.get(str(s), 0) + 1
    (win.artifacts_dir / "selected_designations_strata_counts.json").write_text(
        json.dumps(dict(sorted(counts.items())), indent=2, sort_keys=True) + "\n"
    )

    return {
        "features_parquet": features_path,
        "selected_parquet": out_sel,
        "selected_features_parquet": out_sel_features,
        "selected_meta_json": out_meta,
    }


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Select ~N diverse designations for a local subset window.")
    p.add_argument("--subset-dir", type=str, required=True)
    p.add_argument("--n-total", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out = persist_selection_for_subset(
        subset_dir=Path(args.subset_dir),
        cfg=BqConfig(),
        sampling_cfg=SamplingConfig(n_total=int(args.n_total), seed=int(args.seed)),
    )
    for k, v in out.items():
        print(f"{k}={v}")


if __name__ == "__main__":
    main()

