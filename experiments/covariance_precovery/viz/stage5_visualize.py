from __future__ import annotations

import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def _designation_from_object_id(object_id: str) -> str:
    s = str(object_id).strip()
    if s.startswith("(") and s.endswith(")") and len(s) >= 3:
        return s[1:-1].strip()
    return s.split()[0].strip()


def _slugify(s: str) -> str:
    """
    Make a filesystem-friendly identifier.
    """
    s = str(s).strip().replace(" ", "_")
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unknown"


def _orbit_name_map_from_stage2_meta(*, stage2_run_dir: Path) -> dict[str, str]:
    """
    Attempt to map orbit_id -> human-friendly designation using the Stage 2 meta.json.
    """
    stage2_run_dir = Path(stage2_run_dir)
    meta_path = stage2_run_dir / "meta.json"
    if not meta_path.exists():
        return {}
    try:
        meta = json.loads(meta_path.read_text())
    except Exception:  # noqa: BLE001
        return {}
    orbits_path = meta.get("orbits_parquet_used") or meta.get("orbits_parquet_arg")
    if not orbits_path:
        return {}
    try:
        t = pq.read_table(str(orbits_path), columns=["orbit_id", "object_id"])
    except Exception:  # noqa: BLE001
        try:
            t = pq.read_table(str(orbits_path), columns=["orbit_id"])
        except Exception:  # noqa: BLE001
            return {}

    cols = set(t.column_names)
    orbit_id = [str(x) for x in t["orbit_id"].to_pylist()] if "orbit_id" in cols else []
    if not orbit_id:
        return {}
    if "object_id" not in cols:
        return {str(x): str(x) for x in orbit_id}
    obj = t["object_id"].to_pylist()
    out: dict[str, str] = {}
    for oid, o in zip(orbit_id, obj):
        if o is None:
            continue
        name = _designation_from_object_id(str(o))
        if name:
            out[str(oid)] = str(name)
    return out


def read_stage5_timeseries(*, run_dir: Path, direction: str = "abs") -> pd.DataFrame:
    run_dir = Path(run_dir)
    df = pq.read_table(str(run_dir / "timeseries.parquet")).to_pandas()
    df = df[df["direction"] == str(direction)].copy()
    # Convenience for plotting.
    df["abs_dt_mid_days"] = 0.5 * (df["abs_dt_min_days"].fillna(np.nan) + df["abs_dt_max_days"].fillna(np.nan))
    return df


def read_stage5_meta(*, run_dir: Path) -> dict[str, object]:
    run_dir = Path(run_dir)
    meta_path = run_dir / "meta.json"
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text())
    except Exception:  # noqa: BLE001
        return {}


def _metric_quantiles(df: pd.DataFrame, *, metric: str) -> pd.DataFrame:
    if metric not in df.columns:
        raise ValueError(f"Missing metric column {metric!r} in timeseries")
    q = (
        df.groupby("batch_id")[str(metric)]
        .quantile([0.5, 0.9, 0.99])
        .unstack()
        .reset_index()
        .rename(columns={0.5: "p50", 0.9: "p90", 0.99: "p99"})
    )
    # Attach a representative x-position for the batch, based on observed |Δt| midpoints.
    if "abs_dt_mid_days" in df.columns:
        x = (
            df.groupby("batch_id")["abs_dt_mid_days"]
            .median()
            .reset_index()
            .rename(columns={"abs_dt_mid_days": "abs_dt_mid_days"})
        )
        q = q.merge(x, on="batch_id", how="left")
    return q.sort_values("batch_id").reset_index(drop=True)


def _metric_p99_by_batch(df: pd.DataFrame, *, metric: str) -> pd.DataFrame:
    if metric not in df.columns:
        raise ValueError(f"Missing metric column {metric!r} in timeseries")
    q = (
        df.groupby("batch_id")[str(metric)]
        .quantile([0.99])
        .unstack()
        .reset_index()
        .rename(columns={0.99: "p99"})
    )
    if "abs_dt_mid_days" in df.columns:
        x = (
            df.groupby("batch_id")["abs_dt_mid_days"]
            .median()
            .reset_index()
            .rename(columns={"abs_dt_mid_days": "abs_dt_mid_days"})
        )
        q = q.merge(x, on="batch_id", how="left")
    return q.sort_values("batch_id").reset_index(drop=True)


def plot_area_vs_expected_obs(
    *,
    run_dir: Path,
    direction: str = "abs",
    area_metric: str = "ellipse_area_deg2_mean",
    bytes_per_obs: float | None = None,
    orbit_id: str | None = None,
    out_png: Path | None = None,
    title: str | None = None,
) -> Path:
    """
    Plot covariance ellipse area vs |Δt| and expected returned observations per exposure.

    Expected obs is computed from `weighted_bytes_per_exposure / bytes_per_obs` when possible.
    """
    try:
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("matplotlib is required for visualization") from e

    run_dir = Path(run_dir)
    meta = read_stage5_meta(run_dir=run_dir)
    name_map: dict[str, str] = {}
    try:
        s2 = meta.get("stage2_run_dir")
        if s2:
            name_map = _orbit_name_map_from_stage2_meta(stage2_run_dir=Path(str(s2)))
    except Exception:  # noqa: BLE001
        name_map = {}
    df = read_stage5_timeseries(run_dir=run_dir, direction=str(direction))
    if orbit_id is not None:
        df = df[df["orbit_id"] == str(orbit_id)].copy()

    # Area p99 (deg^2).
    q_area = _metric_p99_by_batch(df, metric=str(area_metric))

    # Expected obs quantiles (density × covariance area).
    bpo = None
    if bytes_per_obs is not None:
        bpo = float(bytes_per_obs)
    else:
        try:
            if meta.get("bytes_per_obs") is not None:
                bpo = float(meta["bytes_per_obs"])  # type: ignore[arg-type]
        except Exception:  # noqa: BLE001
            bpo = None

    if bpo is None or (not np.isfinite(bpo)) or bpo <= 0:
        raise ValueError("bytes_per_obs must be provided (or present in meta.json) and > 0 to compute expected obs")

    try:
        nside = int(meta.get("healpix_nside", 0) or 0)
    except Exception:  # noqa: BLE001
        nside = 0
    if nside <= 0:
        raise ValueError("meta.json is missing a valid healpix_nside")

    # Area of a single HEALPix pixel in deg^2.
    pix_area_deg2 = (4.0 * np.pi) * (180.0 / np.pi) ** 2 / (12.0 * float(nside) * float(nside))

    # Expected observations encompassed by covariance (per exposure):
    # use fractional-coverage weighted bytes if available.
    exp_col = "_expected_obs_in_cov_per_exposure"
    if "weighted_bytes_per_exposure" in df.columns:
        df[exp_col] = df["weighted_bytes_per_exposure"].to_numpy(dtype=float) / float(bpo)
    elif "sum_weighted_data_length_bytes" in df.columns:
        sw = df["sum_weighted_data_length_bytes"].to_numpy(dtype=float)
        n = df["n_targets"].to_numpy(dtype=float)
        df[exp_col] = np.where(n > 0, (sw / n) / float(bpo), np.nan)
    else:
        # Fallback to density × area model, using intersected-pixel density.
        bytes_per_exp = df["bytes_per_exposure"].to_numpy(dtype=float)
        frames_per_exp = df["frames_per_exposure"].to_numpy(dtype=float)
        area = df[str(area_metric)].to_numpy(dtype=float)
        obs_per_deg2 = (bytes_per_exp / float(bpo)) / (frames_per_exp * float(pix_area_deg2))
        exp = area * obs_per_deg2
        df[exp_col] = np.where((frames_per_exp > 0) & np.isfinite(exp), exp, np.nan)

    q_exp = _metric_p99_by_batch(df, metric=str(exp_col))

    # x-axis: |Δt| midpoint (days) when present.
    if "abs_dt_mid_days" in q_area.columns and q_area["abs_dt_mid_days"].notna().any():
        x = q_area["abs_dt_mid_days"].to_numpy(dtype=float)
        x_label = "|Δt| (days)"
    else:
        x = q_area["batch_id"].to_numpy(dtype=float)
        x_label = "batch_id"

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    # Area on left axis.
    ax1.plot(x, q_area["p99"].to_numpy(dtype=float), label=f"p99({area_metric})", color="C0")

    # Expected obs on right axis.
    ylab = "expected_obs_in_cov_per_exposure"
    ax2.plot(x, q_exp["p99"].to_numpy(dtype=float), label=f"p99({ylab})", color="C1")

    ax1.set_xlabel(x_label)
    ax1.set_ylabel(f"{area_metric} (deg^2)")
    ax2.set_ylabel(ylab)

    # Scales when meaningful.
    def _maybe_log_or_symlog(ax, vals: np.ndarray) -> None:
        v = vals[np.isfinite(vals)]
        if v.size <= 0 or np.nanmax(v) <= 0:
            return
        if np.nanmin(v) <= 0:
            # Show zeros on the axis without dropping them.
            ax.set_yscale("symlog", linthresh=1e-12)
        else:
            ax.set_yscale("log")

    _maybe_log_or_symlog(ax1, q_area["p99"].to_numpy(float))
    _maybe_log_or_symlog(ax2, q_exp["p99"].to_numpy(float))

    ax1.grid(True, which="both", linestyle=":", linewidth=0.5)

    # Single legend combining both axes.
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="best")

    if title is None:
        ns = meta.get("n_sigma")
        fp = meta.get("footprint")
        frac = meta.get("fractional_nside")
        bpo_s = "?" if bpo is None else f"{bpo:g}"
        if orbit_id is None:
            extra = ""
        else:
            pretty = name_map.get(str(orbit_id), str(orbit_id))
            extra = f", orbit={pretty} (id={orbit_id})"
        title = (
            f"Stage5 area vs expected obs (dir={direction}{extra}, nσ={ns}, fp={fp}, "
            f"frac_nside={frac}, bytes/obs={bpo_s})"
        )
    ax1.set_title(title)

    if out_png is None:
        if orbit_id is None:
            out_png = run_dir / f"stage5_area_vs_expected_{direction}.png"
        else:
            pretty = name_map.get(str(orbit_id), str(orbit_id))
            out_png = run_dir / f"stage5_area_vs_expected_{direction}_orbit_{_slugify(pretty)}_id{orbit_id}.png"
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    return out_png


def plot_all_orbits(
    *,
    run_dir: Path,
    direction: str = "abs",
    area_metric: str = "ellipse_area_deg2_mean",
    bytes_per_obs: float | None = None,
    out_dir: Path | None = None,
) -> list[Path]:
    run_dir = Path(run_dir)
    if out_dir is None:
        out_dir = run_dir / "orbit_plots_named"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Try to get friendly names once.
    meta = read_stage5_meta(run_dir=run_dir)
    name_map: dict[str, str] = {}
    try:
        s2 = meta.get("stage2_run_dir")
        if s2:
            name_map = _orbit_name_map_from_stage2_meta(stage2_run_dir=Path(str(s2)))
    except Exception:  # noqa: BLE001
        name_map = {}

    df = read_stage5_timeseries(run_dir=run_dir, direction=str(direction))
    orbit_ids = sorted({str(x) for x in df["orbit_id"].dropna().unique().tolist()})
    outs: list[Path] = []
    for oid in orbit_ids:
        pretty = name_map.get(str(oid), str(oid))
        slug = _slugify(pretty)
        outs.append(
            plot_area_vs_expected_obs(
                run_dir=run_dir,
                direction=str(direction),
                area_metric=str(area_metric),
                bytes_per_obs=bytes_per_obs,
                orbit_id=str(oid),
                out_png=out_dir / f"stage5_area_vs_expected_{direction}_orbit_{slug}_id{oid}.png",
            )
        )
    return outs


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Stage 5 visualization: covariance area vs expected observations.")
    p.add_argument("--run-dir", type=str, required=True)
    p.add_argument("--direction", type=str, default="abs", choices=["abs", "backward", "forward"])
    p.add_argument(
        "--orbit-id",
        type=str,
        default=None,
        help="Optional: restrict plot to a single orbit_id (recommended for this exploration).",
    )
    p.add_argument(
        "--area-metric",
        type=str,
        default="ellipse_area_deg2_mean",
        help="Timeseries column to use for covariance area in area_expected plot.",
    )
    p.add_argument(
        "--bytes-per-obs",
        type=float,
        default=None,
        help="Override bytes_per_obs when computing expected observations (area_expected plot).",
    )
    p.add_argument(
        "--all-orbits",
        action="store_true",
        help="Generate one plot per orbit_id into <run_dir>/orbit_plots_named (or --out-dir).",
    )
    p.add_argument("--out-dir", type=str, default=None, help="Output directory when --all-orbits is set.")
    p.add_argument("--out-png", type=str, default=None)
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    if bool(args.all_orbits):
        outs = plot_all_orbits(
            run_dir=run_dir,
            direction=str(args.direction),
            area_metric=str(args.area_metric),
            bytes_per_obs=(None if args.bytes_per_obs is None else float(args.bytes_per_obs)),
            out_dir=(None if args.out_dir is None else Path(args.out_dir)),
        )
        print(f"n_orbits={len(outs)}")
        if outs:
            print(f"out_dir={Path(outs[0]).parent}")
    else:
        out_png = plot_area_vs_expected_obs(
            run_dir=run_dir,
            direction=str(args.direction),
            area_metric=str(args.area_metric),
            bytes_per_obs=(None if args.bytes_per_obs is None else float(args.bytes_per_obs)),
            orbit_id=(None if args.orbit_id is None else str(args.orbit_id)),
            out_png=None if args.out_png is None else Path(args.out_png),
        )
        print(f"out_png={out_png}")


if __name__ == "__main__":
    main()

