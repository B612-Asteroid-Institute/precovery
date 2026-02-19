from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


_EXP_COL = "_expected_obs_in_cov"


def _designation_from_object_id(object_id: str) -> str:
    """
    Extract a human-friendly designation from an SBDB-style `object_id` string.

    Common forms include:
      - "2019 QU127"
      - "3753 Cruithne (1986 TO)"
      - "594913 'Aylo'chaxnim (2020 AV2)"

    Prefer the trailing parenthetical (e.g. "(1986 TO)") when present; otherwise
    fall back to the full string.
    """
    s = str(object_id).strip()
    # If the entire string is parenthetical, strip it.
    if s.startswith("(") and s.endswith(")") and len(s) >= 3:
        return s[1:-1].strip()
    # If there's a trailing "(...)" group, use its contents.
    m = re.search(r"\(([^()]+)\)\s*$", s)
    if m:
        return str(m.group(1)).strip()
    return s


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
        obj_s = str(o)
        name = _designation_from_object_id(obj_s)
        if name:
            out[str(oid)] = str(name)
            # Also allow lookup by the leading SBDB number (often used downstream as orbit_id).
            first = obj_s.strip().split()[0].strip() if obj_s.strip() else ""
            if first:
                out[str(first)] = str(name)
    return out


def read_stage5_timeseries(*, run_dir: Path, direction: str = "abs") -> pd.DataFrame:
    run_dir = Path(run_dir)
    df = pq.read_table(str(run_dir / "timeseries.parquet")).to_pandas()
    df = df[df["direction"] == str(direction)].copy()
    # Convenience for plotting.
    df["abs_dt_mid_days"] = 0.5 * (
        df["abs_dt_min_days"].fillna(np.nan) + df["abs_dt_max_days"].fillna(np.nan)
    )
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
    return _metric_quantile_by_batch(df, metric=metric, q=0.99).rename(columns={"q": "p99"})


def _metric_quantile_by_batch(df: pd.DataFrame, *, metric: str, q: float) -> pd.DataFrame:
    if metric not in df.columns:
        raise ValueError(f"Missing metric column {metric!r} in timeseries")
    q = float(q)
    out = (
        df.groupby("batch_id")[str(metric)]
        .quantile(q)
        .reset_index()
        .rename(columns={str(metric): "q"})
    )
    if "abs_dt_mid_days" in df.columns:
        x = (
            df.groupby("batch_id")["abs_dt_mid_days"]
            .median()
            .reset_index()
            .rename(columns={"abs_dt_mid_days": "abs_dt_mid_days"})
        )
        out = out.merge(x, on="batch_id", how="left")
    return out.sort_values("batch_id").reset_index(drop=True)


def _attach_expected_obs(
    df_in: pd.DataFrame,
    *,
    bytes_per_obs: float,
    area_metric: str,
    pix_area_deg2: float,
) -> tuple[pd.DataFrame, str]:
    """
    Add `_EXP_COL` to `df_in` and return a y-axis label.
    """
    df_out = df_in.copy()
    bpo = float(bytes_per_obs)
    if (not np.isfinite(bpo)) or bpo <= 0:
        raise ValueError("bytes_per_obs must be finite and > 0")

    ylab = "expected_obs_in_cov_per_hit_exposure"
    if "expected_obs_per_hit_exposure" in df_out.columns:
        df_out[_EXP_COL] = df_out["expected_obs_per_hit_exposure"].to_numpy(dtype=float)
    elif "weighted_bytes_per_hit_exposure" in df_out.columns:
        df_out[_EXP_COL] = (
            df_out["weighted_bytes_per_hit_exposure"].to_numpy(dtype=float) / float(bpo)
        )
    elif "weighted_bytes_per_exposure" in df_out.columns:
        # Fallback: unconditional average over all targets in the bin (often ~0).
        ylab = "expected_obs_in_cov_per_exposure"
        df_out[_EXP_COL] = (
            df_out["weighted_bytes_per_exposure"].to_numpy(dtype=float) / float(bpo)
        )
    elif "sum_weighted_data_length_bytes" in df_out.columns:
        # Old fallback.
        ylab = "expected_obs_in_cov_per_exposure"
        sw = df_out["sum_weighted_data_length_bytes"].to_numpy(dtype=float)
        n = df_out["n_targets"].to_numpy(dtype=float)
        df_out[_EXP_COL] = np.where(n > 0, (sw / n) / float(bpo), np.nan)
    else:
        # Fallback to density × area model, using intersected-pixel density.
        ylab = "expected_obs_in_cov_per_exposure"
        bytes_per_exp = df_out["bytes_per_exposure"].to_numpy(dtype=float)
        frames_per_exp = df_out["frames_per_exposure"].to_numpy(dtype=float)
        area = df_out[str(area_metric)].to_numpy(dtype=float)
        obs_per_deg2 = (bytes_per_exp / float(bpo)) / (
            frames_per_exp * float(pix_area_deg2)
        )
        exp = area * obs_per_deg2
        df_out[_EXP_COL] = np.where(
            (frames_per_exp > 0) & np.isfinite(exp), exp, np.nan
        )
    return df_out, ylab


def plot_area_vs_expected_obs(
    *,
    run_dir: Path,
    direction: str = "abs",
    area_metric: str = "ellipse_area_pred_deg2_mean",
    bytes_per_obs: float | None = None,
    distance_metric: str | None = "rho_au_min",
    distance_quantile: float = 0.01,
    orbit_id: str | None = None,
    out_png: Path | None = None,
    title: str | None = None,
) -> Path:
    """
    Plot covariance ellipse area vs Δt and expected returned observations per exposure.

    Supported directions:
      - "abs": plot vs |Δt| (default Stage 5 output).
      - "backward": negative-time side only (requires Stage 5 `--report-signed`).
      - "forward": positive-time side only (requires Stage 5 `--report-signed`).
      - "signed": plot both backward + forward on a signed x-axis (requires Stage 5
        `--report-signed`; typically used with `--all-orbits`).

    Expected obs is computed from Stage 5 bytes metrics when possible.

    Important: for wide target sets, most (orbit,target) pairs will have *no* intersected
    frames (orbit is not in that exposure). In that regime, averaging per-exposure over
    all targets drives expected_obs toward ~0. Prefer the conditional
    `*_per_hit_exposure` metrics when available.
    """
    try:
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("matplotlib is required for visualization") from e

    run_dir = Path(run_dir)
    direction = str(direction)
    meta = read_stage5_meta(run_dir=run_dir)
    name_map: dict[str, str] = {}
    try:
        s2 = meta.get("stage2_run_dir")
        if s2:
            name_map = _orbit_name_map_from_stage2_meta(stage2_run_dir=Path(str(s2)))
    except Exception:  # noqa: BLE001
        name_map = {}

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
        raise ValueError(
            "bytes_per_obs must be provided (or present in meta.json) and > 0 to compute expected obs"
        )

    try:
        nside = int(meta.get("healpix_nside", 0) or 0)
    except Exception:  # noqa: BLE001
        nside = 0
    if nside <= 0:
        raise ValueError("meta.json is missing a valid healpix_nside")

    # Area of a single HEALPix pixel in deg^2.
    pix_area_deg2 = (
        (4.0 * np.pi) * (180.0 / np.pi) ** 2 / (12.0 * float(nside) * float(nside))
    )

    def _plot_expected_scatter(ax, xvals: np.ndarray, yvals: np.ndarray, *, label: str) -> None:
        ax.scatter(
            xvals,
            yvals,
            s=10,
            alpha=0.9,
            color="C1",
            label=label,
        )

    if direction == "signed":
        if orbit_id is None:
            raise ValueError(
                "direction='signed' requires --orbit-id (or use --all-orbits)."
            )
        oid = str(orbit_id)
        df_back = read_stage5_timeseries(run_dir=run_dir, direction="backward")
        df_fwd = read_stage5_timeseries(run_dir=run_dir, direction="forward")
        df_back = df_back[df_back["orbit_id"] == oid].copy()
        df_fwd = df_fwd[df_fwd["orbit_id"] == oid].copy()

        df_back["dt_mid_days"] = -df_back["abs_dt_mid_days"].to_numpy(dtype=float)
        df_fwd["dt_mid_days"] = df_fwd["abs_dt_mid_days"].to_numpy(dtype=float)
        df_back = df_back.sort_values("dt_mid_days").reset_index(drop=True)
        df_fwd = df_fwd.sort_values("dt_mid_days").reset_index(drop=True)

        df_back, ylab = _attach_expected_obs(
            df_back,
            bytes_per_obs=float(bpo),
            area_metric=str(area_metric),
            pix_area_deg2=float(pix_area_deg2),
        )
        df_fwd, ylab2 = _attach_expected_obs(
            df_fwd,
            bytes_per_obs=float(bpo),
            area_metric=str(area_metric),
            pix_area_deg2=float(pix_area_deg2),
        )
        if ylab2 != ylab:
            ylab = f"{ylab} / {ylab2}"

        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax2 = ax1.twinx()

        ax1.plot(
            df_back["dt_mid_days"].to_numpy(dtype=float),
            df_back[str(area_metric)].to_numpy(dtype=float),
            label=f"backward({area_metric})",
            color="C0",
            linestyle="--",
        )
        ax1.plot(
            df_fwd["dt_mid_days"].to_numpy(dtype=float),
            df_fwd[str(area_metric)].to_numpy(dtype=float),
            label=f"forward({area_metric})",
            color="C0",
            linestyle="-",
        )

        _plot_expected_scatter(
            ax2,
            df_back["dt_mid_days"].to_numpy(dtype=float),
            df_back[_EXP_COL].to_numpy(dtype=float),
            label=f"backward({ylab})",
        )
        _plot_expected_scatter(
            ax2,
            df_fwd["dt_mid_days"].to_numpy(dtype=float),
            df_fwd[_EXP_COL].to_numpy(dtype=float),
            label=f"forward({ylab})",
        )

        ax1.set_xlabel("Δt (days)")
        ax1.set_ylabel(f"{area_metric} (deg^2)")
        ax2.set_ylabel(ylab)

        # Keep expected-obs axis linear for interpretability.
        ax2.set_yscale("linear")
        v = np.concatenate(
            [
                df_back[_EXP_COL].to_numpy(float),
                df_fwd[_EXP_COL].to_numpy(float),
            ]
        )
        v = v[np.isfinite(v)]
        if v.size and float(np.nanmin(v)) >= 0.0:
            ax2.set_ylim(bottom=0.0)

        ax1.grid(True, which="both", linestyle=":", linewidth=0.5)

        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc="best")

        if title is None:
            ns = meta.get("n_sigma")
            fp = meta.get("footprint")
            frac = meta.get("fractional_nside")
            bpo_s = "?" if bpo is None else f"{bpo:g}"
            pretty = name_map.get(oid, oid)
            title = (
                f"Stage5 area vs expected obs (dir={direction}, orbit={pretty} (id={oid}), "
                f"nσ={ns}, fp={fp}, frac_nside={frac}, bytes/obs={bpo_s})"
            )
        ax1.set_title(title)

        if out_png is None:
            pretty = name_map.get(oid, oid)
            out_png = (
                run_dir
                / f"stage5_area_vs_expected_{direction}_orbit_{_slugify(pretty)}_id{oid}.png"
            )
    else:
        df = read_stage5_timeseries(run_dir=run_dir, direction=direction)
        if orbit_id is not None:
            df = df[df["orbit_id"] == str(orbit_id)].copy()

        # Area p99 (deg^2).
        q_area = _metric_p99_by_batch(df, metric=str(area_metric))

        df, ylab = _attach_expected_obs(
            df,
            bytes_per_obs=float(bpo),
            area_metric=str(area_metric),
            pix_area_deg2=float(pix_area_deg2),
        )
        q_exp = _metric_p99_by_batch(df, metric=str(_EXP_COL))

        # x-axis: |Δt| midpoint (days) when present.
        if (
            "abs_dt_mid_days" in q_area.columns
            and q_area["abs_dt_mid_days"].notna().any()
        ):
            x = q_area["abs_dt_mid_days"].to_numpy(dtype=float)
            x_label = "|Δt| (days)"
        else:
            x = q_area["batch_id"].to_numpy(dtype=float)
            x_label = "batch_id"

        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax2 = ax1.twinx()
        ax3 = None
        q_dist = None
        dist_label = None
        if distance_metric is not None and str(distance_metric) in df.columns:
            q_dist = _metric_quantile_by_batch(
                df, metric=str(distance_metric), q=float(distance_quantile)
            )
            # Align to the same batch ordering as q_area for plotting.
            q_dist = (
                q_dist.set_index("batch_id")
                .reindex(q_area["batch_id"].to_numpy(dtype=int))
                .reset_index()
            )
            ax3 = ax1.twinx()
            ax3.spines["right"].set_position(("outward", 60))
            qtag = f"q={float(distance_quantile):g}"
            unit = "AU" if "rho_au" in str(distance_metric) else ""
            dist_label = f"{qtag}({distance_metric}){(' ('+unit+')') if unit else ''}"

        ax1.plot(
            x,
            q_area["p99"].to_numpy(dtype=float),
            label=f"p99({area_metric})",
            color="C0",
        )
        _plot_expected_scatter(
            ax2,
            x,
            q_exp["p99"].to_numpy(dtype=float),
            label=f"p99({ylab})",
        )
        if ax3 is not None and q_dist is not None and dist_label is not None:
            ax3.plot(
                x,
                q_dist["q"].to_numpy(dtype=float),
                label=dist_label,
                color="C2",
            )

        ax1.set_xlabel(x_label)
        ax1.set_ylabel(f"{area_metric} (deg^2)")
        ax2.set_ylabel(ylab)
        if ax3 is not None:
            ax3.set_ylabel("range (AU)")

        # Scales when meaningful (area axis only).
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

        # Keep expected-obs axis linear for interpretability.
        ax2.set_yscale("linear")
        v = q_exp["p99"].to_numpy(float)
        v = v[np.isfinite(v)]
        if v.size and float(np.nanmin(v)) >= 0.0:
            ax2.set_ylim(bottom=0.0)

        ax1.grid(True, which="both", linestyle=":", linewidth=0.5)

        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        if ax3 is not None:
            h3, l3 = ax3.get_legend_handles_labels()
            ax1.legend(h1 + h2 + h3, l1 + l2 + l3, loc="best")
        else:
            ax1.legend(h1 + h2, l1 + l2, loc="best")

        if title is None:
            ns = meta.get("n_sigma")
            fp = meta.get("footprint")
            frac = meta.get("fractional_nside")
            bpo_s = "?" if bpo is None else f"{bpo:g}"
            if orbit_id is None:
                extra = ""
            else:
                oid = str(orbit_id)
                pretty = name_map.get(oid, oid)
                extra = f", orbit={pretty} (id={oid})"
            title = (
                f"Stage5 area vs expected obs (dir={direction}{extra}, nσ={ns}, fp={fp}, "
                f"frac_nside={frac}, bytes/obs={bpo_s})"
            )
        ax1.set_title(title)

        if out_png is None:
            if orbit_id is None:
                out_png = run_dir / f"stage5_area_vs_expected_{direction}.png"
            else:
                oid = str(orbit_id)
                pretty = name_map.get(oid, oid)
                out_png = (
                    run_dir
                    / f"stage5_area_vs_expected_{direction}_orbit_{_slugify(pretty)}_id{oid}.png"
                )

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    return out_png


def plot_covsize_and_expected_obs_panels(
    *,
    run_dir: Path,
    direction: str = "abs",
    cov_metric: str = "sigma_major_pred_arcsec_mean",
    bytes_per_obs: float | None = None,
    orbit_id: str | None = None,
    out_png: Path | None = None,
    title: str | None = None,
) -> Path:
    """
    Two-panel plot:
      - top: covariance size metric vs Δt
      - bottom: expected candidate observations per hit exposure vs Δt

    Notes:
      - This is still Stage 5 "index-only": expected observations is an estimate derived
        from `index.db` bytes metrics, not truth counts.
      - For multi-orbit plots (orbit_id=None), curves are p99 across orbit_id per batch.
    """
    try:
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("matplotlib is required for visualization") from e

    run_dir = Path(run_dir)
    direction = str(direction)
    if direction == "signed":
        raise ValueError("panels plot currently supports abs/backward/forward (not signed).")

    meta = read_stage5_meta(run_dir=run_dir)

    # bytes/obs for expected observations
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
        raise ValueError(
            "bytes_per_obs must be provided (or present in meta.json) and > 0 to compute expected obs"
        )

    try:
        nside = int(meta.get("healpix_nside", 0) or 0)
    except Exception:  # noqa: BLE001
        nside = 0
    if nside <= 0:
        raise ValueError("meta.json is missing a valid healpix_nside")
    pix_area_deg2 = (
        (4.0 * np.pi) * (180.0 / np.pi) ** 2 / (12.0 * float(nside) * float(nside))
    )

    df = read_stage5_timeseries(run_dir=run_dir, direction=direction)
    if orbit_id is not None:
        df = df[df["orbit_id"] == str(orbit_id)].copy()

    df, ylab = _attach_expected_obs(
        df,
        bytes_per_obs=float(bpo),
        area_metric="ellipse_area_pred_deg2_mean",
        pix_area_deg2=float(pix_area_deg2),
    )

    # x-axis: |Δt| midpoint (days) when present.
    if "abs_dt_mid_days" in df.columns and df["abs_dt_mid_days"].notna().any():
        # median within batch for x-position
        x_by_batch = (
            df.groupby("batch_id")["abs_dt_mid_days"]
            .median()
            .reset_index()
            .rename(columns={"abs_dt_mid_days": "x"})
        )
        x_label = "|Δt| (days)"
    else:
        x_by_batch = (
            df.groupby("batch_id")["batch_id"]
            .median()
            .reset_index()
            .rename(columns={"batch_id": "x"})
        )
        x_label = "batch_id"

    # For multi-orbit: p99 across orbit_id per batch.
    if orbit_id is None:
        q_cov = _metric_p99_by_batch(df, metric=str(cov_metric))
        q_exp = _metric_p99_by_batch(df, metric=str(_EXP_COL))
        cov_label = f"p99({cov_metric})"
        exp_label = f"p99({ylab})"
    else:
        q_cov = _metric_quantile_by_batch(df, metric=str(cov_metric), q=0.5)
        q_exp = _metric_quantile_by_batch(df, metric=str(_EXP_COL), q=0.5)
        cov_label = str(cov_metric)
        exp_label = str(ylab)

    # Align to the same x ordering
    q_cov = q_cov.merge(x_by_batch, on="batch_id", how="left").sort_values("batch_id")
    q_exp = q_exp.merge(x_by_batch, on="batch_id", how="left").sort_values("batch_id")
    x = q_cov["x"].to_numpy(dtype=float)

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )

    # Top: covariance size metric
    y_cov = (q_cov["p99"] if "p99" in q_cov.columns else q_cov["q"]).to_numpy(dtype=float)
    ax_top.plot(x, y_cov, color="C0", label=cov_label)
    ax_top.set_ylabel(cov_metric)
    ax_top.grid(True, which="both", linestyle=":", linewidth=0.5)

    # log/symlog if meaningful
    def _maybe_log_or_symlog(ax, vals: np.ndarray) -> None:
        v = vals[np.isfinite(vals)]
        if v.size <= 0 or np.nanmax(v) <= 0:
            return
        if np.nanmin(v) <= 0:
            ax.set_yscale("symlog", linthresh=1e-12)
        else:
            ax.set_yscale("log")

    _maybe_log_or_symlog(ax_top, y_cov)
    ax_top.legend(loc="best")

    # Bottom: expected obs
    y_exp = (q_exp["p99"] if "p99" in q_exp.columns else q_exp["q"]).to_numpy(dtype=float)
    ax_bot.plot(x, y_exp, color="C1", label=exp_label)
    ax_bot.set_ylabel(ylab)
    ax_bot.set_xlabel(x_label)
    ax_bot.grid(True, which="both", linestyle=":", linewidth=0.5)
    v = y_exp[np.isfinite(y_exp)]
    if v.size and float(np.nanmin(v)) >= 0.0:
        ax_bot.set_ylim(bottom=0.0)
    ax_bot.legend(loc="best")

    if title is None:
        ns = meta.get("n_sigma")
        fp = meta.get("footprint")
        frac = meta.get("fractional_nside")
        extra = "" if orbit_id is None else f", orbit_id={orbit_id}"
        title = (
            f"Stage5 cov-size + expected obs (dir={direction}{extra}, nσ={ns}, fp={fp}, "
            f"frac_nside={frac}, bytes/obs={bpo:g})"
        )
    fig.suptitle(title)

    if out_png is None:
        if orbit_id is None:
            out_png = run_dir / f"stage5_covsize_and_expected_{direction}.png"
        else:
            out_png = run_dir / f"stage5_covsize_and_expected_{direction}_orbit_{_slugify(str(orbit_id))}.png"

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    return out_png


def plot_metric_vs_expected_obs(
    *,
    run_dir: Path,
    direction: str = "abs",
    metric: str,
    metric_label: str,
    metric_unit: str | None = None,
    bytes_per_obs: float | None = None,
    distance_metric: str | None = None,
    distance_unit: str | None = None,
    orbit_id: str | None = None,
    out_png: Path | None = None,
    title: str | None = None,
) -> Path:
    """
    Single-axes plot with dual-y:
      - y1: `metric` (line)
      - y2: expected observations in the covariance (scatter)
    """
    try:
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("matplotlib is required for visualization") from e

    run_dir = Path(run_dir)
    direction = str(direction)
    meta = read_stage5_meta(run_dir=run_dir)

    # bytes/obs for expected observations
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
        raise ValueError(
            "bytes_per_obs must be provided (or present in meta.json) and > 0 to compute expected obs"
        )

    try:
        nside = int(meta.get("healpix_nside", 0) or 0)
    except Exception:  # noqa: BLE001
        nside = 0
    if nside <= 0:
        raise ValueError("meta.json is missing a valid healpix_nside")
    pix_area_deg2 = (
        (4.0 * np.pi) * (180.0 / np.pi) ** 2 / (12.0 * float(nside) * float(nside))
    )

    is_signed = str(direction) == "signed"
    if is_signed:
        if orbit_id is None:
            raise ValueError("direction='signed' requires --orbit-id (or use --all-orbits).")
        oid = str(orbit_id)
        df_back = read_stage5_timeseries(run_dir=run_dir, direction="backward")
        df_fwd = read_stage5_timeseries(run_dir=run_dir, direction="forward")
        df_back = df_back[df_back["orbit_id"] == oid].copy()
        df_fwd = df_fwd[df_fwd["orbit_id"] == oid].copy()

        df_back["dt_mid_days"] = -df_back["abs_dt_mid_days"].to_numpy(dtype=float)
        df_fwd["dt_mid_days"] = df_fwd["abs_dt_mid_days"].to_numpy(dtype=float)
        df_back = df_back.sort_values("dt_mid_days").reset_index(drop=True)
        df_fwd = df_fwd.sort_values("dt_mid_days").reset_index(drop=True)

        df_back, ylab = _attach_expected_obs(
            df_back,
            bytes_per_obs=float(bpo),
            area_metric="ellipse_area_pred_deg2_mean",
            pix_area_deg2=float(pix_area_deg2),
        )
        df_fwd, ylab2 = _attach_expected_obs(
            df_fwd,
            bytes_per_obs=float(bpo),
            area_metric="ellipse_area_pred_deg2_mean",
            pix_area_deg2=float(pix_area_deg2),
        )
        if ylab2 != ylab:
            ylab = f"{ylab} / {ylab2}"

        x_back = df_back["dt_mid_days"].to_numpy(dtype=float)
        x_fwd = df_fwd["dt_mid_days"].to_numpy(dtype=float)
        y_m_back = df_back[str(metric)].to_numpy(dtype=float)
        y_m_fwd = df_fwd[str(metric)].to_numpy(dtype=float)
        y_e_back = df_back[str(_EXP_COL)].to_numpy(dtype=float)
        y_e_fwd = df_fwd[str(_EXP_COL)].to_numpy(dtype=float)
        y_d_back = (
            df_back[str(distance_metric)].to_numpy(dtype=float)
            if (distance_metric is not None and str(distance_metric) in df_back.columns)
            else None
        )
        y_d_fwd = (
            df_fwd[str(distance_metric)].to_numpy(dtype=float)
            if (distance_metric is not None and str(distance_metric) in df_fwd.columns)
            else None
        )
        x_label = "Δt (days)"
        m_label = str(metric_label)
        e_label = str(ylab)
    else:
        df = read_stage5_timeseries(run_dir=run_dir, direction=direction)
        if orbit_id is not None:
            df = df[df["orbit_id"] == str(orbit_id)].copy()
            # For a single orbit, the timeseries already has one row per batch; use it directly.
            df = df.sort_values("batch_id").reset_index(drop=True)

        df, ylab = _attach_expected_obs(
            df,
            bytes_per_obs=float(bpo),
            area_metric="ellipse_area_pred_deg2_mean",
            pix_area_deg2=float(pix_area_deg2),
        )

        # x-axis
        if "abs_dt_mid_days" in df.columns and df["abs_dt_mid_days"].notna().any():
            x = (
                df.groupby("batch_id")["abs_dt_mid_days"].median().to_numpy(dtype=float)
                if orbit_id is None
                else df["abs_dt_mid_days"].to_numpy(dtype=float)
            )
            x_label = "|Δt| (days)" if direction == "abs" else "Δt (days)"
        else:
            x = (
                df["batch_id"].unique().astype(float)
                if orbit_id is None
                else df["batch_id"].to_numpy(dtype=float)
            )
            x_label = "batch_id"

        # y-values: for multi-orbit, use p99 per batch; for single orbit, use the values.
        if orbit_id is None:
            q_m = _metric_p99_by_batch(df, metric=str(metric))
            q_e = _metric_p99_by_batch(df, metric=str(_EXP_COL))
            y_m = q_m["p99"].to_numpy(dtype=float)
            y_e = q_e["p99"].to_numpy(dtype=float)
            m_label = f"p99({metric_label})"
            e_label = f"p99({ylab})"
            # align x with q_m midpoints when present
            if (
                "abs_dt_mid_days" in q_m.columns
                and q_m["abs_dt_mid_days"].notna().any()
            ):
                x = q_m["abs_dt_mid_days"].to_numpy(dtype=float)
                x_label = "|Δt| (days)"
        else:
            y_m = df[str(metric)].to_numpy(dtype=float)
            y_e = df[str(_EXP_COL)].to_numpy(dtype=float)
            m_label = str(metric_label)
            e_label = str(ylab)

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()
    ax3 = None

    if is_signed:
        # Legend does not need forward/backward; the sign of Δt makes it clear.
        ax1.plot(x_back, y_m_back, color="C0", linestyle="--", label="_nolegend_")
        ax1.plot(x_fwd, y_m_fwd, color="C0", linestyle="-", label=str(m_label))
        ax2.scatter(
            x_back,
            y_e_back,
            s=12,
            alpha=0.9,
            color="C1",
            marker="x",
            label="_nolegend_",
        )
        ax2.scatter(
            x_fwd,
            y_e_fwd,
            s=10,
            alpha=0.9,
            color="C1",
            marker="o",
            label=str(e_label),
        )
        if y_d_back is not None and y_d_fwd is not None:
            ax3 = ax1.twinx()
            ax3.spines["right"].set_position(("outward", 60))
            ax3.plot(
                x_back,
                y_d_back,
                color="C2",
                linestyle="--",
                label="_nolegend_",
            )
            ax3.plot(
                x_fwd,
                y_d_fwd,
                color="C2",
                linestyle="-",
                label=str(distance_metric),
            )
    else:
        ax1.plot(x, y_m, color="C0", label=m_label)
        ax2.scatter(x, y_e, s=10, alpha=0.9, color="C1", label=e_label)
        if distance_metric is not None and str(distance_metric) in df.columns:
            ax3 = ax1.twinx()
            ax3.spines["right"].set_position(("outward", 60))
            y_d = (
                df[str(distance_metric)].to_numpy(dtype=float)
                if orbit_id is not None
                else _metric_quantile_by_batch(df, metric=str(distance_metric), q=0.5)[
                    "q"
                ].to_numpy(dtype=float)
            )
            ax3.plot(x, y_d, color="C2", label=str(distance_metric))

    ax1.set_xlabel(x_label)
    y1 = metric_label + (f" ({metric_unit})" if metric_unit else "")
    ax1.set_ylabel(y1)
    ax2.set_ylabel(ylab)
    if ax3 is not None:
        du = ""
        if distance_unit:
            du = f" ({distance_unit})"
        ax3.set_ylabel(f"{distance_metric}{du}")

    # scaling: keep expected obs linear; metric can be log if positive.
    ax2.set_yscale("linear")
    if is_signed:
        vv = np.concatenate([y_e_back, y_e_fwd])
        v = vv[np.isfinite(vv)]
    else:
        v = y_e[np.isfinite(y_e)]
    if v.size and float(np.nanmin(v)) >= 0.0:
        ax2.set_ylim(bottom=0.0)

    def _maybe_log_or_symlog(ax, vals: np.ndarray) -> None:
        v = vals[np.isfinite(vals)]
        if v.size <= 0 or np.nanmax(v) <= 0:
            return
        if np.nanmin(v) <= 0:
            ax.set_yscale("symlog", linthresh=1e-12)
        else:
            ax.set_yscale("log")

    if is_signed:
        _maybe_log_or_symlog(ax1, np.concatenate([y_m_back, y_m_fwd]))
    else:
        _maybe_log_or_symlog(ax1, y_m)
    ax1.grid(True, which="both", linestyle=":", linewidth=0.5)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    if ax3 is not None:
        h3, l3 = ax3.get_legend_handles_labels()
        ax1.legend(h1 + h2 + h3, l1 + l2 + l3, loc="best")
    else:
        ax1.legend(h1 + h2, l1 + l2, loc="best")

    if title is None:
        ns = meta.get("n_sigma")
        fp = meta.get("footprint")
        frac = meta.get("fractional_nside")
        extra = "" if orbit_id is None else f", orbit_id={orbit_id}"
        title = (
            f"Stage5 {metric_label} vs expected obs (dir={direction}{extra}, nσ={ns}, fp={fp}, "
            f"frac_nside={frac}, bytes/obs={bpo:g})"
        )
    ax1.set_title(title)

    if out_png is None:
        suffix = "" if orbit_id is None else f"_orbit_{_slugify(str(orbit_id))}"
        out_png = run_dir / f"stage5_{_slugify(metric_label)}_vs_expected_{direction}{suffix}.png"
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
    area_metric: str = "ellipse_area_pred_deg2_mean",
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

    list_direction = "abs" if str(direction) == "signed" else str(direction)
    df = read_stage5_timeseries(run_dir=run_dir, direction=list_direction)
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
                out_png=out_dir
                / f"stage5_area_vs_expected_{direction}_orbit_{slug}_id{oid}.png",
            )
        )
    return outs


def plot_orbit_two_plots_all_orbits(
    *,
    run_dir: Path,
    direction: str = "abs",
    cov_metric: str = "sigma_major_pred_arcsec_mean",
    area_metric: str = "ellipse_area_pred_deg2_mean",
    bytes_per_obs: float | None = None,
    out_dir: Path | None = None,
) -> list[Path]:
    """
    Generate *two* plots per orbit_id into a single folder:
      - cov_metric vs expected obs (scatter)
      - area_metric vs expected obs (scatter)
    """
    run_dir = Path(run_dir)
    if out_dir is None:
        out_dir = run_dir / "orbit_two_plots_named"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = read_stage5_meta(run_dir=run_dir)
    name_map: dict[str, str] = {}
    try:
        s2 = meta.get("stage2_run_dir")
        if s2:
            name_map = _orbit_name_map_from_stage2_meta(stage2_run_dir=Path(str(s2)))
    except Exception:  # noqa: BLE001
        name_map = {}

    # Pull orbit_ids from the timeseries parquet. If direction="signed", the underlying
    # parquet has "abs" (always) + "backward"/"forward"; use "abs" just for the list.
    df_dir = "abs" if str(direction) == "signed" else str(direction)
    df = read_stage5_timeseries(run_dir=run_dir, direction=str(df_dir))
    orbit_ids = sorted({str(x) for x in df["orbit_id"].dropna().unique().tolist()})

    outs: list[Path] = []
    for oid in orbit_ids:
        pretty = name_map.get(str(oid), str(oid))
        slug = _slugify(pretty)

        def _unit_for_metric(m: str) -> str | None:
            mm = str(m).lower()
            if "arcsec" in mm:
                return "arcsec"
            if "_km" in mm or mm.endswith("km"):
                return "km"
            if "deg2" in mm:
                return "deg^2"
            return None

        out1 = plot_metric_vs_expected_obs(
            run_dir=run_dir,
            direction=str(direction),
            metric=str(cov_metric),
            metric_label=str(cov_metric),
            metric_unit=_unit_for_metric(str(cov_metric)),
            bytes_per_obs=bytes_per_obs,
            orbit_id=str(oid),
            out_png=out_dir
            / f"stage5_{_slugify(str(cov_metric))}_vs_expected_{direction}_orbit_{slug}_id{oid}.png",
            title=(
                f"Stage5 covsize vs expected obs (dir={direction}, orbit={pretty} (id={oid}))"
            ),
        )
        out2 = plot_metric_vs_expected_obs(
            run_dir=run_dir,
            direction=str(direction),
            metric=str(area_metric),
            metric_label=str(area_metric),
            metric_unit=_unit_for_metric(str(area_metric)),
            distance_metric="rho_au_min" if "rho_au_min" in df.columns else None,
            distance_unit="AU",
            bytes_per_obs=bytes_per_obs,
            orbit_id=str(oid),
            out_png=out_dir
            / f"stage5_{_slugify(str(area_metric))}_vs_expected_{direction}_orbit_{slug}_id{oid}.png",
            title=(
                f"Stage5 area vs expected obs (dir={direction}, orbit={pretty} (id={oid}))"
            ),
        )
        outs.append(Path(out1))
        outs.append(Path(out2))
    return outs


def plot_orbit_two_plots_stacked_signed_all_orbits(
    *,
    run_dir: Path,
    cov_metric: str = "sigma_major_pos_km_mean",
    cov_metric_unit: str | None = None,
    area_metric: str = "ellipse_area_pred_deg2_mean",
    area_metric_unit: str | None = "deg^2",
    distance_metric: str | None = "rho_au_min",
    distance_unit: str | None = "AU",
    bytes_per_obs: float | None = None,
    out_dir: Path | None = None,
) -> list[Path]:
    """
    For each orbit_id, write ONE PNG containing two stacked signed-Δt plots:
      - top: cov_metric vs expected obs
      - bottom: area_metric vs expected obs (+ optional distance on 3rd axis)
    """
    try:
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("matplotlib is required for visualization") from e

    run_dir = Path(run_dir)
    if out_dir is None:
        out_dir = run_dir / "orbit_two_plots_stacked_signed"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = read_stage5_meta(run_dir=run_dir)

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
        raise ValueError("bytes_per_obs must be provided (or present in meta.json) and > 0")

    try:
        nside = int(meta.get("healpix_nside", 0) or 0)
    except Exception:  # noqa: BLE001
        nside = 0
    if nside <= 0:
        raise ValueError("meta.json is missing a valid healpix_nside")
    pix_area_deg2 = (
        (4.0 * np.pi) * (180.0 / np.pi) ** 2 / (12.0 * float(nside) * float(nside))
    )

    # friendly names
    name_map: dict[str, str] = {}
    try:
        s2 = meta.get("stage2_run_dir")
        if s2:
            name_map = _orbit_name_map_from_stage2_meta(stage2_run_dir=Path(str(s2)))
    except Exception:  # noqa: BLE001
        name_map = {}

    # orbit list from abs direction (always present)
    df_abs = read_stage5_timeseries(run_dir=run_dir, direction="abs")
    orbit_ids = sorted({str(x) for x in df_abs["orbit_id"].dropna().unique().tolist()})

    outs: list[Path] = []
    for oid in orbit_ids:
        pretty = name_map.get(str(oid), str(oid))
        slug = _slugify(pretty)

        df_back = read_stage5_timeseries(run_dir=run_dir, direction="backward")
        df_fwd = read_stage5_timeseries(run_dir=run_dir, direction="forward")
        df_back = df_back[df_back["orbit_id"] == str(oid)].copy()
        df_fwd = df_fwd[df_fwd["orbit_id"] == str(oid)].copy()
        if df_back.empty and df_fwd.empty:
            continue

        df_back["dt_mid_days"] = -df_back["abs_dt_mid_days"].to_numpy(dtype=float)
        df_fwd["dt_mid_days"] = df_fwd["abs_dt_mid_days"].to_numpy(dtype=float)
        df_back = df_back.sort_values("dt_mid_days").reset_index(drop=True)
        df_fwd = df_fwd.sort_values("dt_mid_days").reset_index(drop=True)

        df_back, ylab = _attach_expected_obs(
            df_back,
            bytes_per_obs=float(bpo),
            area_metric="ellipse_area_pred_deg2_mean",
            pix_area_deg2=float(pix_area_deg2),
        )
        df_fwd, _ylab2 = _attach_expected_obs(
            df_fwd,
            bytes_per_obs=float(bpo),
            area_metric="ellipse_area_pred_deg2_mean",
            pix_area_deg2=float(pix_area_deg2),
        )

        # Prepare arrays
        x_b = df_back["dt_mid_days"].to_numpy(dtype=float)
        x_f = df_fwd["dt_mid_days"].to_numpy(dtype=float)
        e_b = df_back[_EXP_COL].to_numpy(dtype=float)
        e_f = df_fwd[_EXP_COL].to_numpy(dtype=float)

        fig, (ax_top, ax_bot) = plt.subplots(
            2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [1, 1]}
        )
        ax_top_r = ax_top.twinx()
        ax_bot_r = ax_bot.twinx()
        ax_bot_dist = None

        # --- Top: physical cov metric
        yb = df_back[str(cov_metric)].to_numpy(dtype=float)
        yf = df_fwd[str(cov_metric)].to_numpy(dtype=float)
        ax_top.plot(x_b, yb, color="C0", linestyle="--", label="_nolegend_")
        ax_top.plot(x_f, yf, color="C0", linestyle="-", label=str(cov_metric))
        ax_top_r.scatter(x_b, e_b, s=12, alpha=0.9, color="C1", marker="x", label="_nolegend_")
        ax_top_r.scatter(x_f, e_f, s=10, alpha=0.9, color="C1", marker="o", label=str(ylab))
        ax_top.set_ylabel(
            f"{cov_metric}{(' ('+cov_metric_unit+')') if cov_metric_unit else ''}"
        )
        ax_top_r.set_ylabel(ylab)
        ax_top_r.set_ylim(bottom=0.0)
        ax_top.grid(True, which="both", linestyle=":", linewidth=0.5)

        # --- Bottom: on-sky area (+ optional distance)
        yb2 = df_back[str(area_metric)].to_numpy(dtype=float)
        yf2 = df_fwd[str(area_metric)].to_numpy(dtype=float)
        ax_bot.plot(x_b, yb2, color="C0", linestyle="--", label="_nolegend_")
        ax_bot.plot(x_f, yf2, color="C0", linestyle="-", label=str(area_metric))
        ax_bot_r.scatter(x_b, e_b, s=12, alpha=0.9, color="C1", marker="x", label="_nolegend_")
        ax_bot_r.scatter(x_f, e_f, s=10, alpha=0.9, color="C1", marker="o", label=str(ylab))
        ax_bot.set_ylabel(
            f"{area_metric}{(' ('+area_metric_unit+')') if area_metric_unit else ''}"
        )
        ax_bot_r.set_ylabel(ylab)
        ax_bot_r.set_ylim(bottom=0.0)
        ax_bot.grid(True, which="both", linestyle=":", linewidth=0.5)

        if (
            distance_metric is not None
            and str(distance_metric) in df_back.columns
            and str(distance_metric) in df_fwd.columns
        ):
            ax_bot_dist = ax_bot.twinx()
            ax_bot_dist.spines["right"].set_position(("outward", 60))
            db = df_back[str(distance_metric)].to_numpy(dtype=float)
            dfv = df_fwd[str(distance_metric)].to_numpy(dtype=float)
            ax_bot_dist.plot(x_b, db, color="C2", linestyle="--", label="_nolegend_")
            ax_bot_dist.plot(x_f, dfv, color="C2", linestyle="-", label=str(distance_metric))
            ax_bot_dist.set_ylabel(
                f"{distance_metric}{(' ('+distance_unit+')') if distance_unit else ''}"
            )

        ax_bot.set_xlabel("Δt (days)")

        # Legends: only include one label per series (no backward/forward text)
        def _legend(ax):
            hs, ls = ax.get_legend_handles_labels()
            hs = [h for h, l in zip(hs, ls) if l and l != "_nolegend_"]
            ls = [l for l in ls if l and l != "_nolegend_"]
            return hs, ls

        h1, l1 = _legend(ax_top)
        h2, l2 = _legend(ax_top_r)
        ax_top.legend(h1 + h2, l1 + l2, loc="best")

        hb1, lb1 = _legend(ax_bot)
        hb2, lb2 = _legend(ax_bot_r)
        if ax_bot_dist is not None:
            hb3, lb3 = _legend(ax_bot_dist)
            ax_bot.legend(hb1 + hb2 + hb3, lb1 + lb2 + lb3, loc="best")
        else:
            ax_bot.legend(hb1 + hb2, lb1 + lb2, loc="best")

        ns = meta.get("n_sigma")
        fp = meta.get("footprint")
        frac = meta.get("fractional_nside")
        fig.suptitle(
            f"Stage5 signed Δt (orbit={pretty} (id={oid}), nσ={ns}, fp={fp}, frac_nside={frac}, bytes/obs={bpo:g})"
        )

        out_png = out_dir / f"stage5_signed_stacked_orbit_{slug}_id{oid}.png"
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(out_png, dpi=200)
        plt.close(fig)
        outs.append(Path(out_png))

    return outs


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(
        description="Stage 5 visualization: covariance area vs expected observations."
    )
    p.add_argument("--run-dir", type=str, required=True)
    p.add_argument(
        "--plot",
        type=str,
        default="area_expected",
        choices=["area_expected", "panels", "orbit_two_plots", "orbit_two_plots_stacked"],
        help="Which plot to generate.",
    )
    p.add_argument(
        "--direction",
        type=str,
        default="abs",
        choices=["abs", "backward", "forward", "signed"],
    )
    p.add_argument(
        "--orbit-id",
        type=str,
        default=None,
        help="Optional: restrict plot to a single orbit_id (recommended for this exploration).",
    )
    p.add_argument(
        "--area-metric",
        type=str,
        default="ellipse_area_pred_deg2_mean",
        help="Timeseries column to use for covariance area in area_expected plot.",
    )
    p.add_argument(
        "--cov-metric",
        type=str,
        default="sigma_major_pred_arcsec_mean",
        help="Timeseries column to use for covariance size in panels plot.",
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
    p.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory when --all-orbits is set.",
    )
    p.add_argument("--out-png", type=str, default=None)
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    if bool(args.all_orbits):
        if str(args.plot) == "orbit_two_plots_stacked":
            outs = plot_orbit_two_plots_stacked_signed_all_orbits(
                run_dir=run_dir,
                cov_metric=str(args.cov_metric),
                cov_metric_unit=(
                    "km" if "_km" in str(args.cov_metric).lower() else None
                ),
                area_metric=str(args.area_metric),
                area_metric_unit="deg^2" if "deg2" in str(args.area_metric).lower() else None,
                distance_metric="rho_au_min",
                distance_unit="AU",
                bytes_per_obs=(
                    None if args.bytes_per_obs is None else float(args.bytes_per_obs)
                ),
                out_dir=(None if args.out_dir is None else Path(args.out_dir)),
            )
        elif str(args.plot) == "orbit_two_plots":
            outs = plot_orbit_two_plots_all_orbits(
                run_dir=run_dir,
                direction=str(args.direction),
                cov_metric=str(args.cov_metric),
                area_metric=str(args.area_metric),
                bytes_per_obs=(
                    None if args.bytes_per_obs is None else float(args.bytes_per_obs)
                ),
                out_dir=(None if args.out_dir is None else Path(args.out_dir)),
            )
        else:
            outs = plot_all_orbits(
                run_dir=run_dir,
                direction=str(args.direction),
                area_metric=str(args.area_metric),
                bytes_per_obs=(
                    None if args.bytes_per_obs is None else float(args.bytes_per_obs)
                ),
                out_dir=(None if args.out_dir is None else Path(args.out_dir)),
            )
        print(f"n_orbits={len(outs)}")
        if outs:
            print(f"out_dir={Path(outs[0]).parent}")
    else:
        if str(args.plot) == "orbit_two_plots":
            if args.orbit_id is None:
                raise ValueError("--plot orbit_two_plots requires --orbit-id")
            oid = str(args.orbit_id)
            # 1) absolute covariance size (sigma-major) + expected obs
            out1 = plot_metric_vs_expected_obs(
                run_dir=run_dir,
                direction=str(args.direction),
                metric=str(args.cov_metric),
                metric_label=str(args.cov_metric),
                metric_unit=(
                    "arcsec"
                    if "arcsec" in str(args.cov_metric).lower()
                    else ("km" if "_km" in str(args.cov_metric).lower() else None)
                ),
                bytes_per_obs=(
                    None if args.bytes_per_obs is None else float(args.bytes_per_obs)
                ),
                orbit_id=oid,
                out_png=None,
            )
            # 2) on-sky covariance area + expected obs
            out2 = plot_metric_vs_expected_obs(
                run_dir=run_dir,
                direction=str(args.direction),
                metric=str(args.area_metric),
                metric_label=str(args.area_metric),
                metric_unit="deg^2" if "deg2" in str(args.area_metric) else None,
                distance_metric="rho_au_min",
                distance_unit="AU",
                bytes_per_obs=(
                    None if args.bytes_per_obs is None else float(args.bytes_per_obs)
                ),
                orbit_id=oid,
                out_png=None,
            )
            print(f"out_png_1={out1}")
            print(f"out_png_2={out2}")
            return
        if str(args.plot) == "panels":
            out_png = plot_covsize_and_expected_obs_panels(
                run_dir=run_dir,
                direction=str(args.direction),
                cov_metric=str(args.cov_metric),
                bytes_per_obs=(
                    None if args.bytes_per_obs is None else float(args.bytes_per_obs)
                ),
                orbit_id=(None if args.orbit_id is None else str(args.orbit_id)),
                out_png=None if args.out_png is None else Path(args.out_png),
            )
        else:
            out_png = plot_area_vs_expected_obs(
                run_dir=run_dir,
                direction=str(args.direction),
                area_metric=str(args.area_metric),
                bytes_per_obs=(
                    None if args.bytes_per_obs is None else float(args.bytes_per_obs)
                ),
                orbit_id=(None if args.orbit_id is None else str(args.orbit_id)),
                out_png=None if args.out_png is None else Path(args.out_png),
            )
        print(f"out_png={out_png}")


if __name__ == "__main__":
    main()
