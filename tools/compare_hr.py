"""Compare rPPG-predicted HR windows against a ground-truth HR time series.

Ground-truth formats supported
-------------------------------
* CSV with a UTC datetime column and one or more numeric HR columns (e.g.,
  the signal.csv files produced by the vHRM data-preprocessing tool).
* CSV with a plain numeric seconds column and one or more HR columns.
* CSV with no time column; rows are assumed equally spaced and --gt-fs is
  required.

Time alignment
--------------
Ground-truth timestamps are absolute (UTC); prediction times are relative to
the video start.  To overlay them you must supply one of:

  --video-start-utc  ISO-8601 string for when the video recording began
                     (e.g. "2026-06-04T14:59:18.428000+00:00").  The tool
                     converts video-relative seconds to GT elapsed time.

  --gt-offset-sec    Already-computed offset: the number of GT elapsed seconds
                     that corresponds to video t = 0.  Positive means the GT
                     started N seconds before the video; negative means after.

If neither is given the tool still plots both signals on independent elapsed-
time axes but skips numeric comparison metrics.

Outputs (all written to --output-dir)
--------------------------------------
  hr_comparison.png        HR time-series overlay (GT + all selected methods)
  bland_altman_<m>.png     Bland-Altman plot for each method vs GT
  scatter_<m>.png          Predicted vs GT scatter for each method
  aligned_signals.csv      GT interpolated to each prediction window midpoint
  metrics.csv              MAE, RMSE, MAPE, Pearson r, mean bias per method

Usage example
-------------
  cd C:\\Projects\\vHRM2\\code\\rPPG-Toolbox
  python tools/compare_hr.py ^
      "C:\\Projects\\vHRM2\\data\\jzc14_baseline\\0001_Baseline_part2\\signal.csv" ^
      "C:\\Projects\\vHRM2\\runs\\jzc14_baseline_cam2_Front_unsupervised\\hr_predictions.csv" ^
      --output-dir "C:\\Projects\\vHRM2\\runs\\jzc14_baseline_cam2_Front_unsupervised\\comparison" ^
      --gt-column averaged ^
      --video-start-utc "2026-06-04T14:59:18.428000+00:00"
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import scipy.stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare ground-truth HR against rPPG toolbox predictions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage example")[1] if "Usage example" in __doc__ else "",
    )
    parser.add_argument("gt_csv", type=Path, help="Ground-truth HR CSV path.")
    parser.add_argument("pred_csv", type=Path, help="Predictions CSV from single_video_unsupervised.py.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Folder for all output files (default: <pred_csv parent>/comparison/).",
    )
    parser.add_argument(
        "--gt-time-column",
        default=None,
        help="Name of the timestamp/time column in gt_csv. Auto-detected if omitted.",
    )
    parser.add_argument(
        "--gt-column",
        default=None,
        help="Name of the HR column to use from gt_csv. Auto-detected if omitted "
             "(prefers 'averaged').",
    )
    parser.add_argument(
        "--gt-fs",
        type=float,
        default=None,
        help="GT sampling rate in Hz. Used only if gt_csv has no time column.",
    )
    parser.add_argument(
        "--video-start-utc",
        default=None,
        help="ISO-8601 UTC timestamp when the video started recording. "
             "Used to align GT timestamps with video-relative prediction times.",
    )
    parser.add_argument(
        "--gt-offset-sec",
        type=float,
        default=None,
        help="GT elapsed seconds that corresponds to video t = 0. "
             "Alternative to --video-start-utc when timestamps are not available.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Prediction methods to include (default: all found in pred_csv).",
    )
    parser.add_argument(
        "--no-bland-altman",
        action="store_true",
        help="Skip Bland-Altman plots.",
    )
    parser.add_argument(
        "--no-scatter",
        action="store_true",
        help="Skip scatter plots.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _detect_time_column(df):
    """Return the name of the first column that looks like a time axis."""
    for col in df.columns:
        low = col.lower()
        if "time" in low or "timestamp" in low or "date" in low:
            return col
    return None


def _detect_hr_column(df, time_col):
    """Return a numeric HR column, preferring 'averaged'."""
    candidates = [
        c for c in df.columns
        if c != time_col and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not candidates:
        raise ValueError(
            f"No numeric columns found in GT CSV (excluding time column '{time_col}')."
        )
    if "averaged" in candidates:
        return "averaged"
    return candidates[-1]


def load_gt(gt_csv_path, time_col_hint, hr_col_hint, fs_hint):
    """Load ground-truth CSV.

    Returns
    -------
    elapsed_sec : np.ndarray   seconds from the first sample
    hr_bpm      : np.ndarray
    hr_col      : str          name of the HR column used
    gt_start_utc: pd.Timestamp | None
    """
    df = pd.read_csv(gt_csv_path)
    if df.empty:
        raise ValueError(f"GT CSV is empty: {gt_csv_path}")

    time_col = time_col_hint or _detect_time_column(df)
    hr_col = hr_col_hint or _detect_hr_column(df, time_col)

    if hr_col not in df.columns:
        raise ValueError(
            f"HR column '{hr_col}' not found in GT CSV. "
            f"Available columns: {list(df.columns)}"
        )

    gt_start_utc = None
    if time_col and time_col in df.columns:
        try:
            times = pd.to_datetime(df[time_col], utc=True, format="mixed")
            gt_start_utc = times.iloc[0]
            elapsed_sec = (times - gt_start_utc).dt.total_seconds().values.astype(float)
        except Exception:
            # Fallback: treat as plain numeric seconds
            elapsed_sec = df[time_col].astype(float).values
    elif fs_hint is not None:
        elapsed_sec = np.arange(len(df), dtype=float) / fs_hint
    else:
        # Last resort: assume 2 Hz (common for ECG/HR monitors)
        print("Warning: no time column detected and --gt-fs not given. Assuming 2 Hz.")
        elapsed_sec = np.arange(len(df), dtype=float) / 2.0

    hr_bpm = df[hr_col].astype(float).values
    return elapsed_sec, hr_bpm, hr_col, gt_start_utc


def load_predictions(pred_csv_path, methods_filter):
    """Return {method: DataFrame} with one entry per method."""
    df = pd.read_csv(pred_csv_path)
    required = {"method", "start_time_sec", "end_time_sec", "predicted_hr_bpm"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"pred_csv is missing columns: {missing}. "
            "Expected output from single_video_unsupervised.py."
        )
    all_methods = list(df["method"].unique())
    if methods_filter:
        chosen = [m for m in all_methods if m in methods_filter]
        unknown = [m for m in methods_filter if m not in all_methods]
        if unknown:
            print(f"Warning: requested methods not found in pred_csv: {unknown}")
    else:
        chosen = all_methods
    return {m: df[df["method"] == m].copy().reset_index(drop=True) for m in chosen}


# ---------------------------------------------------------------------------
# Time alignment
# ---------------------------------------------------------------------------

def compute_offset(video_start_utc_str, gt_start_utc, gt_offset_sec):
    """Return GT elapsed seconds corresponding to video t = 0, or None."""
    if gt_offset_sec is not None:
        return float(gt_offset_sec)
    if video_start_utc_str is not None:
        if gt_start_utc is None:
            print(
                "Warning: --video-start-utc supplied but GT has no parseable UTC "
                "timestamps. Alignment skipped."
            )
            return None
        video_start = pd.to_datetime(video_start_utc_str, utc=True)
        return float((video_start - gt_start_utc).total_seconds())
    return None


def interp_gt_at(gt_elapsed, gt_hr, query_sec):
    """Linearly interpolate GT HR at query_sec (GT-elapsed axis)."""
    return np.interp(query_sec, gt_elapsed, gt_hr, left=np.nan, right=np.nan)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(gt_hr, pred_hr):
    valid = ~np.isnan(gt_hr) & ~np.isnan(pred_hr)
    n = int(valid.sum())
    if n < 2:
        return dict(n=n, mae=np.nan, rmse=np.nan, mape=np.nan,
                    mean_bias=np.nan, pearson_r=np.nan, pearson_p=np.nan)
    g, p = gt_hr[valid], pred_hr[valid]
    mae = float(np.mean(np.abs(p - g)))
    rmse = float(np.sqrt(np.mean((p - g) ** 2)))
    mape = float(np.mean(np.abs((p - g) / g)) * 100) if np.all(g != 0) else np.nan
    bias = float(np.mean(p - g))
    r, pval = scipy.stats.pearsonr(g, p)
    return dict(n=n, mae=mae, rmse=rmse, mape=mape,
                mean_bias=bias, pearson_r=float(r), pearson_p=float(pval))


# ---------------------------------------------------------------------------
# CSV outputs
# ---------------------------------------------------------------------------

def save_aligned_csv(path, gt_elapsed, gt_hr, gt_col, pred_by_method,
                     gt_at_midpoints, offset):
    """Save one row per prediction window midpoint with GT + all methods."""
    # Use the first method to build the time grid
    first_df = next(iter(pred_by_method.values()))
    midpoints_video = (
        (first_df["start_time_sec"] + first_df["end_time_sec"]) / 2
    ).values
    midpoints_gt = midpoints_video + offset

    rows = []
    for i, (t_video, t_gt) in enumerate(zip(midpoints_video, midpoints_gt)):
        row = {
            "window_midpoint_video_sec": round(float(t_video), 4),
            "window_midpoint_gt_elapsed_sec": round(float(t_gt), 4),
            f"gt_{gt_col}_bpm": round(float(gt_at_midpoints[next(iter(pred_by_method))][i]), 4),
        }
        for method, df_pred in pred_by_method.items():
            if i < len(df_pred):
                row[f"pred_{method}_bpm"] = round(
                    float(df_pred["predicted_hr_bpm"].iloc[i]), 4
                )
            else:
                row[f"pred_{method}_bpm"] = ""
        rows.append(row)

    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved aligned signals: {path}")


def save_metrics_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved metrics: {path}")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

_METHOD_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45",
]


def plot_hr_comparison(path, gt_elapsed, gt_hr, gt_col, pred_by_method,
                       offset, has_alignment):
    fig, ax = plt.subplots(figsize=(14, 5))

    # Ground truth (continuous line)
    ax.plot(gt_elapsed, gt_hr, color="black", linewidth=1.5,
            label=f"GT — {gt_col}", zorder=10)

    for i, (method, df_pred) in enumerate(pred_by_method.items()):
        midpoints = (df_pred["start_time_sec"] + df_pred["end_time_sec"]) / 2
        x = (midpoints + offset).values if has_alignment else midpoints.values
        y = df_pred["predicted_hr_bpm"].values
        color = _METHOD_COLORS[i % len(_METHOD_COLORS)]
        ax.step(x, y, where="mid", color=color, linewidth=1.5,
                marker="o", markersize=5, label=method)

    if has_alignment:
        ax.set_xlabel("GT elapsed time (s)")
        ax.set_title("Heart rate comparison — aligned")
    else:
        ax.set_xlabel("Elapsed time (s) — GT and video axes are independent")
        ax.set_title("Heart rate comparison — no time alignment")

    ax.set_ylabel("Heart rate (bpm)")
    ax.legend(ncol=4, fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    print(f"Saved HR comparison plot: {path}")


def plot_bland_altman(output_dir, pred_by_method, gt_at_midpoints, gt_col):
    for i, (method, df_pred) in enumerate(pred_by_method.items()):
        gt_hr = gt_at_midpoints.get(method)
        if gt_hr is None:
            continue
        pred_hr = df_pred["predicted_hr_bpm"].values.astype(float)
        valid = ~np.isnan(gt_hr) & ~np.isnan(pred_hr)
        if valid.sum() < 2:
            print(f"  Skipping Bland-Altman for {method}: fewer than 2 valid pairs.")
            continue

        g, p = gt_hr[valid], pred_hr[valid]
        mean = (g + p) / 2.0
        diff = p - g
        md = float(np.mean(diff))
        sd = float(np.std(diff, ddof=1))
        lo = md - 1.96 * sd
        hi = md + 1.96 * sd

        color = _METHOD_COLORS[i % len(_METHOD_COLORS)]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(mean, diff, color=color, alpha=0.75, s=60, zorder=5)
        ax.axhline(md, color="navy", linewidth=1.5, linestyle="--",
                   label=f"Bias = {md:+.2f} bpm")
        ax.axhline(hi, color="firebrick", linewidth=1.2, linestyle=":",
                   label=f"+1.96 SD = {hi:+.2f} bpm")
        ax.axhline(lo, color="firebrick", linewidth=1.2, linestyle=":",
                   label=f"−1.96 SD = {lo:+.2f} bpm")
        ax.axhline(0, color="grey", linewidth=0.8, linestyle="-")
        ax.set_xlabel("Mean of GT and predicted (bpm)")
        ax.set_ylabel("Predicted − GT (bpm)")
        ax.set_title(f"Bland-Altman: {method} vs GT ({gt_col})")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        out = output_dir / f"bland_altman_{method}.png"
        plt.savefig(out, dpi=160)
        plt.close()
        print(f"Saved Bland-Altman: {out}")


def plot_scatter(output_dir, pred_by_method, gt_at_midpoints, gt_col):
    for i, (method, df_pred) in enumerate(pred_by_method.items()):
        gt_hr = gt_at_midpoints.get(method)
        if gt_hr is None:
            continue
        pred_hr = df_pred["predicted_hr_bpm"].values.astype(float)
        valid = ~np.isnan(gt_hr) & ~np.isnan(pred_hr)
        if valid.sum() < 2:
            print(f"  Skipping scatter for {method}: fewer than 2 valid pairs.")
            continue

        g, p = gt_hr[valid], pred_hr[valid]
        r, pval = scipy.stats.pearsonr(g, p)

        lo = min(g.min(), p.min()) - 5
        hi = max(g.max(), p.max()) + 5
        color = _METHOD_COLORS[i % len(_METHOD_COLORS)]

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(g, p, color=color, alpha=0.75, s=60, zorder=5)
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="Identity")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel(f"GT HR — {gt_col} (bpm)")
        ax.set_ylabel(f"Predicted HR — {method} (bpm)")
        ax.set_title(f"Scatter: {method} vs GT  (r = {r:.3f}, p = {pval:.3f})")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")
        plt.tight_layout()
        out = output_dir / f"scatter_{method}.png"
        plt.savefig(out, dpi=160)
        plt.close()
        print(f"Saved scatter: {out}")


def plot_combined_bland_altman(path, pred_by_method, gt_at_midpoints, gt_col):
    """One figure with all methods as subplots side by side."""
    valid_methods = [
        m for m in pred_by_method
        if m in gt_at_midpoints and gt_at_midpoints[m] is not None
        and (~np.isnan(gt_at_midpoints[m])).sum() >= 2
    ]
    if not valid_methods:
        return

    ncols = min(4, len(valid_methods))
    nrows = int(np.ceil(len(valid_methods) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows),
                              squeeze=False)

    for idx, method in enumerate(valid_methods):
        ax = axes[idx // ncols][idx % ncols]
        gt_hr = gt_at_midpoints[method]
        pred_hr = pred_by_method[method]["predicted_hr_bpm"].values.astype(float)
        valid = ~np.isnan(gt_hr) & ~np.isnan(pred_hr)
        g, p = gt_hr[valid], pred_hr[valid]
        mean = (g + p) / 2.0
        diff = p - g
        md = float(np.mean(diff))
        sd = float(np.std(diff, ddof=1))
        color = _METHOD_COLORS[idx % len(_METHOD_COLORS)]

        ax.scatter(mean, diff, color=color, alpha=0.75, s=40)
        ax.axhline(md, color="navy", linewidth=1.3, linestyle="--",
                   label=f"Bias {md:+.1f}")
        ax.axhline(md + 1.96 * sd, color="firebrick", linewidth=1, linestyle=":")
        ax.axhline(md - 1.96 * sd, color="firebrick", linewidth=1, linestyle=":",
                   label=f"±1.96 SD")
        ax.axhline(0, color="grey", linewidth=0.7)
        ax.set_title(method, fontsize=10)
        ax.set_xlabel("Mean (bpm)", fontsize=8)
        ax.set_ylabel("Pred − GT (bpm)", fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(len(valid_methods), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle(f"Bland-Altman — all methods vs GT ({gt_col})", fontsize=12)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    print(f"Saved combined Bland-Altman: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    gt_path = args.gt_csv.resolve()
    pred_path = args.pred_csv.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else pred_path.parent / "comparison"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"GT CSV:          {gt_path}")
    print(f"Predictions CSV: {pred_path}")
    print(f"Output dir:      {output_dir}")

    # --- load ---
    gt_elapsed, gt_hr, gt_col, gt_start_utc = load_gt(
        gt_path,
        time_col_hint=args.gt_time_column,
        hr_col_hint=args.gt_column,
        fs_hint=args.gt_fs,
    )
    print(
        f"\nGround truth: {len(gt_elapsed)} samples, column='{gt_col}', "
        f"span={gt_elapsed[-1]:.1f}s"
        + (f", starts at {gt_start_utc}" if gt_start_utc else "")
    )

    pred_by_method = load_predictions(pred_path, args.methods)
    print(f"Predictions:  {list(pred_by_method.keys())}")

    # --- align ---
    offset = compute_offset(args.video_start_utc, gt_start_utc, args.gt_offset_sec)
    has_alignment = offset is not None
    if has_alignment:
        print(f"\nTime alignment: video t=0  ↔  GT elapsed {offset:.3f}s")
        if gt_start_utc is not None and args.video_start_utc:
            print(f"  GT starts at:    {gt_start_utc.isoformat()}")
            print(f"  Video starts at: {pd.to_datetime(args.video_start_utc, utc=True).isoformat()}")
    else:
        print(
            "\nNo time alignment. Plots will show GT and predictions on independent "
            "elapsed-time axes. Metrics will not be computed.\n"
            "Tip: supply --video-start-utc or --gt-offset-sec to enable alignment."
        )

    # --- interpolate GT at prediction midpoints ---
    gt_at_midpoints: dict[str, np.ndarray] = {}
    if has_alignment:
        for method, df_pred in pred_by_method.items():
            midpoints = (
                (df_pred["start_time_sec"] + df_pred["end_time_sec"]) / 2
            ).values
            gt_query = midpoints + offset  # GT-elapsed seconds
            gt_at_midpoints[method] = interp_gt_at(gt_elapsed, gt_hr, gt_query)
            n_valid = int(np.sum(~np.isnan(gt_at_midpoints[method])))
            n_total = len(gt_at_midpoints[method])
            print(
                f"  {method}: {n_valid}/{n_total} windows have GT coverage"
            )

    # --- metrics ---
    metrics_rows = []
    if has_alignment:
        print()
        for method, df_pred in pred_by_method.items():
            gt_vals = gt_at_midpoints.get(method)
            if gt_vals is None:
                continue
            pred_vals = df_pred["predicted_hr_bpm"].values.astype(float)
            m = compute_metrics(gt_vals, pred_vals)
            row = {"method": method, "gt_column": gt_col, **m}
            metrics_rows.append(row)
            print(
                f"  {method:8s}  n={m['n']:3d}  "
                f"MAE={m['mae']:.2f}  RMSE={m['rmse']:.2f}  "
                f"r={m['pearson_r']:.3f}  bias={m['mean_bias']:+.2f} bpm"
            )

    # --- save CSVs ---
    if has_alignment and pred_by_method:
        save_aligned_csv(
            output_dir / "aligned_signals.csv",
            gt_elapsed, gt_hr, gt_col,
            pred_by_method, gt_at_midpoints, offset,
        )
    if metrics_rows:
        save_metrics_csv(output_dir / "metrics.csv", metrics_rows)

    # --- plots ---
    print()
    plot_hr_comparison(
        output_dir / "hr_comparison.png",
        gt_elapsed, gt_hr, gt_col,
        pred_by_method, offset, has_alignment,
    )

    if has_alignment and gt_at_midpoints:
        if not args.no_bland_altman:
            plot_combined_bland_altman(
                output_dir / "bland_altman_all.png",
                pred_by_method, gt_at_midpoints, gt_col,
            )
            plot_bland_altman(output_dir, pred_by_method, gt_at_midpoints, gt_col)
        if not args.no_scatter:
            plot_scatter(output_dir, pred_by_method, gt_at_midpoints, gt_col)
    elif has_alignment:
        print("No aligned GT values available — skipping Bland-Altman and scatter plots.")

    print(f"\nDone. Outputs: {output_dir}")


if __name__ == "__main__":
    main()
