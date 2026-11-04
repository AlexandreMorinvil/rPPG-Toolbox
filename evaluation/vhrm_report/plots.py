"""Figures: per-participant time series, mosaics of every participant, and global summaries."""

import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

from .data import condition_name, label_name
from .evaluate import METRICS, conditions, label_order, metric_pairs, paired, recording_order

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
ESTIMATE = "#2a78d6"                       # categorical slot 1
# Categorical slots in fixed order, one per segment label (protocol order). The first three pass the
# all-pairs colour checks used by the scatter plots.
LABEL_SLOTS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
# Sequential blue ramp (steps 100-700) for the error heatmaps.
ERROR_RAMP = LinearSegmentedColormap.from_list("error", [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
    "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"])
ERROR_RAMP.set_bad(SURFACE)
GT_LABEL = {"hr": "Measured HR", "rmssd": "HRV from ECG R-peaks", "sdnn": "HRV from ECG R-peaks",
            "ibi": "ECG R-R interval"}
AXIS_LABELS = {"hr": "HR (bpm)", "rmssd": "RMSSD (ms)", "sdnn": "SDNN (ms)", "ibi": "RR / IBI (ms)"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
    "font.size": 9, "text.color": INK, "axes.labelcolor": INK_SECONDARY, "axes.titlesize": 9.5,
    "axes.titlecolor": INK, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
    "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK_SECONDARY,
    "ytick.labelcolor": INK_SECONDARY, "lines.linewidth": 1.6, "lines.solid_capstyle": "round",
    "legend.frameon": False, "legend.fontsize": 8.5,
})


def _save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _fmt(value, digits=1):
    return "n/a" if value is None or not np.isfinite(value) else f"{value:.{digits}f}"


def _finish(fig, title, handles=(), header_in=0.75):
    """Lay out the axes below a header: the title, then the legend (both left-aligned)."""
    height = fig.get_figheight()
    fig.tight_layout(rect=(0, 0, 1, 1 - header_in / height))
    fig.text(0.01, 1 - 0.10 / height, title, ha="left", va="top", fontsize=10.5, fontweight="semibold", color=INK)
    if handles:
        fig.legend(handles=list(handles), loc="upper left", bbox_to_anchor=(0.005, 1 - 0.36 / height),
                   ncol=len(handles), handlelength=1.8, columnspacing=1.6)


def _line(color, label, width=1.6, style="-", marker=None):
    return Line2D([], [], color=color, linewidth=width, linestyle=style, marker=marker,
                  markersize=4, label=label)


def label_colors(results):
    return {label: LABEL_SLOTS[index % len(LABEL_SLOTS)] for index, label in enumerate(label_order(results))}


def _several_views(results):
    return len({r["view"] for r in results}) > 1


def _condition(condition, with_view):
    label, repetition, view = condition
    return condition_name({"label": label, "repetition": repetition, "view": view}, with_view)


def condition_file(condition, with_view):
    label, repetition, view = condition
    return f"{label}{repetition}" + (f"_{view}" if with_view else "")


# ----------------------------------------------------------------------------- per-metric drawing

def draw_metric(ax, result, metric, detailed=True):
    """Estimate vs ground truth through time for one recording. Returns the panel's error stats."""
    s = result["series"]
    if metric == "hr":
        if detailed:
            ax.plot(s["hr_times"], s["hr_untracked"], color=MUTED, linewidth=0.8, alpha=0.7)
        ax.plot(s["hr_times"], s["hr_gt"], color=INK)
        ax.plot(s["hr_times"], s["hr"], color=ESTIMATE)
    elif metric in ("rmssd", "sdnn"):
        ax.plot(s["hrv_times"], s[f"{metric}_gt"], color=INK)
        ax.plot(s["hrv_times"], s[metric], color=ESTIMATE)
    elif metric == "ibi":
        valid = s["ibi_est_valid"].astype(bool)
        if detailed:
            ax.plot(s["ibi_est_times"][~valid], s["ibi_est"][~valid], "x", color=MUTED, markersize=3.5,
                    markeredgewidth=0.9)
        ax.plot(s["ibi_gt_times"], s["ibi_gt"], color=INK, linewidth=1.2)
        ax.plot(s["ibi_est_times"][valid], s["ibi_est"][valid], "o", color=ESTIMATE,
                markersize=3.2 if detailed else 2.2, markeredgecolor=SURFACE, markeredgewidth=0.5)
    ax.set_xlim(0, result["duration_s"])
    return paired(*metric_pairs(result, metric))


def _metric_limits(results, metric):
    """Shared y-range: ground truth range, widened by the bulk (5-95 %) of the estimates."""
    estimates, truths = [], []
    for result in results:
        estimate, truth = metric_pairs(result, metric)
        if metric == "ibi":
            valid = result["series"]["ibi_est_valid"].astype(bool)
            estimate = result["series"]["ibi_est"][valid]
        estimates.append(np.asarray(estimate, dtype=float))
        truths.append(np.asarray(truth, dtype=float))
    truth = np.concatenate(truths)
    estimate = np.concatenate(estimates)
    truth, estimate = truth[np.isfinite(truth)], estimate[np.isfinite(estimate)]
    if not truth.size:
        return None
    low = min(np.min(truth), np.percentile(estimate, 5) if estimate.size else np.inf)
    high = max(np.max(truth), np.percentile(estimate, 95) if estimate.size else -np.inf)
    pad = 0.08 * (high - low or 1)
    return max(0.0, low - pad), high + pad


def _metric_handles(metric, detailed):
    handles = [_line(INK, GT_LABEL[metric])]
    if metric == "ibi":
        handles.append(_line(ESTIMATE, "rPPG inter-beat interval", style="none", marker="o"))
        if detailed:
            handles.append(Line2D([], [], color=MUTED, linestyle="none", marker="x", markersize=4,
                                  label="rPPG interval rejected (artefact rule)"))
    else:
        handles.append(_line(ESTIMATE, "rPPG HR (tracked)" if metric == "hr" else "HRV from rPPG beats"))
        if metric == "hr" and detailed:
            handles.append(_line(MUTED, "rPPG HR, untracked PSD peak", width=0.8))
    return handles


# ----------------------------------------------------------------------------- per participant

def _name(result):
    """Participant, with its training split when the checkpoint has one (its fold in a cross-validation report)."""
    if result.get("fold"):
        return f"{result['participant']} · fold {result['fold']}"
    return f"{result['participant']} · {result['split']}" if result.get("split") else result["participant"]


def participant_figures(results, out_dir, model_title):
    """hr.png, hrv.png, rr.png and pulse.png for one participant (one row per recording)."""
    results = recording_order(results)
    participant = _name(results[0])
    rows = len(results)
    for name, metrics in (("hr", ["hr"]), ("hrv", ["rmssd", "sdnn"]), ("rr", ["ibi"])):
        fig, axes = plt.subplots(rows, len(metrics), figsize=(6.8 * len(metrics) ** 0.75, 2.6 * rows + 0.4),
                                 squeeze=False, sharey="col")
        for row, result in enumerate(results):
            for col, metric in enumerate(metrics):
                ax = axes[row, col]
                stats = draw_metric(ax, result, metric)
                unit = METRICS[metric][1]
                extra = (f" · rejected {_fmt(result['ibi_rejected_pct'], 0)} %" if metric == "ibi"
                         else f" · r {_fmt(stats['pearson'], 2)}")
                ax.set_title(f"{result['condition']} · {METRICS[metric][0]} · MAE {_fmt(stats['mae'])} {unit} · "
                             f"bias {_fmt(stats['bias'])}{extra}", loc="left")
                ax.set_ylabel(AXIS_LABELS[metric])
                if metric == "ibi" and _metric_limits(results, metric):
                    ax.set_ylim(*_metric_limits(results, metric))
                if row == rows - 1:
                    ax.set_xlabel("Time since segment start (s)")
        handles = []
        for metric in metrics:
            handles += [h for h in _metric_handles(metric, True) if h.get_label() not in {x.get_label() for x in handles}]
        _finish(fig, f"{participant} · {model_title}", handles)
        _save(fig, out_dir / f"{name}.png")
    _pulse_figure(results, out_dir / "pulse.png", f"{participant} · {model_title}")


def _pulse_figure(results, path, title, excerpt_sec=15.0):
    fig, axes = plt.subplots(len(results), 1, figsize=(9.5, 2.3 * len(results) + 0.4), squeeze=False)
    for ax, result in zip(axes[:, 0], results):
        s = result["series"]
        start = max(0.0, result["duration_s"] / 2 - excerpt_sec / 2)
        end = min(result["duration_s"], start + excerpt_sec)
        view = (s["time"] >= start) & (s["time"] <= end)
        wave = s["bvp"] / (np.std(s["bvp"]) or 1)
        for peak in s["rpeaks_gt"][(s["rpeaks_gt"] >= start) & (s["rpeaks_gt"] <= end)]:
            ax.axvline(peak, color=INK, linewidth=0.8, alpha=0.35)
        ax.plot(s["time"][view], wave[view], color=ESTIMATE, linewidth=1.3)
        beats = s["beats_est"][(s["beats_est"] >= start) & (s["beats_est"] <= end)]
        ax.plot(beats, np.interp(beats, s["time"], wave), "o", color=ESTIMATE, markersize=4.5,
                markeredgecolor=SURFACE, markeredgewidth=1.0)
        ax.set_xlim(start, end)
        ax.set_ylabel("Filtered rPPG (z)")
        ax.set_title(f"{result['condition']} · {excerpt_sec:.0f} s excerpt from the middle of the segment", loc="left")
    axes[-1, 0].set_xlabel("Time since segment start (s)")
    _finish(fig, title, [_line(ESTIMATE, "rPPG pulse signal (0.7-3.5 Hz)"),
                         _line(ESTIMATE, "detected rPPG beat", style="none", marker="o"),
                         _line(INK, "ECG R-peak", width=0.8)])
    _save(fig, path)


# ----------------------------------------------------------------------------- mosaics

def mosaic(results, metric, condition, path, model_title, columns=5):
    """One recording condition (e.g. curl 1), every participant on one page."""
    members = sorted((r for r in results if (r["label"], r["repetition"], r["view"]) == condition),
                     key=lambda r: (r["experiment"], r["participant"]))
    if not members:
        return False
    rows = math.ceil(len(members) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(3.1 * columns, 2.05 * rows + 0.6), squeeze=False)
    limits = _metric_limits(members, metric)
    unit = METRICS[metric][1]
    for ax, result in zip(axes.flat, members):
        stats = draw_metric(ax, result, metric, detailed=False)
        if limits:
            ax.set_ylim(*limits)
        ax.set_title(f"{_name(result)} · MAE {_fmt(stats['mae'])} {unit}", loc="left", fontsize=8.5)
        ax.tick_params(labelsize=7.5)
    for ax in axes.flat[len(members):]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel(AXIS_LABELS[metric], fontsize=8)
    for ax in axes[-1, :]:
        ax.set_xlabel("Time (s)", fontsize=8)
    pooled = paired(np.concatenate([metric_pairs(r, metric)[0] for r in members]),
                    np.concatenate([metric_pairs(r, metric)[1] for r in members]))
    _finish(fig, f"{METRICS[metric][0]} · {_condition(condition, _several_views(results))} · {model_title} · "
                 f"pooled MAE {_fmt(pooled['mae'])} {unit} (n = {pooled['n']})", _metric_handles(metric, False))
    _save(fig, path)
    return True


def face_crop_mosaic(results, path):
    """The model input at mid-segment: one row per participant, one column per recording condition."""
    found = conditions(results)
    participants = sorted({(r["experiment"], r["participant"]) for r in results})
    with_view = _several_views(results)
    fig, axes = plt.subplots(len(participants), len(found), squeeze=False,
                             figsize=(1.45 * len(found) + 0.3, 1.95 * len(participants) + 0.2))
    for ax in axes.flat:
        ax.axis("off")
    for result in results:
        row = participants.index((result["experiment"], result["participant"]))
        ax = axes[row, found.index((result["label"], result["repetition"], result["view"]))]
        ax.imshow(result["thumbnail"])
        face = "" if not np.isfinite(result["face_found_pct"]) else f"\nface {result['face_found_pct']:.0f} %"
        ax.set_title(f"{result['participant']} · {_condition((result['label'], result['repetition'], result['view']), with_view)}"
                     f"{face}", fontsize=6.5, color=INK_SECONDARY)
    _finish(fig, "Model input (mid-segment frame) · face = share of successful face detections", header_in=0.4)
    _save(fig, path)


# ----------------------------------------------------------------------------- global summary

def overview(results, path, model_title):
    """Scatter (estimate vs ground truth) and Bland-Altman for HR, RMSSD and IBI, pooled, coloured by label."""
    metrics = ["hr", "rmssd", "ibi"]
    colors = label_colors(results)
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for col, metric in enumerate(metrics):
        label, unit, _, _ = METRICS[metric]
        top, bottom = axes[0, col], axes[1, col]
        all_estimate, all_truth = [], []
        for segment_label, color in colors.items():
            members = [r for r in results if r["label"] == segment_label]
            estimate = np.concatenate([metric_pairs(r, metric)[0] for r in members])
            truth = np.concatenate([metric_pairs(r, metric)[1] for r in members])
            keep = np.isfinite(estimate) & np.isfinite(truth)
            estimate, truth = estimate[keep], truth[keep]
            all_estimate.append(estimate)
            all_truth.append(truth)
            style = dict(s=7, alpha=0.35, color=color, linewidths=0)
            top.scatter(truth, estimate, **style)
            bottom.scatter((truth + estimate) / 2, estimate - truth, **style)
        estimate, truth = np.concatenate(all_estimate), np.concatenate(all_truth)
        if estimate.size < 3:
            top.set_title(f"{label} · no comparable values", loc="left")
            bottom.set_visible(False)
            continue
        stats = paired(estimate, truth)
        low = min(np.percentile(truth, 0.5), np.percentile(estimate, 0.5))
        high = max(np.percentile(truth, 99.5), np.percentile(estimate, 99.5))
        top.plot([low, high], [low, high], color=INK_SECONDARY, linewidth=0.9)
        top.set_xlim(low, high)
        top.set_ylim(low, high)
        top.set_aspect("equal", adjustable="box")
        top.set_xlabel(f"Ground truth {label} ({unit})")
        top.set_ylabel(f"rPPG estimate ({unit})")
        top.set_title(f"{label} · MAE {_fmt(stats['mae'])} {unit} · r {_fmt(stats['pearson'], 2)} · n {stats['n']}",
                      loc="left")
        difference = estimate - truth
        bias, spread = float(np.mean(difference)), 1.96 * float(np.std(difference, ddof=1))
        bottom.axhline(bias, color=INK_SECONDARY, linewidth=1.0)
        for limit in (bias - spread, bias + spread):
            bottom.axhline(limit, color=MUTED, linewidth=0.9, linestyle=(0, (4, 3)))
        bottom.set_xlim(low, high)
        span = max(abs(np.percentile(difference, 0.5)), abs(np.percentile(difference, 99.5)), spread + abs(bias))
        bottom.set_ylim(-1.1 * span, 1.1 * span)
        bottom.set_xlabel(f"Mean of estimate and ground truth ({unit})")
        bottom.set_ylabel(f"Estimate - ground truth ({unit})")
        bottom.set_title(f"Bland-Altman · bias {_fmt(bias)} · 95 % limits ±{_fmt(spread)} {unit}", loc="left")
    _finish(fig, f"All recordings pooled · {model_title} · one point per HR window / HRV window / beat",
            [Line2D([], [], color=color, marker="o", linestyle="none", markersize=5, label=label_name(segment_label))
             for segment_label, color in colors.items()]
            + [_line(INK_SECONDARY, "identity / mean bias", width=0.9),
               _line(MUTED, "95 % limits of agreement", width=0.9, style=(0, (4, 3)))])
    _save(fig, path)


def participant_errors(table, results, path, model_title):
    """MAE heatmaps, one row per participant and one column per recording condition: HR, RMSSD and IBI."""
    metrics = [("hr_mae_bpm", "Heart rate MAE (bpm)", 1), ("rmssd_mae_ms", "HRV RMSSD MAE (ms)", 0),
               ("ibi_mae_ms", "RR / IBI MAE (ms)", 0)]
    found = conditions(results)
    with_view = _several_views(results)
    participants = sorted({(r["experiment"], r["participant"]) for r in results})
    name_of = {(r["experiment"], r["participant"]): _name(r) for r in results}
    names = [name_of[participant] for participant in participants]
    width = 0.62 * len(found) + 1.3
    fig, axes = plt.subplots(1, len(metrics), figsize=(width * len(metrics), 0.3 * len(participants) + 1.9),
                             squeeze=False)
    for ax, (column, title, digits) in zip(axes[0], metrics):
        values = np.full((len(participants), len(found)), np.nan)
        for row in table.itertuples():
            values[participants.index((row.experiment, row.participant)),
                   found.index((row.label, row.repetition, row.view))] = getattr(row, column)
        finite = values[np.isfinite(values)]
        top = float(np.percentile(finite, 95)) if finite.size else 1.0
        ax.pcolormesh(np.ma.masked_invalid(values), cmap=ERROR_RAMP, vmin=0, vmax=max(top, 1e-9),
                      edgecolors=SURFACE, linewidth=2)
        for (r, c), value in np.ndenumerate(values):
            dark = np.isfinite(value) and value / max(top, 1e-9) > 0.55
            ax.text(c + 0.5, r + 0.5, "–" if not np.isfinite(value) else f"{value:.{digits}f}", ha="center",
                    va="center", fontsize=7.5, color=SURFACE if dark else (INK if np.isfinite(value) else MUTED))
        ax.set_xticks(np.arange(len(found)) + 0.5, [_condition(c, with_view) for c in found], rotation=35,
                      ha="right", fontsize=7.5)
        ax.set_yticks(np.arange(len(participants)) + 0.5, names, fontsize=7.5)
        ax.invert_yaxis()
        ax.grid(False)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(title, loc="left")
    _finish(fig, f"Error per participant and recording · {model_title} · darker = larger error "
                 "(scale capped at each panel's 95th percentile)", header_in=0.45)
    _save(fig, path)
