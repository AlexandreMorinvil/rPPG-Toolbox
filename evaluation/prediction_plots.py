"""Static per-recording "highlight" figures saved with test outputs.

Only a handful of recordings are drawn (worst, best, median, and the worst of every
movement x view scenario); everything else is browsed on demand in the launcher.
"""

import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evaluation import signal_analysis as sa
from evaluation.prediction_store import parse_recording_id

WINDOW_S = 10.0
STEP_S = 2.0
EXCERPT_S = 20.0


def _array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def _join(chunks):
    if isinstance(chunks, dict):
        return np.concatenate([_array(chunks[c]).reshape(-1) for c in sorted(chunks, key=int)])
    return _array(chunks).reshape(-1)


def outputs_recordings(outputs):
    """{recording: (prediction, label, gt or None)} trimmed to a common length."""
    gt_all = outputs.get("gt") or {}
    # Pickles written before vHRM training support stored (normalized) HR channels as labels.
    labels = {} if outputs.get("label_columns") else outputs.get("labels") or {}
    result = {}
    for name in outputs["predictions"]:
        prediction = _join(outputs["predictions"][name])
        label = _join(labels[name]) if name in labels else None
        gt = gt_all.get(name)
        lengths = [len(prediction)] + ([len(label)] if label is not None else []) + ([len(gt)] if gt is not None else [])
        n = min(lengths)
        result[str(name)] = (prediction[:n], label[:n] if label is not None else None,
                             np.asarray(gt[:n], dtype=np.float64) if gt is not None else None)
    return result


def diff_flags(outputs):
    meta = outputs.get("meta") or {}
    label_diff = meta.get("label_is_diff", outputs.get("label_type") == "DiffNormalized")
    return bool(meta.get("prediction_is_diff", label_diff)), bool(label_diff)


def _official_mae(output_dir):
    """Per-recording MAE from vHRM window_results.csv (the reported metric), when present."""
    path = os.path.join(output_dir, "window_results.csv")
    if not os.path.isfile(path):
        return {}
    errors = {}
    with open(path, newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            try:
                value = float(row["absolute_error_bpm"])
            except (KeyError, ValueError):
                continue
            if np.isfinite(value):
                errors.setdefault(row["recording_id"], []).append(value)
    return {name: float(np.mean(values)) for name, values in errors.items()}


def select_highlights(mae_by_recording, worst=3):
    """[(recording, reason)]: worst N, best, median, and the worst per movement x view."""
    ranked = sorted((name for name, mae in mae_by_recording.items() if np.isfinite(mae)),
                    key=lambda name: (-mae_by_recording[name], name))
    if not ranked:
        return []
    picks = [(name, f"worst{i + 1}") for i, name in enumerate(ranked[:max(0, worst)])]
    picks.append((ranked[-1], "best"))
    picks.append((ranked[len(ranked) // 2], "median"))
    scenarios = {}
    for name in ranked:
        meta = parse_recording_id(name)
        if meta["movement"] or meta["view"]:
            scenarios.setdefault((meta["movement"], meta["view"]), name)
    for (movement, view), name in sorted(scenarios.items()):
        picks.append((name, f"scenario-{movement}-{view}"))
    chosen, seen = [], set()
    for name, reason in picks:
        if name not in seen:
            seen.add(name)
            chosen.append((name, reason))
    return chosen


def plot_recording(name, prediction, label, gt, fs, pred_diff, label_diff, title_extra="", path=None):
    """Notebook-style figure: HR over time, BVP excerpt with beats, spectrogram."""
    measured = gt[:, 0] if gt is not None else None
    rows = sa.recording_windows(prediction, fs, WINDOW_S, STEP_S, pred_diff, label=label, label_diff=label_diff,
                                measured_hr=measured, track=True)
    metrics = sa.window_metrics(rows)
    tracked_metrics = sa.window_metrics(rows, "tracked")
    centres = np.array([(r["start"] + r["end"]) / 2 for r in rows])
    duration = len(prediction) / fs

    fig, (ax_hr, ax_bvp, ax_spec) = plt.subplots(3, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [1.2, 1, 1]})

    if measured is not None:
        t = np.arange(len(measured)) / fs
        ax_hr.plot(t, measured, color="black", linewidth=1.8, alpha=0.85, label="Measured HR")
    if label is not None:
        ax_hr.plot(centres, [r["label_hr"] for r in rows], color="0.55", linestyle="--", linewidth=1.2,
                   label="HR of PPG label")
    ax_hr.plot(centres, [r["pred"] for r in rows], color="tab:red", marker="o", markersize=3, linewidth=1.3,
               label="Predicted (FFT per window)")
    if rows and "tracked" in rows[0]:
        ax_hr.plot(centres, [r["tracked"] for r in rows], color="tab:blue", linewidth=1.3, alpha=0.8,
                   label="Predicted (tracked)")
    reference = "measured HR" if measured is not None else "label HR"
    if metrics.get("windows"):
        summary = (f"vs {reference}: MAE {metrics['mae']:.1f} bpm | bias {metrics['bias']:+.1f} | "
                   f"within ±5 bpm {metrics.get('within5', np.nan):.0f}% | max jump {metrics.get('max_jump', np.nan):.1f}\n"
                   f"tracked: MAE {tracked_metrics.get('mae', np.nan):.1f} bpm | "
                   f"max jump {tracked_metrics.get('max_jump', np.nan):.1f}")
    else:
        summary = (f"No ground truth in this file | max jump {sa.max_jump([r['pred'] for r in rows]):.1f} bpm "
                   f"(tracked {sa.max_jump([r.get('tracked', np.nan) for r in rows]):.1f})")
    ax_hr.text(0.01, 0.97, summary,
               transform=ax_hr.transAxes, va="top", ha="left", fontsize=9,
               bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.9})
    ax_hr.set_xlim(0, duration)
    ax_hr.set_ylabel("HR [bpm]")
    ax_hr.set_title(f"{name}{title_extra}  ({WINDOW_S:g} s windows, {STEP_S:g} s step)")
    ax_hr.legend(loc="upper right", fontsize=8)

    # BVP excerpt around the window with the largest error.
    errors = [abs(r["pred"] - r["gt"]) if np.isfinite(r["gt"]) else -1 for r in rows]
    centre = centres[int(np.argmax(errors))] if rows else duration / 2
    start = max(0.0, min(centre - EXCERPT_S / 2, duration - EXCERPT_S))
    end = min(duration, start + EXCERPT_S)
    ax_hr.axvspan(start, end, color="tab:red", alpha=0.08)
    pred_wave = sa.zscore(sa.pulse_waveform(prediction, fs, pred_diff))
    t = np.arange(len(pred_wave)) / fs
    view = (t >= start) & (t <= end)
    caption = []
    if label is not None:
        label_wave = sa.zscore(sa.pulse_waveform(label, fs, label_diff))
        ax_bvp.plot(t[view], label_wave[view], color="black", linewidth=1.2, label="PPG label")
        peaks = sa.beats(label_wave, fs)
        peaks = peaks[view[peaks]]
        ax_bvp.plot(t[peaks], label_wave[peaks], "v", color="black", markersize=5)
        caption.append(f"label {len(peaks)} beats, {sa.beat_rate(peaks, fs):.0f} bpm")
    ax_bvp.plot(t[view], pred_wave[view], color="tab:red", linewidth=1.2, label="Prediction")
    peaks = sa.beats(pred_wave, fs)
    peaks = peaks[view[peaks]]
    ax_bvp.plot(t[peaks], pred_wave[peaks], "^", color="tab:red", markersize=5)
    caption.append(f"prediction {len(peaks)} beats, {sa.beat_rate(peaks, fs):.0f} bpm")
    if gt is not None and gt.shape[1] > 3:
        for a, b in sa.invalid_spans(gt[:, 3] > 0.5, fs):
            if b >= start and a <= end:
                ax_bvp.axvspan(max(a, start), min(b, end), color="0.85", alpha=0.6, linewidth=0)
    ax_bvp.set_xlim(start, end)
    ax_bvp.set_ylabel("Amplitude [z]")
    ax_bvp.set_title(f"BVP {start:.0f}-{end:.0f} s (largest error) | " + " | ".join(caption), fontsize=10)
    ax_bvp.legend(loc="upper right", fontsize=8)

    times, freqs, power = sa.spectrogram(sa.pulse_waveform(prediction, fs, pred_diff), fs, WINDOW_S, STEP_S / 2)
    if power.size:
        ax_spec.pcolormesh(times, freqs * 60, power, shading="nearest", cmap="viridis")
        if measured is not None:
            t_gt = np.arange(len(measured)) / fs
            ax_spec.plot(t_gt, measured, color="white", linewidth=1.2, label="Measured HR")
            ax_spec.plot(t_gt, 2 * measured, color="white", linewidth=0.8, linestyle=":", label="2× HR")
        elif label is not None:
            ax_spec.plot(centres, [r["label_hr"] for r in rows], color="white", linewidth=1.2, label="Label HR")
        if ax_spec.get_legend_handles_labels()[0]:
            ax_spec.legend(loc="upper right", fontsize=8)
    ax_spec.set_xlim(0, duration)
    ax_spec.set_ylim(0, 240)
    ax_spec.set_xlabel("Time [s]")
    ax_spec.set_ylabel("Frequency [bpm]")
    ax_spec.set_title("Prediction spectrogram", fontsize=10)

    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=110)
    plt.close(fig)
    return metrics


def render_highlights(outputs, output_dir, worst=3):
    """Write highlights/<rank>_<reason>_<recording>.png and index.csv; never raises."""
    try:
        fs = float(outputs.get("fs") or 30)
        pred_diff, label_diff = diff_flags(outputs)
        recordings = outputs_recordings(outputs)
        official = _official_mae(output_dir)
        mae = {}
        for name, (prediction, label, gt) in recordings.items():
            if name in official:
                mae[name] = official[name]
                continue
            rows = sa.recording_windows(prediction, fs, WINDOW_S, STEP_S, pred_diff, label=label,
                                        label_diff=label_diff, measured_hr=gt[:, 0] if gt is not None else None)
            mae[name] = sa.window_metrics(rows).get("mae", float("nan"))
        picks = select_highlights(mae, worst)
        if not picks:
            return []
        folder = os.path.join(output_dir, "highlights")
        os.makedirs(folder, exist_ok=True)
        index = []
        for rank, (name, reason) in enumerate(picks):
            path = os.path.join(folder, f"{rank:02d}_{reason}_{name}.png".replace("/", "_"))
            prediction, label, gt = recordings[name]
            try:
                plot_recording(name, prediction, label, gt, fs, pred_diff, label_diff,
                               title_extra=f"  [{reason}, recording MAE {mae[name]:.1f} bpm]", path=path)
            except Exception as error:
                print(f"[outputs] Highlight plot failed for {name}: {error}")
                continue
            index.append({"rank": rank, "reason": reason, "recording": name, "mae_bpm": mae[name],
                          "file": os.path.basename(path)})
        with open(os.path.join(folder, "index.csv"), "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["rank", "reason", "recording", "mae_bpm", "file"])
            writer.writeheader()
            writer.writerows(index)
        print(f"Saved {len(index)} highlight plots to: {folder}")
        try:
            from neural_methods import wandb_logger
            if wandb_logger.is_enabled():
                for item in index:
                    wandb_logger.log_image(f"highlights/{item['reason']}", os.path.join(folder, item["file"]))
        except Exception:
            pass
        return index
    except Exception as error:  # plots are a convenience; never fail the run
        print(f"[outputs] Highlight plots skipped: {error}")
        return []
