"""Evaluation and grouped reports for vHRM's BPM ground truth."""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal

from evaluation.post_process import _calculate_SNR, _calculate_fft_hr, _calculate_peak_hr, _detrend


def _reform(data, flatten=True):
    tensors = [value for _, value in sorted(data.items())]
    combined = np.concatenate([tensor.cpu().numpy() for tensor in tensors], axis=0)
    return np.reshape(combined, (-1,)) if flatten else combined


def _parse_recording_id(recording_id):
    participant, segment_index, movement, view = recording_id.split("--", 3)
    return participant, segment_index, movement, view


def _estimate_hr(prediction, fs, method, prediction_is_diff):
    waveform = np.cumsum(prediction) if prediction_is_diff else prediction
    waveform = _detrend(waveform, 100)
    b, a = scipy.signal.butter(1, [0.6 / fs * 2, 3.3 / fs * 2], btype="bandpass")
    waveform = scipy.signal.filtfilt(b, a, waveform.astype(np.float64))
    if method == "FFT":
        return float(_calculate_fft_hr(waveform, fs=fs)), waveform
    if method == "peak detection":
        return float(_calculate_peak_hr(waveform, fs=fs)), waveform
    raise ValueError(f"Unsupported vHRM evaluation method: {method}")


def _pulse_interval_metrics(waveform, fs):
    minimum_distance = max(1, int(fs * 60 / 200))
    prominence = max(np.std(waveform) * 0.1, np.finfo(np.float64).eps)
    peaks, _ = scipy.signal.find_peaks(
        waveform, distance=minimum_distance, prominence=prominence)
    intervals_ms = np.diff(peaks) / fs * 1000
    mean_rr = float(np.mean(intervals_ms)) if intervals_ms.size else np.nan
    rmssd = float(np.sqrt(np.mean(np.square(np.diff(intervals_ms))))) \
        if intervals_ms.size >= 2 else np.nan
    return mean_rr, rmssd


def _summary(frame, group_columns):
    def summarize(group):
        error = group["predicted_hr_bpm"] - group["ground_truth_hr_bpm"]
        denominator = group["ground_truth_hr_bpm"].replace(0, np.nan)
        if len(group) > 1 and group["predicted_hr_bpm"].nunique() > 1 and group["ground_truth_hr_bpm"].nunique() > 1:
            correlation = group["predicted_hr_bpm"].corr(group["ground_truth_hr_bpm"])
        else:
            correlation = np.nan
        return pd.Series({
            "windows": len(group),
            "mae_bpm": error.abs().mean(),
            "rmse_bpm": np.sqrt(np.square(error).mean()),
            "mape_percent": (error.abs() / denominator).mean() * 100,
            "pearson": correlation,
            "mean_snr_db": group["snr_db"].mean(),
            "mean_ground_truth_hr_bpm": group["ground_truth_hr_bpm"].mean(),
            "mean_predicted_hr_bpm": group["predicted_hr_bpm"].mean(),
            "hrv_windows": group[["ground_truth_hrv_ms", "predicted_hrv_rmssd_ms"]].dropna().shape[0],
            "hrv_mae_ms": (
                group["predicted_hrv_rmssd_ms"] - group["ground_truth_hrv_ms"]
            ).abs().mean(),
            "mean_ground_truth_hrv_ms": group["ground_truth_hrv_ms"].mean(),
            "mean_predicted_hrv_rmssd_ms": group["predicted_hrv_rmssd_ms"].mean(),
        })

    if group_columns:
        return frame.groupby(group_columns, dropna=False, sort=True).apply(summarize, include_groups=False).reset_index()
    return summarize(frame).to_frame().T


def _save_plots(windows, summaries, output_dir):
    plt.figure(figsize=(6, 6))
    plt.scatter(windows["ground_truth_hr_bpm"], windows["predicted_hr_bpm"], alpha=0.45)
    bounds = [windows[["ground_truth_hr_bpm", "predicted_hr_bpm"]].min().min(),
              windows[["ground_truth_hr_bpm", "predicted_hr_bpm"]].max().max()]
    plt.plot(bounds, bounds, "k--", linewidth=1)
    plt.xlabel("Ground-truth HR (bpm)")
    plt.ylabel("Predicted HR (bpm)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "heart_rate_scatter.png"), dpi=180)
    plt.close()

    difference = windows["predicted_hr_bpm"] - windows["ground_truth_hr_bpm"]
    average = (windows["predicted_hr_bpm"] + windows["ground_truth_hr_bpm"]) / 2
    bias, spread = difference.mean(), difference.std(ddof=1)
    plt.figure(figsize=(7, 5))
    plt.scatter(average, difference, alpha=0.45)
    for value, style in ((bias, "-"), (bias + 1.96 * spread, "--"), (bias - 1.96 * spread, "--")):
        plt.axhline(value, color="black", linestyle=style, linewidth=1)
    plt.xlabel("Mean of predicted and ground-truth HR (bpm)")
    plt.ylabel("Prediction error (bpm)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "bland_altman.png"), dpi=180)
    plt.close()

    for name, summary in summaries.items():
        label_column = name.removeprefix("by_")
        plt.figure(figsize=(max(6, len(summary) * 1.2), 4.5))
        plt.bar(summary[label_column].astype(str), summary["mae_bpm"])
        plt.xlabel(label_column.replace("_", " ").title())
        plt.ylabel("MAE (bpm)")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"mae_{name}.png"), dpi=180)
        plt.close()


def calculate_vhrm_metrics(predictions, labels, config):
    """Compare model-derived HR with frame-aligned measured BPM and save reports."""
    fs = config.TEST.DATA.FS
    use_windows = config.INFERENCE.EVALUATION_WINDOW.USE_SMALLER_WINDOW
    configured_window = int(config.INFERENCE.EVALUATION_WINDOW.WINDOW_SIZE * fs)
    prediction_is_diff = config.TEST.DATA.VHRM.PREDICTION_IS_DIFF
    rows = []

    for recording_id in sorted(predictions):
        prediction = _reform(predictions[recording_id])
        ground_truth = _reform(labels[recording_id], flatten=False)
        if ground_truth.ndim == 1:
            ground_truth = ground_truth[:, None]
        participant, segment_index, movement, view = _parse_recording_id(recording_id)
        window_size = configured_window if use_windows else len(prediction)
        for window_index, start in enumerate(range(0, len(prediction), window_size)):
            end = min(start + window_size, len(prediction))
            if end - start < max(9, fs * 2):
                continue
            predicted_hr, waveform = _estimate_hr(
                prediction[start:end], fs, config.INFERENCE.EVALUATION_METHOD, prediction_is_diff)
            measured_hr = float(np.nanmean(ground_truth[start:end, 0]))
            predicted_mean_ibi, predicted_hrv = _pulse_interval_metrics(waveform, fs)
            measured_hrv = float(np.nanmean(ground_truth[start:end, 1])) \
                if ground_truth.shape[1] > 1 and np.any(np.isfinite(ground_truth[start:end, 1])) else np.nan
            measured_respiration_rate = float(np.nanmean(ground_truth[start:end, 2])) \
                if ground_truth.shape[1] > 2 and np.any(np.isfinite(ground_truth[start:end, 2])) else np.nan
            rows.append({
                "recording_id": recording_id,
                "participant": participant,
                "segment_index": segment_index,
                "movement": movement,
                "view": view,
                "window_index": window_index,
                "start_seconds": start / fs,
                "end_seconds": end / fs,
                "ground_truth_hr_bpm": measured_hr,
                "predicted_hr_bpm": predicted_hr,
                "error_bpm": predicted_hr - measured_hr,
                "absolute_error_bpm": abs(predicted_hr - measured_hr),
                "snr_db": float(_calculate_SNR(waveform, measured_hr, fs=fs)),
                "ground_truth_hrv_ms": measured_hrv,
                "predicted_hrv_rmssd_ms": predicted_hrv,
                "ground_truth_respiration_rate_bpm": measured_respiration_rate,
                "predicted_mean_ibi_ms": predicted_mean_ibi,
            })

    windows = pd.DataFrame(rows)
    if windows.empty:
        raise ValueError("vHRM evaluation produced no valid windows")
    output_dir = config.TEST.OUTPUT_SAVE_DIR
    os.makedirs(output_dir, exist_ok=True)
    windows.to_csv(os.path.join(output_dir, "window_results.csv"), index=False)

    report_groups = {
        "recording_summary": ["participant", "segment_index", "movement", "view"],
        "by_movement": ["movement"],
        "by_participant": ["participant"],
        "by_view": ["view"],
    }
    reports = {name: _summary(windows, columns) for name, columns in report_groups.items()}
    overall = _summary(windows, [])
    for name, report in reports.items():
        report.to_csv(os.path.join(output_dir, f"{name}.csv"), index=False)
    overall.to_csv(os.path.join(output_dir, "overall_summary.csv"), index=False)
    _save_plots(windows, {key: value for key, value in reports.items() if key.startswith("by_")}, output_dir)

    summary = overall.iloc[0].replace({np.nan: None}).to_dict()
    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2)
    print(f"Saved vHRM CSV, JSON, and plot reports to: {output_dir}")
    return {f"test/vHRM/{key}": value for key, value in summary.items() if value is not None}
