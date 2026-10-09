"""Evaluation and grouped reports for the vHRM dataset.

The primary ground truth is the measured heart rate (Zephyr by default) stored in the
per-clip ground-truth files written by VHRMLoader. The HR derived from the PPG training
label is reported as a secondary reference, comparable with the usual toolbox metrics.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal

from evaluation.post_process import _calculate_SNR, _calculate_fft_hr, _calculate_peak_hr, _detrend

GT_HR, GT_HRV, GT_BR, GT_PPG_VALID = range(4)


def _to_numpy(value):
    return value.cpu().numpy() if hasattr(value, "cpu") else np.asarray(value)


def _concatenate(chunks, chunk_ids):
    return np.concatenate([np.reshape(_to_numpy(chunks[chunk_id]), (-1,)) for chunk_id in chunk_ids])


def _parse_recording_id(recording_id):
    parts = recording_id.split("--")
    if len(parts) == 5:
        return dict(zip(("experiment", "participant", "segment_index", "movement", "view"), parts))
    if len(parts) == 4:
        return dict(zip(("participant", "segment_index", "movement", "view"), parts), experiment="")
    return {"experiment": "", "participant": recording_id, "segment_index": "", "movement": "", "view": ""}


def _ground_truth_files(file_list_path):
    file_list = pd.read_csv(file_list_path)
    if "gt_files" not in file_list:
        raise ValueError(f"{file_list_path} has no gt_files column; re-run vHRM preprocessing.")
    return {(row.recording_id, int(row.chunk)): row.gt_files for row in file_list.itertuples()}


def _estimate_hr(signal, fs, method, is_diff):
    waveform = np.cumsum(signal) if is_diff else signal
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


def _nanmean(values):
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if values.size else np.nan


def _correlation(frame, first, second):
    pair = frame[[first, second]].dropna()
    if len(pair) > 1 and pair[first].nunique() > 1 and pair[second].nunique() > 1:
        return pair[first].corr(pair[second])
    return np.nan


def _summary(frame, group_columns):
    def summarize(group):
        error = group["predicted_hr_bpm"] - group["ground_truth_hr_bpm"]
        denominator = group["ground_truth_hr_bpm"].replace(0, np.nan)
        label_error = group["predicted_hr_bpm"] - group["label_hr_bpm"]
        return pd.Series({
            "windows": len(group),
            "hr_windows": int(group["ground_truth_hr_bpm"].notna().sum()),
            "mae_bpm": error.abs().mean(),
            "rmse_bpm": np.sqrt(np.square(error).mean()),
            "mape_percent": (error.abs() / denominator).mean() * 100,
            "pearson": _correlation(group, "predicted_hr_bpm", "ground_truth_hr_bpm"),
            "mean_snr_db": group["snr_db"].mean(),
            "mean_ground_truth_hr_bpm": group["ground_truth_hr_bpm"].mean(),
            "mean_predicted_hr_bpm": group["predicted_hr_bpm"].mean(),
            "label_windows": int(group["label_hr_bpm"].notna().sum()),
            "mae_vs_label_bpm": label_error.abs().mean(),
            "rmse_vs_label_bpm": np.sqrt(np.square(label_error).mean()),
            "pearson_vs_label": _correlation(group, "predicted_hr_bpm", "label_hr_bpm"),
            "mean_label_hr_bpm": group["label_hr_bpm"].mean(),
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
    measured = windows.dropna(subset=["ground_truth_hr_bpm", "predicted_hr_bpm"])
    if measured.empty:
        return
    plt.figure(figsize=(6, 6))
    plt.scatter(measured["ground_truth_hr_bpm"], measured["predicted_hr_bpm"], alpha=0.45)
    bounds = [measured[["ground_truth_hr_bpm", "predicted_hr_bpm"]].min().min(),
              measured[["ground_truth_hr_bpm", "predicted_hr_bpm"]].max().max()]
    plt.plot(bounds, bounds, "k--", linewidth=1)
    plt.xlabel("Ground-truth HR (bpm)")
    plt.ylabel("Predicted HR (bpm)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "heart_rate_scatter.png"), dpi=180)
    plt.close()

    difference = measured["predicted_hr_bpm"] - measured["ground_truth_hr_bpm"]
    average = (measured["predicted_hr_bpm"] + measured["ground_truth_hr_bpm"]) / 2
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


def calculate_vhrm_metrics(predictions, labels, config, data_config=None, output_dir=None,
                           prediction_is_diff=None, prefix="test/vHRM"):
    """Compare model-derived HR with the measured HR and the label HR, and save reports.

    predictions / labels: {recording_id: {chunk_id: tensor}} as built by the trainers.
    """
    data_config = data_config if data_config is not None else config.TEST.DATA
    output_dir = output_dir or config.TEST.OUTPUT_SAVE_DIR
    fs = data_config.FS
    if prediction_is_diff is None:
        prediction_is_diff = data_config.VHRM.PREDICTION_IS_DIFF
    label_is_diff = data_config.PREPROCESS.LABEL_TYPE == "DiffNormalized"
    method = config.INFERENCE.EVALUATION_METHOD
    use_windows = config.INFERENCE.EVALUATION_WINDOW.USE_SMALLER_WINDOW
    configured_window = int(config.INFERENCE.EVALUATION_WINDOW.WINDOW_SIZE * fs)
    ground_truth_files = _ground_truth_files(data_config.FILE_LIST_PATH)
    rows = []

    for recording_id in sorted(predictions):
        chunk_ids = sorted(predictions[recording_id])
        prediction = _concatenate(predictions[recording_id], chunk_ids)
        label = None
        if labels is not None and recording_id in labels:
            label = _concatenate(labels[recording_id], chunk_ids)
        try:
            ground_truth = np.concatenate([
                np.load(ground_truth_files[(recording_id, int(chunk_id))]) for chunk_id in chunk_ids])
        except KeyError as error:
            raise ValueError(f"No ground-truth file for {error.args[0]} in {data_config.FILE_LIST_PATH}")
        length = min(len(prediction), len(ground_truth), len(label) if label is not None else len(prediction))
        prediction, ground_truth = prediction[:length], ground_truth[:length]
        label = label[:length] if label is not None else None

        recording = _parse_recording_id(recording_id)
        window_size = configured_window if use_windows else length
        for window_index, start in enumerate(range(0, length, window_size)):
            end = min(start + window_size, length)
            if end - start < max(9, fs * 2):
                continue
            predicted_hr, waveform = _estimate_hr(prediction[start:end], fs, method, prediction_is_diff)
            window_truth = ground_truth[start:end]
            measured_hr = _nanmean(window_truth[:, GT_HR])
            label_hr = np.nan
            if label is not None and np.all(window_truth[:, GT_PPG_VALID] > 0.5):
                label_hr, _ = _estimate_hr(label[start:end], fs, method, label_is_diff)
            predicted_mean_ibi, predicted_hrv = _pulse_interval_metrics(waveform, fs)
            rows.append({
                "recording_id": recording_id,
                **recording,
                "window_index": window_index,
                "start_seconds": start / fs,
                "end_seconds": end / fs,
                "ground_truth_hr_bpm": measured_hr,
                "predicted_hr_bpm": predicted_hr,
                "error_bpm": predicted_hr - measured_hr,
                "absolute_error_bpm": abs(predicted_hr - measured_hr),
                "label_hr_bpm": label_hr,
                "label_error_bpm": predicted_hr - label_hr,
                "snr_db": float(_calculate_SNR(waveform, measured_hr, fs=fs)) if np.isfinite(measured_hr) else np.nan,
                "ground_truth_hrv_ms": _nanmean(window_truth[:, GT_HRV]),
                "predicted_hrv_rmssd_ms": predicted_hrv,
                "ground_truth_respiration_rate_bpm": _nanmean(window_truth[:, GT_BR]),
                "predicted_mean_ibi_ms": predicted_mean_ibi,
            })

    windows = pd.DataFrame(rows)
    if windows.empty:
        raise ValueError("vHRM evaluation produced no valid windows")
    os.makedirs(output_dir, exist_ok=True)
    windows.to_csv(os.path.join(output_dir, "window_results.csv"), index=False)

    report_groups = {
        "recording_summary": ["experiment", "participant", "segment_index", "movement", "view"],
        "by_experiment": ["experiment"],
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
    print("vHRM results (ground truth = measured HR; 'vs label' = HR of the PPG label):")
    for key in ("windows", "mae_bpm", "rmse_bpm", "mape_percent", "pearson", "mae_vs_label_bpm", "pearson_vs_label"):
        print(f"  {key}: {summary.get(key)}")
    print(f"Saved vHRM CSV, JSON, and plot reports to: {output_dir}")
    return {f"{prefix}/{key}": value for key, value in summary.items() if value is not None}
