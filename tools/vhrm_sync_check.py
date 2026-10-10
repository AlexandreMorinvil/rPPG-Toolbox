"""Estimate the video <-> signal lag of vHRM recordings.

For each selected recording, the face region is tracked on the first frame, POS is run on
its mean colour, and the result is cross-correlated with the training PPG over a range of
lags. The output CSV can be passed to VHRM.LAG_TABLE_PATH: lag_sec is the shift that makes
the label match the video (a positive lag means the video pulse arrives later than the label).

Rows are marked use=True only when the estimate looks reliable: the correlation is high
enough, it beats the peaks one heartbeat away, and the windows of the recording agree on
the same lag. Inspect the other rows manually before using them.

Example (run from the toolbox root):
  python tools/vhrm_sync_check.py --data-path "F:/Datasets preparation" --experiments JZC \
      --labels Resting --views front --max-seconds 120 --output runs/sync/front_resting.csv
"""

import argparse
import os
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np
import pandas as pd
import scipy.signal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from config import _vhrm_node  # noqa: E402
from dataset.data_loader.VHRMLoader import (  # noqa: E402
    HR_COLUMNS, MAX_GAP_PPG, MAX_GAP_SLOW, VALID_RANGES, VHRMLoader, _read_values,
    discover_segments, read_timed_csv, resolve_signal_files, rpeak_waveform,
    sample_series, select_recordings)
from evaluation.post_process import _calculate_fft_hr  # noqa: E402
from unsupervised_methods.methods.POS_WANG import POS_WANG  # noqa: E402


class _Crop:
    """Minimal stand-in for the loader's face-detection state."""

    def __init__(self, backend, device, large_box_coef):
        self.detector = VHRMLoader.__new__(VHRMLoader)
        self.detector._face_det_device = device
        self.detector.Y5FObj = None
        self.crop = _vhrm_crop_config(backend, large_box_coef)

    def box(self, frame, crop_mode):
        stats = {"attempts": 0, "failures": 0}
        box = self.detector._resolve_crop(frame, crop_mode, _Preprocess(self.crop), None, stats)
        return box, stats["failures"] > 0


class _Preprocess:
    def __init__(self, crop):
        self.CROP_FACE = crop


def _vhrm_crop_config(backend, large_box_coef):
    from yacs.config import CfgNode
    crop = CfgNode()
    crop.BACKEND = backend
    crop.USE_LARGE_FACE_BOX = large_box_coef > 1.0
    crop.LARGE_BOX_COEF = large_box_coef
    return crop


def _first_frame(video_path):
    """First frame as displayed (container rotation applied) and its size."""
    capture, rotation = VHRMLoader._open_capture(video_path)
    success, frame = capture.read()
    capture.release()
    if not success:
        raise ValueError(f"Cannot read {video_path}")
    frame = VHRMLoader._rotate_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), rotation)
    return frame, frame.shape[1], frame.shape[0]


def mean_rgb_ffmpeg(video_path, box, width, height, fs, max_seconds, ffmpeg, use_cuda):
    """Mean colour of the crop per output frame, decoded and resampled by FFmpeg.

    FFmpeg auto-rotates, so box, width and height are in displayed-frame pixels."""
    filters = []
    if box is not None:
        x, y, w, h = (int(value) for value in box)
        x, y = max(0, x), max(0, y)
        w, h = max(2, min(w, width - x)), max(2, min(h, height - y))
        filters.append(f"crop={w}:{h}:{x}:{y}")
    filters += ["scale=16:16:flags=area", f"fps={fs}"]
    command = [ffmpeg, "-v", "error", "-nostdin"]
    if use_cuda:
        command += ["-hwaccel", "cuda"]
    command += ["-i", video_path]
    if max_seconds:
        command += ["-t", str(max_seconds)]
    command += ["-vf", ",".join(filters), "-an", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    output = subprocess.run(command, check=True, capture_output=True).stdout
    frames = np.frombuffer(output, dtype=np.uint8).reshape(-1, 16, 16, 3)
    return frames.reshape(len(frames), -1, 3).mean(axis=1)


def mean_rgb_opencv(video_path, box, fs, max_seconds):
    """box is in displayed-frame pixels; the mean colour is taken from the stored frame."""
    capture, rotation = VHRMLoader._open_capture(video_path)
    if box is not None:
        box = VHRMLoader._box_to_source(box, rotation, int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                        int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    source_fps = capture.get(cv2.CAP_PROP_FPS)
    values, source_index, next_time = [], 0, 0.0
    success, frame = capture.read()
    while success:
        frame_time = source_index / source_fps
        if max_seconds and frame_time >= max_seconds:
            break
        if frame_time + 0.5 / source_fps >= next_time:
            if box is not None:
                x, y, w, h = (int(value) for value in box)
                frame = frame[max(y, 0):y + h, max(x, 0):x + w]
            values.append(frame.reshape(-1, 3).mean(axis=0)[::-1])
            next_time += 1.0 / fs
        source_index += 1
        success, frame = capture.read()
    capture.release()
    return np.asarray(values, dtype=np.float64)


def _bandpass(signal, fs):
    b, a = scipy.signal.butter(2, [0.7 / fs * 2, 3.0 / fs * 2], btype="bandpass")
    filtered = scipy.signal.filtfilt(b, a, signal - np.mean(signal))
    return (filtered - filtered.mean()) / (filtered.std() + 1e-12)


def lag_correlation(pulse, label_at, lags):
    """Pearson correlation between pulse(t) and label(t - lag) for every lag."""
    values = []
    for lag in lags:
        label = label_at(lag)
        valid = np.isfinite(label)
        if valid.sum() < len(label) * 0.8:
            values.append(np.nan)
            continue
        values.append(np.corrcoef(pulse[valid], label[valid])[0, 1])
    return np.asarray(values)


def analyse(recording, args, crop):
    frame, width, height = _first_frame(recording["path"])
    box, face_missing = crop.box(frame, recording["crop_mode"])
    ffmpeg = shutil.which(args.ffmpeg) if args.decoder != "opencv" else None
    if ffmpeg:
        rgb = mean_rgb_ffmpeg(recording["path"], box, width, height, args.fs, args.max_seconds, ffmpeg,
                              args.cuda)
    else:
        rgb = mean_rgb_opencv(recording["path"], box, args.fs, args.max_seconds)
    count = len(rgb)
    if count < args.fs * 20:
        raise ValueError(f"only {count} frames decoded")
    pulse = _bandpass(np.asarray(POS_WANG(rgb.reshape(count, 1, 1, 3), args.fs)).reshape(-1), args.fs)

    files = resolve_signal_files(recording)
    times = recording["start_epoch"] + np.arange(count) / args.fs
    if args.ppg_source == "synthetic":
        if not files["ppg"]:
            raise ValueError("no synthetic PPG")
        source_times, values = _read_values(files["ppg"], ("synthetic_ppg",))

        def label_at(lag):
            return sample_series(source_times, values, times - lag, MAX_GAP_PPG)
    else:
        if not files["rpeaks"]:
            raise ValueError("no R-peaks")
        peaks = read_timed_csv(files["rpeaks"])["epoch"].to_numpy(dtype=np.float64)

        def label_at(lag):
            return rpeak_waveform(peaks + args.rpeak_delay, times - lag, args.ppg_source, 0.15)

    def filtered_label_at(lag):
        label = label_at(lag)
        valid = np.isfinite(label)
        if valid.sum() < len(label) * 0.8:
            return label
        result = np.full_like(label, np.nan)
        start, end = np.flatnonzero(valid)[[0, -1]]
        segment = label[start:end + 1]
        segment = np.interp(np.arange(len(segment)), np.flatnonzero(np.isfinite(segment)),
                            segment[np.isfinite(segment)])
        result[start:end + 1] = _bandpass(segment, args.fs)
        return result

    lags = np.arange(-args.search, args.search + 0.5 / args.fs, 1.0 / args.fs)
    curve = lag_correlation(pulse, filtered_label_at, lags)
    if not np.any(np.isfinite(curve)):
        raise ValueError("signals do not overlap")
    best = int(np.nanargmax(curve))
    peaks, _ = scipy.signal.find_peaks(np.nan_to_num(curve, nan=-1.0))
    others = [curve[index] for index in peaks if abs(lags[index] - lags[best]) > 0.25]
    second = max(others) if others else np.nan
    ratio = curve[best] / second if np.isfinite(second) and second > 0 else np.inf

    window = int(args.window * args.fs)
    window_lags = []
    for start in range(0, count - window + 1, window):
        segment_pulse = pulse[start:start + window]

        def window_label(lag, start=start):
            return filtered_label_at(lag)[start:start + window]
        window_curve = lag_correlation(segment_pulse, window_label, lags)
        if np.any(np.isfinite(window_curve)):
            window_lags.append(lags[int(np.nanargmax(window_curve))])

    hr_truth = np.nan
    if files["zephyr"]:
        hr_times, hr_values = _read_values(files["zephyr"], HR_COLUMNS["zephyr"], VALID_RANGES["hr"])
        if hr_values is not None:
            hr_truth = float(np.nanmean(sample_series(hr_times, hr_values, times, MAX_GAP_SLOW)))
    hr_pos = float(_calculate_fft_hr(pulse, fs=args.fs))
    window_median = float(np.median(window_lags)) if window_lags else np.nan
    window_iqr = float(np.subtract(*np.percentile(window_lags, [75, 25]))) if len(window_lags) > 1 else np.nan
    windows_agree = (not np.isfinite(window_iqr) or window_iqr <= args.max_window_iqr) and         (not np.isfinite(window_median) or abs(window_median - lags[best]) <= args.max_window_iqr)
    confident = bool(curve[best] >= args.min_correlation and ratio >= args.min_peak_ratio
                     and windows_agree and not face_missing)
    return {
        "lag_sec": round(float(lags[best]), 4),
        "correlation": round(float(curve[best]), 4),
        "second_peak_correlation": round(float(second), 4) if np.isfinite(second) else None,
        "peak_ratio": round(float(ratio), 3) if np.isfinite(ratio) else None,
        "inverted": bool(np.nanmin(curve) < -curve[best]),
        "window_lag_median": round(window_median, 4) if np.isfinite(window_median) else None,
        "window_lag_iqr": round(window_iqr, 4) if np.isfinite(window_iqr) else None,
        "pos_hr_bpm": round(hr_pos, 2),
        "zephyr_hr_bpm": round(hr_truth, 2) if np.isfinite(hr_truth) else None,
        "hr_abs_error_bpm": round(abs(hr_pos - hr_truth), 2) if np.isfinite(hr_truth) else None,
        "face_missing": face_missing,
        "analysed_seconds": round(count / args.fs, 2),
        "use": confident,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--experiments", nargs="*", default=[])
    parser.add_argument("--participants", nargs="*", default=[])
    parser.add_argument("--exclude-participants", nargs="*", default=[])
    parser.add_argument("--labels", nargs="*", default=[])
    parser.add_argument("--views", nargs="*", default=["front"])
    parser.add_argument("--crop-mode", nargs="*", default=["front=face", "back-left=face_or_full",
                                                           "back-right=face_or_full"],
                        help="Same syntax as VHRM.CROP_MODE_BY_VIEW.")
    parser.add_argument("--ppg-source", default="synthetic", choices=["synthetic", "rpeak_sine", "rpeak_gaussian"])
    parser.add_argument("--rpeak-delay", type=float, default=0.2)
    parser.add_argument("--fs", type=float, default=30)
    parser.add_argument("--max-seconds", type=float, default=120,
                        help="Analyse at most this many seconds from the start of each video (0 = all).")
    parser.add_argument("--search", type=float, default=2.0, help="Largest lag tried, in seconds.")
    parser.add_argument("--window", type=float, default=30, help="Window for the per-window lag spread.")
    parser.add_argument("--min-correlation", type=float, default=0.3)
    parser.add_argument("--min-peak-ratio", type=float, default=1.02,
                        help="Best correlation / best correlation one beat away.")
    parser.add_argument("--max-window-iqr", type=float, default=0.15,
                        help="Largest spread (s) of the per-window lags for a confident estimate.")
    parser.add_argument("--backend", default="Y5F")
    parser.add_argument("--large-box-coef", type=float, default=1.5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--decoder", default="ffmpeg", choices=["ffmpeg", "opencv"])
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--cuda", action="store_true", help="Decode with -hwaccel cuda.")
    parser.add_argument("--output", required=True, help="CSV to write (usable as VHRM.LAG_TABLE_PATH).")
    args = parser.parse_args()
    os.chdir(ROOT)

    node = _vhrm_node()
    node.EXPERIMENTS, node.PARTICIPANTS = args.experiments, args.participants
    node.EXCLUDE_PARTICIPANTS, node.LABELS, node.VIEWS = args.exclude_participants, args.labels, args.views
    node.CROP_MODE_BY_VIEW = args.crop_mode
    recordings = select_recordings(discover_segments(args.data_path, args.experiments), node)
    print(f"{len(recordings)} recordings selected.")
    crop = _Crop(args.backend, args.device, args.large_box_coef)

    rows = []
    for number, recording in enumerate(recordings, start=1):
        started = time.perf_counter()
        row = {key: recording[key] for key in ("experiment", "participant", "segment_index", "label", "view")}
        try:
            row.update(analyse(recording, args, crop))
            print(f"[{number}/{len(recordings)}] {recording['index']}: lag {row['lag_sec']:+.3f}s "
                  f"r={row['correlation']:.2f} ratio={row['peak_ratio']} POS {row['pos_hr_bpm']} bpm "
                  f"vs Zephyr {row['zephyr_hr_bpm']} ({time.perf_counter() - started:.0f}s)")
        except Exception as error:  # keep going: one unreadable segment should not stop the run
            row.update({"error": str(error), "use": False})
            print(f"[{number}/{len(recordings)}] {recording['index']}: FAILED ({error})")
        rows.append(row)
        frame = pd.DataFrame(rows)
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        frame.to_csv(args.output, index=False)

    frame = pd.DataFrame(rows)
    if "lag_sec" in frame and frame["use"].any():
        confident = frame[frame["use"]]
        print(f"\n{len(confident)}/{len(frame)} confident estimates; lag median {confident['lag_sec'].median():+.3f}s, "
              f"IQR {confident['lag_sec'].quantile(0.75) - confident['lag_sec'].quantile(0.25):.3f}s.")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
