"""Analysis of one recording: cached input -> model or unsupervised method -> pulse signal -> HR, RR, HRV vs
ground truth."""

import hashlib
import json
from pathlib import Path

import numpy as np

from . import data, signals

# Recording fields kept in data/<key>.npz besides the time series.
INFO_FIELDS = ("id", "key", "experiment", "participant", "segment_index", "label", "repetition", "view",
               "condition", "split", "fold", "video_path", "duration_s", "fs", "face_found_pct", "polarity_flips",
               "beats_gt", "beats_est", "ibi_rejected_pct", "has_ecg")


def model_fingerprint(checkpoint, model_name, inference):
    """What the saved model output depends on, besides the recording's cached input."""
    stat = Path(checkpoint).stat()
    settings = {"checkpoint": str(Path(checkpoint).resolve()), "size": stat.st_size, "mtime": stat.st_mtime_ns,
                "model": model_name, **inference}
    return hashlib.sha256(json.dumps(settings, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def method_fingerprint(method, settings):
    """What the saved output of an unsupervised method depends on, besides the recording's cached input."""
    return hashlib.sha256(json.dumps({"method": method, **settings}, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def _output_key(model_key, recording):
    return f"{model_key}-{recording['video_fingerprint'][:16]}-{len(recording['inputs'])}"


def predicted_bvp(recording, video, run, data_dir):
    """Pulse signal of a recording (run["predict"]); reused from data/<key>.npz when the checkpoint or method and
    the input are unchanged."""
    key = _output_key(run["model_key"], recording)
    saved = Path(data_dir) / f"{recording['key']}.npz"
    if saved.is_file():
        with np.load(saved) as previous:
            if str(previous["model_fingerprint"]) == key:
                return previous["bvp_raw"], int(previous["polarity_flips"]), key
    bvp, flips = run["predict"](video)
    return bvp, flips, key


def analyse_recording(recording, run, data_dir):
    settings = run["settings"]
    video = data.ChunkedVideo(recording["inputs"])
    bvp, flips, key = predicted_bvp(recording, video, run, data_dir)
    fs = recording["fs"]
    duration = len(bvp) / fs
    band = settings["signal"]["bandpass_hz"]
    filtered = signals.prepare_bvp(bvp, fs, band, settings["signal"]["bandpass_order"], run["is_diff"])
    frame_times = np.arange(len(bvp)) / fs

    # Heart rate: one estimate per 8 s window (1 s step) vs the measured HR at the window centre.
    hr = signals.heart_rate_windows(filtered, fs, settings["hr"], band)
    hr_gt = np.interp(hr["times"], frame_times, data.measured_hr(recording)[:len(bvp)])

    # Beats and inter-beat intervals: rPPG pulse peaks vs ECG R-peaks, cleaned by the same rule.
    rpeaks = data.rpeak_times(recording, duration, settings["ibi"]["rpeak_margin_sec"])
    has_ecg = rpeaks is not None and len(rpeaks) >= 2
    rpeaks = rpeaks if has_ecg else np.asarray([])
    beats = signals.detect_beats(filtered, fs, hr["times"], hr["hr"], settings["beats"])
    est = signals.intervals(beats, settings["ibi"])
    gt = signals.intervals(rpeaks, settings["ibi"])
    gt_inside = (gt["times"] >= 0) & (gt["times"] <= duration)
    gt_used = gt_inside & gt["valid"]
    ibi_est_at_gt = signals.interpolate(est["times"][est["valid"]], est["ibi"][est["valid"]], gt["times"],
                                        settings["ibi"]["max_gap_sec"])
    ibi_est_at_gt[~gt_used] = np.nan

    # HRV: RMSSD / SDNN over sliding windows, and over the whole segment.
    hrv_est = signals.hrv_windows(est, duration, settings["hrv"])
    hrv_gt = signals.hrv_windows(gt, duration, settings["hrv"])
    min_intervals = settings["hrv"]["min_intervals"]
    segment_est = signals.hrv_values(est["times"], est["ibi"], est["valid"], 0, duration, min_intervals)
    segment_gt = signals.hrv_values(gt["times"], gt["ibi"], gt["valid"], 0, duration, min_intervals)

    series = {
        "time": frame_times,
        "bvp_raw": bvp,
        "bvp": filtered,
        "beats_est": beats,
        "rpeaks_gt": rpeaks[(rpeaks >= 0) & (rpeaks <= duration)],
        "hr_times": hr["times"], "hr": hr["hr"], "hr_untracked": hr["hr_untracked"], "hr_gt": hr_gt,
        "hrv_times": hrv_est["times"], "rmssd": hrv_est["rmssd"], "sdnn": hrv_est["sdnn"],
        "rmssd_gt": hrv_gt["rmssd"], "sdnn_gt": hrv_gt["sdnn"],
        "ibi_est_times": est["times"], "ibi_est": est["ibi"] * 1000, "ibi_est_valid": est["valid"],
        "ibi_gt_times": gt["times"], "ibi_gt": np.where(gt_used, gt["ibi"] * 1000, np.nan),
        "ibi_est_at_gt": ibi_est_at_gt * 1000,
    }
    result = {
        **{field: recording.get(field) for field in INFO_FIELDS if field in recording},
        "duration_s": duration,
        "polarity_flips": flips,
        "thumbnail": data.thumbnail(video, run["data_types"]),
        "beats_gt": int(gt_inside.sum()),
        "beats_est": int(len(beats)),
        "ibi_rejected_pct": 100.0 * (1 - est["valid"].mean()) if est["valid"].size else np.nan,
        "has_ecg": has_ecg,
        "segment_hrv": {"rmssd": segment_est[0], "sdnn": segment_est[1],
                        "rmssd_gt": segment_gt[0], "sdnn_gt": segment_gt[1]},
        "series": series,
    }
    save_result(result, key, data_dir)
    return result


def save_result(result, key, data_dir):
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    info = {field: result.get(field) for field in INFO_FIELDS}
    info.update({f"segment_{name}": value for name, value in result["segment_hrv"].items()})
    np.savez_compressed(Path(data_dir) / f"{result['key']}.npz", model_fingerprint=np.asarray(key),
                        polarity_flips=np.int64(result["polarity_flips"]), thumbnail=result["thumbnail"],
                        info=np.asarray(json.dumps(info, default=float)), **result["series"])


def load_result(path):
    """A recording result read back from data/<key>.npz (to redraw figures or merge reports)."""
    with np.load(path) as saved:
        info = json.loads(str(saved["info"]))
        series = {name: saved[name] for name in saved.files
                  if name not in ("info", "thumbnail", "model_fingerprint", "polarity_flips")}
        thumbnail = saved["thumbnail"]
    segment = {name: info.pop(f"segment_{name}", np.nan) for name in ("rmssd", "sdnn", "rmssd_gt", "sdnn_gt")}
    numbers = {key: (np.nan if value is None else value) for key, value in info.items()
               if key in ("face_found_pct", "ibi_rejected_pct")}
    return {**info, **numbers, "thumbnail": thumbnail, "series": series,
            "segment_hrv": {name: np.nan if value is None else value for name, value in segment.items()}}
