"""Discovery and analysis of run outputs (checkpoints, plots, reports, prediction pickles)."""

import csv
import json
import math
import os
import pickle
import sys
import threading
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np

from . import paths, settings, signals

MARKERS = {"PreTrainedModels", "saved_test_outputs", "saved_outputs", "plots"}
SKIP_DIRS = {"wandb", ".git", "__pycache__", "node_modules"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp"}
TABLE_LIMIT = 300_000
_pickle_cache = OrderedDict()
_cache_lock = threading.Lock()


def _roots(extra=()):
    return list(settings.results_roots()) + [Path(p) for p in extra]


def check_path(path, extra_roots=()):
    path = Path(path).resolve(strict=False)
    if not paths.is_within(path, _roots(extra_roots) + [paths.JOBS_DIR]):
        raise ValueError("Path is outside the configured results folders")
    return path


def _kind(path):
    suffix = path.suffix.lower()
    if suffix == ".pth":
        return "checkpoint"
    if suffix in IMAGE_EXT:
        return "image"
    if suffix == ".pdf":
        return "pdf"
    if suffix in (".csv", ".json"):
        return suffix[1:]
    if suffix in (".pickle", ".pkl"):
        return "pickle"
    if suffix in (".log", ".txt", ".yaml", ".yml"):
        return "text"
    return "other"


def scan(job_outputs=None, max_depth=6):
    """Find experiment folders (LOG.PATH/<cache name>) under the results roots."""
    found = {}
    for root in _roots():
        stack = [(root, 0)]
        while stack:
            current, depth = stack.pop()
            try:
                entries = [e for e in os.scandir(current) if e.is_dir(follow_symlinks=False)]
            except OSError:
                continue
            names = {e.name for e in entries}
            markers = names & MARKERS
            if markers:
                path = Path(current).resolve()
                found[str(path)] = _summarize(path, root, markers)
            if depth >= max_depth:
                continue
            for entry in entries:
                if entry.name not in SKIP_DIRS and entry.name not in MARKERS and not entry.name.startswith("."):
                    stack.append((entry.path, depth + 1))
    experiments = sorted(found.values(), key=lambda item: item["modified"], reverse=True)
    for experiment in experiments:
        experiment["jobs"] = [job for job, dirs in (job_outputs or {}).items()
                              if any(paths.is_within(d, [experiment["path"]]) or d == experiment["path"]
                                     for d in dirs)]
    return experiments


def _summarize(path, root, markers):
    modified = path.stat().st_mtime
    checkpoints = []
    model_dir = path / "PreTrainedModels"
    if model_dir.is_dir():
        checkpoints = sorted(p.name for p in model_dir.glob("*.pth"))
        modified = max([modified] + [p.stat().st_mtime for p in model_dir.glob("*.pth")])
    summary = None
    for folder in ("saved_test_outputs", "saved_outputs"):
        summary_file = path / folder / "summary.json"
        if summary_file.is_file():
            try:
                summary = json.loads(summary_file.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                summary = None
        if (path / folder).is_dir():
            modified = max([modified] + [p.stat().st_mtime for p in (path / folder).iterdir()])
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError:
        relative = path.name
    return {
        "path": str(path),
        "root": str(root),
        "name": relative,
        "markers": sorted(markers),
        "checkpoints": len(checkpoints),
        "last_checkpoint": checkpoints[-1] if checkpoints else None,
        "modified": modified,
        "summary": summary,
    }


def _read_table(path, limit_rows=500):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        rows = []
        for index, row in enumerate(reader):
            if index > limit_rows:
                break
            rows.append(row)
    return {"header": rows[0] if rows else [], "rows": rows[1:], "truncated": len(rows) > limit_rows}


def detail(path):
    path = check_path(path)
    if not path.is_dir():
        raise ValueError("Experiment folder not found")
    files = []
    for file in sorted(path.rglob("*")):
        if not file.is_file() or any(part in SKIP_DIRS for part in file.relative_to(path).parts):
            continue
        entry = {"path": str(file), "name": file.relative_to(path).as_posix(), "size": file.stat().st_size,
                 "modified": file.stat().st_mtime, "kind": _kind(file)}
        if entry["kind"] in ("csv", "json") and entry["size"] <= TABLE_LIMIT:
            try:
                entry["content"] = _read_table(file) if entry["kind"] == "csv" else \
                    json.loads(file.read_text(encoding="utf-8"))
            except (OSError, ValueError, csv.Error):
                pass
        files.append(entry)
        if len(files) >= 2000:
            break
    return {"path": str(path), "files": files}


def _to_array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def _join_chunks(chunks):
    if isinstance(chunks, dict):
        arrays = [_to_array(v) for _, v in sorted(chunks.items(), key=lambda item: int(item[0]))]
    else:
        arrays = [_to_array(chunks)]
    arrays = [a.reshape(-1) if a.ndim == 1 or (a.ndim == 2 and a.shape[-1] == 1) else a for a in arrays]
    return np.concatenate(arrays, axis=0)


def load_pickle(path):
    path = check_path(path)
    stamp = path.stat().st_mtime
    key = str(path)
    with _cache_lock:
        if key in _pickle_cache and _pickle_cache[key][0] == stamp:
            _pickle_cache.move_to_end(key)
            return _pickle_cache[key][1]
    try:
        with path.open("rb") as handle:
            data = pickle.load(handle)
    except ModuleNotFoundError as error:
        raise ValueError(f"This output pickle needs the '{error.name}' package in the launcher's Python "
                         f"({sys.executable}). Start the launcher from the toolbox "
                         "environment to analyse predictions.") from error
    if not isinstance(data, dict) or not {"predictions", "labels"} <= set(data):
        raise ValueError("Unsupported output pickle structure")
    recordings = {}
    for name in data["predictions"]:
        prediction = _join_chunks(data["predictions"][name]).reshape(-1)
        label = _join_chunks(data["labels"][name])
        if label.ndim > 1 and label.shape[-1] == 1:
            label = label.reshape(-1)
        n = min(len(prediction), len(label))
        recordings[str(name)] = (prediction[:n], label[:n])
    label_type = data.get("label_type", "")
    label_columns = data.get("label_columns")
    sample = [label for _, label in list(recordings.values())[:5]]
    label_is_hr = bool(label_columns) or (label_type == "Raw" and bool(sample) and
                                          all(signals.looks_like_heart_rate(label) for label in sample))
    parsed = {"recordings": recordings, "fs": float(data.get("fs") or 30), "label_type": label_type,
              "label_columns": label_columns, "label_is_hr": label_is_hr}
    with _cache_lock:
        _pickle_cache[key] = (stamp, parsed)
        while len(_pickle_cache) > 3:
            _pickle_cache.popitem(last=False)
    return parsed


def _recording_meta(name):
    parts = name.split("--")
    if len(parts) == 4:
        return {"participant": parts[0], "segment": parts[1], "movement": parts[2], "view": parts[3]}
    return {}


def _windows(prediction, label, fs, window_s, step_s, is_diff, label_is_hr):
    n = len(prediction)
    size = int(round(window_s * fs)) if window_s else n
    size = min(size, n)
    step = int(round(step_s * fs)) if step_s else size
    if size < 9:
        return []
    rows = []
    for start in range(0, n - size + 1, max(step, 1)):
        segment = slice(start, start + size)
        wave = signals.pulse_waveform(prediction[segment], fs, is_diff)
        predicted = signals.fft_hr(wave, fs)
        if label_is_hr:
            channel = label[segment, 0] if label.ndim > 1 else label[segment]
            finite = channel[np.isfinite(channel)]
            truth = float(finite.mean()) if finite.size else float("nan")
        else:
            channel = label[segment, 0] if label.ndim > 1 else label[segment]
            truth = signals.fft_hr(signals.pulse_waveform(channel, fs, is_diff), fs)
        rows.append({"start": start / fs, "end": (start + size) / fs, "gt": truth, "pred": predicted,
                     "snr": signals.snr_db(wave, fs, truth)})
    return rows


def _metrics(rows):
    pairs = np.array([(r["gt"], r["pred"]) for r in rows if math.isfinite(r["gt"]) and math.isfinite(r["pred"])])
    if not len(pairs):
        return {"windows": 0}
    error = pairs[:, 1] - pairs[:, 0]
    result = {
        "windows": int(len(pairs)),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "bias": float(np.mean(error)),
        "mean_gt": float(np.mean(pairs[:, 0])),
        "mean_pred": float(np.mean(pairs[:, 1])),
    }
    nonzero = pairs[:, 0] != 0
    if nonzero.any():
        result["mape"] = float(np.mean(np.abs(error[nonzero] / pairs[nonzero, 0])) * 100)
    if len(pairs) >= 3 and np.std(pairs[:, 0]) > 0 and np.std(pairs[:, 1]) > 0:
        result["pearson"] = float(np.corrcoef(pairs[:, 0], pairs[:, 1])[0, 1])
    snrs = [r["snr"] for r in rows if math.isfinite(r["snr"])]
    if snrs:
        result["snr"] = float(np.mean(snrs))
    return result


def _diff_flag(parsed, diff):
    if diff in ("1", "true", True):
        return True
    if diff in ("0", "false", False):
        return False
    if parsed["label_is_hr"]:
        return True
    return parsed["label_type"] == "DiffNormalized"


def analyse(path, window_s=10.0, step_s=None, diff="auto"):
    parsed = load_pickle(path)
    fs = parsed["fs"]
    is_diff = _diff_flag(parsed, diff)
    label_is_hr = parsed["label_is_hr"]
    started = time.time()
    recordings = []
    all_rows = []
    for name, (prediction, label) in sorted(parsed["recordings"].items()):
        rows = _windows(prediction, label, fs, window_s, step_s, is_diff, label_is_hr)
        for row in rows:
            row["recording"] = name
        all_rows.extend(rows)
        recordings.append({"id": name, "duration": len(prediction) / fs, **_recording_meta(name),
                           **_metrics(rows)})
    points = [{"recording": r["recording"], "gt": r["gt"], "pred": r["pred"]} for r in all_rows
              if math.isfinite(r["gt"]) and math.isfinite(r["pred"])]
    return {
        "path": str(path), "fs": fs, "label_type": parsed["label_type"], "label_columns": parsed["label_columns"],
        "is_diff": is_diff, "label_is_hr": label_is_hr, "window": window_s, "step": step_s or window_s,
        "overall": _metrics(all_rows), "recordings": recordings, "points": points,
        "seconds": round(time.time() - started, 2),
    }


def recording(path, name, window_s=10.0, step_s=None, diff="auto"):
    parsed = load_pickle(path)
    if name not in parsed["recordings"]:
        raise ValueError(f"Unknown recording: {name}")
    prediction, label = parsed["recordings"][name]
    fs = parsed["fs"]
    is_diff = _diff_flag(parsed, diff)
    label_is_hr = parsed["label_is_hr"]
    rows = _windows(prediction, label, fs, window_s, step_s, is_diff, label_is_hr)
    wave = signals.pulse_waveform(prediction, fs, is_diff)
    wave = (wave - wave.mean()) / (wave.std() or 1)
    payload = {
        "id": name, "fs": fs, "duration": len(prediction) / fs, "windows": rows, "metrics": _metrics(rows),
        "prediction": signals.finite_list(signals.downsample(wave)),
        "is_diff": is_diff, "label_is_hr": label_is_hr,
    }
    if label_is_hr:
        channels = label if label.ndim > 1 else label[:, None]
        names = parsed["label_columns"] or ["heart_rate_bpm", "HRV", "respiration_rate_bpm"][:channels.shape[1]]
        payload["label_channels"] = {names[i]: signals.finite_list(signals.downsample(channels[:, i]), 3)
                                     for i in range(min(len(names), channels.shape[1]))}
    else:
        label_wave = signals.pulse_waveform(label if label.ndim == 1 else label[:, 0], fs, is_diff)
        label_wave = (label_wave - label_wave.mean()) / (label_wave.std() or 1)
        payload["label"] = signals.finite_list(signals.downsample(label_wave))
    freqs, power = signals.spectrum(wave, fs)
    mask = freqs <= 4
    payload["spectrum"] = {"freqs": signals.finite_list(freqs[mask]), "power": signals.finite_list(
        power[mask] / (power[mask].max() or 1))}
    return payload


def window_csv(path):
    """Fallback HR series from vHRM window_results.csv."""
    path = check_path(path)
    table = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            try:
                item = {"start": float(row["start_seconds"]), "end": float(row["end_seconds"]),
                        "gt": float(row["ground_truth_hr_bpm"]), "pred": float(row["predicted_hr_bpm"]),
                        "snr": float(row.get("snr_db") or "nan")}
            except (KeyError, ValueError):
                continue
            table.setdefault(row.get("recording_id", "all"), []).append(item)
    recordings = [{"id": name, **_recording_meta(name), **_metrics(rows), "duration": rows[-1]["end"]}
                  for name, rows in sorted(table.items())]
    points = [{"recording": name, "gt": r["gt"], "pred": r["pred"]} for name, rows in table.items() for r in rows]
    return {"path": str(path), "overall": _metrics([r for rows in table.values() for r in rows]),
            "recordings": recordings, "points": points, "windows_by_recording": table, "source": "csv"}
