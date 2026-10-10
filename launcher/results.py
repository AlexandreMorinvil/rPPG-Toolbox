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

MARKERS = {"PreTrainedModels", "saved_test_outputs", "saved_outputs", "saved_valid_outputs", "plots"}
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
    """Parsed *_outputs.pickle: recordings {id: (prediction, PPG label or None, gt[N, C] or None)}."""
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
    label_type = data.get("label_type", "")
    label_columns = data.get("label_columns")
    gt_all = data.get("gt") or {}
    joined = {}
    for name in data["predictions"]:
        prediction = _join_chunks(data["predictions"][name]).reshape(-1)
        label = _join_chunks(data["labels"][name]) if name in data["labels"] else None
        if label is not None and label.ndim > 1 and label.shape[-1] == 1:
            label = label.reshape(-1)
        joined[str(name)] = (prediction, label, gt_all.get(name))
    sample = [label for _, label, _ in list(joined.values())[:5] if label is not None]
    # Older vHRM outputs stored heart-rate channels as labels instead of a PPG waveform.
    label_is_hr = not gt_all and (bool(label_columns) or (label_type == "Raw" and bool(sample) and
                                                          all(signals.looks_like_heart_rate(x) for x in sample)))
    recordings = {}
    for name, (prediction, label, gt) in joined.items():
        if label_is_hr and label is not None:
            gt, label = (label if label.ndim > 1 else label[:, None]), None
        n = min([len(prediction)] + [len(x) for x in (label, gt) if x is not None])
        gt = np.asarray(gt, dtype=np.float64)[:n] if gt is not None else None
        recordings[name] = (prediction[:n], label[:n] if label is not None else None, gt)
    meta = data.get("meta") or {}
    gt_columns = list(data.get("gt_columns") or label_columns or
                      ["heart_rate_bpm", "HRV", "respiration_rate_bpm", "ppg_valid"])
    parsed = {"recordings": recordings, "fs": float(data.get("fs") or 30), "label_type": label_type,
              "gt_columns": gt_columns, "has_gt": any(gt is not None for _, _, gt in recordings.values()),
              "label_is_hr": label_is_hr, "meta": meta, "split": data.get("split") or meta.get("split") or "test"}
    with _cache_lock:
        _pickle_cache[key] = (stamp, parsed)
        while len(_pickle_cache) > 6:
            _pickle_cache.popitem(last=False)
    return parsed


def _recording_meta(name):
    parts = name.split("--")
    if len(parts) == 5:
        return {"experiment": parts[0], "participant": parts[1], "segment": parts[2], "movement": parts[3],
                "view": parts[4]}
    if len(parts) == 4:
        return {"participant": parts[0], "segment": parts[1], "movement": parts[2], "view": parts[3]}
    return {}


def _flags(parsed, diff):
    """(prediction is differenced, label is differenced)."""
    meta = parsed["meta"]
    label_diff = bool(meta.get("label_is_diff", parsed["label_type"] == "DiffNormalized"))
    if diff in ("1", "true", True):
        return True, label_diff
    if diff in ("0", "false", False):
        return False, label_diff
    if "prediction_is_diff" in meta:
        return bool(meta["prediction_is_diff"]), label_diff
    if parsed["label_is_hr"]:
        return True, label_diff
    return label_diff, label_diff


def _rows(parsed, name, window_s, step_s, diff, truth, track=False):
    prediction, label, gt = parsed["recordings"][name]
    pred_diff, label_diff = _flags(parsed, diff)
    return signals.recording_windows(prediction, parsed["fs"], window_s, step_s, pred_diff, label=label,
                                     label_diff=label_diff, measured_hr=gt[:, 0] if gt is not None else None,
                                     truth=truth, track=track)


def _run_name(path, parsed):
    path = Path(path)
    name = parsed["meta"].get("name") or path.stem.removesuffix("_outputs")
    return f"{path.parent.parent.name} / {name}"


def analyse(path, window_s=10.0, step_s=None, diff="auto", truth="measured", track=False):
    parsed = load_pickle(path)
    pred_diff, label_diff = _flags(parsed, diff)
    started = time.time()
    recordings = []
    points = []
    for name in sorted(parsed["recordings"]):
        prediction = parsed["recordings"][name][0]
        rows = _rows(parsed, name, window_s, step_s, diff, truth, track)
        item = {"id": name, "duration": len(prediction) / parsed["fs"], **_recording_meta(name),
                **signals.window_metrics(rows)}
        if track:
            item["mae_tracked"] = signals.window_metrics(rows, "tracked").get("mae")
        recordings.append(item)
        points.extend({"recording": name, "gt": r["gt"], "pred": r["pred"]} for r in rows
                      if math.isfinite(r["gt"]) and math.isfinite(r["pred"]))
    overall = signals.window_metrics([{"gt": p["gt"], "pred": p["pred"], "snr": float("nan")} for p in points])
    overall.pop("max_jump", None)
    snrs = [r["snr"] for r in recordings if math.isfinite(r.get("snr", float("nan")))]
    if snrs:
        overall["snr"] = float(np.mean(snrs))
    return {
        "path": str(path), "name": _run_name(path, parsed), "fs": parsed["fs"], "label_type": parsed["label_type"],
        "split": parsed["split"], "meta": parsed["meta"], "has_gt": parsed["has_gt"],
        "truth": "measured" if parsed["has_gt"] and truth == "measured" else "label",
        "is_diff": pred_diff, "label_is_diff": label_diff, "label_is_hr": parsed["label_is_hr"],
        "window": window_s, "step": step_s or window_s, "track": bool(track),
        "overall": overall, "recordings": recordings, "points": points,
        "seconds": round(time.time() - started, 2),
    }


WAVE_LIMIT = 30000


def _wave(values):
    return signals.finite_list(signals.downsample(values, WAVE_LIMIT), 3)


def recording(path, name, window_s=10.0, step_s=None, diff="auto", truth="measured", track=False):
    parsed = load_pickle(path)
    if name not in parsed["recordings"]:
        raise ValueError(f"Unknown recording: {name}")
    prediction, label, gt = parsed["recordings"][name]
    fs = parsed["fs"]
    pred_diff, label_diff = _flags(parsed, diff)
    rows = _rows(parsed, name, window_s, step_s, diff, truth, track)
    filtered = signals.pulse_waveform(prediction, fs, pred_diff)
    wave = signals.zscore(filtered)
    raw = signals.fill_nonfinite(prediction)
    raw = signals.zscore(signals.detrend(np.cumsum(raw) if pred_diff else raw))
    payload = {
        "id": name, "fs": fs, "duration": len(prediction) / fs, "windows": rows,
        "metrics": signals.window_metrics(rows), "is_diff": pred_diff, "label_is_hr": parsed["label_is_hr"],
        "truth": "measured" if gt is not None and truth == "measured" else "label",
        "prediction": _wave(wave), "prediction_raw": _wave(raw),
        "pred_beats": signals.finite_list(signals.beats(wave, fs) / fs, 3),
    }
    if track:
        payload["metrics_tracked"] = signals.window_metrics(rows, "tracked")
    if label is not None:
        label_wave = signals.zscore(signals.pulse_waveform(label, fs, label_diff))
        payload["label"] = _wave(label_wave)
        payload["label_beats"] = signals.finite_list(signals.beats(label_wave, fs) / fs, 3)
    if gt is not None:
        names = list(parsed["gt_columns"]) + [f"channel_{i}" for i in range(len(parsed["gt_columns"]), gt.shape[1])]
        payload["gt_channels"] = {names[i]: signals.finite_list(signals.downsample(gt[:, i]), 3)
                                  for i in range(gt.shape[1]) if names[i] != "ppg_valid"}
        if "ppg_valid" in names[:gt.shape[1]]:
            payload["invalid_spans"] = signals.invalid_spans(gt[:, names.index("ppg_valid")] > 0.5, fs)
    window = window_s or 10.0
    times, freqs, power = signals.spectrogram(filtered, fs, window, max((step_s or window) / 2, 0.5))
    payload["spectrogram"] = {"times": signals.finite_list(times, 2), "freqs": signals.finite_list(freqs * 60, 2),
                              "power": [signals.finite_list(row, 3) for row in power]}
    return payload


def compare(paths, name=None, window_s=10.0, step_s=None, diff="auto", truth="measured", track=False):
    """Several runs on the same recordings: a per-recording MAE matrix, or one recording's overlays."""
    runs = [(path, load_pickle(path)) for path in paths]
    if not name:
        common = sorted(set.intersection(*(set(parsed["recordings"]) for _, parsed in runs))) if runs else []
        matrix = [{"id": recording_id, **_recording_meta(recording_id),
                   "mae": [signals.window_metrics(_rows(parsed, recording_id, window_s, step_s, diff, truth)).get("mae")
                           for _, parsed in runs]}
                  for recording_id in common]
        return {"runs": [{"path": str(path), "name": _run_name(path, parsed), "split": parsed["split"]}
                         for path, parsed in runs], "recordings": matrix}
    result = []
    for path, parsed in runs:
        if name not in parsed["recordings"]:
            continue
        prediction = parsed["recordings"][name][0]
        pred_diff, _ = _flags(parsed, diff)
        rows = _rows(parsed, name, window_s, step_s, diff, truth, track)
        result.append({"path": str(path), "name": _run_name(path, parsed), "windows": rows,
                       "metrics": signals.window_metrics(rows), "duration": len(prediction) / parsed["fs"],
                       "prediction": _wave(signals.zscore(signals.pulse_waveform(prediction, parsed["fs"], pred_diff)))})
    return {"id": name, "runs": result}


def sources():
    """Every *_outputs.pickle under the results roots (for the comparison picker)."""
    found = []
    for experiment in scan():
        root = Path(experiment["path"])
        for folder in ("saved_test_outputs", "saved_outputs", "saved_valid_outputs"):
            directory = root / folder
            if directory.is_dir():
                for file in sorted(directory.rglob("*_outputs.pickle")):
                    found.append({"path": str(file), "experiment": experiment["name"],
                                  "name": file.relative_to(root).as_posix(), "modified": file.stat().st_mtime})
    return sorted(found, key=lambda item: item["modified"], reverse=True)


def history(path, name, window_s=10.0, step_s=None, diff="auto", truth="measured"):
    """Probe-recording predictions per epoch from saved_valid_outputs/probe_history.npz."""
    path = check_path(path)
    with np.load(path, allow_pickle=False) as data:
        ids = [str(i) for i in data["ids"]]
        index = ids.index(name) if name in ids else -1
        epochs = [int(e) for e in data["epochs"]]
        predictions = data[f"pred_{index}"].astype(np.float64) if f"pred_{index}" in data.files else None
        label = data[f"label_{index}"].astype(np.float64) if f"label_{index}" in data.files else None
        gt = data[f"gt_{index}"].astype(np.float64) if f"gt_{index}" in data.files else None
    if predictions is None:
        return {"id": name, "epochs": [], "probes": ids}
    best = path.parent / "best_valid_outputs.pickle"
    parsed = load_pickle(best) if best.is_file() else {"meta": {}, "fs": 30.0, "label_type": "", "label_is_hr": False}
    fs = parsed["fs"]
    pred_diff, label_diff = _flags(parsed, diff)
    per_epoch = []
    for epoch, prediction in zip(epochs, predictions):
        if not np.isfinite(prediction).any():
            continue  # probe added after this epoch
        rows = signals.recording_windows(prediction, fs, window_s, step_s, pred_diff, label=label,
                                         label_diff=label_diff, measured_hr=gt[:, 0] if gt is not None else None,
                                         truth=truth)
        per_epoch.append({"epoch": epoch, "windows": rows, "metrics": signals.window_metrics(rows),
                          "prediction": _wave(signals.zscore(signals.pulse_waveform(prediction, fs, pred_diff)))})
    return {"id": name, "fs": fs, "epochs": per_epoch, "probes": ids}


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
    recordings = [{"id": name, **_recording_meta(name), **signals.window_metrics(rows), "duration": rows[-1]["end"]}
                  for name, rows in sorted(table.items())]
    points = [{"recording": name, "gt": r["gt"], "pred": r["pred"]} for name, rows in table.items() for r in rows]
    return {"path": str(path), "overall": signals.window_metrics([r for rows in table.values() for r in rows]),
            "recordings": recordings, "points": points, "windows_by_recording": table, "source": "csv"}
