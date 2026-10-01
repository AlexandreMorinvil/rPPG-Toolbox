"""Browsing of preprocessed caches (DataFileLists CSVs and *_input/*_label .npy clips)."""

import csv
import os
from functools import lru_cache
from pathlib import Path, PurePosixPath, PureWindowsPath

import numpy as np

from . import paths, settings, signals

MAX_FRAME_BYTES = 64 * 1024 * 1024


def _roots():
    return list(settings.cache_roots())


def check_path(path):
    path = Path(path).resolve(strict=False)
    if not paths.is_within(path, _roots()):
        raise ValueError("Path is outside the configured preprocessed-data folders")
    return path


def scan(max_depth=4):
    """Cache folders are directories containing a DataFileLists sub-folder."""
    sets = []
    for root in _roots():
        stack = [(root, 0)]
        while stack:
            current, depth = stack.pop()
            current = Path(current)
            lists_dir = current / "DataFileLists"
            if lists_dir.is_dir():
                lists = []
                for csv_path in sorted(lists_dir.glob("*.csv")):
                    lists.append({"name": csv_path.stem, "path": str(csv_path.resolve()),
                                  "modified": csv_path.stat().st_mtime})
                experiments = sorted(e.name for e in os.scandir(current)
                                     if e.is_dir() and e.name != "DataFileLists")
                sets.append({"path": str(current.resolve()), "root": str(root), "lists": lists,
                             "experiments": experiments,
                             "name": current.resolve().relative_to(root).as_posix() if current != root else
                             current.name})
            if depth >= max_depth:
                continue
            try:
                children = [e.path for e in os.scandir(current) if e.is_dir() and e.name != "DataFileLists"
                            and not e.name.startswith(".")]
            except OSError:
                continue
            if not lists_dir.is_dir():
                stack.extend((child, depth + 1) for child in children)
    return sets


def _parts(raw):
    raw = raw.strip()
    if raw.startswith("/"):
        return PurePosixPath(raw).parts
    return PureWindowsPath(raw).parts


def _remap(raw, cache_dir, experiments):
    """Find the readable host file for a CSV entry written on another machine or in Docker."""
    candidate = Path(raw)
    if candidate.is_file():
        return candidate
    mapped = paths.container_to_host(raw) if raw.startswith("/") else None
    if mapped is not None and mapped.is_file():
        return mapped
    parts = _parts(raw)
    for index in range(len(parts) - 1, -1, -1):
        if parts[index] in experiments:
            remapped = Path(cache_dir).joinpath(*parts[index:])
            if remapped.is_file():
                return remapped
    return None


@lru_cache(maxsize=8)
def _samples(list_path, stamp):
    list_path = Path(list_path)
    cache_dir = list_path.parent.parent
    experiments = {e.name for e in os.scandir(cache_dir) if e.is_dir()}
    samples = []
    missing = 0
    with list_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if "input_files" not in (reader.fieldnames or []):
            raise ValueError("The file list has no input_files column")
        for row in reader:
            raw = row["input_files"]
            host = _remap(raw, cache_dir, experiments)
            if host is None:
                missing += 1
                continue
            label = host.with_name(host.name.replace("_input", "_label"))
            try:
                relative = host.relative_to(cache_dir).as_posix()
            except ValueError:
                relative = host.name
            parts = relative.split("/")
            samples.append({
                "input": str(host),
                "label": str(label) if label.is_file() else None,
                "name": relative,
                "group": parts[1] if len(parts) > 2 else host.name.split("_input")[0],
            })
            if len(samples) >= 50000:
                break
    return {"samples": samples, "missing": missing, "cache_dir": str(cache_dir)}


def samples(list_path):
    list_path = check_path(list_path)
    if list_path.suffix.lower() != ".csv" or not list_path.is_file():
        raise ValueError("File list not found")
    return _samples(str(list_path), list_path.stat().st_mtime)


def sample_info(input_path):
    path = check_path(input_path)
    if path.suffix != ".npy" or not path.is_file():
        raise ValueError("Sample not found")
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    info = {"path": str(path), "shape": list(array.shape), "dtype": str(array.dtype),
            "size_bytes": path.stat().st_size}
    if array.ndim == 4:
        step = max(1, array.shape[0] // 32)
        sample = np.asarray(array[::step], dtype=np.float32)
        info["frames"], info["height"], info["width"], info["channels"] = array.shape
        info["groups"] = []
        for start in range(0, array.shape[3], 3):
            group = sample[..., start:start + 3]
            finite = group[np.isfinite(group)]
            info["groups"].append({
                "start": start, "channels": int(group.shape[-1]),
                "min": float(finite.min()) if finite.size else None,
                "max": float(finite.max()) if finite.size else None,
                "mean": float(finite.mean()) if finite.size else None,
                "std": float(finite.std()) if finite.size else None,
            })
    label_path = path.with_name(path.name.replace("_input", "_label"))
    if label_path.is_file():
        label = np.load(label_path, mmap_mode="r", allow_pickle=False)
        info["label"] = {"path": str(label_path), "shape": list(label.shape), "dtype": str(label.dtype)}
    return info


def frames(input_path, group=0, stride=1):
    """Return (bytes, meta): uint8 RGB frames for one 3-channel group, normalised per clip."""
    path = check_path(input_path)
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.ndim != 4:
        raise ValueError(f"Expected T,H,W,C input, got shape {array.shape}")
    count, height, width, channels = array.shape
    start = int(group) * 3
    if start >= channels:
        raise ValueError("Channel group out of range")
    stride = max(1, int(stride))
    data = np.asarray(array[::stride, :, :, start:start + 3], dtype=np.float32)
    scale = 1
    while data.nbytes / 4 / (scale * scale) > MAX_FRAME_BYTES:
        scale += 1
    if scale > 1:
        data = data[:, ::scale, ::scale]
    finite = data[np.isfinite(data)]
    if finite.size:
        low, high = np.percentile(finite[:: max(1, finite.size // 200000)], [1, 99])
    else:
        low, high = 0.0, 1.0
    if high <= low:
        high = low + 1
    scaled = np.clip((np.nan_to_num(data) - low) / (high - low) * 255, 0, 255).astype(np.uint8)
    if scaled.shape[-1] == 1:
        scaled = np.repeat(scaled, 3, axis=-1)
    elif scaled.shape[-1] == 2:
        scaled = np.concatenate([scaled, np.zeros_like(scaled[..., :1])], axis=-1)
    scaled = np.ascontiguousarray(scaled)
    meta = {"frames": scaled.shape[0], "height": scaled.shape[1], "width": scaled.shape[2], "stride": stride,
            "low": float(low), "high": float(high)}
    return scaled.tobytes(), meta


def label(input_path, fs=30.0, diff="auto"):
    path = check_path(input_path)
    label_path = path.with_name(path.name.replace("_input", "_label"))
    if not label_path.is_file():
        return {"available": False}
    data = np.asarray(np.load(label_path, allow_pickle=False), dtype=np.float64)
    fs = float(fs)
    result = {"available": True, "shape": list(data.shape), "fs": fs}
    if (data.ndim == 2 and data.shape[1] > 1) or signals.looks_like_heart_rate(data):
        if data.ndim == 1:
            data = data[:, None]
        names = ["heart_rate_bpm", "HRV", "respiration_rate_bpm"] if data.shape[1] <= 3 else \
            [f"channel {i}" for i in range(data.shape[1])]
        result["channels"] = {names[i]: signals.finite_list(data[:, i], 3) for i in range(data.shape[1])}
        hr = data[:, 0][np.isfinite(data[:, 0])]
        result["kind"] = "physiology"
        result["mean_hr"] = float(hr.mean()) if hr.size else None
        return result
    series = data.reshape(-1)
    result["kind"] = "waveform"
    result["signal"] = signals.finite_list(series)
    is_diff = diff in ("1", "true") or (diff == "auto" and "LabelTypeDiffNormalized" in str(path))
    wave = signals.pulse_waveform(series, fs, is_diff)
    freqs, power = signals.spectrum(wave, fs)
    mask = freqs <= 5
    result["is_diff"] = bool(is_diff)
    result["spectrum"] = {"freqs": signals.finite_list(freqs[mask]),
                          "power": signals.finite_list(power[mask] / (power[mask].max() or 1))}
    result["hr"] = signals.fft_hr(wave, fs)
    return result
