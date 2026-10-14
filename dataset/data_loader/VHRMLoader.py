"""Data loader for the vHRM lab dataset.

Expected layout (produced by the vHRM video research tool):

    <DATA_PATH>/<experiment>/<participant>/
        zephyr_1.csv, polar_1.csv, synthetic_ppg/*.csv        (full-session signals)
        labelled_segments/<NNNN>_<Label>/
            meta.json, cam<N>_Camera_<View>.mp4, zephyr_1.csv, synthetic_ppg/*.csv

One recording is one (segment, camera view). Videos are resampled to DATA.FS and cached
as clips shared by every split. Labels are a PPG waveform sampled on the absolute time of
each frame; a ground-truth sidecar holds the measured HR, HRV, and respiration rate used
for evaluation. Labels live in a sub-folder per label setting, so changing the waveform
source or the sync correction never requires decoding the videos again.
"""

import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
import traceback

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from dataset.data_loader.BaseLoader import BaseLoader

CACHE_VERSION = 2
KNOWN_VIEWS = ("front", "back-left", "back-right")
CROP_MODES = ("face", "face_or_full", "full", "roi")
PPG_SOURCES = ("synthetic", "rpeak_sine", "rpeak_gaussian")
HR_SOURCES = ("zephyr", "polar", "synthetic_hr")
GT_COLUMNS = ("heart_rate_bpm", "HRV", "respiration_rate_bpm", "ppg_valid")

HR_COLUMNS = {
    "zephyr": ("heart_rate_bpm", "Zephyr"),
    "polar": ("heart_rate_bpm", "Polar"),
    "synthetic_hr": ("heart_rate_bpm",),
}
HRV_COLUMNS = ("HRV", "aux__Zephyr__HRV")
BR_COLUMNS = ("BR", "aux__Zephyr__BR")
VALID_RANGES = {"hr": (20.0, 250.0), "hrv": (0.0, 65534.0), "br": (0.0, 100.0)}
# Largest gap between two source samples that may still be interpolated (seconds).
MAX_GAP_PPG = 0.1
MAX_GAP_SLOW = 3.0
# Plausible R-R interval range for the R-peak based waveforms (seconds).
RR_RANGE = (0.3, 2.0)
# Clockwise display rotation (container metadata) -> cv2.rotate code. The vHRM cameras
# record landscape frames tagged with a -90 degree display matrix (portrait).
ROTATE_CODES = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
# Bump when the decoded frame geometry changes so cached clips are decoded again.
FRAME_GEOMETRY = "display-rotation-v1"


def slug(value):
    return re.sub(r"[^a-z0-9]+", "-", str(value).strip().lower()).strip("-")


def parse_aliases(items):
    aliases = {}
    for item in items:
        if not str(item).strip():
            continue
        if "=" not in str(item):
            raise ValueError(f"vHRM alias must look like 'old=new': {item!r}")
        old, new = str(item).split("=", 1)
        aliases[slug(old)] = slug(new)
    return aliases


def parse_crop_modes(items):
    """Parse ['front=face', 'back-left=roi:0.3,0.1,0.4,0.4', '*=full']."""
    modes = {}
    for item in items:
        if not str(item).strip():
            continue
        if "=" not in str(item):
            raise ValueError(f"vHRM CROP_MODE_BY_VIEW entries must look like 'view=mode': {item!r}")
        view, mode = str(item).split("=", 1)
        mode = mode.strip().lower()
        roi = None
        if mode.startswith("roi"):
            _, _, values = mode.partition(":")
            roi = tuple(float(value) for value in values.split(","))
            if len(roi) != 4 or roi[2] <= 0 or roi[3] <= 0:
                raise ValueError(f"vHRM roi must be roi:x,y,w,h with a positive size: {item!r}")
            mode = "roi"
        if mode not in CROP_MODES:
            raise ValueError(f"Unsupported vHRM crop mode {mode!r}; use one of {CROP_MODES}")
        key = "*" if view.strip() == "*" else slug(view)
        modes[key] = (mode, roi)
    return modes


def _parse_utc(value):
    if value in (None, ""):
        return None
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    return timestamp.value / 1e9


def _first_existing(*candidates):
    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return candidate
    return None


def _sorted_glob(directory, pattern):
    return sorted(glob.glob(os.path.join(directory, pattern)))


def discover_segments(data_path, experiments=()):
    """List every labelled segment under DATA_PATH, without filtering views or labels."""
    if not os.path.isdir(data_path):
        raise ValueError(f"vHRM data path does not exist: {data_path}")
    experiments = {slug(experiment) for experiment in experiments}
    segments = []
    for experiment_entry in sorted(os.scandir(data_path), key=lambda entry: entry.name):
        if not experiment_entry.is_dir():
            continue
        if experiments and slug(experiment_entry.name) not in experiments:
            continue
        for participant_entry in sorted(os.scandir(experiment_entry.path), key=lambda entry: entry.name):
            segments_path = os.path.join(participant_entry.path, "labelled_segments")
            if not participant_entry.is_dir() or not os.path.isdir(segments_path):
                continue
            for segment_entry in sorted(os.scandir(segments_path), key=lambda entry: entry.name):
                metadata_path = os.path.join(segment_entry.path, "meta.json")
                if not segment_entry.is_dir() or not os.path.isfile(metadata_path):
                    continue
                with open(metadata_path, "r", encoding="utf-8") as metadata_file:
                    metadata = json.load(metadata_file)
                anchor = _parse_utc(metadata.get("signal_anchor_datetime_utc"))
                if anchor is not None and metadata.get("start_sec") is not None:
                    start_epoch = anchor + float(metadata["start_sec"])
                else:
                    start_epoch = _parse_utc(metadata.get("segment_start_datetime_utc"))
                cameras = []
                for camera in metadata.get("cameras", []):
                    label = str(camera.get("label", ""))
                    video_path = os.path.join(segment_entry.path, camera.get("video_file", ""))
                    cameras.append({
                        "view": slug(label.removeprefix("Camera_")),
                        "camera_label": label,
                        "path": video_path,
                        "exists": bool(camera.get("video_file")) and os.path.isfile(video_path),
                        "fps": camera.get("fps"),
                        "dimensions": camera.get("dimensions"),
                    })
                segments.append({
                    "experiment": slug(experiment_entry.name),
                    "experiment_folder": experiment_entry.name,
                    "participant": slug(participant_entry.name),
                    "participant_folder": participant_entry.name,
                    "participant_dir": participant_entry.path,
                    "segment_dir": segment_entry.path,
                    "segment_folder": segment_entry.name,
                    "segment_index": str(metadata.get("index", segment_entry.name.split("_", 1)[0])).zfill(4),
                    "label": str(metadata.get("label", segment_entry.name.split("_", 1)[-1])),
                    "start_epoch": start_epoch,
                    "duration_sec": float(metadata.get("duration_sec", 0) or 0),
                    "meta_paths": {
                        key: metadata.get(key) for key in (
                            "synthetic_ppg_path", "synthetic_rr_path", "synthetic_ppg_hr_path")
                    },
                    "cameras": cameras,
                })
    return segments


def resolve_signal_files(recording):
    """Signal files of a recording. Full-session files are preferred so that shifted
    signals stay available at the segment edges; segment slices are the fallback."""
    participant_dir = recording["participant_dir"]
    segment_dir = recording["segment_dir"]
    meta_paths = recording.get("meta_paths", {})

    def from_meta(key):
        value = meta_paths.get(key)
        return os.path.join(segment_dir, value) if value else None

    return {
        "ppg": _first_existing(
            os.path.join(participant_dir, "synthetic_ppg", "synthetic_ppg.csv"),
            from_meta("synthetic_ppg_path"),
            os.path.join(segment_dir, "synthetic_ppg", "synthetic_ppg.csv")),
        "rpeaks": _first_existing(
            os.path.join(participant_dir, "synthetic_ppg", "synthetic_ppg_rr.csv"),
            from_meta("synthetic_rr_path"),
            os.path.join(segment_dir, "synthetic_ppg", "synthetic_ppg_rr.csv")),
        "zephyr": _first_existing(
            *_sorted_glob(participant_dir, "zephyr*.csv"),
            *_sorted_glob(segment_dir, "zephyr*.csv"),
            os.path.join(participant_dir, "signal_synced.csv"),
            os.path.join(segment_dir, "signal.csv")),
        "polar": _first_existing(
            *_sorted_glob(participant_dir, "polar*.csv"),
            *_sorted_glob(segment_dir, "polar*.csv")),
        "synthetic_hr": _first_existing(
            os.path.join(participant_dir, "synthetic_ppg", "synthetic_ppg_hr.csv"),
            from_meta("synthetic_ppg_hr_path"),
            os.path.join(segment_dir, "synthetic_ppg", "synthetic_ppg_hr.csv")),
    }


def select_recordings(segments, vhrm, warn=print):
    """Apply the experiment / participant / label / view filters of a VHRM config node."""
    experiments = {slug(item) for item in vhrm.EXPERIMENTS if str(item).strip()}
    participants = {slug(item) for item in vhrm.PARTICIPANTS if str(item).strip()}
    excluded_participants = {slug(item) for item in vhrm.EXCLUDE_PARTICIPANTS if str(item).strip()}
    label_aliases = parse_aliases(vhrm.LABEL_ALIASES)
    view_aliases = parse_aliases(vhrm.VIEW_ALIASES)

    def canonical_label(value):
        return label_aliases.get(slug(value), slug(value))

    def canonical_view(value):
        return view_aliases.get(slug(value), slug(value))

    labels = {canonical_label(item) for item in vhrm.LABELS if str(item).strip()}
    excluded_labels = {canonical_label(item) for item in vhrm.EXCLUDE_LABELS if str(item).strip()}
    views = {canonical_view(item) for item in vhrm.VIEWS if str(item).strip()}
    crop_modes = parse_crop_modes(vhrm.CROP_MODE_BY_VIEW)

    recordings = []
    for segment in segments:
        if experiments and segment["experiment"] not in experiments:
            continue
        if participants and segment["participant"] not in participants:
            continue
        if segment["participant"] in excluded_participants:
            continue
        label = canonical_label(segment["label"])
        if (labels and label not in labels) or label in excluded_labels:
            continue
        if segment["start_epoch"] is None:
            warn(f"vHRM: skipping {segment['segment_dir']} (meta.json has no start time).")
            continue
        for camera in segment["cameras"]:
            view = canonical_view(camera["view"])
            if views and view not in views:
                continue
            if not views and view not in KNOWN_VIEWS:
                warn(f"vHRM: skipping unknown view {camera['camera_label']!r} in {segment['segment_dir']} "
                     f"(map it with VIEW_ALIASES).")
                continue
            if not camera["exists"]:
                warn(f"vHRM: missing video for view {view!r} in {segment['segment_dir']}.")
                continue
            recording_id = "--".join((
                segment["experiment"], segment["participant"], segment["segment_index"], label, view))
            recordings.append({
                **{key: value for key, value in segment.items() if key != "cameras"},
                "index": recording_id,
                "label": label,
                "view": view,
                "path": camera["path"],
                "camera_fps": camera["fps"],
                "crop_mode": crop_modes.get(view, crop_modes.get("*", ("face", None))),
            })
    return recordings


def load_lag_table(path):
    """Per-segment sync corrections: columns participant, segment_index, lag_sec and
    optionally experiment, view (empty = every view) and use (rows with false are ignored)."""
    if not path:
        return {}
    if not os.path.isfile(path):
        raise ValueError(f"vHRM LAG_TABLE_PATH does not exist: {path}")
    table = pd.read_csv(path, dtype=str).fillna("")
    for column in ("participant", "segment_index", "lag_sec"):
        if column not in table:
            raise ValueError(f"vHRM lag table {path} has no {column!r} column")
    lags = {}
    for _, row in table.iterrows():
        if "use" in table and row["use"].strip().lower() in {"0", "false", "no", "n"}:
            continue
        if not row["lag_sec"].strip():
            continue
        key = (
            slug(row.get("experiment", "")),
            slug(row["participant"]),
            str(row["segment_index"]).strip().zfill(4),
            slug(row.get("view", "")),
        )
        lags[key] = float(row["lag_sec"])
    return lags


def lookup_lag(lags, recording):
    for experiment in (recording["experiment"], ""):
        for view in (recording["view"], ""):
            key = (experiment, recording["participant"], recording["segment_index"], view)
            if key in lags:
                return lags[key]
    return 0.0


_CSV_CACHE = {}


def read_timed_csv(path):
    """Read a signal CSV and add an 'epoch' column (seconds since 1970, UTC)."""
    key = (path, os.path.getmtime(path))
    frame = _CSV_CACHE.get(key)
    if frame is None:
        frame = pd.read_csv(path)
        if "timestamp_utc" not in frame:
            raise ValueError(f"Signal file has no timestamp_utc column: {path}")
        timestamps = pd.to_datetime(frame["timestamp_utc"], utc=True, format="ISO8601")
        frame["epoch"] = (timestamps - pd.Timestamp(0, tz="UTC")) / pd.Timedelta(seconds=1)
        if len(_CSV_CACHE) >= 16:
            _CSV_CACHE.clear()
        _CSV_CACHE[key] = frame
    return frame


def sample_series(source_times, source_values, query_times, max_gap):
    """Linear interpolation that returns NaN outside the data and across gaps > max_gap."""
    result = np.full(len(query_times), np.nan)
    valid = np.isfinite(source_times) & np.isfinite(source_values)
    times = np.asarray(source_times[valid], dtype=np.float64)
    values = np.asarray(source_values[valid], dtype=np.float64)
    if len(times) < 2:
        return result
    order = np.argsort(times, kind="stable")
    times, values = times[order], values[order]
    right = np.clip(np.searchsorted(times, query_times, side="left"), 1, len(times) - 1)
    gaps = times[right] - times[right - 1]
    inside = (query_times >= times[0]) & (query_times <= times[-1]) & (gaps <= max_gap)
    result[inside] = np.interp(query_times[inside], times, values)
    return result


def _column(frame, candidates):
    return next((name for name in candidates if name in frame), None)


def _read_values(path, candidates, valid_range=None):
    frame = read_timed_csv(path)
    column = _column(frame, candidates)
    if column is None:
        return None, None
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
    if valid_range is not None:
        values = np.where((values > valid_range[0]) & (values < valid_range[1]), values, np.nan)
    return frame["epoch"].to_numpy(dtype=np.float64), values


def rpeak_waveform(peak_times, query_times, shape, sigma_fraction):
    """Beat-locked waveform with its maximum on each (already delayed) R-peak time."""
    result = np.full(len(query_times), np.nan)
    peaks = np.sort(np.asarray(peak_times[np.isfinite(peak_times)], dtype=np.float64))
    if len(peaks) < 2:
        return result
    beat = np.searchsorted(peaks, query_times, side="right") - 1
    inside = (beat >= 0) & (beat < len(peaks) - 1)
    beat = np.clip(beat, 0, len(peaks) - 2)
    start, end = peaks[beat], peaks[beat + 1]
    rr = end - start
    inside &= (rr >= RR_RANGE[0]) & (rr <= RR_RANGE[1])
    if shape == "rpeak_sine":
        phase = (query_times - start) / rr
        waveform = 0.5 * (1.0 + np.cos(2.0 * np.pi * phase))
    else:
        sigma = sigma_fraction * rr
        waveform = (np.exp(-0.5 * ((query_times - start) / sigma) ** 2)
                    + np.exp(-0.5 * ((query_times - end) / sigma) ** 2))
    result[inside] = waveform[inside]
    return result


def build_signals(recording, frame_count, fs, settings):
    """Sample the training waveform and the ground-truth series at each frame time.

    settings: ppg_source, rpeak_delay, rpeak_sigma, sync_delay, hr_source.
    A positive sync_delay shifts every signal later relative to the video.
    """
    files = resolve_signal_files(recording)
    times = recording["start_epoch"] + np.arange(frame_count, dtype=np.float64) / fs - settings["sync_delay"]
    signals = {}

    ppg = np.full(frame_count, np.nan)
    if settings["ppg_source"] == "synthetic":
        if files["ppg"]:
            source_times, values = _read_values(files["ppg"], ("synthetic_ppg",))
            if values is not None:
                ppg = sample_series(source_times, values, times, MAX_GAP_PPG)
    elif files["rpeaks"]:
        peak_times = read_timed_csv(files["rpeaks"])["epoch"].to_numpy(dtype=np.float64)
        ppg = rpeak_waveform(peak_times + settings["rpeak_delay"], times,
                             settings["ppg_source"], settings["rpeak_sigma"])
    signals["ppg"] = ppg

    def sampled(path, candidates, valid_range):
        if not path:
            return np.full(frame_count, np.nan)
        source_times, values = _read_values(path, candidates, valid_range)
        if values is None:
            return np.full(frame_count, np.nan)
        return sample_series(source_times, values, times, MAX_GAP_SLOW)

    hr_source = settings["hr_source"]
    signals["hr"] = sampled(files[hr_source], HR_COLUMNS[hr_source], VALID_RANGES["hr"])
    signals["hrv"] = sampled(files["zephyr"], HRV_COLUMNS, VALID_RANGES["hrv"])
    signals["br"] = sampled(files["zephyr"], BR_COLUMNS, VALID_RANGES["br"])
    return signals


def transform_label(ppg, label_type):
    """NaN-aware equivalents of BaseLoader's label transforms (NaN marks missing PPG)."""
    if label_type == "Raw":
        return ppg.astype(np.float64)
    if label_type == "DiffNormalized":
        difference = np.diff(ppg)
        deviation = np.nanstd(difference) if np.any(np.isfinite(difference)) else np.nan
        if np.isfinite(deviation) and deviation > 0:
            difference = difference / deviation
        last = 0.0 if len(ppg) and np.isfinite(ppg[-1]) else np.nan
        return np.append(difference, last)
    if label_type == "Standardized":
        if not np.any(np.isfinite(ppg)):
            return ppg.astype(np.float64)
        deviation = np.nanstd(ppg)
        centred = ppg - np.nanmean(ppg)
        return centred / deviation if deviation > 0 else centred
    raise ValueError(f"Unsupported label type: {label_type}")


def _fingerprint(settings):
    serialized = json.dumps(settings, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _file_signature(path):
    if not path or not os.path.isfile(path):
        return None
    stat = os.stat(path)
    return [stat.st_size, stat.st_mtime_ns]


class VHRMLoader(BaseLoader):
    """Load vHRM videos with a PPG training label and measured HR for evaluation."""

    def __init__(self, name, data_path, config_data, device=None):
        vhrm = config_data.VHRM
        if not config_data.FS or config_data.FS <= 0:
            raise ValueError("vHRM resamples videos to DATA.FS; set FS (e.g. 30).")
        if vhrm.PPG_SOURCE not in PPG_SOURCES:
            raise ValueError(f"VHRM.PPG_SOURCE must be one of {PPG_SOURCES}")
        if vhrm.HR_SOURCE not in HR_SOURCES:
            raise ValueError(f"VHRM.HR_SOURCE must be one of {HR_SOURCES}")
        self.vhrm = vhrm
        self.require_ppg = vhrm.REQUIRE_PPG
        self.lag_table = load_lag_table(vhrm.LAG_TABLE_PATH)
        self.video_decoder = vhrm.VIDEO_DECODER.lower()
        self.ffmpeg_path = vhrm.FFMPEG_PATH
        self.ffprobe_path = vhrm.FFPROBE_PATH
        self.incremental_preprocess = vhrm.INCREMENTAL_PREPROCESS
        self.manifest_filename = vhrm.MANIFEST_FILENAME
        self.cache_dtype_name = vhrm.CACHE_DTYPE.lower()
        self.show_video_progress = vhrm.SHOW_VIDEO_PROGRESS
        if self.cache_dtype_name not in {"float16", "float32"}:
            raise ValueError("VHRM.CACHE_DTYPE must be 'float16' or 'float32'")
        self.cache_dtype = np.dtype(self.cache_dtype_name)
        parse_crop_modes(vhrm.CROP_MODE_BY_VIEW)  # fail early on a malformed entry
        self.gt_files = []
        self._video_codec_cache = {}
        self._nvdec_decoders = None
        super().__init__(name, data_path, config_data, device)

    # ------------------------------------------------------------------ selection

    def get_raw_data(self, data_path):
        segments = discover_segments(data_path, self.vhrm.EXPERIMENTS)
        recordings = select_recordings(segments, self.vhrm)
        if not recordings:
            raise ValueError(f"{self.dataset_name}: no vHRM recordings match the VHRM selection "
                             f"(experiments, participants, labels, views) under {data_path}")
        return recordings

    def split_raw_data(self, data_dirs, begin, end):
        if begin == 0 and end == 1:
            return data_dirs
        participants = sorted({recording["participant"] for recording in data_dirs})
        selected = set(participants[int(begin * len(participants)):int(end * len(participants))])
        return [recording for recording in data_dirs if recording["participant"] in selected]

    # ------------------------------------------------------------------ cache layout

    def _recording_dir(self, recording):
        # Kept shallow: Windows paths are limited to 260 characters and file names
        # already carry the full recording id.
        return os.path.join(self.cached_path, recording["experiment"], recording["participant"])

    def _input_path(self, recording, chunk):
        return os.path.join(self._recording_dir(recording), f"{recording['index']}_input{chunk}.npy")

    def _label_dir(self, recording, label_key):
        return os.path.join(self._recording_dir(recording), f"labels_{label_key}")

    def _label_paths(self, recording, label_key, chunk):
        directory = self._label_dir(recording, label_key)
        return (os.path.join(directory, f"{recording['index']}_label{chunk}.npy"),
                os.path.join(directory, f"{recording['index']}_gt{chunk}.npy"))

    def _crop_mode(self, recording, config_preprocess):
        if not config_preprocess.CROP_FACE.DO_CROP_FACE:
            return ("full", None)
        mode, roi = recording["crop_mode"]
        return (mode, tuple(roi) if roi else None)

    def _video_fingerprint(self, recording, config_preprocess):
        crop = config_preprocess.CROP_FACE
        detection = crop.DETECTION
        mode, roi = self._crop_mode(recording, config_preprocess)
        settings = {
            "version": CACHE_VERSION,
            "frame_geometry": FRAME_GEOMETRY,
            "fs": self.config_data.FS,
            "data_type": list(config_preprocess.DATA_TYPE),
            "data_aug": list(config_preprocess.DATA_AUG),
            "do_chunk": config_preprocess.DO_CHUNK,
            "chunk_length": config_preprocess.CHUNK_LENGTH,
            "resize": [config_preprocess.RESIZE.H, config_preprocess.RESIZE.W],
            "crop_mode": mode,
            "roi": list(roi) if roi else None,
            "cache_dtype": self.cache_dtype_name,
        }
        if mode in ("face", "face_or_full"):
            settings.update({
                "crop_backend": crop.BACKEND,
                "large_box": crop.USE_LARGE_FACE_BOX,
                "large_box_coef": crop.LARGE_BOX_COEF,
                "dynamic_detection": detection.DO_DYNAMIC_DETECTION,
                "detection_frequency": detection.DYNAMIC_DETECTION_FREQUENCY,
            })
        return _fingerprint(settings)

    def _sync_delay(self, recording):
        return float(self.vhrm.PPG_DELAY_SEC) + lookup_lag(self.lag_table, recording)

    def _label_settings(self, recording):
        settings = {
            "ppg_source": self.vhrm.PPG_SOURCE,
            "sync_delay": round(self._sync_delay(recording), 6),
            "hr_source": self.vhrm.HR_SOURCE,
            "rpeak_delay": 0.0,
            "rpeak_sigma": 0.0,
        }
        if self.vhrm.PPG_SOURCE.startswith("rpeak"):
            settings["rpeak_delay"] = float(self.vhrm.RPEAK_PULSE_DELAY_SEC)
        if self.vhrm.PPG_SOURCE == "rpeak_gaussian":
            settings["rpeak_sigma"] = float(self.vhrm.RPEAK_GAUSSIAN_SIGMA)
        return settings

    def _label_key(self, recording, config_preprocess):
        settings = {
            "version": CACHE_VERSION,
            "fs": self.config_data.FS,
            "label_type": config_preprocess.LABEL_TYPE,
            "do_chunk": config_preprocess.DO_CHUNK,
            "chunk_length": config_preprocess.CHUNK_LENGTH,
            **self._label_settings(recording),
        }
        return _fingerprint(settings)[:10]

    @staticmethod
    def _signal_signature(recording):
        return {kind: _file_signature(path) for kind, path in resolve_signal_files(recording).items()}

    # ------------------------------------------------------------------ preprocessing

    def preprocess_dataset(self, data_dirs, config_preprocess, begin, end, allow_video_processing=True):
        recordings = self.split_raw_data(data_dirs, begin, end)
        os.makedirs(self.cached_path, exist_ok=True)
        manifest_path = os.path.join(self.cached_path, self.manifest_filename)
        manifest = self._load_manifest(manifest_path)
        entries = manifest["recordings"]

        pending = [recording for recording in recordings
                   if not (self.incremental_preprocess
                           and self._video_outputs_valid(entries.get(recording["index"]), recording,
                                                         config_preprocess))]
        failed = []
        if pending:
            if not allow_video_processing:
                raise ValueError(
                    f"{self.dataset_name}: {len(pending)} vHRM recordings are not in the cache "
                    f"{self.cached_path}. Set DO_PREPROCESS: True.")
            num_workers = getattr(config_preprocess, "NUM_WORKERS", 4)
            results = dict(self.multi_process_manager(
                pending, config_preprocess, multi_process_quota=num_workers))
            for index, recording in enumerate(pending):
                result = results.get(index)
                if result is None:
                    failed.append(recording["index"])
                    entries.pop(recording["index"], None)
                    continue
                result = dict(result)
                self._remove_stale_outputs(recording, result["chunk_count"])
                entries[recording["index"]] = {
                    "source": _file_signature(recording["path"]),
                    "video_fingerprint": self._video_fingerprint(recording, config_preprocess),
                    "frame_count": result["frame_count"],
                    "chunk_count": result["chunk_count"],
                    "crop": result["crop"],
                    "labels": {},
                }
            self._write_manifest(manifest_path, manifest)

        rows = []
        dropped_chunks = 0
        relabelled = 0
        for recording in recordings:
            entry = entries.get(recording["index"])
            if entry is None:
                continue
            label_key = self._label_key(recording, config_preprocess)
            variant = entry.setdefault("labels", {}).get(label_key)
            if (variant is None or variant.get("signals") != self._signal_signature(recording)
                    or not self._label_outputs_exist(recording, label_key, entry["chunk_count"])):
                variant = self._write_labels(recording, entry, config_preprocess, label_key)
                entry["labels"][label_key] = variant
                relabelled += 1
            for chunk in range(entry["chunk_count"]):
                if self.require_ppg and not variant["ppg_valid"][chunk]:
                    dropped_chunks += 1
                    continue
                label_path, gt_path = self._label_paths(recording, label_key, chunk)
                rows.append({
                    "input_files": self._input_path(recording, chunk),
                    "label_files": label_path,
                    "gt_files": gt_path,
                    "recording_id": recording["index"],
                    "chunk": chunk,
                })
        if relabelled:
            self._write_manifest(manifest_path, manifest)

        if failed:
            print(f"WARNING: {len(failed)} vHRM recordings failed to preprocess and are excluded: {failed}")
        print(f"vHRM {self.dataset_name}: {len(recordings) - len(failed)} recordings, "
              f"{len(pending) - len(failed)} decoded, {relabelled} (re)labelled, {len(rows)} clips kept, "
              f"{dropped_chunks} clips dropped (no valid PPG).", end="\n\n")
        if not rows:
            raise ValueError(f"{self.dataset_name}: no vHRM clips available for this split.")
        self._write_file_list(rows)
        self.load_preprocessed_data()

    def build_file_list_retroactive(self, data_dirs, begin, end):
        self.preprocess_dataset(data_dirs, self.config_data.PREPROCESS, begin, end,
                                allow_video_processing=False)

    def preprocess_dataset_subprocess(self, data_dirs, config_preprocess, index, file_list_dict):
        recording = data_dirs[index]
        recording_id = recording["index"]
        started_at = time.perf_counter()
        progress_position = 1 + index % max(1, config_preprocess.NUM_WORKERS)
        expected_frames = round(recording["duration_sec"] * self.config_data.FS)
        decode_progress = tqdm(
            total=expected_frames or None,
            desc=f"{recording_id} | decode",
            unit="frame",
            position=progress_position,
            leave=False,
            dynamic_ncols=True,
            disable=not self.show_video_progress,
        )
        self._progress_message(
            f"[{recording_id}] Starting: {recording['duration_sec']:.1f}s, "
            f"target {self.config_data.FS} FPS ({expected_frames} frames).")
        try:
            crop_mode = self._crop_mode(recording, config_preprocess)
            crop_stats = {"mode": crop_mode[0], "attempts": 0, "failures": 0}
            try:
                frames = self.read_video(recording["path"], self.config_data.FS, config_preprocess,
                                         decode_progress, crop_mode, crop_stats)
            finally:
                decode_progress.close()
            decoded_seconds = len(frames) / self.config_data.FS
            if recording["duration_sec"] and abs(decoded_seconds - recording["duration_sec"]) > 1.0:
                print(f"WARNING: [{recording_id}] decoded {decoded_seconds:.2f}s but meta.json says "
                      f"{recording['duration_sec']:.2f}s; labels assume the video starts at the segment start.")
            if crop_stats["attempts"] and crop_stats["failures"] > crop_stats["attempts"] / 2:
                print(f"WARNING: [{recording_id}] no face in {crop_stats['failures']}/{crop_stats['attempts']} "
                      f"detections (crop mode {crop_mode[0]}).")
            chunk_count = self.transform_and_save(frames, config_preprocess, recording, progress_position)
            file_list_dict[index] = {
                "frame_count": int(len(frames)),
                "chunk_count": int(chunk_count),
                "crop": crop_stats,
            }
            self._progress_message(
                f"[{recording_id}] Complete: {chunk_count} clips in {time.perf_counter() - started_at:.1f}s.")
        except Exception:
            print(f"ERROR: [{recording_id}] preprocessing failed:\n{traceback.format_exc()}")

    def _progress_message(self, message):
        if self.show_video_progress:
            tqdm.write(message)

    def _video_outputs_valid(self, entry, recording, config_preprocess):
        if not entry or entry.get("video_fingerprint") != self._video_fingerprint(recording, config_preprocess):
            return False
        if entry.get("source") != _file_signature(recording["path"]):
            return False
        chunk_count = entry.get("chunk_count", 0)
        paths = [self._input_path(recording, chunk) for chunk in range(chunk_count)]
        if not all(os.path.isfile(path) for path in paths):
            return False
        channels = 3 * len(config_preprocess.DATA_TYPE)
        for path in paths[:1] + paths[-1:]:
            try:
                array = np.load(path, mmap_mode="r")
                shape, dtype = array.shape, array.dtype
                if hasattr(array, "_mmap"):
                    array._mmap.close()
            except (OSError, ValueError):
                return False
            if dtype != self.cache_dtype or shape[1:] != (
                    config_preprocess.RESIZE.H, config_preprocess.RESIZE.W, channels):
                return False
            if config_preprocess.DO_CHUNK and shape[0] != config_preprocess.CHUNK_LENGTH:
                return False
        return True

    def _label_outputs_exist(self, recording, label_key, chunk_count):
        return all(os.path.isfile(path) for chunk in range(chunk_count)
                   for path in self._label_paths(recording, label_key, chunk))

    def _remove_stale_outputs(self, recording, chunk_count):
        """Drop clips beyond the new clip count and every label written for the old clips."""
        directory = self._recording_dir(recording)
        prefix = glob.escape(recording["index"])
        for path in glob.glob(os.path.join(directory, f"{prefix}_input*.npy")):
            match = re.search(r"_input(\d+)\.npy$", path)
            if match and int(match.group(1)) >= chunk_count:
                os.remove(path)
        for pattern in (f"{prefix}_label*.npy", f"{prefix}_gt*.npy"):
            for path in glob.glob(os.path.join(directory, "labels_*", pattern)):
                os.remove(path)

    def _write_labels(self, recording, entry, config_preprocess, label_key):
        frame_count = entry["frame_count"]
        settings = self._label_settings(recording)
        signals = build_signals(recording, frame_count, self.config_data.FS, settings)
        ppg = transform_label(signals["ppg"], config_preprocess.LABEL_TYPE)
        ppg_valid = np.isfinite(ppg)
        ground_truth = np.column_stack(
            (signals["hr"], signals["hrv"], signals["br"], ppg_valid.astype(np.float64)))
        chunk_length = config_preprocess.CHUNK_LENGTH if config_preprocess.DO_CHUNK else frame_count
        os.makedirs(self._label_dir(recording, label_key), exist_ok=True)
        chunk_ppg_valid = []
        chunk_hr_valid = []
        for chunk in range(entry["chunk_count"]):
            start, end = chunk * chunk_length, (chunk + 1) * chunk_length
            label_path, gt_path = self._label_paths(recording, label_key, chunk)
            np.save(label_path, np.nan_to_num(ppg[start:end], nan=0.0).astype(np.float32))
            np.save(gt_path, ground_truth[start:end].astype(np.float32))
            chunk_ppg_valid.append(bool(np.all(ppg_valid[start:end])))
            chunk_hr_valid.append(bool(np.any(np.isfinite(signals["hr"][start:end]))))
        return {
            "signals": self._signal_signature(recording),
            "settings": settings,
            "ppg_coverage": float(np.mean(np.isfinite(signals["ppg"]))) if frame_count else 0.0,
            "ppg_valid": chunk_ppg_valid,
            "hr_valid": chunk_hr_valid,
        }

    def _write_file_list(self, rows):
        os.makedirs(os.path.dirname(self.file_list_path), exist_ok=True)
        pd.DataFrame(rows).to_csv(self.file_list_path, index=False)

    def load_preprocessed_data(self):
        file_list = pd.read_csv(self.file_list_path)
        if file_list.empty:
            raise ValueError(self.dataset_name + " dataset loading data error!")
        if "label_files" not in file_list:
            raise ValueError(f"{self.file_list_path} was written by an older vHRM loader; "
                             f"delete it or set DO_PREPROCESS: True.")
        file_list = file_list.sort_values("input_files")
        self.inputs = file_list["input_files"].tolist()
        self.labels = file_list["label_files"].tolist()
        self.gt_files = file_list["gt_files"].tolist()
        self.preprocessed_data_len = len(self.inputs)

    @staticmethod
    def _load_manifest(manifest_path):
        empty = {"version": CACHE_VERSION, "recordings": {}}
        if not os.path.isfile(manifest_path):
            return empty
        try:
            with open(manifest_path, "r", encoding="utf-8") as manifest_file:
                manifest = json.load(manifest_file)
            if manifest.get("version") != CACHE_VERSION or not isinstance(manifest.get("recordings"), dict):
                raise ValueError("unsupported manifest format")
            return manifest
        except (json.JSONDecodeError, OSError, ValueError) as error:
            print(f"Ignoring invalid incremental preprocessing manifest: {error}")
            return empty

    @staticmethod
    def _write_manifest(manifest_path, manifest):
        temporary_path = manifest_path + ".tmp"
        with open(temporary_path, "w", encoding="utf-8") as manifest_file:
            json.dump(manifest, manifest_file, indent=2, sort_keys=True)
        os.replace(temporary_path, manifest_path)

    # ------------------------------------------------------------------ cropping

    def _detect_face(self, frame, crop):
        box = self.face_detection(frame, crop.BACKEND, False, 1.0)
        box = [int(value) for value in box]
        # BaseLoader returns [0, 0, frame height, frame width] when no face is found.
        if box == [0, 0, frame.shape[0], frame.shape[1]]:
            return None
        if crop.USE_LARGE_FACE_BOX:
            coef = crop.LARGE_BOX_COEF
            box = [max(0, box[0] - (coef - 1.0) / 2 * box[2]), max(0, box[1] - (coef - 1.0) / 2 * box[3]),
                   coef * box[2], coef * box[3]]
        return [int(value) for value in box]

    def _resolve_crop(self, frame, crop_mode, config_preprocess, previous_box, stats):
        """Crop box [x, y, w, h] in source pixels, or None for the full frame."""
        mode, roi = crop_mode
        if mode == "full":
            return None
        if mode == "roi":
            height, width = frame.shape[:2]
            if all(value <= 1.0 for value in roi):
                return [int(roi[0] * width), int(roi[1] * height), int(roi[2] * width), int(roi[3] * height)]
            return [int(value) for value in roi]
        stats["attempts"] += 1
        box = self._detect_face(frame, config_preprocess.CROP_FACE)
        if box is None:
            stats["failures"] += 1
            return previous_box if mode == "face" else None
        return box

    # ------------------------------------------------------------------ decoding

    @staticmethod
    def _open_capture(video_path):
        """Open a video without auto-rotation; return it with its clockwise display rotation.

        Whether OpenCV applies the container rotation depends on the build, so it is
        disabled here and applied explicitly with _rotate_frame."""
        capture = cv2.VideoCapture(video_path)
        capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
        rotation = int(round(capture.get(cv2.CAP_PROP_ORIENTATION_META))) % 360
        if rotation not in (0, *ROTATE_CODES):
            capture.release()
            raise ValueError(f"Unsupported display rotation {rotation} degrees: {video_path}")
        return capture, rotation

    @staticmethod
    def _rotate_frame(frame, rotation):
        return cv2.rotate(frame, ROTATE_CODES[rotation]) if rotation else frame

    @staticmethod
    def _box_to_source(box, rotation, source_width, source_height):
        """Map a crop box [x, y, w, h] from the displayed frame to the stored (source) frame."""
        display_width, display_height = ((source_height, source_width) if rotation in (90, 270)
                                         else (source_width, source_height))
        x, y, width, height = (int(value) for value in box)
        left, top = max(x, 0), max(y, 0)
        right, bottom = min(x + width, display_width), min(y + height, display_height)
        width, height = right - left, bottom - top
        if rotation == 90:
            return [top, source_height - right, height, width]
        if rotation == 180:
            return [source_width - right, source_height - bottom, width, height]
        if rotation == 270:
            return [source_width - bottom, left, height, width]
        return [left, top, width, height]

    def read_video(self, video_path, target_fps, config_preprocess, progress=None,
                   crop_mode=("face", None), crop_stats=None):
        crop_stats = crop_stats if crop_stats is not None else {"attempts": 0, "failures": 0}
        if self._can_use_nvdec(video_path, config_preprocess):
            try:
                if progress is not None:
                    progress.set_description(f"{os.path.basename(video_path)} | NVDEC", refresh=True)
                frames = self._read_video_nvdec(
                    video_path, target_fps, config_preprocess, progress, crop_mode, crop_stats)
                self._progress_message(f"Using FFmpeg/NVDEC: {video_path}")
                return frames
            except (OSError, RuntimeError, ValueError) as error:
                self._progress_message(f"FFmpeg/NVDEC unavailable for {video_path}: {error}")
                self._progress_message("Falling back to OpenCV decoding.")
                crop_stats["attempts"] = crop_stats["failures"] = 0
                if progress is not None:
                    progress.reset()
        if progress is not None:
            progress.set_description(f"{os.path.basename(video_path)} | OpenCV", refresh=True)
        return self._read_video_opencv(
            video_path, target_fps, config_preprocess, progress, crop_mode, crop_stats)

    def _can_use_nvdec(self, video_path, config_preprocess):
        if self.video_decoder == "opencv":
            return False
        if self.video_decoder not in {"auto", "nvdec"}:
            raise ValueError("VHRM.VIDEO_DECODER must be 'auto', 'nvdec', or 'opencv'")
        detection = config_preprocess.CROP_FACE.DETECTION
        if detection.DO_DYNAMIC_DETECTION or detection.USE_MEDIAN_FACE_BOX:
            return False
        if not str(self._face_det_device).lower().startswith("cuda") or not torch.cuda.is_available():
            return False

        ffmpeg = shutil.which(self.ffmpeg_path)
        ffprobe = shutil.which(self.ffprobe_path)
        if not ffmpeg or not ffprobe:
            return False
        try:
            codec = self._probe_codec(video_path)
            decoder = {"hevc": "hevc_cuvid", "h264": "h264_cuvid"}.get(codec)
            if decoder is None:
                return False
            if self._nvdec_decoders is None:
                self._nvdec_decoders = subprocess.run(
                    [ffmpeg, "-v", "error", "-decoders"],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout
            return decoder in self._nvdec_decoders
        except (json.JSONDecodeError, OSError, subprocess.SubprocessError):
            return False

    def _read_video_nvdec(self, video_path, target_fps, config_preprocess, progress=None,
                          crop_mode=("face", None), crop_stats=None):
        capture, rotation = self._open_capture(video_path)
        source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        success, first_frame = capture.read()
        capture.release()
        if not success or source_width <= 0 or source_height <= 0:
            raise ValueError("OpenCV could not read the reference frame for face detection")
        # Face detection runs on the displayed (upright) frame; FFmpeg crops and resizes the
        # stored frame (-noautorotate), and the rotation is applied to the decoded frames.
        first_frame = self._rotate_frame(cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB), rotation)
        transposed = rotation in (90, 270)
        decoded_width, decoded_height = (
            (config_preprocess.RESIZE.H, config_preprocess.RESIZE.W) if transposed
            else (config_preprocess.RESIZE.W, config_preprocess.RESIZE.H))

        codec = self._probe_codec(video_path)
        decoder = {"hevc": "hevc_cuvid", "h264": "h264_cuvid"}.get(codec)
        if decoder is None:
            raise ValueError(f"NVDEC does not support input codec: {codec}")

        command = [
            shutil.which(self.ffmpeg_path), "-v", "error", "-nostdin", "-noautorotate",
            "-hwaccel", "cuda", "-hwaccel_output_format", "cuda", "-c:v", decoder,
        ]
        region = self._resolve_crop(first_frame, crop_mode, config_preprocess, None, crop_stats)
        if region is not None:
            x, y, width, height = self._box_to_source(region, rotation, source_width, source_height)
            left, top, right, bottom = self._even_crop_bounds(
                x, y, width, height, source_width, source_height)
            command.extend([
                "-crop",
                f"{top}x{source_height - bottom}x{left}x{source_width - right}",
            ])
        command.extend([
            "-resize", f"{decoded_width}x{decoded_height}",
            "-i", video_path,
            "-vf", f"fps={target_fps},hwdownload,format=nv12,format=rgb24",
            "-an", "-sn", "-dn", "-f", "rawvideo", "pipe:1",
        ])

        frame_size = decoded_width * decoded_height * 3
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        frames = []
        try:
            while True:
                frame_bytes = process.stdout.read(frame_size)
                if not frame_bytes:
                    break
                while len(frame_bytes) < frame_size:
                    remainder = process.stdout.read(frame_size - len(frame_bytes))
                    if not remainder:
                        break
                    frame_bytes += remainder
                if len(frame_bytes) != frame_size:
                    raise RuntimeError("FFmpeg returned a partial video frame")
                frames.append(self._rotate_frame(np.frombuffer(frame_bytes, dtype=np.uint8).reshape(
                    decoded_height, decoded_width, 3), rotation))
                if progress is not None:
                    progress.update(1)
            stderr = process.stderr.read().decode("utf-8", errors="replace").strip()
            return_code = process.wait()
        finally:
            if process.poll() is None:
                process.kill()
        if return_code != 0:
            raise RuntimeError(stderr or f"FFmpeg exited with status {return_code}")
        if not frames:
            raise ValueError("FFmpeg returned no video frames")
        return np.asarray(frames)

    def _probe_codec(self, video_path):
        codec_cache = getattr(self, "_video_codec_cache", {})
        if video_path in codec_cache:
            return codec_cache[video_path]
        probe = subprocess.run(
            [shutil.which(self.ffprobe_path), "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=codec_name", "-of", "json", video_path],
            check=True,
            capture_output=True,
            text=True,
        )
        streams = json.loads(probe.stdout).get("streams", [])
        if not streams:
            raise ValueError("ffprobe found no video stream")
        codec = streams[0].get("codec_name")
        codec_cache[video_path] = codec
        self._video_codec_cache = codec_cache
        return codec

    @staticmethod
    def _even_crop_bounds(x, y, width, height, frame_width, frame_height):
        left = max(0, min(int(x), frame_width - 2))
        top = max(0, min(int(y), frame_height - 2))
        right = max(left + 2, min(int(x + width), frame_width))
        bottom = max(top + 2, min(int(y + height), frame_height))
        left -= left % 2
        top -= top % 2
        right -= right % 2
        bottom -= bottom % 2
        return left, top, right, bottom

    def _read_video_opencv(self, video_path, target_fps, config_preprocess, progress=None,
                           crop_mode=("face", None), crop_stats=None):
        crop_stats = crop_stats if crop_stats is not None else {"attempts": 0, "failures": 0}
        capture, rotation = self._open_capture(video_path)
        source_fps = capture.get(cv2.CAP_PROP_FPS)
        if not capture.isOpened() or source_fps <= 0 or target_fps <= 0:
            capture.release()
            raise ValueError(f"Cannot read video or determine FPS: {video_path}")

        frames = []
        source_index = 0
        next_sample_time = 0.0
        region = None
        sampled_index = 0
        detection = config_preprocess.CROP_FACE.DETECTION
        if detection.USE_MEDIAN_FACE_BOX:
            capture.release()
            raise ValueError("vHRM streaming preprocessing does not support USE_MEDIAN_FACE_BOX")
        success, frame = capture.read()
        while success:
            frame_time = source_index / source_fps
            if frame_time + (0.5 / source_fps) >= next_sample_time:
                frame = self._rotate_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), rotation)
                resolve_now = sampled_index == 0 or (
                    crop_mode[0] in ("face", "face_or_full")
                    and detection.DO_DYNAMIC_DETECTION
                    and sampled_index % detection.DYNAMIC_DETECTION_FREQUENCY == 0
                )
                if resolve_now:
                    region = self._resolve_crop(frame, crop_mode, config_preprocess, region, crop_stats)
                if region is not None:
                    x, y, width, height = (int(value) for value in region)
                    frame = frame[max(y, 0):min(y + height, frame.shape[0]),
                                  max(x, 0):min(x + width, frame.shape[1])]
                frame = cv2.resize(
                    frame,
                    (config_preprocess.RESIZE.W, config_preprocess.RESIZE.H),
                    interpolation=cv2.INTER_AREA,
                )
                frames.append(frame)
                next_sample_time += 1.0 / target_fps
                sampled_index += 1
                if progress is not None:
                    progress.update(1)
            source_index += 1
            success, frame = capture.read()
        capture.release()
        if not frames:
            raise ValueError(f"Video contains no readable frames: {video_path}")
        return np.asarray(frames)

    # ------------------------------------------------------------------ input transforms

    def transform_and_save(self, frames, config_preprocess, recording, progress_position=1):
        """Apply recording-wide normalization and save one clip at a time. Returns the clip count."""
        supported_types = {"Raw", "DiffNormalized", "Standardized"}
        unsupported = set(config_preprocess.DATA_TYPE) - supported_types
        if unsupported:
            raise ValueError(f"Unsupported data type: {sorted(unsupported)}")

        standard_mean = None
        standard_deviation = None
        if "Standardized" in config_preprocess.DATA_TYPE:
            standard_mean = float(np.mean(frames, dtype=np.float64))
            standard_deviation = float(np.std(frames, dtype=np.float64))
        difference_deviation = None
        if "DiffNormalized" in config_preprocess.DATA_TYPE:
            difference_deviation = self._difference_deviation(
                frames, config_preprocess.CHUNK_LENGTH)

        if config_preprocess.DO_CHUNK:
            chunk_length = config_preprocess.CHUNK_LENGTH
            clip_count = len(frames) // chunk_length
        else:
            chunk_length = len(frames)
            clip_count = 1

        os.makedirs(self._recording_dir(recording), exist_ok=True)
        channels = 3 * len(config_preprocess.DATA_TYPE)
        clip_range = tqdm(
            range(clip_count),
            desc=f"{recording['index']} | transform/save",
            unit="clip",
            position=progress_position,
            leave=False,
            dynamic_ncols=True,
            disable=not self.show_video_progress,
        )
        for clip_index in clip_range:
            start = clip_index * chunk_length
            end = min(start + chunk_length, len(frames))
            transformed = np.empty(
                (end - start, frames.shape[1], frames.shape[2], channels),
                dtype=self.cache_dtype,
            )
            for data_index, data_type in enumerate(config_preprocess.DATA_TYPE):
                output = transformed[..., data_index * 3:(data_index + 1) * 3]
                if data_type == "Raw":
                    output[...] = frames[start:end]
                elif data_type == "Standardized":
                    self._standardize_into(
                        frames[start:end], output, standard_mean, standard_deviation)
                else:
                    self._difference_normalize_into(
                        frames, output, start, end, difference_deviation)
            np.save(self._input_path(recording, clip_index), transformed)
        return clip_count

    @staticmethod
    def _raw_differences(frames, start, end):
        current = frames[start:end].astype(np.float32)
        following = frames[start + 1:end + 1].astype(np.float32)
        numerator = following - current
        following += current
        following += np.float32(1e-7)
        np.divide(numerator, following, out=numerator)
        return numerator

    @classmethod
    def _difference_deviation(cls, frames, block_length):
        count = 0
        mean = 0.0
        sum_squared_deviations = 0.0
        for start in range(0, len(frames) - 1, block_length):
            end = min(start + block_length, len(frames) - 1)
            differences = cls._raw_differences(frames, start, end)
            block_count = differences.size
            block_mean = float(np.mean(differences, dtype=np.float64))
            block_deviations = differences.astype(np.float64) - block_mean
            block_sum_squared = float(np.sum(np.square(block_deviations)))
            delta = block_mean - mean
            combined_count = count + block_count
            sum_squared_deviations += (
                block_sum_squared + delta * delta * count * block_count / combined_count)
            mean += delta * block_count / combined_count
            count = combined_count
        return np.sqrt(sum_squared_deviations / count) if count else 0.0

    @staticmethod
    def _standardize_into(source, output, mean, deviation):
        if not deviation or not np.isfinite(deviation):
            output.fill(0)
            return
        temporary = source.astype(np.float32)
        temporary -= np.float32(mean)
        temporary /= np.float32(deviation)
        output[...] = temporary

    @classmethod
    def _difference_normalize_into(cls, frames, output, start, end, deviation):
        output.fill(0)
        difference_end = min(end, len(frames) - 1)
        if difference_end <= start or not deviation or not np.isfinite(deviation):
            return
        differences = cls._raw_differences(frames, start, difference_end)
        differences /= np.float32(deviation)
        output[:len(differences)] = differences
