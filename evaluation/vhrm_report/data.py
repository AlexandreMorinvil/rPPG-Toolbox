"""The recordings of a report: the toolbox cache (model input, measured HR) and the ECG R-peaks.

The report covers the experiments and views of the config's TEST data, for every participant and
(by default) every segment label. The vHRM loader prepares them like any test split: recordings
already in the cache are reused, missing ones are decoded once and stay cached. Each recording is
then read back as the exact clips the checkpoint was evaluated on, its measured HR per frame
(HR_SOURCE) and the R-peaks of the ECG, on the same timeline as the frames.
"""

import json
import os
from collections import OrderedDict

import numpy as np
import pandas as pd

from config import _vhrm_selection_suffix
from dataset.data_loader import VHRMLoader as vhrm

SPLIT_ORDER = ("test", "validation", "training")
FACE_MODES = vhrm.FACE_CROP_MODES


def report_data_config(config, labels=None, participants=None):
    """TEST.DATA widened to every participant and every label (or the report's) of its experiments and views."""
    if config.TEST.DATA.DATASET != "vHRM":
        raise ValueError("The analysis report needs vHRM test data (TEST.DATA.DATASET: vHRM).")
    data = config.TEST.DATA.clone()
    data.defrost()
    data.VHRM.PARTICIPANTS = list(config.REPORT.PARTICIPANTS if participants is None else participants)
    data.VHRM.EXCLUDE_PARTICIPANTS = []
    data.VHRM.LABELS = list(config.REPORT.LABELS if labels is None else labels)
    data.VHRM.EXCLUDE_LABELS = []
    data.VHRM.REQUIRE_PPG = False
    data.VHRM.INCREMENTAL_PREPROCESS = True
    data.BEGIN, data.END = 0.0, 1.0
    data.DO_PREPROCESS = True
    data.FILE_LIST_PATH = os.path.join(os.path.dirname(config.TEST.DATA.FILE_LIST_PATH),
                                       f"{data.EXP_DATA_NAME}_report{_vhrm_selection_suffix(data)}.csv")
    data.freeze()
    return data


def participant_splits(config):
    """{participant: training | validation | test} from the run's data selections.

    A participant in several splits takes the first of training, validation, test: it was seen.
    Empty when the run did not train on vHRM (the training participants are then unknown).
    """
    if config.TOOLBOX_MODE != "train_and_test" or config.TRAIN.DATA.DATASET != "vHRM":
        return {}
    blocks = [("training", config.TRAIN.DATA)]
    if not config.TEST.USE_LAST_EPOCH:
        blocks.append(("validation", config.VALID.DATA))
    blocks.append(("test", config.TEST.DATA))
    discovered = {}
    split_of = {}
    for split, data in blocks:
        if data.DATASET != "vHRM" or not data.DATA_PATH:
            continue
        key = (data.DATA_PATH, tuple(data.VHRM.EXPERIMENTS), vhrm.source_suffix(data),
               tuple(data.VHRM.EXPERIMENT_PATHS))
        if key not in discovered:
            discovered[key] = vhrm.discover_segments(data.DATA_PATH, data.VHRM.EXPERIMENTS, key[2], key[3])
        recordings = vhrm.select_recordings(discovered[key], data.VHRM, warn=lambda message: None)
        # split_raw_data only uses its arguments (the BEGIN/END participant fractions).
        for recording in vhrm.VHRMLoader.split_raw_data(None, recordings, data.BEGIN, data.END):
            split_of.setdefault(recording["participant"], split)
    return split_of


def _repetitions(recordings):
    """1-based occurrence of each label for a participant and view, in segment order (curl 1, curl 2)."""
    counts = {}
    for recording in sorted(recordings, key=lambda r: (r["experiment"], r["participant"], r["view"],
                                                       r["segment_index"])):
        key = (recording["experiment"], recording["participant"], recording["view"], recording["label"])
        counts[key] = counts.get(key, 0) + 1
        recording["repetition"] = counts[key]


def assign_keys(recordings):
    """Short file names: <participant>_<label><repetition>, with the experiment / view when there are several."""
    several_experiments = len({r["experiment"] for r in recordings}) > 1
    several_views = len({r["view"] for r in recordings}) > 1
    for recording in recordings:
        parts = [recording["participant"], f"{recording['label']}{recording['repetition']}"]
        if several_experiments:
            parts.insert(0, recording["experiment"])
        if several_views:
            parts.append(recording["view"])
        recording["key"] = "_".join(parts)
        recording["condition"] = condition_name(recording, several_views)


def label_name(label):
    return label.replace("-", " ").capitalize()


def condition_name(recording, with_view=False):
    name = f"{label_name(recording['label'])} {recording['repetition']}"
    return f"{name} · {recording['view']}" if with_view else name


def load_recordings(config, data_config, device=None):
    """Prepare the report's recordings with the vHRM loader and describe each one."""
    loader = vhrm.VHRMLoader("report", data_config.DATA_PATH, data_config, device=device)
    rows = pd.read_csv(data_config.FILE_LIST_PATH)
    meta = {recording["index"]: recording for recording in loader.raw_data_dirs}
    manifest_path = os.path.join(data_config.CACHED_PATH, data_config.VHRM.MANIFEST_FILENAME)
    with open(manifest_path, "r", encoding="utf-8") as handle:
        entries = json.load(handle)["recordings"]
    recordings = []
    for recording_id, chunks in rows.sort_values(["recording_id", "chunk"]).groupby("recording_id", sort=True):
        source = meta[recording_id]
        entry = entries.get(recording_id, {})
        crop = entry.get("crop") or {}
        attempts = crop.get("attempts") or 0
        face_found = (100.0 * (1 - crop.get("failures", 0) / attempts)
                      if crop.get("mode") in FACE_MODES and attempts else np.nan)
        recordings.append({
            "id": recording_id,
            "experiment": source["experiment"],
            "participant": source["participant"],
            "segment_index": source["segment_index"],
            "label": source["label"],
            "view": source["view"],
            "inputs": chunks["input_files"].tolist(),
            "gt_files": chunks["gt_files"].tolist(),
            "fs": float(data_config.FS),
            "start_epoch": source["start_epoch"],
            "sync_delay": loader._sync_delay(source),
            "rpeak_file": vhrm.resolve_signal_files(source)["rpeaks"],
            "video_path": source["path"],
            "video_fingerprint": entry.get("video_fingerprint", ""),
            "face_found_pct": face_found,
        })
    _repetitions(recordings)
    assign_keys(recordings)
    return recordings


class ChunkedVideo:
    """A recording's cached clips (contiguous slices of one recording-wide transform) as one frame sequence."""

    def __init__(self, paths, keep=3):
        self.paths = list(paths)
        self.chunk_length = int(np.load(self.paths[0], mmap_mode="r").shape[0])
        self.keep = keep
        self._chunks = OrderedDict()

    def __len__(self):
        return self.chunk_length * len(self.paths)

    def _chunk(self, index):
        if index not in self._chunks:
            self._chunks[index] = np.load(self.paths[index]).astype(np.float32)
            while len(self._chunks) > self.keep:
                self._chunks.popitem(last=False)
        self._chunks.move_to_end(index)
        return self._chunks[index]

    def clip(self, start, length):
        parts = []
        position, end = start, start + length
        while position < end:
            index, offset = divmod(position, self.chunk_length)
            take = min(self.chunk_length - offset, end - position)
            parts.append(self._chunk(index)[offset:offset + take])
            position += take
        return parts[0] if len(parts) == 1 else np.concatenate(parts)


def thumbnail(video, data_types):
    """The middle frame as an 8-bit RGB image, from the most viewable input channels."""
    frame = video.clip(len(video) // 2, 1)[0]
    for preferred in ("Raw", "Standardized", "DiffNormalized"):
        if preferred in data_types:
            index = list(data_types).index(preferred)
            break
    image = frame[..., 3 * index:3 * index + 3].astype(np.float64)
    low, high = np.percentile(image, [1, 99])
    return np.clip((image - low) / max(high - low, 1e-9) * 255, 0, 255).astype(np.uint8)


def measured_hr(recording):
    """Measured HR (HR_SOURCE) at every frame, NaN where the sensor has no value."""
    return np.concatenate([np.load(path)[:, 0] for path in recording["gt_files"]]).astype(np.float64)


def rpeak_times(recording, duration, margin):
    """ECG R-peak times on the frame timeline (seconds since the segment start), or None without an ECG."""
    path = recording.get("rpeak_file")
    if not path:
        return None
    epochs = vhrm.read_timed_csv(path)["epoch"].to_numpy(dtype=np.float64)
    # Frame i samples the signals at start_epoch + i / FS - sync_delay (VHRMLoader.build_signals).
    times = np.sort(epochs - recording["start_epoch"] + recording["sync_delay"])
    return times[(times >= -margin) & (times <= duration + margin)]
