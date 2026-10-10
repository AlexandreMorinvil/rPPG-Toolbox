"""Compact prediction outputs for later visualization (launcher Results tab, highlight plots).

Signals are cheap (one float per frame); images are what fill disks. Runs therefore save
per-frame predictions/labels (and the vHRM measured ground truth) and the launcher draws
plots on demand. During training only the best-epoch validation set and a few "probe"
recordings every N epochs are kept.

*_outputs.pickle keys: predictions/labels {recording: {chunk: tensor}}, label_type, fs
(original toolbox format) plus split, meta, and for vHRM gt {recording: float32[N, 4]}
with gt_columns.
"""

import os
import pickle
from collections import defaultdict

import numpy as np

GT_COLUMNS = ("heart_rate_bpm", "HRV", "respiration_rate_bpm", "ppg_valid")
VALID_DIR_NAME = "saved_valid_outputs"
BEST_VALID_FILE = "best_valid_outputs.pickle"
PROBE_HISTORY_FILE = "probe_history.npz"


def _to_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def parse_recording_id(recording_id):
    parts = str(recording_id).split("--")
    if len(parts) == 5:
        return dict(zip(("experiment", "participant", "segment_index", "movement", "view"), parts))
    if len(parts) == 4:
        return dict(zip(("participant", "segment_index", "movement", "view"), parts), experiment="")
    return {"experiment": "", "participant": str(recording_id), "segment_index": "", "movement": "", "view": ""}


def ground_truth_files(file_list_path):
    """{(recording_id, chunk): gt .npy path} from a vHRM file list."""
    import pandas as pd

    file_list = pd.read_csv(file_list_path)
    if "gt_files" not in file_list:
        raise ValueError(f"{file_list_path} has no gt_files column; re-run vHRM preprocessing.")
    return {(row.recording_id, int(row.chunk)): row.gt_files for row in file_list.itertuples()}


def load_vhrm_ground_truth(file_list_path, predictions):
    """Per-recording measured ground truth, concatenated in the same chunk order as predictions."""
    files = ground_truth_files(file_list_path)
    result = {}
    for recording_id, chunks in predictions.items():
        arrays = []
        for chunk_id in sorted(chunks, key=int):
            path = files.get((recording_id, int(chunk_id)))
            if path is None or not os.path.isfile(path):
                arrays = None
                break
            arrays.append(np.load(path).astype(np.float32))
        if arrays:
            result[recording_id] = np.concatenate(arrays, axis=0)
    return result


def prediction_is_diff(config):
    """Whether the model output is a first-difference signal (DiffNormalized training labels)."""
    if config.TOOLBOX_MODE == "train_and_test":
        return config.TRAIN.DATA.PREPROCESS.LABEL_TYPE == "DiffNormalized"
    if config.TEST.DATA.DATASET == "vHRM":
        return bool(config.TEST.DATA.VHRM.PREDICTION_IS_DIFF)
    return config.TEST.DATA.PREPROCESS.LABEL_TYPE == "DiffNormalized"


def run_meta(config, data_config, split, **extra):
    meta = {
        "split": split,
        "mode": config.TOOLBOX_MODE,
        "model": config.MODEL.NAME,
        "dataset": data_config.DATASET,
        "train_dataset": config.TRAIN.DATA.DATASET if config.TOOLBOX_MODE == "train_and_test" else "",
        "checkpoint": config.INFERENCE.MODEL_PATH if config.TOOLBOX_MODE == "only_test" else "",
        "label_is_diff": data_config.PREPROCESS.LABEL_TYPE == "DiffNormalized",
        "evaluation_method": config.INFERENCE.EVALUATION_METHOD,
        "evaluation_window_s": config.INFERENCE.EVALUATION_WINDOW.WINDOW_SIZE
        if config.INFERENCE.EVALUATION_WINDOW.USE_SMALLER_WINDOW else 0,
    }
    meta.update(extra)
    return meta


def write_outputs_pickle(path, predictions, labels, label_type, fs, split="test", meta=None, gt=None):
    data = {"predictions": predictions, "labels": labels, "label_type": label_type, "fs": fs,
            "split": split, "meta": meta or {}}
    if gt:
        data["gt"] = gt
        data["gt_columns"] = list(GT_COLUMNS)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)
    print("Saving outputs to:", path)
    return data


def vhrm_ground_truth_or_none(data_config, predictions):
    if data_config.DATASET != "vHRM":
        return None
    try:
        return load_vhrm_ground_truth(data_config.FILE_LIST_PATH, predictions)
    except Exception as error:  # never fail a run because of visualization extras
        print(f"[outputs] vHRM ground truth not attached: {error}")
        return None


def select_probes(recording_ids, count):
    """Deterministic, stratified pick: one recording per (movement, view), rotating participants."""
    recording_ids = sorted(str(r) for r in recording_ids)
    if count <= 0 or not recording_ids:
        return []
    groups = defaultdict(list)
    for recording_id in recording_ids:
        meta = parse_recording_id(recording_id)
        groups[(meta["movement"], meta["view"])].append(recording_id)
    used_participants = defaultdict(int)
    chosen = []
    while len(chosen) < min(count, len(recording_ids)):
        progressed = False
        for key in sorted(groups):
            remaining = [r for r in groups[key] if r not in chosen]
            if not remaining or len(chosen) >= count:
                continue
            pick = min(remaining, key=lambda r: (used_participants[parse_recording_id(r)["participant"]], r))
            used_participants[parse_recording_id(pick)["participant"]] += 1
            chosen.append(pick)
            progressed = True
        if not progressed:
            break
    return chosen


def _concatenate(chunks):
    return np.concatenate([np.reshape(_to_numpy(chunks[c]), (-1,)) for c in sorted(chunks, key=int)])


def load_probe_history(path):
    if not os.path.isfile(path):
        return None
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def update_probe_history(path, epoch, probes):
    """Append one epoch of probe predictions; epochs >= epoch from an interrupted attempt are dropped.

    probes: {recording_id: (prediction[N], label[N], gt[N, 4] or None)}.
    """
    history = load_probe_history(path) or {}
    epochs = history.get("epochs", np.zeros(0, dtype=np.int32))
    keep = epochs < epoch
    if not keep.any():
        history = {}  # a new run (or a restart from epoch 0) starts a new history
    ids = [str(i) for i in history.get("ids", [])] or sorted(probes)
    out = {"ids": np.asarray(ids), "epochs": np.concatenate([epochs[keep], [epoch]]).astype(np.int32)}
    for index, recording_id in enumerate(ids):
        if recording_id not in probes:
            continue
        prediction, label, gt = probes[recording_id]
        previous = history.get(f"pred_{index}")
        if previous is not None:
            previous = previous[keep]
            length = min(previous.shape[1], len(prediction))
            stacked = np.concatenate([previous[:, :length], prediction[None, :length]], axis=0)
        else:
            # A probe first seen now gets NaN rows for the epochs already in the history.
            stacked = np.full((int(keep.sum()) + 1, len(prediction)), np.nan, dtype=np.float32)
            stacked[-1] = prediction
        out[f"pred_{index}"] = stacked.astype(np.float16)
        out[f"label_{index}"] = np.asarray(label[:stacked.shape[1]], dtype=np.float32)
        if gt is not None:
            out[f"gt_{index}"] = np.asarray(gt[:stacked.shape[1]], dtype=np.float32)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "wb") as handle:
        np.savez_compressed(handle, **out)
    os.replace(temporary, path)
    return out


class ValidationCapture:
    """Collects validation predictions; commit() writes the best-epoch set and probe history."""

    def __init__(self, config):
        self.config = config
        visualize = config.VISUALIZE
        self.keep_best = bool(visualize.VALID_BEST)
        self.every = int(visualize.PROBE_EVERY_N_EPOCHS)
        self.probe_count = int(visualize.PROBE_COUNT)
        self.explicit_probes = [str(r) for r in visualize.PROBE_RECORDINGS]
        self.total_epochs = int(config.TRAIN.EPOCHS)
        self.output_dir = os.path.join(config.LOG.PATH, config.TRAIN.DATA.EXP_DATA_NAME, VALID_DIR_NAME)
        self.predictions = {}
        self.labels = {}

    @property
    def enabled(self):
        return self.keep_best or self.every > 0

    def probe_epoch(self, epoch):
        return self.every > 0 and ((epoch + 1) % self.every == 0 or epoch == 0 or epoch == self.total_epochs - 1)

    def add_batch(self, recording_ids, chunk_ids, prediction, label, frames=None):
        if not self.enabled:
            return
        size = len(recording_ids)
        prediction = _to_numpy(prediction).astype(np.float32)
        label = _to_numpy(label).astype(np.float32)
        if label.ndim >= 3:
            label = label[..., 0]
        if size == 0 or prediction.size % size or label.size % size:
            return  # cannot split per sample
        if frames is not None and (prediction.size != size * frames or label.size != size * frames):
            return  # trimmed batch (EfficientPhys/TSCAN base_len): samples are not aligned
        prediction = prediction.reshape(size, -1)
        label = label.reshape(size, -1)
        for index in range(size):
            recording_id = str(recording_ids[index])
            chunk_id = int(chunk_ids[index])
            self.predictions.setdefault(recording_id, {})[chunk_id] = prediction[index]
            self.labels.setdefault(recording_id, {})[chunk_id] = label[index]

    def _probe_ids(self, epoch):
        history = load_probe_history(os.path.join(self.output_dir, PROBE_HISTORY_FILE))
        if history is not None and np.any(history.get("epochs", []) < epoch):
            return [str(i) for i in history["ids"]]  # resumed run: keep the same probes
        if self.explicit_probes:
            return [r for r in self.explicit_probes if r in self.predictions]
        return select_probes(self.predictions, self.probe_count)

    def commit(self, epoch, is_best):
        try:
            if not self.predictions:
                return
            data_config = self.config.VALID.DATA
            need_probes = self.probe_epoch(epoch)
            need_best = self.keep_best and is_best
            if not (need_best or need_probes):
                return
            gt = vhrm_ground_truth_or_none(data_config, self.predictions)
            if need_best:
                write_outputs_pickle(
                    os.path.join(self.output_dir, BEST_VALID_FILE), self.predictions, self.labels,
                    data_config.PREPROCESS.LABEL_TYPE, data_config.FS, split="valid",
                    meta=run_meta(self.config, data_config, "valid", epoch=int(epoch),
                                  prediction_is_diff=self.config.TRAIN.DATA.PREPROCESS.LABEL_TYPE == "DiffNormalized"),
                    gt=gt)
            if need_probes:
                probes = {}
                for recording_id in self._probe_ids(epoch):
                    if recording_id not in self.predictions:
                        continue
                    prediction = _concatenate(self.predictions[recording_id])
                    label = _concatenate(self.labels[recording_id])
                    probes[recording_id] = (prediction, label, (gt or {}).get(recording_id))
                if probes:
                    update_probe_history(os.path.join(self.output_dir, PROBE_HISTORY_FILE), int(epoch), probes)
        except Exception as error:  # visualization extras must never stop training
            print(f"[outputs] Validation outputs not saved for epoch {epoch}: {error}")
        finally:
            self.predictions = {}
            self.labels = {}
