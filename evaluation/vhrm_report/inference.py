"""The checkpoint's network, and inference over whole recordings with overlapping clips.

The network is built by the model's own trainer in only_test mode, so its constructor arguments come
from the config exactly as in the toolbox's test. FORWARD mirrors each trainer's test-time call: input
layout, appended last frame, frame-wise models, output shape.

Clips start every `stride` frames and are merged with a phase-aligned Hann overlap-add
(exploration/deep_learning_test/HR_ESTIMATION_METHODOLOGY.md): each new clip is sign-flipped if it
anti-correlates with what is already reconstructed in the shared overlap, then Hann-weighted so clip
edges count less.

An unsupervised method (POS, CHROM, ...) has no clips: it runs once over the whole recording, as in the
toolbox's unsupervised run (method_predictor).
"""

import importlib

import numpy as np
import torch


def build_trainer(config):
    """The model's trainer, built as for evaluating a checkpoint (no data loaders needed)."""
    name = config.MODEL.NAME
    if name == "BigSmall":
        raise ValueError("The analysis report does not support BigSmall (multi-task input and output).")
    test_config = config.clone()
    test_config.defrost()
    test_config.TOOLBOX_MODE = "only_test"
    test_config.freeze()
    try:
        module = importlib.import_module(f"neural_methods.trainer.{name}Trainer")
    except ImportError as error:
        raise ValueError(f"Cannot load the {name} trainer: {error}") from error
    return getattr(module, f"{name}Trainer")(test_config, {})


def load_weights(trainer, checkpoint):
    """Load an epoch checkpoint (or released weights) into the trainer's network, strictly."""
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    for container in ("state_dict", "model_state_dict"):
        if isinstance(state, dict) and isinstance(state.get(container), dict):
            state = state[container]
    model = trainer.model
    wrapped = isinstance(model, torch.nn.DataParallel)
    prefixed = bool(state) and all(key.startswith("module.") for key in state)
    if wrapped and not prefixed:
        state = {f"module.{key}": value for key, value in state.items()}
    elif prefixed and not wrapped:
        state = {key.removeprefix("module."): value for key, value in state.items()}
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError:
        if trainer.config.MODEL.NAME != "FactorizePhys":
            raise
        # The FactorizePhys trainer loads non-strictly: FSAM weights may be absent when it is off.
        result = model.load_state_dict(state, strict=False)
        print(f"[report] FactorizePhys weights loaded non-strictly: {len(result.missing_keys)} missing, "
              f"{len(result.unexpected_keys)} unexpected keys.")
    model.eval()
    return trainer


def _first(output):
    return output[0] if isinstance(output, (tuple, list)) else output


def _append_last_frame(data, trainer):
    """FactorizePhys and iBVPNet take one frame more than the clip (NCDHW, along D)."""
    return torch.cat((data, data[:, :, -1:].repeat(1, 1, max(trainer.num_of_gpu, 1), 1, 1)), 2)


def _frames(data, trainer):
    """Frame-wise models (TS-CAN, EfficientPhys, DeepPhys) see the clips as one batch of frames (NDCHW)."""
    if data.ndim != 5:
        raise ValueError("Frame-wise models need DATA_FORMAT NDCHW.")
    count, depth = data.shape[:2]
    base = max(getattr(trainer, "base_len", 1), 1)
    if depth % base:
        raise ValueError(f"The clip length ({depth}) must be a multiple of {base} (FRAME_DEPTH x GPUs).")
    return data.reshape(count * depth, *data.shape[2:]), count


def _tscan(trainer, data):
    frames, count = _frames(data, trainer)
    return trainer.model(frames).reshape(count, -1)


def _efficientphys(trainer, data):
    frames, count = _frames(data, trainer)
    frames = torch.cat((frames, frames[-1:].repeat(max(trainer.num_of_gpu, 1), 1, 1, 1)), 0)
    return trainer.model(frames).reshape(count, -1)


def _deepphys(trainer, data):
    frames, count = _frames(data, trainer)
    size = trainer.config.INFERENCE.FRAME_BATCH_SIZE
    output = (torch.cat([trainer.model(batch) for batch in torch.split(frames, size)]) if size > 0
              else trainer.model(frames))
    return output.reshape(count, -1)


# MODEL.NAME -> fn(trainer, data in the config's DATA_FORMAT) -> (clips, frames) prediction
FORWARD = {
    "Physnet": lambda trainer, data: _first(trainer.model(data)),
    "PhysFormer": lambda trainer, data: _first(trainer.model(data, 2.0)),
    "PhysMamba": lambda trainer, data: _first(trainer.model(data)),
    "RhythmFormer": lambda trainer, data: _first(trainer.model(data)),
    "iBVPNet": lambda trainer, data: _first(trainer.model(_append_last_frame(data, trainer))),
    "FactorizePhys": lambda trainer, data: _first(trainer.model(_append_last_frame(data, trainer))),
    "Tscan": _tscan,
    "EfficientPhys": _efficientphys,
    "DeepPhys": _deepphys,
}


def to_layout(clips, data_format):
    """(clips, T, H, W, C) cache layout -> the DATA_FORMAT of the config, as BaseLoader.__getitem__."""
    if data_format == "NDCHW":
        return clips.permute(0, 1, 4, 2, 3)
    if data_format == "NCDHW":
        return clips.permute(0, 4, 1, 2, 3)
    if data_format == "NDHWC":
        return clips
    raise ValueError(f"Unsupported DATA_FORMAT: {data_format}")


class Network:
    """A trainer with loaded weights, called on batches of clips in the cache layout."""

    def __init__(self, trainer, data_format):
        name = trainer.config.MODEL.NAME
        if name not in FORWARD:
            raise ValueError(f"The analysis report does not support {name}.")
        self.trainer = trainer
        self.forward = FORWARD[name]
        self.data_format = data_format
        self.device = next(trainer.model.parameters()).device

    def __call__(self, clips):
        data = to_layout(torch.from_numpy(np.ascontiguousarray(clips, dtype=np.float32)).to(self.device),
                         self.data_format).contiguous()
        with torch.inference_mode():
            output = self.forward(self.trainer, data)
        return output.reshape(len(clips), -1).double().cpu().numpy()


def clip_starts(frame_count, clip_length, stride):
    if frame_count < clip_length:
        raise ValueError(f"Recording has {frame_count} frames, fewer than one {clip_length}-frame clip")
    starts = list(range(0, frame_count - clip_length + 1, stride))
    if starts[-1] != frame_count - clip_length:
        starts.append(frame_count - clip_length)  # anchor the last clip to the end
    return starts


def predict_bvp(network, video, clip_length, stride, batch_size, zscore=True, window_floor=1e-3):
    """Full-length predicted pulse signal (one value per frame) and the number of sign flips.

    video: anything with len() and clip(start, length) -> (length, H, W, C) array.
    """
    starts = clip_starts(len(video), clip_length, stride)
    window = np.maximum(np.hanning(clip_length), window_floor)
    summed = np.zeros(len(video))
    weight = np.zeros(len(video))
    flips = 0
    for batch_start in range(0, len(starts), batch_size):
        batch = starts[batch_start:batch_start + batch_size]
        output = network(np.stack([video.clip(start, clip_length) for start in batch]))
        if output.shape[1] < clip_length:
            raise ValueError(f"The network returned {output.shape[1]} samples for a {clip_length}-frame clip")
        output = output[:, :clip_length]
        if zscore:
            output = (output - output.mean(axis=1, keepdims=True)) / np.maximum(output.std(axis=1, keepdims=True), 1e-12)
        for start, clip in zip(batch, output):
            span = slice(start, start + clip_length)
            overlap = weight[span] > 0
            if overlap.sum() >= 3:
                reference = summed[span][overlap] / weight[span][overlap]
                reference = reference - reference.mean()
                candidate = clip[overlap] - clip[overlap].mean()
                if np.dot(reference, candidate) < 0:
                    clip, flips = -clip, flips + 1
            summed[span] += clip * window
            weight[span] += window
    return summed / weight, flips


def clip_predictor(network, clip_length, stride, batch_size, inference):
    """predict(video) -> (pulse signal, sign flips): the network over overlapping clips (predict_bvp)."""
    return lambda video: predict_bvp(network, video, clip_length, stride, batch_size, inference["zscore_clips"],
                                     inference["overlap_window_floor"])


# Unsupervised methods whose output falls when the blood volume rises: negated, so that their peaks are pulse peaks
# as in the PPG labels (VHRMLoader.pos_pseudo_ppg flips POS for the same reason).
FALLING_METHODS = {"POS"}


def method_sign(method):
    return -1.0 if method in FALLING_METHODS else 1.0


def method_predictor(method, fs, masked):
    """predict(video) -> (pulse signal, 0): an unsupervised method on the whole recording's first 3 channels.

    masked: average only the skin pixels of each frame (UV maps), as unsupervised_predict does.
    """
    from unsupervised_methods.unsupervised_predictor import METHODS, unsupervised_bvp
    if method not in METHODS:
        raise ValueError(f"Unknown unsupervised method {method!r} (one of {', '.join(METHODS)}).")
    sign = method_sign(method)

    def predict(video):
        frames = video.clip(0, len(video))[..., :3]
        return sign * np.asarray(unsupervised_bvp(method, frames, fs, masked), dtype=np.float64).reshape(-1), 0
    return predict
