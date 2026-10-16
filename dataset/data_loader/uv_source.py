"""UV input mode: frames from the skin-UV HDF5 files written by vHRM Studio.

vHRM Studio maps the visible skin of every video frame onto the SMPL-X surface and stores it as a 1024x1024 texture:
the standard SMPL-X layout (`uv_skin_rgb`) and the compact "packed" layout (`packed_uv_skin_rgb`, 22 semantic charts
- head, torso, arms, legs, hands, fingers, feet - repacked upright into the square). A texel is 0 where no skin was
seen. Here a texture becomes a toolbox frame:

* masked area pooling to S x S: every output pixel is the mean of the *skin* texels of its block (the selected charts
  only), so a 128 px frame averages ~64 skin samples per pixel instead of point-sampling the video once;
* an output pixel is valid when at least MIN_COVERAGE of the block's region texels are skin; invalid pixels are 0;
* rows the studio could not process (frame_status != 1) are kept as all-invalid frames, so the timing is unchanged;
* optional resampling to a target frame rate from `source_timestamp_s`;
* `fill_invalid` replaces invalid pixels for the neural methods (unsupervised methods keep the zeros and average
  over valid pixels only, see unsupervised_methods.utils.masked_spatial_mean).

Only numpy, h5py and OpenCV: the module is imported by the loaders, the tools and the tests.
"""
import json
import os
from pathlib import Path

import cv2
import numpy as np

H5_SUFFIX = "_skin_uv.h5"
MANIFEST_NAME = "vhrm_manifest.json"
LAYOUT_PREFIXES = {"packed": "packed_uv_", "standard": "uv_"}
FILL_MODES = ("none", "temporal_mean", "hold")
# Loaders whose sources the studio cannot convert (.mat files) or whose read_video has another signature.
UNSUPPORTED_DATASETS = ("SCAMPS", "MMPD", "BP4DPlus", "BP4DPlusBigSmall")

_manifest_cache = {}


# ---------------------------------------------------------------- settings

def uv_settings(uv_node):
    """Plain dict of the UV settings that change the frames (cache identities, fingerprints)."""
    return {
        "layout": str(uv_node.LAYOUT).lower(),
        "charts": sorted(str(name).strip().lower() for name in uv_node.CHARTS if str(name).strip()),
        "exclude_charts": sorted(str(name).strip().lower() for name in uv_node.EXCLUDE_CHARTS if str(name).strip()),
        "min_coverage": float(uv_node.MIN_COVERAGE),
        "fill": str(uv_node.FILL).lower(),
        "min_pixel_valid_fraction": float(uv_node.MIN_PIXEL_VALID_FRACTION),
        "resample_to_fs": bool(uv_node.RESAMPLE_TO_FS),
    }


def validate_settings(uv_node, preprocess, dataset=""):
    """Raise on settings the UV mode cannot honour."""
    settings = uv_settings(uv_node)
    if settings["layout"] not in LAYOUT_PREFIXES:
        raise ValueError(f"UV.LAYOUT must be one of {tuple(LAYOUT_PREFIXES)}, got {uv_node.LAYOUT!r}")
    if settings["fill"] not in FILL_MODES:
        raise ValueError(f"UV.FILL must be one of {FILL_MODES}, got {uv_node.FILL!r}")
    if settings["layout"] == "standard" and (settings["charts"] or settings["exclude_charts"]):
        raise ValueError("UV.CHARTS / UV.EXCLUDE_CHARTS need LAYOUT: packed (the standard layout has no charts)")
    if not 0.0 <= settings["min_coverage"] <= 1.0:
        raise ValueError("UV.MIN_COVERAGE must be within [0, 1]")
    if preprocess.RESIZE.H != preprocess.RESIZE.W:
        raise ValueError("UV mode needs a square frame (RESIZE.H == RESIZE.W): the UV texture is square")
    if any(aug != "None" for aug in preprocess.DATA_AUG):
        raise ValueError("UV mode does not support DATA_AUG (the augmented .npy videos are not UV maps)")
    if dataset in UNSUPPORTED_DATASETS:
        raise ValueError(f"UV mode is not supported for {dataset}")


# ---------------------------------------------------------------- locating the HDF5 file

def _load_manifest(directory):
    """{source relative path (lower case, posix): output relative path} of the vhrm_manifest.json in directory."""
    path = os.path.join(directory, MANIFEST_NAME)
    if path not in _manifest_cache:
        mapping = {}
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    items = json.load(handle).get("items", [])
                mapping = {str(item["source"]).lower(): str(item["output"]) for item in items
                           if item.get("source") and item.get("output")}
            except (OSError, ValueError, KeyError, TypeError, AttributeError):
                mapping = {}
        _manifest_cache[path] = mapping
    return _manifest_cache[path]


def candidate_paths(source_path, suffix=H5_SUFFIX):
    """HDF5 names vHRM Studio gives a source (vhrm_studio.dataset.plan_outputs): video `dir/x.ext` ->
    `dir/x_skin_uv.h5` (or `dir/x_ext_skin_uv.h5` when two sources share a stem); image folder `dir/F/` ->
    `dir/F_skin_uv.h5`."""
    source = Path(str(source_path).rstrip("/\\"))
    if source.suffix and not source.is_dir():
        return [source.with_name(source.stem + suffix),
                source.with_name(f"{source.stem}_{source.suffix.lstrip('.')}{suffix}")]
    return [source.with_name(source.name + suffix), source / (source.name + suffix)]


def resolve_uv_path(source_path, data_root=None, suffix=H5_SUFFIX):
    """HDF5 file converted from source_path (a video file or an image folder a loader would read).

    The vhrm_manifest.json of the converted dataset is used first (searched from the source folder up to
    data_root), then the naming rule. The source itself does not need to exist (mirror conversions)."""
    source = Path(str(source_path).rstrip("/\\")).absolute()
    if source.name.lower().endswith(suffix.lower()) and source.is_file():
        return str(source)
    stop = Path(data_root).absolute() if data_root else None
    folder = source.parent
    while True:
        mapping = _load_manifest(str(folder))
        if mapping:
            relative = os.path.relpath(source, folder).replace(os.sep, "/").lower()
            output = mapping.get(relative)
            if output and os.path.isfile(folder / output):
                return str(folder / output)
        if folder == stop or folder.parent == folder:
            break
        folder = folder.parent
    candidates = candidate_paths(source, suffix)
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    raise FileNotFoundError(
        f"No vHRM Studio file for {source_path}: looked for {', '.join(str(c) for c in candidates)} "
        f"and in {MANIFEST_NAME}. Convert the dataset with vHRM Studio (mirror mode) and point DATA_PATH to it.")


# ---------------------------------------------------------------- reading

def region_mask(h5, layout="packed", charts=(), exclude_charts=()):
    """(H, W) bool: texels of the body (restricted to the selected charts for the packed layout)."""
    prefix = LAYOUT_PREFIXES[layout]
    body = np.asarray(h5[prefix + "body_mask"][()], dtype=bool)
    if not charts and not exclude_charts:
        return body
    names = [str(name).lower() for name in h5.attrs["packed_uv_chart_names"]]
    unknown = sorted((set(charts) | set(exclude_charts)) - set(names))
    if unknown:
        raise ValueError(f"Unknown UV charts {unknown}; the file has {names}")
    texel_chart = np.asarray(h5["packed_uv_texel_chart"][()])
    keep = np.ones(len(names), dtype=bool) if not charts else np.isin(names, list(charts))
    keep &= ~np.isin(names, list(exclude_charts))
    mask = np.zeros(body.size, dtype=bool)
    mask[np.flatnonzero(body)] = keep[texel_chart]
    return mask.reshape(body.shape)


def _area_mean(image, size):
    """Mean of each output pixel's area (exact block mean for integer factors, area-weighted otherwise)."""
    return cv2.resize(np.ascontiguousarray(image, dtype=np.float32), (size, size), interpolation=cv2.INTER_AREA)


def pool_masked(rgb, weight, size, region_count=None):
    """Mean RGB of the weighted texels per output pixel, and the weighted fraction of the region texels.

    rgb (H, W, 3), weight (H, W) bool -> (size, size, 3) float32, (size, size) float32. region_count is the
    pooled region mask (pool_region, computed once); without it the fraction is relative to the whole block."""
    mask = np.ascontiguousarray(weight, dtype=np.uint8)
    if rgb.dtype == np.uint8:
        weighted = cv2.bitwise_and(rgb, rgb, mask=mask)
    else:
        weighted = rgb * mask[..., None]
    numerator = _area_mean(weighted, size)
    count = _area_mean(mask, size)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(count[..., None] > 0, numerator / count[..., None], 0.0).astype(np.float32)
        if region_count is None:
            coverage = count
        else:
            coverage = np.where(region_count > 0, count / region_count, 0.0).astype(np.float32)
    return mean, coverage


def pool_region(region, size):
    """Pooled region mask (fraction of each output pixel's area that belongs to the region)."""
    return _area_mean(region, size)


def skin_texels(rgb):
    """Texels holding a colour (vHRM Studio writes 0 where no skin was seen)."""
    return (rgb[..., 0] | rgb[..., 1] | rgb[..., 2]) != 0


def resample_rows(timestamps, target_fps):
    """Row index of every output frame: nearest stored row to k / target_fps (k = 0, 1, ...)."""
    timestamps = np.asarray(timestamps, dtype=np.float64)
    if len(timestamps) < 2:
        return np.zeros(len(timestamps), dtype=np.int64)
    timestamps = timestamps - timestamps[0]
    count = int(np.floor(timestamps[-1] * target_fps + 1e-6)) + 1
    times = np.arange(count) / float(target_fps)
    right = np.clip(np.searchsorted(timestamps, times), 1, len(timestamps) - 1)
    left = right - 1
    return np.where(times - timestamps[left] <= timestamps[right] - times, left, right)


def _unpack_mask(packed):
    return np.unpackbits(np.asarray(packed), axis=-1).astype(bool)


def read_uv_frames(h5_path, settings, size, target_fps=None, progress=None):
    """Pooled UV frames of one file.

    settings: uv_settings(...) dict. Returns frames (T, size, size, 3) float32 (0 = invalid), valid (T, size, size)
    bool and an info dict (fps, stride, rows, coverage)."""
    import h5py

    layout = settings["layout"]
    prefix = LAYOUT_PREFIXES[layout]
    with h5py.File(h5_path, "r") as h5:
        if prefix + "skin_rgb" not in h5:
            raise ValueError(f"{h5_path} has no {prefix}skin_rgb (layout {layout!r}); re-convert it with vHRM Studio")
        rgb_dataset = h5[prefix + "skin_rgb"]
        mask_dataset = h5.get(prefix + "skin_mask_packed")
        texture_size = rgb_dataset.shape[1]
        if size > texture_size:
            raise ValueError(f"UV size {size} is larger than the stored texture ({texture_size})")
        status = np.asarray(h5["frame_status"][()]) if "frame_status" in h5 else np.ones(rgb_dataset.shape[0], np.uint8)
        timestamps = np.asarray(h5["source_timestamp_s"][()]) if "source_timestamp_s" in h5 else None
        effective_fps = float(h5.attrs.get("effective_fps", 0.0) or 0.0)
        frame_stride = int(h5.attrs.get("frame_stride", 1) or 1)
        if target_fps:
            if timestamps is None:
                if not effective_fps:
                    raise ValueError(f"{h5_path} has neither timestamps nor effective_fps")
                timestamps = np.arange(len(status)) / effective_fps
            rows = resample_rows(timestamps, target_fps)
        else:
            rows = np.arange(len(status))

        region = region_mask(h5, layout, settings["charts"], settings["exclude_charts"])
        region_count = pool_region(region, size)
        frames = np.zeros((len(rows), size, size, 3), dtype=np.float32)
        valid = np.zeros((len(rows), size, size), dtype=bool)
        cache_row, cache = None, None
        for index, row in enumerate(rows):
            if row != cache_row:
                cache_row = row
                cache = None
                if status[row] == 1:
                    rgb = rgb_dataset[row]
                    skin = _unpack_mask(mask_dataset[row]) if mask_dataset is not None else skin_texels(rgb)
                    mean, coverage = pool_masked(rgb, skin & region, size, region_count)
                    ok = coverage >= max(settings["min_coverage"], 1e-9)
                    mean[~ok] = 0.0
                    cache = (mean, ok)
            if cache is not None:
                frames[index], valid[index] = cache
            if progress is not None:
                progress.update(1)

    if settings["min_pixel_valid_fraction"] > 0 and len(valid):
        rare = valid.mean(axis=0) < settings["min_pixel_valid_fraction"]
        valid[:, rare] = False
        frames[:, rare] = 0.0
    valid_fraction = valid.reshape(len(valid), -1).mean(axis=1) if len(valid) else np.zeros(0)
    info = {
        "path": str(h5_path),
        "layout": layout,
        "rows": int(len(status)),
        "frames": int(len(rows)),
        "bad_rows": int(np.sum(status != 1)),
        "effective_fps": effective_fps,
        "frame_stride": frame_stride,
        "mean_valid_fraction": float(valid_fraction.mean()) if len(valid_fraction) else 0.0,
        "empty_frames": int(np.sum(valid_fraction == 0)),
    }
    return frames, valid, info


def fill_invalid(frames, valid, mode):
    """Replace invalid pixels in place (frames (T, H, W, 3), valid (T, H, W)).

    none: keep 0. temporal_mean: the pixel's mean over its valid frames. hold: the last valid value (the first
    valid value before it was ever seen). Pixels never valid stay 0."""
    if mode == "none" or not len(frames):
        return frames
    if mode not in FILL_MODES:
        raise ValueError(f"Unknown UV fill mode {mode!r}")
    invalid = ~valid
    if not invalid.any():
        return frames
    if mode == "temporal_mean":
        counts = valid.sum(axis=0, dtype=np.float32)
        sums = frames.sum(axis=0, dtype=np.float64)  # invalid pixels are 0
        with np.errstate(invalid="ignore", divide="ignore"):
            means = np.where(counts[..., None] > 0, sums / counts[..., None], 0.0).astype(np.float32)
        time_index, rows, cols = np.nonzero(invalid)
        frames[time_index, rows, cols] = means[rows, cols]
        return frames
    steps = np.arange(len(frames), dtype=np.int32)[:, None, None]
    last = np.maximum.accumulate(np.where(valid, steps, -1), axis=0)
    first = np.where(valid, steps, len(frames))[::-1]
    first = np.minimum.accumulate(first, axis=0)[::-1]
    source = np.where(last >= 0, last, first)
    seen = source < len(frames)
    rows, cols = np.indices(frames.shape[1:3])
    filled = frames[np.where(seen, source, 0), rows, cols]
    filled[~seen] = 0.0
    frames[...] = filled
    return frames


def load_uv_video(source_path, uv_node, size, data_root=None, target_fps=None, progress=None):
    """Resolve, read and fill: the frames a loader gets instead of the decoded video."""
    settings = uv_settings(uv_node)
    h5_path = resolve_uv_path(source_path, data_root, uv_node.H5_SUFFIX or H5_SUFFIX)
    frames, valid, info = read_uv_frames(h5_path, settings, size, target_fps=target_fps, progress=progress)
    fill_invalid(frames, valid, settings["fill"])
    return frames, info


class UVVideoReader:
    """Drop-in for a loader's read_video(path, ...) in UV mode (a plain object, so loaders stay picklable for the
    spawned preprocessing workers). Extra arguments of the loader-specific signature are ignored."""

    def __init__(self, uv_node, size, data_root=None, target_fps=None, dataset_name=""):
        self.uv_node = uv_node
        self.size = int(size)
        self.data_root = data_root
        self.target_fps = target_fps
        self.dataset_name = dataset_name

    def __call__(self, source_path, *args, **kwargs):
        frames, info = load_uv_video(source_path, self.uv_node, self.size, self.data_root, self.target_fps)
        if not self.target_fps and info["frame_stride"] != 1:
            print(f"WARNING: {self.dataset_name} UV: {info['path']} keeps every {info['frame_stride']}th source "
                  f"frame ({info['effective_fps']:.2f} fps). Loaders that do not resample their labels to the frame "
                  f"count need UV.RESAMPLE_TO_FS: True and FS set to that rate.")
        if info["mean_valid_fraction"] < 0.01:
            print(f"WARNING: {self.dataset_name} UV: {info['path']} has almost no valid pixels "
                  f"({100 * info['mean_valid_fraction']:.2f} % on average).")
        return frames
