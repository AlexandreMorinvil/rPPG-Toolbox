"""Movement and skin visibility of every evaluation window, from the vHRM Studio files of the recordings.

Movement and visibility belong to the recording, not to the input a model used: they are measured once per window
and joined to any run by (recording_id, start_frame) (tools/paired_comparison.py --features ... --bin-by ...).

    python tools/vhrm_window_features.py --windows <run folder or window_results.csv> --output runs/features/b2.csv

Columns (means over the window's frames):
    processed_fraction        frames vHRM Studio fitted a body to (frame_status 1)
    body_speed_deg_s          rotation speed of the 21 SMPL-X body joints (relative to their parent), averaged
    global_speed_deg_s        rotation speed of the whole body (SMPL-X global orientation)
    head_speed_deg_s          rotation speed of the head (FLAME pose)
    translation_speed_per_s   body translation speed in the camera frame, in body distances per second
    head_skin_fraction        share of the head chart's texels with visible skin (face visibility)
    nonhead_skin_fraction     same for every other chart, and body_skin_fraction for the whole body
The skin fractions read the UV texture every --visibility-step frames (14 ms per read); 0 skips them.

The files are found like the loaders find them: thesis/paths.yaml (uv section) by default, or --experiment-path
experiment=folder (lab data) and --uv-root DATASET=folder (UBFC-rPPG, PURE, COHFACE). Frame k of a lab recording is
the stored row nearest to k / 30 s (VHRMLoader resampling); literature recordings have one row per frame.
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import _vhrm_node  # noqa: E402
from dataset.data_loader import uv_source  # noqa: E402
from dataset.data_loader.VHRMLoader import discover_segments, select_recordings  # noqa: E402
from tools.paired_comparison import find_windows  # noqa: E402

FEATURES = ("processed_fraction", "body_speed_deg_s", "global_speed_deg_s", "head_speed_deg_s",
            "translation_speed_per_s", "head_skin_fraction", "nonhead_skin_fraction", "body_skin_fraction")
LAB_FS = 30


def rotation_speeds(matrices, times, valid):
    """Per-row rotation speed (deg/s) between each row and the previous one; (T, ..., 3, 3) -> (T, ...)."""
    relative = np.einsum("t...ji,t...jk->t...ik", matrices[:-1], matrices[1:])
    angles = np.degrees(np.arccos(np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0)))
    steps = np.diff(times).reshape((-1,) + (1,) * (angles.ndim - 1))
    speeds = np.full((len(matrices),) + angles.shape[1:], np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        speeds[1:] = np.where(steps > 0, angles / steps, np.nan)
    both = valid[1:] & valid[:-1]
    speeds[1:][~both] = np.nan
    return speeds


def motion_per_row(h5):
    """{feature: per-row values} of the movement features, and the row times (s)."""
    from scipy.spatial.transform import Rotation

    status = np.asarray(h5["frame_status"][()]) == 1
    count = len(status)
    if "source_timestamp_s" in h5:
        times = np.asarray(h5["source_timestamp_s"][()], dtype=np.float64)
    else:
        times = np.arange(count) / float(h5.attrs.get("effective_fps", LAB_FS) or LAB_FS)
    rows = {"processed_fraction": status.astype(np.float64)}
    nan = np.full(count, np.nan)
    if "body_param/body_pose" in h5:
        joints = rotation_speeds(np.asarray(h5["body_param/body_pose"][()], dtype=np.float64), times, status)
        with warnings.catch_warnings():  # rows without a body are NaN for every joint
            warnings.simplefilter("ignore", RuntimeWarning)
            rows["body_speed_deg_s"] = np.nanmean(joints, axis=1)
        rows["global_speed_deg_s"] = rotation_speeds(
            np.asarray(h5["body_param/global_pose"][()], dtype=np.float64)[:, 0], times, status)
    if "flame_param/pose_params" in h5:
        head = Rotation.from_rotvec(np.asarray(h5["flame_param/pose_params"][()], dtype=np.float64)).as_matrix()
        rows["head_speed_deg_s"] = rotation_speeds(head, times, status)
    if "pd_cam" in h5:
        translation = np.asarray(h5["pd_cam"][()], dtype=np.float64)[:, :3, 3]
        depth = np.abs(translation[:, 2])
        step = np.linalg.norm(np.diff(translation, axis=0), axis=1)
        speed = nan.copy()
        with np.errstate(divide="ignore", invalid="ignore"):
            speed[1:] = step / np.maximum(depth[1:], 1e-6) / np.diff(times)
        speed[1:][~(status[1:] & status[:-1])] = np.nan
        rows["translation_speed_per_s"] = speed
    return rows, times


class Visibility:
    """Skin fractions of the packed charts for single rows (read on demand, cached)."""

    def __init__(self, h5):
        self.h5 = h5
        self.rgb = h5["packed_uv_skin_rgb"]
        self.mask = h5.get("packed_uv_skin_mask_packed")
        self.status = np.asarray(h5["frame_status"][()]) == 1
        self.head = uv_source.region_mask(h5, "packed", ["head"], [])
        self.body = uv_source.region_mask(h5, "packed")
        self.nonhead = self.body & ~self.head
        self.cache = {}

    def row(self, index):
        if index not in self.cache:
            if not self.status[index]:
                self.cache[index] = (np.nan, np.nan, np.nan)
            else:
                skin = (uv_source._unpack_mask(self.mask[index]) if self.mask is not None
                        else uv_source.skin_texels(self.rgb[index]))
                self.cache[index] = tuple(float(skin[region].mean()) if region.any() else np.nan
                                          for region in (self.head, self.nonhead, self.body))
        return self.cache[index]


def lab_files(experiment_paths):
    """{recording_id: .h5 path} of every lab recording in the experiment folders (all views and labels)."""
    node = _vhrm_node()
    node.VIEWS = []
    segments = discover_segments("", (), uv_source.H5_SUFFIX, experiment_paths)
    return {recording["index"]: recording["path"] for recording in select_recordings(segments, node, warn=lambda m: None)}


def literature_file(dataset, recording_id, root):
    """The .h5 file of a literature recording (ids as in window_results.csv)."""
    if dataset == "UBFC-rPPG":
        source = Path(root) / recording_id / "vid.avi"
    elif dataset == "PURE":
        name = f"{int(recording_id) // 100:02d}-{int(recording_id) % 100:02d}"
        nested = Path(root) / name / name
        source = nested if (Path(root) / name / f"{name}.json").is_file() else Path(root) / name
    elif dataset == "COHFACE":
        subject, session = recording_id.split("-")
        source = Path(root) / str(int(subject)) / session / "data.avi"
    else:
        raise ValueError(f"No vHRM Studio layout known for {dataset}")
    return uv_source.resolve_uv_path(source, root)


def window_features(path, windows, lab, visibility_step):
    """Feature rows of the windows of one recording (.h5 file)."""
    import h5py

    with h5py.File(path, "r") as h5:
        per_row, times = motion_per_row(h5)
        rows = (uv_source.resample_rows(times, LAB_FS) if lab else np.arange(len(times)))
        visibility = Visibility(h5) if visibility_step > 0 and "packed_uv_skin_rgb" in h5 else None
        result = []
        for window in windows.itertuples(index=False):
            selected = rows[int(window.start_frame):int(window.end_frame)]
            features = {"recording_id": window.recording_id, "start_frame": int(window.start_frame),
                        "end_frame": int(window.end_frame)}
            with warnings.catch_warnings():  # a window without a fitted body has NaN features
                warnings.simplefilter("ignore", RuntimeWarning)
                for name, values in per_row.items():
                    features[name] = float(np.nanmean(values[selected])) if len(selected) else np.nan
                if visibility is not None and len(selected):
                    samples = np.array([visibility.row(int(row)) for row in selected[::visibility_step]])
                    for column, name in enumerate(("head_skin_fraction", "nonhead_skin_fraction",
                                                   "body_skin_fraction")):
                        features[name] = float(np.nanmean(samples[:, column]))
            result.append(features)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--windows", required=True, help="A run folder or its window_results.csv.")
    parser.add_argument("--output", required=True, help="CSV file to write.")
    parser.add_argument("--paths", default=str(ROOT / "thesis" / "paths.yaml"),
                        help="Where the vHRM Studio files are (its uv section).")
    parser.add_argument("--experiment-path", action="append", default=[], metavar="EXPERIMENT=FOLDER",
                        help="Lab experiment folder of .h5 files (replaces the paths file for it).")
    parser.add_argument("--uv-root", action="append", default=[], metavar="DATASET=FOLDER",
                        help="Converted literature dataset (replaces the paths file for it).")
    parser.add_argument("--visibility-step", type=int, default=10,
                        help="Read the UV texture every N frames for the skin fractions (0 = skip them).")
    args = parser.parse_args(argv)

    uv = {}
    if args.paths and os.path.isfile(args.paths):
        with open(args.paths, encoding="utf-8") as handle:
            uv = (yaml.safe_load(handle) or {}).get("uv") or {}
    experiment_paths = {name: folder for name, folder in (uv.get("lab") or {}).items() if folder}
    experiment_paths.update(dict(item.split("=", 1) for item in args.experiment_path))
    roots = {name: folder for name, folder in uv.items() if name != "lab" and folder}
    roots.update(dict(item.split("=", 1) for item in args.uv_root))

    windows = pd.read_csv(find_windows(args.windows))
    windows["recording_id"] = windows["recording_id"].astype(str)
    windows = windows.drop_duplicates(["recording_id", "start_frame"])
    is_lab = "experiment" in windows and windows["experiment"].fillna("").astype(str).str.len().gt(0).all()
    files = {}
    if is_lab:
        existing = [f"{name}={folder}" for name, folder in experiment_paths.items() if os.path.isdir(folder)]
        files = lab_files(existing)
    rows, missing = [], []
    groups = list(windows.groupby("recording_id", sort=True))
    for number, (recording_id, recording_windows) in enumerate(groups, start=1):
        try:
            if is_lab:
                path = files[recording_id]
            else:
                dataset = str(recording_windows["dataset"].iloc[0])
                path = literature_file(dataset, recording_id, roots[dataset])
            rows += window_features(path, recording_windows, is_lab, args.visibility_step)
        except (KeyError, FileNotFoundError, OSError, ValueError) as error:
            missing.append(f"{recording_id} ({error})")
        if number % 25 == 0 or number == len(groups):
            print(f"{number}/{len(groups)} recordings", flush=True)
    frame = pd.DataFrame(rows, columns=["recording_id", "start_frame", "end_frame", *FEATURES])
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f"Wrote {len(frame)} windows of {len(groups) - len(missing)} recordings to {args.output}")
    if missing:
        print(f"WARNING: no vHRM Studio file for {len(missing)} recordings: " + "; ".join(missing[:10])
              + (" ..." if len(missing) > 10 else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
