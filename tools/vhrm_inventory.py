"""Inventory of the vHRM dataset and dry run of a config's split selection.

Never opens video files: it only reads meta.json and the signal CSV headers.

Examples (run from the toolbox root):
  # Every segment, its views and the ground truth available.
  python tools/vhrm_inventory.py --data-path "F:/Datasets preparation" --csv inventory.csv

  # Which recordings each split of a config selects, and whether splits overlap.
  python tools/vhrm_inventory.py --config configs/train_configs/vHRM_vHRM_vHRM_PHYSNET_FRONT.yaml
"""

import argparse
import collections
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dataset.data_loader.VHRMLoader import (  # noqa: E402
    KNOWN_VIEWS, discover_segments, resolve_signal_files, select_recordings)

SPLITS = ("TRAIN", "VALID", "TEST", "UNSUPERVISED")


def inventory(data_path):
    rows = []
    for segment in discover_segments(data_path):
        files = resolve_signal_files(segment)
        views = {camera["view"]: camera for camera in segment["cameras"]}
        row = {
            "experiment": segment["experiment_folder"],
            "participant": segment["participant_folder"],
            "segment": segment["segment_folder"],
            "label": segment["label"],
            "duration_sec": segment["duration_sec"],
            "has_start_time": segment["start_epoch"] is not None,
        }
        for view in KNOWN_VIEWS:
            camera = views.get(view)
            row[f"{view}_fps"] = round(camera["fps"], 2) if camera and camera["exists"] and camera["fps"] else None
        row["other_views"] = ",".join(sorted(set(views) - set(KNOWN_VIEWS)))
        for kind in ("ppg", "rpeaks", "zephyr", "polar"):
            row[f"has_{kind}"] = files[kind] is not None
        rows.append(row)
    return pd.DataFrame(rows)


def print_inventory(frame):
    print(f"{len(frame)} segments, {frame['duration_sec'].sum() / 3600:.2f} h of labelled time "
          f"(per view), {frame['participant'].nunique()} participant folders.\n")
    summary = frame.groupby("experiment").agg(
        participants=("participant", "nunique"), segments=("segment", "count"),
        hours=("duration_sec", lambda values: values.sum() / 3600),
        with_ppg=("has_ppg", "sum"), with_zephyr=("has_zephyr", "sum"))
    print(summary.round(2).to_string(), end="\n\n")
    labels = frame.groupby("label").agg(segments=("segment", "count"),
                                        hours=("duration_sec", lambda values: values.sum() / 3600))
    print(labels.round(2).to_string(), end="\n\n")
    for view in KNOWN_VIEWS:
        missing = frame[frame[f"{view}_fps"].isna()]
        if len(missing):
            print(f"Missing {view}: " + ", ".join(sorted(set(missing["participant"]))))
    odd_fps = frame[[f"{view}_fps" for view in KNOWN_VIEWS]].apply(
        lambda column: column.notna() & (column.round() != 60))
    for view in KNOWN_VIEWS:
        participants = sorted(set(frame.loc[odd_fps[f"{view}_fps"], "participant"]))
        if participants:
            print(f"{view} not at 59.94 fps: " + ", ".join(participants))
    unknown = frame[frame["other_views"] != ""]
    if len(unknown):
        print("Unrecognised camera labels (use VIEW_ALIASES): " + ", ".join(
            f"{row.participant}/{row.segment} [{row.other_views}]" for row in unknown.itertuples()))
    for kind in ("ppg", "zephyr"):
        missing = frame[~frame[f"has_{kind}"]]
        if len(missing):
            print(f"No {kind} signal: " + ", ".join(sorted(set(missing["participant"]))))


def split_selection(config, split):
    data = getattr(config, split).DATA
    if data.DATASET != "vHRM" or not data.DATA_PATH:
        return None
    segments = discover_segments(data.DATA_PATH, data.VHRM.EXPERIMENTS)
    recordings = select_recordings(segments, data.VHRM, warn=lambda message: None)
    if not (data.BEGIN == 0 and data.END == 1):
        participants = sorted({recording["participant"] for recording in recordings})
        selected = set(participants[int(data.BEGIN * len(participants)):int(data.END * len(participants))])
        recordings = [recording for recording in recordings if recording["participant"] in selected]
    return recordings


def print_config_selection(config_path, show_recordings):
    from config import get_config
    config = get_config(argparse.Namespace(config_file=config_path))
    active = {
        "train_and_test": ("TRAIN", "VALID", "TEST") if not config.TEST.USE_LAST_EPOCH else ("TRAIN", "TEST"),
        "only_test": ("TEST",),
        "unsupervised_method": ("UNSUPERVISED",),
    }.get(config.TOOLBOX_MODE, SPLITS)
    selections = {}
    for split in active:
        recordings = split_selection(config, split)
        if recordings is None:
            print(f"{split}: not a vHRM split ({getattr(config, split).DATA.DATASET or 'unset'}).\n")
            continue
        selections[split] = recordings
        data = getattr(config, split).DATA
        hours = sum(recording["duration_sec"] for recording in recordings) / 3600
        print(f"{split}: {len(recordings)} recordings, {hours:.2f} h of video, "
              f"PPG source {data.VHRM.PPG_SOURCE}, HR source {data.VHRM.HR_SOURCE}, "
              f"REQUIRE_PPG {data.VHRM.REQUIRE_PPG}")
        print("  participants: " + ", ".join(sorted({recording["participant"] for recording in recordings})))
        print("  labels:       " + ", ".join(f"{label} ({count})" for label, count in sorted(
            collections.Counter(recording["label"] for recording in recordings).items())))
        print("  views:        " + ", ".join(f"{view} ({count})" for view, count in sorted(
            collections.Counter(recording["view"] for recording in recordings).items())))
        modes = collections.Counter((recording["view"], recording["crop_mode"][0]) for recording in recordings)
        print("  crop modes:   " + ", ".join(f"{view}={mode}" for view, mode in sorted(modes)))
        if show_recordings:
            for recording in recordings:
                print(f"    {recording['index']}  ({recording['duration_sec']:.0f}s)")
        print()

    participants = {split: {recording["participant"] for recording in recordings}
                    for split, recordings in selections.items()}
    names = list(participants)
    overlap_found = False
    for position, first in enumerate(names):
        for second in names[position + 1:]:
            shared = participants[first] & participants[second]
            if shared:
                overlap_found = True
                print(f"WARNING: {first} and {second} share participants: {', '.join(sorted(shared))}")
    if not overlap_found and len(names) > 1:
        print("No participant appears in more than one split.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-path", help="Dataset root (folder containing JZC, baseline, ...).")
    parser.add_argument("--config", help="Toolbox YAML config to dry-run.")
    parser.add_argument("--csv", help="Write the segment inventory to this CSV file.")
    parser.add_argument("--recordings", action="store_true", help="List every selected recording.")
    args = parser.parse_args()
    if not args.data_path and not args.config:
        parser.error("give --data-path and/or --config")
    os.chdir(ROOT)
    if args.data_path:
        frame = inventory(args.data_path)
        print_inventory(frame)
        if args.csv:
            frame.to_csv(args.csv, index=False)
            print(f"\nWrote {args.csv}")
        print()
    if args.config:
        print_config_selection(args.config, args.recordings)


if __name__ == "__main__":
    main()
