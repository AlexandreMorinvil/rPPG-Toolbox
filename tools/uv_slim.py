"""Shrink vHRM Studio *_skin_uv.h5 files to what the toolbox UV input mode reads (packed layout).

Every *_skin_uv.h5 file under the given folder (recursively) is rewritten in place with only:

    packed_uv_*            the packed texture, its masks, chart map (uv_source.read_uv_frames / region_mask)
    frame_status           frames the studio could not process
    source_timestamp_s     resampling to DATA.FS
    all root attributes    effective_fps, frame_stride, packed_uv_chart_names, ...

Dropped: the standard SMPL-X layout (uv_*), the body / face fit parameters and anything else. Afterwards the files
work only with UV.LAYOUT: packed. --keep-motion also keeps body_param, flame_param and pd_cam, which
tools/vhrm_window_features.py reads for its motion features.

HDF5 does not give back the space of deleted datasets, so each file is copied (kept items only, chunks copied
as stored, no recompression) to a temporary file next to it, which then replaces the original. Other files
(labels, vhrm_manifest.json) are not touched. The originals are overwritten: run it on a copy.

    python tools/uv_slim.py D:/rPPG_Datasets_uv --dry-run
    python tools/uv_slim.py D:/rPPG_Datasets_uv
"""
import argparse
import os
import sys
from pathlib import Path

import h5py

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dataset.data_loader.uv_source import H5_SUFFIX  # noqa: E402

KEEP_PREFIXES = ("packed_uv_",)
KEEP_NAMES = ("frame_status", "source_timestamp_s")
MOTION_NAMES = ("body_param", "flame_param", "pd_cam")
REQUIRED = ("packed_uv_skin_rgb", "packed_uv_body_mask", "frame_status")
TMP_SUFFIX = ".slim-tmp"


def is_kept(name, keep_motion=False):
    names = KEEP_NAMES + (MOTION_NAMES if keep_motion else ())
    return name.startswith(KEEP_PREFIXES) or name in names


def storage_size(obj):
    """Bytes an HDF5 dataset (or every dataset of a group) occupies in the file."""
    if isinstance(obj, h5py.Dataset):
        return obj.id.get_storage_size()
    total = 0

    def add(_, item):
        nonlocal total
        if isinstance(item, h5py.Dataset):
            total += item.id.get_storage_size()

    obj.visititems(add)
    return total


def plan(h5, keep_motion=False):
    """(kept names, {dropped name: bytes}) of the root items of an open file."""
    kept, dropped = [], {}
    for name in h5.keys():
        if is_kept(name, keep_motion):
            kept.append(name)
        else:
            dropped[name] = storage_size(h5[name])
    return kept, dropped


def slim_file(path, keep_motion=False, dry_run=False):
    """Rewrite one file with the kept items. Returns (status, bytes before, bytes after, dropped names)."""
    path = Path(path)
    before = path.stat().st_size
    with h5py.File(path, "r") as h5:
        missing = [name for name in REQUIRED if name not in h5]
        if missing:
            return f"skipped: no {', '.join(missing)}", before, before, []
        kept, dropped = plan(h5, keep_motion)
        if not dropped:
            return "already slim", before, before, []
        if dry_run:
            return "would drop", before, before - sum(dropped.values()), sorted(dropped)
        tmp = path.with_name(path.name + TMP_SUFFIX)
        try:
            with h5py.File(tmp, "w") as out:
                for key, value in h5.attrs.items():
                    out.attrs[key] = value
                for name in kept:
                    h5.copy(h5[name], out, name=name)
                for name in kept:
                    if isinstance(h5[name], h5py.Dataset):
                        source, copy = h5[name], out[name]
                        if source.shape != copy.shape or source.dtype != copy.dtype:
                            raise RuntimeError(f"copy of {name} differs ({copy.shape} {copy.dtype})")
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
    os.replace(tmp, path)
    return "slimmed", before, path.stat().st_size, sorted(dropped)


def find_files(folder, suffix=H5_SUFFIX):
    return sorted(p for p in Path(folder).rglob(f"*{suffix}") if p.is_file())


def remove_leftovers(folder):
    """Temporary files of an interrupted run."""
    for leftover in Path(folder).rglob(f"*{TMP_SUFFIX}"):
        leftover.unlink()
        print(f"removed leftover {leftover}")


def gb(size):
    return f"{size / 1e9:.2f} GB"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", help="Folder searched recursively for *_skin_uv.h5 files.")
    parser.add_argument("--dry-run", action="store_true", help="Only report what would be dropped.")
    parser.add_argument("--keep-motion", action="store_true",
                        help="Also keep body_param, flame_param and pd_cam (tools/vhrm_window_features.py).")
    parser.add_argument("--suffix", default=H5_SUFFIX, help=f"File suffix to look for (default {H5_SUFFIX}).")
    args = parser.parse_args(argv)

    if not os.path.isdir(args.folder):
        parser.error(f"not a folder: {args.folder}")
    if not args.dry_run:
        remove_leftovers(args.folder)
    files = find_files(args.folder, args.suffix)
    if not files:
        print(f"No *{args.suffix} files under {args.folder}")
        return 1
    total_before = total_after = 0
    failures = 0
    for index, path in enumerate(files, 1):
        prefix = f"[{index}/{len(files)}] {os.path.relpath(path, args.folder)}"
        try:
            status, before, after, dropped = slim_file(path, args.keep_motion, args.dry_run)
        except Exception as error:  # keep going: one unreadable or locked file should not stop the run
            failures += 1
            print(f"{prefix}: FAILED ({type(error).__name__}: {error})")
            continue
        total_before += before
        total_after += after
        detail = f" [{', '.join(dropped)}]" if dropped else ""
        print(f"{prefix}: {status}, {gb(before)} -> {gb(after)}{detail}", flush=True)
    verb = "would go" if args.dry_run else "went"
    print(f"\n{len(files)} files: {gb(total_before)} {verb} to {gb(total_after)} "
          f"({gb(total_before - total_after)} freed){f', {failures} failed' if failures else ''}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
