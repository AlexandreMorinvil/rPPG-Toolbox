"""Participant-wise cross-validation on the vHRM dataset.

From one base config (TRAIN/TEST, optionally VALID, all with DATASET: vHRM), this writes one
config per fold, runs main.py for each fold in turn, and merges the evaluation windows into a
cross-validation summary. The participant pool is the participants selected by the TRAIN
block of the base config (its EXPERIMENTS / LABELS / VIEWS / PARTICIPANTS filters apply).

Preprocessing is cached per video, so only the first fold decodes videos.

Folds:
  --folds loso                 one fold per participant folder of the pool
  --fold-file folds.yaml       named groups, e.g.  jzc02: [jzc02, jzc02_baseline]
                               (participants not listed in any group stay in training)

Examples (run from the toolbox root):
  python tools/vhrm_cross_validation.py --config configs/train_configs/vHRM_vHRM_vHRM_PHYSNET_FRONT.yaml \
      --folds loso --valid-count 2 --output-root runs/cv_physnet_front
  python tools/vhrm_cross_validation.py ... --dry-run          # only write the fold configs
  python tools/vhrm_cross_validation.py ... --aggregate-only   # only merge existing results
"""

import argparse
import glob
import os
import subprocess
import sys

import pandas as pd
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dataset.data_loader.VHRMLoader import discover_segments, select_recordings, slug, source_suffix  # noqa: E402
from evaluation.vhrm_metrics import _summary  # noqa: E402


def _load_yaml(path):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _absolute_bases(raw, config_path):
    bases = raw.get("BASE") or [""]
    raw["BASE"] = [os.path.abspath(os.path.join(os.path.dirname(config_path), base)) if base else ""
                   for base in bases]
    return raw


def _set(raw, dotted, value):
    node = raw
    keys = dotted.split(".")
    for key in keys[:-1]:
        node = node.setdefault(key, {})
    node[keys[-1]] = value


def participant_pool(config):
    data = config.TRAIN.DATA
    if data.DATASET != "vHRM":
        raise ValueError("TRAIN.DATA.DATASET must be vHRM for vHRM cross-validation")
    recordings = select_recordings(discover_segments(data.DATA_PATH, data.VHRM.EXPERIMENTS, source_suffix(data)),
                                   data.VHRM, warn=lambda message: None)
    return sorted({recording["participant"] for recording in recordings})


def build_folds(args, pool):
    if args.fold_file:
        groups = _load_yaml(args.fold_file)
        if isinstance(groups, list):
            groups = {f"fold{index:02d}": members for index, members in enumerate(groups)}
        folds = {str(name): sorted({slug(member) for member in members}) for name, members in groups.items()}
        unknown = sorted({member for members in folds.values() for member in members} - set(pool))
        if unknown:
            print(f"WARNING: fold members not in the TRAIN selection: {', '.join(unknown)}")
        return folds
    return {participant: [participant] for participant in pool}


def fold_config(raw, args, name, test_group, pool, fold_names, folds):
    remaining = [participant for participant in pool if participant not in test_group]
    valid_group = []
    if args.valid_count:
        # Validation participants: the next folds' members (cyclic), never the test group.
        position = fold_names.index(name)
        for offset in range(1, len(fold_names)):
            candidate = folds[fold_names[(position + offset) % len(fold_names)]]
            for participant in candidate:
                if participant in remaining and participant not in valid_group:
                    valid_group.append(participant)
            if len(valid_group) >= args.valid_count:
                break
        valid_group = valid_group[:args.valid_count]
    train_group = [participant for participant in remaining if participant not in valid_group]

    fold = yaml.safe_load(yaml.safe_dump(raw))
    log_path = os.path.join(os.path.abspath(args.output_root), f"fold_{name}")
    _set(fold, "LOG.PATH", log_path)
    _set(fold, "TRAIN.DATA.VHRM.PARTICIPANTS", train_group)
    _set(fold, "TRAIN.DATA.VHRM.EXCLUDE_PARTICIPANTS", [])
    _set(fold, "TEST.DATA.VHRM.PARTICIPANTS", test_group)
    _set(fold, "TEST.DATA.VHRM.EXCLUDE_PARTICIPANTS", [])
    if valid_group:
        _set(fold, "VALID.DATA.VHRM.PARTICIPANTS", valid_group)
        _set(fold, "VALID.DATA.VHRM.EXCLUDE_PARTICIPANTS", [])
        _set(fold, "TEST.USE_LAST_EPOCH", False)
    model_name = (fold.get("TRAIN") or {}).get("MODEL_FILE_NAME") or "vHRM"
    _set(fold, "TRAIN.MODEL_FILE_NAME", f"{model_name}_fold-{name}")
    if (fold.get("WANDB") or {}).get("ENABLED"):
        _set(fold, "WANDB.GROUP", os.path.basename(os.path.abspath(args.output_root)))
        _set(fold, "WANDB.RUN_NAME", f"{model_name}_fold-{name}")
    return fold, log_path, train_group, valid_group


def aggregate(output_root):
    frames = []
    for fold_dir in sorted(glob.glob(os.path.join(output_root, "fold_*"))):
        for path in glob.glob(os.path.join(fold_dir, "**", "window_results.csv"), recursive=True):
            frame = pd.read_csv(path)
            frame.insert(0, "fold", os.path.basename(fold_dir).removeprefix("fold_"))
            frames.append(frame)
    if not frames:
        print("No window_results.csv found; nothing to aggregate.")
        return
    windows = pd.concat(frames, ignore_index=True)
    for column in ("experiment", "participant", "movement", "view"):
        windows[column] = windows[column].fillna("") if column in windows else ""
    summary_dir = os.path.join(output_root, "cv_summary")
    os.makedirs(summary_dir, exist_ok=True)
    windows.to_csv(os.path.join(summary_dir, "all_windows.csv"), index=False)
    overall = _summary(windows, [])
    overall.to_csv(os.path.join(summary_dir, "overall_summary.csv"), index=False)
    for name, columns in {"by_fold": ["fold"], "by_participant": ["participant"], "by_movement": ["movement"],
                          "by_view": ["view"], "by_experiment": ["experiment"]}.items():
        _summary(windows, columns).to_csv(os.path.join(summary_dir, f"{name}.csv"), index=False)
    row = overall.iloc[0]
    print(f"Cross-validation over {windows['fold'].nunique()} folds, {len(windows)} windows: "
          f"MAE {row['mae_bpm']:.2f} bpm, RMSE {row['rmse_bpm']:.2f} bpm, Pearson {row['pearson']:.3f} "
          f"(vs measured HR); MAE vs label HR {row['mae_vs_label_bpm']:.2f} bpm.")
    print(f"Summaries written to {summary_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="Base YAML config.")
    fold_source = parser.add_mutually_exclusive_group()
    fold_source.add_argument("--folds", choices=["loso"], default="loso")
    fold_source.add_argument("--fold-file", help="YAML mapping fold name -> participant folders.")
    parser.add_argument("--valid-count", type=int, default=0,
                        help="Participants moved from training to validation in each fold (0 = none).")
    parser.add_argument("--output-root", required=True, help="Folder for fold configs, logs and results.")
    parser.add_argument("--only", nargs="*", default=[], help="Run only these fold names.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true", help="Write fold configs without running them.")
    parser.add_argument("--keep-going", action="store_true", help="Continue with the next fold after a failure.")
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    output_root = os.path.abspath(args.output_root)

    if args.aggregate_only:
        aggregate(output_root)
        return

    from config import get_config
    config_path = os.path.abspath(args.config)
    config = get_config(argparse.Namespace(config_file=config_path))
    if config.TOOLBOX_MODE != "train_and_test":
        raise ValueError("Cross-validation needs TOOLBOX_MODE: train_and_test")
    if args.valid_count and config.VALID.DATA.DATASET != "vHRM":
        raise ValueError("--valid-count needs a VALID block with DATASET: vHRM in the base config")
    pool = participant_pool(config)
    folds = build_folds(args, pool)
    fold_names = list(folds)
    raw = _absolute_bases(_load_yaml(config_path), config_path)
    os.makedirs(output_root, exist_ok=True)
    print(f"{len(pool)} participants in the pool, {len(folds)} folds.")

    failures = []
    for name in fold_names:
        if args.only and name not in args.only:
            continue
        fold, log_path, train_group, valid_group = fold_config(
            raw, args, name, folds[name], pool, fold_names, folds)
        fold_path = os.path.join(output_root, "configs", f"fold_{name}.yaml")
        os.makedirs(os.path.dirname(fold_path), exist_ok=True)
        with open(fold_path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(fold, handle, sort_keys=False)
        print(f"\n=== fold {name}: test {folds[name]} | valid {valid_group} | train {len(train_group)} participants")
        if not train_group:
            print("  skipped: no training participants left")
            continue
        if args.dry_run:
            print(f"  wrote {fold_path}")
            continue
        os.makedirs(log_path, exist_ok=True)
        with open(os.path.join(log_path, "run.log"), "w", encoding="utf-8") as log:
            # Force UTF-8 in the child: progress bars do not fit legacy Windows code pages.
            environment = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
            process = subprocess.Popen([args.python, "main.py", "--config_file", fold_path], cwd=ROOT,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                       encoding="utf-8", errors="replace", env=environment)
            console_encoding = sys.stdout.encoding or "utf-8"
            for line in process.stdout:
                sys.stdout.write(line.encode(console_encoding, errors="replace").decode(console_encoding))
                sys.stdout.flush()
                log.write(line)
            return_code = process.wait()
        if return_code != 0:
            failures.append(name)
            print(f"Fold {name} failed with exit code {return_code} (log: {os.path.join(log_path, 'run.log')})")
            if not args.keep_going:
                break

    if not args.dry_run:
        aggregate(output_root)
    if failures:
        print(f"Failed folds: {', '.join(failures)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
