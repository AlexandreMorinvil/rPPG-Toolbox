"""Participant-wise cross-validation on the vHRM dataset.

From one base config (TRAIN/TEST, optionally VALID, all with DATASET: vHRM), this writes one
config per fold, runs main.py for each fold in turn, and merges the evaluation windows into a
cross-validation summary. The participant pool is the participants selected by the TRAIN
block of the base config (its EXPERIMENTS / LABELS / VIEWS / PARTICIPANTS filters apply).
The launcher (New run -> Cross-validation) builds the same folds (cv_folds.py).

Preprocessing is cached per video, so only the first fold decodes videos.

Folds (default: the CROSS_VALIDATION block of the config; command-line options override it):
  --folds kfold --k 5          K balanced groups (participants dealt round-robin in sorted order)
  --folds loso                 one fold per participant (leave one out)
  --fold-file folds.yaml       named groups, e.g.  a: [jzc02, jzc05]
                               (participants not listed in any group stay in training)
  --no-group-sessions          test jzc02 and jzc02_baseline in different folds (default: same fold)

Examples (run from the toolbox root):
  python tools/vhrm_cross_validation.py --config configs/train_configs/vHRM_vHRM_vHRM_PHYSNET_FRONT.yaml \
      --folds loso --valid-count 2 --output-root runs/cv_physnet_front
  python tools/vhrm_cross_validation.py ... --dry-run              # only write the fold configs
  python tools/vhrm_cross_validation.py ... --plan-json plan.json  # only write the folds as JSON
  python tools/vhrm_cross_validation.py --output-root runs/cv_physnet_front --aggregate-only
"""

import argparse
import glob
import json
import os
import subprocess
import sys

import pandas as pd
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from cv_folds import SCHEMES, fold_overrides, plan as fold_plan  # noqa: E402
from dataset.data_loader.VHRMLoader import discover_segments, select_recordings, source_suffix  # noqa: E402
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


def fold_settings(args, config):
    """Fold options: the command-line values, else the CROSS_VALIDATION block of the config."""
    cv = config.CROSS_VALIDATION
    return {
        "scheme": "groups" if args.fold_file else (args.folds or cv.SCHEME),
        "k": cv.K if args.k is None else args.k,
        "valid_count": cv.VALID_COUNT if args.valid_count is None else args.valid_count,
        "groups": _load_yaml(args.fold_file) if args.fold_file else list(cv.GROUPS),
        "group_sessions": cv.GROUP_SESSIONS if args.group_sessions is None else args.group_sessions,
    }


def fold_config(raw, fold, output_root):
    config = yaml.safe_load(yaml.safe_dump(raw))
    # A fold is an ordinary run: keeping the block would make the fold config start a cross-validation.
    config.pop("CROSS_VALIDATION", None)
    log_path = os.path.join(os.path.abspath(output_root), f"fold_{fold['name']}")
    model_name = (config.get("TRAIN") or {}).get("MODEL_FILE_NAME") or "vHRM"
    group = os.path.basename(os.path.abspath(output_root)) if (config.get("WANDB") or {}).get("ENABLED") else None
    for key, value in fold_overrides(fold, model_name, log_path, group).items():
        _set(config, key, value)
    return config, log_path


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
    parser.add_argument("--config", help="Base YAML config (not needed with --aggregate-only).")
    fold_source = parser.add_mutually_exclusive_group()
    fold_source.add_argument("--folds", choices=SCHEMES, help="Fold scheme (default: CROSS_VALIDATION.SCHEME).")
    fold_source.add_argument("--fold-file", help="YAML mapping fold name -> participant folders.")
    parser.add_argument("--k", type=int, help="kfold: number of folds (default: CROSS_VALIDATION.K).")
    parser.add_argument("--valid-count", type=int,
                        help="Participants moved from training to validation in each fold, 0 = none "
                             "(default: CROSS_VALIDATION.VALID_COUNT).")
    parser.add_argument("--group-sessions", action=argparse.BooleanOptionalAction, default=None,
                        help="Keep the sessions of one person (jzc02, jzc02_baseline) in the same fold "
                             "(default: CROSS_VALIDATION.GROUP_SESSIONS).")
    parser.add_argument("--output-root", help="Folder for fold configs, logs and results.")
    parser.add_argument("--only", nargs="*", default=[], help="Run only these fold names.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true", help="Write fold configs without running them.")
    parser.add_argument("--plan-json", help="Write the folds (participant pool, test/valid/train) to this JSON file "
                                            "and stop.")
    parser.add_argument("--keep-going", action="store_true", help="Continue with the next fold after a failure.")
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)

    if args.aggregate_only:
        if not args.output_root:
            parser.error("--aggregate-only needs --output-root")
        aggregate(os.path.abspath(args.output_root))
        return
    if not args.config:
        parser.error("--config is required")
    if not args.output_root and not args.plan_json:
        parser.error("--output-root is required")

    from config import get_config
    config_path = os.path.abspath(args.config)
    config = get_config(argparse.Namespace(config_file=config_path))
    if config.TOOLBOX_MODE != "train_and_test":
        raise ValueError("Cross-validation needs TOOLBOX_MODE: train_and_test")
    settings = fold_settings(args, config)
    if settings["valid_count"] and config.VALID.DATA.DATASET != "vHRM":
        raise ValueError("Validation participants need a VALID block with DATASET: vHRM in the base config")
    pool = participant_pool(config)
    folds = fold_plan(pool, **settings)
    if args.plan_json:
        with open(args.plan_json, "w", encoding="utf-8") as handle:
            json.dump(folds, handle, indent=2)
        return
    for warning in folds["warnings"]:
        print(f"WARNING: {warning}")
    output_root = os.path.abspath(args.output_root)
    raw = _absolute_bases(_load_yaml(config_path), config_path)
    os.makedirs(output_root, exist_ok=True)
    print(f"{len(pool)} participants in the pool, {len(folds['folds'])} folds ({settings['scheme']}).")

    failures = []
    for fold in folds["folds"]:
        name = fold["name"]
        if args.only and name not in args.only:
            continue
        fold_yaml, log_path = fold_config(raw, fold, output_root)
        fold_path = os.path.join(output_root, "configs", f"fold_{name}.yaml")
        os.makedirs(os.path.dirname(fold_path), exist_ok=True)
        with open(fold_path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(fold_yaml, handle, sort_keys=False)
        print(f"\n=== fold {name}: test {fold['test']} | valid {fold['valid']} | "
              f"train {len(fold['train'])} participants")
        if not fold["train"]:
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
