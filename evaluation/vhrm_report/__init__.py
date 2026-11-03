"""Analysis report of a checkpoint on vHRM: HR, beat-to-beat RR intervals and HRV against the chest sensors.

    generate_report(config, checkpoint)       tools/vhrm_report.py and the launcher's Report button
    report_after_training(config, trainer)    main.py, when REPORT.ENABLED

Every recording of the test data's experiments and views is analysed (every participant, every segment label
unless REPORT.LABELS narrows it), and each participant is tagged with its split of the run. A report goes to
<experiment folder>/report/<checkpoint>/ (see report_folder); when the run has test participants,
test_set/ repeats every output on them alone. The method is in METHOD.md, copied into every report.
"""

import json
import os
import platform
import shutil
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from . import data, plots
from .evaluate import (conditions, global_table, label_order, label_split_table, recording_order, recording_row,
                       split_order)
from .export import export
from .inference import Network, build_trainer, load_weights
from .pipeline import analyse_recording, load_result, model_fingerprint, save_result
from .report import write_report
from .settings import analysis_settings

REPORT_DIR = "report"
TEST_SET = "test_set"
CV_REPORT = "cross_validation"  # <cross-validation folder>/report/cross_validation
METHOD = Path(__file__).with_name("METHOD.md")
GENERATED = ("summary", "mosaics", "participants", "data_csv", TEST_SET)
SCOPES = {"all": "all recordings", "test": "test participants only",
          "cv": "test participants of every cross-validation fold"}


def report_folder(config, checkpoint):
    """<experiment folder>/report/<checkpoint name, without the run's checkpoint prefix>.

    The experiment folder is the one holding the checkpoint's PreTrainedModels/ folder; for a checkpoint stored
    elsewhere (released weights), the config's: PreTrainedModels/ of a training, saved_test_outputs/ of an evaluation.
    The launcher (launcher/reports.report_location) follows the same rule.
    """
    checkpoint = Path(checkpoint)
    if checkpoint.parent.name == Path(config.MODEL.MODEL_DIR).name:
        experiment = checkpoint.parent.parent
    elif config.TOOLBOX_MODE == "train_and_test":
        experiment = Path(config.MODEL.MODEL_DIR).parent
    else:
        experiment = Path(config.TEST.OUTPUT_SAVE_DIR).parent
    name = checkpoint.stem
    prefix = f"{config.TRAIN.MODEL_FILE_NAME}_"
    if config.TRAIN.MODEL_FILE_NAME and name.startswith(prefix) and len(name) > len(prefix):
        name = name[len(prefix):]
    return experiment / REPORT_DIR / name


def tested_checkpoint(config, trainer):
    """The checkpoint the toolbox's test used: INFERENCE.MODEL_PATH, the last epoch or the best validation epoch."""
    if config.TOOLBOX_MODE == "only_test":
        return config.INFERENCE.MODEL_PATH
    epoch = trainer.max_epoch_num - 1 if config.TEST.USE_LAST_EPOCH else trainer.best_epoch
    return os.path.join(trainer.model_dir, f"{trainer.model_file_name}_Epoch{epoch}.pth")


def report_after_training(config, trainer):
    """main.py hook: the report of the tested checkpoint. A failed report is logged and never fails the run."""
    if not config.REPORT.ENABLED:
        return None
    print("\n===Analysis report===", flush=True)
    try:
        return generate_report(config, tested_checkpoint(config, trainer), trainer=trainer, origin="after_training")
    except Exception as error:
        traceback.print_exc()
        print(f"[report] FAILED: {error}", flush=True)
        return None


def _versions():
    import torch
    return {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
            "pandas": pd.__version__, "cuda": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def write_outputs(output, results, info, failures, run_info, settings, scope="all", export_csv=True):
    """summary/, figures, participants/, REPORT.md, run_config.json (and data_csv/) for these recordings.

    scope: all (every recording, pooled per split too), test (test_set/: the test participants), cv (the test
    participants of every fold of a cross-validation).
    """
    within = settings["hr"]["within_bpm"]
    for name in GENERATED if scope == "all" else GENERATED[:-1]:
        shutil.rmtree(output / name, ignore_errors=True)
    ordered = recording_order(results)
    table = pd.DataFrame([recording_row(result, within) for result in ordered])
    global_metrics = global_table(results, within, by_split=scope == "all")
    matrix = label_split_table(global_metrics, label_order(results), split_order(results) if scope == "all" else [])
    summary = output / "summary"
    summary.mkdir(parents=True, exist_ok=True)
    table.to_csv(summary / "per_recording_metrics.csv", index=False, float_format="%.3f")
    global_metrics.to_csv(summary / "global_metrics.csv", index=False, float_format="%.3f")
    matrix.to_csv(summary / "label_split_mae.csv", index=False, float_format="%.3f")

    several_experiments = len({r["experiment"] for r in results}) > 1
    for experiment, participant in sorted({(r["experiment"], r["participant"]) for r in results}):
        own = [r for r in results if (r["experiment"], r["participant"]) == (experiment, participant)]
        folder = output / "participants" / (f"{experiment}-{participant}" if several_experiments else participant)
        plots.participant_figures(own, folder, info["title"])
        table[(table.experiment == experiment) & (table.participant == participant)].to_csv(
            folder / "metrics.csv", index=False, float_format="%.3f")
    several_views = len({r["view"] for r in results}) > 1
    for condition in conditions(results):
        for metric, name in (("hr", "hr"), ("rmssd", "hrv"), ("ibi", "rr")):
            plots.mosaic(results, metric, condition, output / "mosaics" /
                         f"{name}_{plots.condition_file(condition, several_views)}.png", info["title"])
    plots.overview(results, summary / "overview.png", info["title"])
    plots.participant_errors(table, results, summary / "participant_errors.png", info["title"])
    plots.face_crop_mosaic(results, summary / "face_crops.png")

    write_report(output / "REPORT.md", info, global_metrics, matrix, table, list(failures), within, scope)
    if scope != "test":
        shutil.copy2(METHOD, output / "METHOD.md")
    (output / "run_config.json").write_text(json.dumps(
        {**run_info, "title": info["title"], "scope": SCOPES[scope],
         "recordings": [result["key"] for result in ordered], "failures": failures},
        indent=2, default=str), encoding="utf-8")
    if export_csv:
        export(output / "data", output / "data_csv", info["title"], settings, run_info["hr_source"])
    return global_metrics


def generate_report(config, checkpoint, output=None, labels=None, participants=None, export_csv=None,
                    trainer=None, origin="manual"):
    """Analyse every recording of the report scope with this checkpoint; returns the report folder.

    origin: after_training for main.py's report of the tested checkpoint (preferred by merge_cross_validation).
    """
    checkpoint = str(checkpoint)
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    output = Path(output) if output else report_folder(config, checkpoint)
    data_config = data.report_data_config(config, labels, participants)
    if not data_config.PREPROCESS.DO_CHUNK:
        raise ValueError("The analysis report needs clips of a fixed length (TEST.DATA.PREPROCESS.DO_CHUNK: True).")
    print(f"[report] {config.MODEL.NAME}: {checkpoint}", flush=True)
    print(f"[report] Output: {output}", flush=True)
    splits = data.participant_splits(config)
    recordings = data.load_recordings(config, data_config, config.DEVICE)
    return run_report(config, data_config, checkpoint, recordings, splits, output,
                      config.REPORT.EXPORT_CSV if export_csv is None else export_csv, trainer, origin)


def run_report(config, data_config, checkpoint, recordings, splits, output, export_csv=True, trainer=None,
               origin="manual"):
    """The report of prepared recordings (data.load_recordings) with their participant splits."""
    started = time.time()
    output = Path(output)
    settings = analysis_settings()
    preprocess = data_config.PREPROCESS
    for recording in recordings:
        recording["split"] = splits.get(recording["participant"], "unused") if splits else ""

    from evaluation.prediction_store import prediction_is_diff
    network = Network(load_weights(trainer or build_trainer(config), checkpoint), data_config.DATA_FORMAT)
    clip_length = preprocess.CHUNK_LENGTH
    stride = max(1, int(round(clip_length * (1 - settings["inference"]["clip_overlap"]))))
    run = {
        "settings": settings, "clip_length": clip_length, "stride": stride,
        "batch_size": config.INFERENCE.BATCH_SIZE, "is_diff": prediction_is_diff(config),
        "data_types": list(preprocess.DATA_TYPE),
        "model_key": model_fingerprint(checkpoint, config.MODEL.NAME,
                                       {**settings["inference"], "stride": stride,
                                        "format": data_config.DATA_FORMAT}),
    }
    print(f"[report] {len(recordings)} recordings, {clip_length}-frame clips every {stride} frames", flush=True)

    output.mkdir(parents=True, exist_ok=True)
    results, failures = [], {}
    within = settings["hr"]["within_bpm"]
    progress = tqdm(recordings, desc="Report", unit="recording", dynamic_ncols=True)
    for recording in progress:
        begun = time.time()
        try:
            result = analyse_recording(recording, network, run, output / "data")
        except Exception as error:  # keep going: one broken recording must not stop the report
            failures[recording["key"]] = repr(error)
            tqdm.write(f"[report] {recording['key']} FAILED: {error}")
            traceback.print_exc()
            continue
        results.append(result)
        row = recording_row(result, within)
        tqdm.write(f"[report] {recording['key']}: HR MAE {row['hr_mae_bpm']:.1f} bpm, "
                   f"RMSSD MAE {row['rmssd_mae_ms']:.1f} ms, IBI MAE {row['ibi_mae_ms']:.0f} ms "
                   f"({time.time() - begun:.0f} s)")
    progress.close()
    if not results:
        raise RuntimeError("No recording could be analysed.")
    keep = {f"{result['key']}.npz" for result in results}
    for stale in (output / "data").glob("*.npz"):  # recordings of an earlier report with another scope
        if stale.name not in keep:
            stale.unlink()

    vhrm = data_config.VHRM
    split_members = {split: sorted(p for p, s in splits.items() if s == split)
                     for split in data.SPLIT_ORDER if split in splits.values()}
    tested = sorted({r["participant"] for r in results if r["split"] == "test"})
    info = {
        "title": f"{config.MODEL.NAME} · {Path(checkpoint).stem}", "model": config.MODEL.NAME,
        "checkpoint": checkpoint, "data_path": data_config.DATA_PATH,
        "experiments": list(vhrm.EXPERIMENTS), "views": list(vhrm.VIEWS), "cache": data_config.EXP_DATA_NAME,
        "input": f"{preprocess.RESIZE.W}x{preprocess.RESIZE.H} {'+'.join(preprocess.DATA_TYPE)}, "
                 f"{clip_length}-frame clips every {stride} frames, {data_config.DATA_FORMAT}",
        "splits": bool(splits), "split_members": split_members, "test_participants": tested,
    }
    run_info = {
        "checkpoint": checkpoint, "model": config.MODEL.NAME, "config_mode": config.TOOLBOX_MODE, "origin": origin,
        "input": info["input"],
        "cache": data_config.CACHED_PATH, "data_path": data_config.DATA_PATH, "hr_source": vhrm.HR_SOURCE,
        "selection": {"experiments": list(vhrm.EXPERIMENTS), "views": list(vhrm.VIEWS),
                      "labels": list(vhrm.LABELS), "participants": list(vhrm.PARTICIPANTS)},
        "splits": split_members, "clip_length": clip_length, "stride": stride, "settings": settings,
        "versions": _versions(), "seconds": round(time.time() - started),
    }
    print("[report] Writing figures ...", flush=True)
    sets = {"All recordings": (output, write_outputs(output, results, info, failures, run_info, settings,
                                                     export_csv=export_csv))}

    # The same outputs on the test participants only, with copies of their data/ files.
    test_results = [result for result in results if result["split"] == "test"]
    if test_results:
        test_output = output / TEST_SET
        (test_output / "data").mkdir(parents=True)
        for result in test_results:
            shutil.copy2(output / "data" / f"{result['key']}.npz", test_output / "data")
        test_failures = {key: error for key, error in failures.items()
                         if next((r["split"] for r in recordings if r["key"] == key), "") == "test"}
        sets["Test participants only"] = (test_output, write_outputs(
            test_output, test_results, info, test_failures, run_info, settings, scope="test",
            export_csv=export_csv))

    for name, (folder, global_metrics) in sets.items():
        overall = global_metrics[global_metrics.group == "All recordings"].set_index("metric_key")
        print(f"[report] {name}: HR MAE {overall.mae.get('hr', np.nan):.2f} bpm, "
              f"RMSSD MAE {overall.mae.get('rmssd', np.nan):.1f} ms, RR MAE {overall.mae.get('ibi', np.nan):.0f} ms "
              f"({overall.recordings.get('hr', 0)} recordings)", flush=True)
    print(f"[report] Done in {fmt_seconds(time.time() - started)}: {output / 'REPORT.md'}", flush=True)
    return output


def merge_cross_validation(root, export_csv=True):
    """<root>/report/cross_validation: the test participants of every fold, each analysed by its own fold's checkpoint.

    Every participant is tested in one fold, so this is the report where no participant was seen. Per fold it uses
    the newest report that has test participants, preferring the one main.py wrote after the training. Returns the
    folder, or None when no fold has a report.
    """
    root = Path(root)
    chosen = []
    for fold_dir in sorted(root.glob("fold_*")):
        candidates = []
        for marker in fold_dir.glob(f"*/{REPORT_DIR}/*/{TEST_SET}/run_config.json"):
            folder = marker.parent.parent
            info = json.loads((folder / "run_config.json").read_text(encoding="utf-8"))
            candidates.append(((info.get("origin") == "after_training", marker.stat().st_mtime), folder, info))
        if candidates:
            _, folder, info = max(candidates, key=lambda item: item[0])
            chosen.append((fold_dir.name.removeprefix("fold_"), folder, info))
    if not chosen:
        return None
    output = root / REPORT_DIR / CV_REPORT
    shutil.rmtree(output / "data", ignore_errors=True)
    (output / "data").mkdir(parents=True)
    results, folds = [], []
    for fold, folder, info in chosen:
        folds.append({"fold": fold, "report": str(folder), "checkpoint": info.get("checkpoint")})
        for path in sorted((folder / TEST_SET / "data").glob("*.npz")):
            result = load_result(path)
            result["fold"] = fold
            if any(other["key"] == result["key"] for other in results):  # tested in two folds (custom groups)
                result["key"] = f"{result['key']}_fold{fold}"
            with np.load(path) as saved:
                save_result(result, str(saved["model_fingerprint"]), output / "data")
            results.append(result)
    first = chosen[0][2]
    settings = first["settings"]
    participants = sorted({r["participant"] for r in results})
    selection = first.get("selection", {})
    info = {
        "title": f"{first['model']} · cross-validation, {len(chosen)} folds", "model": first["model"],
        "checkpoint": "; ".join(f"fold {f['fold']}: {Path(str(f['checkpoint'])).name}" for f in folds),
        "data_path": first.get("data_path", ""), "experiments": selection.get("experiments", []),
        "views": selection.get("views", []), "cache": Path(str(first.get("cache", ""))).name,
        "input": first.get("input", ""), "splits": True, "split_members": {"test": participants},
        "test_participants": participants,
    }
    run_info = {key: first.get(key) for key in ("model", "cache", "data_path", "hr_source", "selection", "clip_length",
                                                 "stride", "settings", "input")}
    run_info.update({"checkpoint": None, "origin": "cross_validation", "folds": folds, "splits": {"test": participants},
                     "versions": _versions()})
    write_outputs(output, results, info, {}, run_info, settings, scope="cv", export_csv=export_csv)
    print(f"[report] Cross-validation report of {len(chosen)} folds ({len(participants)} test participants, "
          f"{len(results)} recordings): {output / 'REPORT.md'}", flush=True)
    return output


def fmt_seconds(seconds):
    minutes, seconds = divmod(int(seconds), 60)
    return f"{minutes} min {seconds} s" if minutes else f"{seconds} s"
