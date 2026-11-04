"""Cross-validation runs: fold plans, pre-launch checks, fold configs and merged summaries.

The folds come from tools/vhrm_cross_validation.py --plan-json, run with the local Python of the Settings
tab: listing the participants reads the dataset with the toolbox's vHRM loader, which needs the toolbox
environment (PyTorch, OpenCV) rather than the launcher's. Merging the fold results uses the same tool, so
the launcher and the command line build the same folds and summaries. This holds for Docker runs too.
"""

import copy
import csv
import hashlib
import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

from cv_folds import fold_overrides
from . import configs, paths, preflight, probes, schema, settings, validation

TOOL = paths.TOOLBOX_ROOT / "tools" / "vhrm_cross_validation.py"
PLAN_TTL = 300
PLAN_TIMEOUT = 600
MERGE_TIMEOUT = 1800
SUMMARY_TABLES = ("by_fold", "by_participant", "by_movement", "by_view", "by_experiment")
_plans = {}
_plan_lock = threading.Lock()


def _python():
    python = settings.load()["local_python"]
    if not python or not (Path(python).is_file() or shutil.which(python)):
        raise ValueError("Cross-validation lists the participants and merges the results with the local Python "
                         f"environment of the toolbox (Settings → Local Python), also for Docker runs: {python!r} "
                         "was not found.")
    return python


def _run_tool(arguments, timeout):
    """Run tools/vhrm_cross_validation.py; returns (exit code, output)."""
    environment = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1", "MPLBACKEND": "Agg"}
    try:
        completed = subprocess.run([_python(), str(TOOL), *arguments], cwd=str(paths.TOOLBOX_ROOT),
                                   capture_output=True, text=True, encoding="utf-8", errors="replace",
                                   timeout=timeout, env=environment, stdin=subprocess.DEVNULL,
                                   creationflags=probes.NO_WINDOW)
    except subprocess.TimeoutExpired:
        return 124, f"tools/vhrm_cross_validation.py did not finish within {timeout} s."
    except OSError as error:
        return 127, f"Could not start the local Python: {error}"
    return completed.returncode, (completed.stdout or "") + (completed.stderr or "")


def _tail(text, lines=12):
    return "\n".join(line for line in text.strip().splitlines()[-lines:])


def plan(values, extras=None, backend="local", refresh=False):
    """Participant pool and folds of a cross-validation config (cached for a few minutes)."""
    if not schema.cross_validation_enabled(values):
        raise ValueError("Cross-validation is not enabled for this config.")
    host_values = copy.deepcopy(values)
    # The tool runs on the host: read the dataset where the host sees it.
    host, problem = paths.config_host_path(values.get("TRAIN.DATA.DATA_PATH"), backend)
    if problem or host is None:
        raise ValueError(f"Training data folder: {problem or 'not set'}")
    host_values["TRAIN.DATA.DATA_PATH"] = str(host)
    relevant = {key: value for key, value in host_values.items()
                if key.startswith(("TRAIN.DATA.", "CROSS_VALIDATION.")) or key in ("TOOLBOX_MODE", "VALID.DATA.DATASET")}
    key = hashlib.sha1(json.dumps([relevant, extras or {}], sort_keys=True, default=str).encode()).hexdigest()[:16]
    with _plan_lock:
        cached = _plans.get(key)
        if cached and not refresh and time.time() - cached[0] < PLAN_TTL:
            return copy.deepcopy(cached[1])
        folder = paths.CV_DIR / "plans"
        folder.mkdir(parents=True, exist_ok=True)
        config_path, plan_path = folder / f"{key}.yaml", folder / f"{key}.json"
        config_path.write_text(configs.dump_yaml(configs.build_tree(host_values, extras)), encoding="utf-8")
        plan_path.unlink(missing_ok=True)
        try:
            code, output = _run_tool(["--config", str(config_path), "--plan-json", str(plan_path)], PLAN_TIMEOUT)
            if code != 0 or not plan_path.is_file():
                raise ValueError("Could not plan the folds (tools/vhrm_cross_validation.py --plan-json):\n" +
                                 _tail(output))
            result = json.loads(plan_path.read_text(encoding="utf-8"))
        finally:
            config_path.unlink(missing_ok=True)
            plan_path.unlink(missing_ok=True)
        _plans[key] = (time.time(), result)
        return copy.deepcopy(result)


def fold_runs(values, folds, backend, wandb_group):
    """Config values and output folders of every fold; the fold configs are ordinary single runs."""
    root = values["LOG.PATH"]
    runs = []
    for fold in folds:
        fold_values = copy.deepcopy(values)
        fold_values["CROSS_VALIDATION.ENABLED"] = False
        fold_values.update(fold_overrides(fold, values.get("TRAIN.MODEL_FILE_NAME"),
                                          validation._join(root, f"fold_{fold['name']}"),
                                          wandb_group if values.get("WANDB.ENABLED") else None))
        runs.append({**fold, "values": fold_values, "outputs": validation.output_paths(fold_values, backend)})
    return runs


def _sizes(folds, key):
    sizes = sorted({len(fold[key]) for fold in folds})
    return str(sizes[0]) if len(sizes) == 1 else f"{sizes[0]}–{sizes[-1]}"


def run_checks(values, extras, backend, answers, secrets, manager, group_id=None, refresh=False):
    """Pre-launch checks of a cross-validation: the usual checks of the base config, the folds, and one
    output-folder question for the whole cross-validation."""
    answers = dict(answers or {})
    result = preflight.run_checks(values, extras, backend, answers, secrets, manager)
    checks = preflight.Checks()
    effective = result["effective_values"]
    locations = validation.cross_validation_locations(values, backend)
    existing = validation.existing_cv_folds(locations["root_host"])
    if existing:
        checks.question("outputs", "Existing cross-validation results",
                        f"{locations['root_host']} already has fold results ({', '.join(existing[:6])}"
                        f"{'…' if len(existing) > 6 else ''}). The merged summary reads every fold_* folder, "
                        "so mixing two runs mixes their results.",
                        [{"value": "new_folder", "label": "Write to a new timestamped sub-folder",
                          "description": "LOG.PATH/run_<date-time>/fold_<name>"},
                         {"value": "overwrite", "label": "Overwrite / add to the existing folder",
                          "description": "Only for re-running the same cross-validation."}],
                        answers, default="new_folder")
    cv = {"root": effective["LOG.PATH"], "root_host": validation.cross_validation_locations(
        effective, backend)["root_host"], "scheme": values.get("CROSS_VALIDATION.SCHEME")}
    runs = []
    try:
        folds = plan(values, extras, backend, refresh)
        cv.update({key: folds[key] for key in ("pool", "units", "folds", "warnings")})
        group = values.get("WANDB.GROUP") or f"cv_{group_id or '<launch id>'}"
        runs = fold_runs(effective, folds["folds"], backend, group)
        empty = [run["name"] for run in runs if not run["train"]]
        if empty:
            checks.add("cv_plan", "Cross-validation folds", "error",
                       f"No training participants left in fold(s) {', '.join(empty)}.")
        else:
            label = schema.CV_SCHEME_LABELS.get(cv["scheme"], cv["scheme"])
            checks.add("cv_plan", "Cross-validation folds", "warn" if folds["warnings"] else "ok",
                       f"{len(runs)} folds over {len(folds['units'])} participants ({label}): each fold tests "
                       f"{_sizes(runs, 'test')}, validates on {_sizes(runs, 'valid')} and trains on "
                       f"{_sizes(runs, 'train')} participant folders. The folds run one after another.",
                       details=folds["warnings"])
        # Folds differ from each other only by participant lists and names: checking one covers the shape.
        report = validation.validate(runs[0]["values"], extras, backend)
        if report["errors"]:
            checks.add("cv_fold", "Fold configuration", "error",
                       f"The config of fold {runs[0]['name']} has {len(report['errors'])} problem(s).",
                       details=[item["message"] for item in report["errors"]])
        clash = [run["name"] for run in runs if manager is not None and
                 manager.checkpoint_folder_busy(run["outputs"].get("model_dir_host"))]
        if clash:
            checks.add("cv_busy", "Fold outputs", "error",
                       f"Active jobs write to the checkpoint folder of fold(s) {', '.join(clash)}.")
    except ValueError as error:
        checks.add("cv_plan", "Cross-validation folds", "error", str(error))
    checks.items.extend(result["checks"])
    blocking = [check for check in checks.items if check["status"] in ("error", "question")]
    return {**result, "checks": checks.items, "can_launch": not blocking and bool(runs), "cv": cv,
            "fold_runs": runs, "answers": {**result["answers"], **answers}}


def _number(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return text


def _csv_rows(path, limit=500):
    try:
        with open(path, newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            return [{key: _number(value) for key, value in row.items()} for _, row in zip(range(limit), reader)]
    except OSError:
        return None


def read_summary(root_host, tables=False):
    """Merged results of a cross-validation (cv_summary/*.csv), or None before the first merge."""
    if not root_host:
        return None
    folder = Path(root_host) / "cv_summary"
    overall = _csv_rows(folder / "overall_summary.csv", 1)
    if not overall:
        return None
    result = {"overall": overall[0], "modified": (folder / "overall_summary.csv").stat().st_mtime}
    if tables:
        result["tables"] = {name: _csv_rows(folder / f"{name}.csv") or [] for name in SUMMARY_TABLES}
    return result


def merge(root_host):
    """Merge the fold results into <root>/cv_summary with the command-line tool; returns (ok, output)."""
    if not root_host or not Path(root_host).is_dir():
        return False, f"The cross-validation folder does not exist yet: {root_host}"
    try:
        code, output = _run_tool(["--output-root", str(root_host), "--aggregate-only"], MERGE_TIMEOUT)
    except ValueError as error:
        return False, str(error)
    return code == 0, _tail(output, 20)
