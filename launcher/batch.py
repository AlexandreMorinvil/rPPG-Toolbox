"""Batches: launch many saved configs at once, with one round of pre-launch checks for all of them.

Every config goes through the single-run checks (preflight.run_checks). The questions they share (existing results,
existing caches, W&B login, GPU, Docker image, run order) are asked once for the whole batch, and the problems are
grouped by message, so an issue common to many configs shows once with the configs it affects.
"""

import os
import posixpath
from collections import Counter

import yaml

from . import configs, preflight, schema, settings

MAX_CONFIGS = 400
# Questions answered once for the batch, in display order; "cache" stands for the cache:<split> question of every split.
SHARED_QUESTIONS = ("build", "gpu", "wandb", "outputs", "cache")
LAUNCHABLE = ("ready", "warn")
OUTPUT_KEYS = ("model_dir_host", "test_outputs_host", "outputs_host")
SKIP_OPTION = {"value": "skip", "label": "Skip these configs",
               "description": "Launch only the configs whose output folder is still empty."}


# Answers that change the config (preflight.apply_answers): a config only gets them for the questions it asked, as when
# it is launched alone. A new sub-folder for a config without results, or "reuse" for a split without a cache, is wrong.
SCOPED_QUESTIONS = ("outputs", "cache", "wandb")


def own_answers(answers, asked):
    """The batch answers to the questions one config asked (asked: ids such as outputs, cache:TRAIN.DATA, wandb)."""
    own = {}
    for question in asked:
        key = "cache" if question.startswith("cache:") else question
        if key in SCOPED_QUESTIONS and answers.get(key) not in (None, "", "skip"):
            own[question] = answers[key]
    return own


def common_folder(config_paths):
    """Deepest folder holding every config (job names are the config paths relative to it)."""
    folders = [posixpath.dirname(path) for path in config_paths]
    try:
        return posixpath.commonpath(folders) if folders else ""
    except ValueError:  # absolute and relative paths mixed
        return ""


def short_name(path, folder):
    relative = posixpath.relpath(path, folder) if folder else path
    return relative.rsplit(".", 1)[0]


def default_name(folder):
    for prefix in ("docker/configs/", "configs/"):
        if folder.startswith(prefix):
            return folder[len(prefix):]
    return folder or "Batch"


def _first_line(text):
    return str(text or "").strip().split("\n", 1)[0][:200]


def _folders(outputs):
    return {os.path.normcase(str(outputs[key])) for key in OUTPUT_KEYS if outputs.get(key)}


def _caches(effective, derived):
    """Cache folders a run preprocesses into (two runs filling one cache must not preprocess at the same time)."""
    folders = []
    for prefix in schema.active_splits(effective):
        cache = (derived.get(prefix) or {}).get("cache_dir_host")
        if effective.get(prefix + ".DO_PREPROCESS") and cache:
            folders.append(os.path.normcase(cache))
    return sorted(set(folders))


def _summary(values):
    mode = values.get("TOOLBOX_MODE")
    if mode == "unsupervised_method":
        return "+".join(values.get("UNSUPERVISED.METHOD") or []), values.get("UNSUPERVISED.DATA.DATASET", "")
    datasets = [values.get("TRAIN.DATA.DATASET")] if mode == "train_and_test" else []
    test = values.get("TEST.DATA.DATASET")
    if test and test not in datasets:
        datasets.append(test)
    return values.get("MODEL.NAME", ""), " → ".join(d for d in datasets if d)


def _check_config(entry, path, backend, answers, secrets, manager):
    """Single-run checks of one config; fills entry and returns (checks, run) where run is what launching needs."""
    try:
        data = configs.load_config(path)
    except (ValueError, OSError, yaml.YAMLError) as error:
        entry["errors"].append(f"Cannot read the config: {error}")
        return [], None
    values = data["values"]
    entry["mode"] = values.get("TOOLBOX_MODE")
    entry["model"], entry["datasets"] = _summary(values)
    entry["description"] = _first_line(values.get("DESCRIPTION"))
    if data["type_errors"]:
        entry["errors"] += [f"{key}: {message}" for key, message in data["type_errors"].items()]
        return [], None
    if schema.cross_validation_enabled(values):
        entry["errors"].append("Cross-validation configs run as one job per fold: launch it from the editor.")
        return [], None
    common = {key: value for key, value in answers.items() if key not in SCOPED_QUESTIONS}
    try:
        # The questions asked depend on the config, not on the answers: check once, then again with its own answers.
        result = preflight.run_checks(values, data["extras"], backend, common, secrets, manager)
        own = own_answers(answers, [check["id"] for check in result["checks"] if check.get("options")])
        if own:
            result = preflight.run_checks(values, data["extras"], backend, {**common, **own}, secrets, manager)
    except Exception as error:  # one broken config must not hide the checks of the others
        entry["errors"].append(f"The checks failed: {type(error).__name__}: {error}")
        return [], None
    effective, derived = result["effective_values"], result["validation"]["derived"]
    outputs = derived.get("outputs", {})
    entry["log_path"] = outputs.get("log_path_host") or effective.get("LOG.PATH")
    run = {"values": effective, "extras": data["extras"], "outputs": outputs, "answers": result["answers"],
           "caches": _caches(effective, derived)}
    return result["checks"], run


def _shared_question(key, items, answers, total):
    """One batch question from the same question asked by several configs."""
    first = items[0][1]
    options = list(first["options"]) + ([SKIP_OPTION] if key == "outputs" else [])
    names = [entry["name"] for entry, _ in items]
    share = "" if len(items) == total else f" ({len(items)} of {total} configs)"
    title, message, details = first["title"], first["message"] + share, []
    if key == "outputs":
        title = "Existing results"
        message = f"{len(items)} config(s) already have results in their output folder."
        details = [f"{entry['name']}: {entry.get('log_path') or ''}" for entry, _ in items]
    elif key == "cache":
        title = "Existing caches"
        message = f"{len(items)} config(s) can reuse a preprocessed cache that already exists."
        details = [f"{entry['name']} · {check['title']}: {check['message']}" for entry, check in items]
    item = {"id": key, "title": title, "message": message, "options": options, "default": first.get("default"),
            "details": details, "configs": names}
    if first.get("secret"):
        item["secret"] = first["secret"]
    chosen = answers.get(key)
    item["answer"] = chosen if chosen in {option["value"] for option in options} else None
    problem = next((check for _, check in items if check["status"] == "error"), None)
    pending = next((check for _, check in items if check["status"] == "question"), None)
    if problem:
        item.update(status="error", message=problem["message"])
    elif item["answer"] is None or (key in ("wandb", "build", "gpu") and pending):
        # Per-config questions are only open while the batch answer is missing (or the W&B key is not typed yet).
        item["status"] = "question"
        if pending and item["answer"] is not None:
            item["message"] = pending["message"]
    else:
        item["status"] = "answered"
        if key == "wandb":
            item["message"] = first["message"]
    return item


def run_checks(config_paths, backend, answers, secrets, manager, exclude=()):
    """Pre-launch checks of a batch: the single-run checks of every config, with shared questions asked once.

    Returns the batch-level checks, one entry per config (status ready / warn / blocked / skipped / busy), the
    problems grouped by message, and, under "runs", what launching each config needs (internal: not for the client).
    """
    config_paths = list(dict.fromkeys(str(path) for path in config_paths or [] if path))
    if not config_paths:
        raise ValueError("Choose at least one config.")
    if len(config_paths) > MAX_CONFIGS:
        raise ValueError(f"A batch holds at most {MAX_CONFIGS} configs.")
    if backend not in ("docker", "local"):
        raise ValueError("Backend must be docker or local.")
    answers = dict(answers or {})
    exclude = set(exclude or ())
    prefs = settings.load()
    folder = common_folder(config_paths)
    total = len(config_paths)

    entries, runs, shared, general = [], {}, {}, {}
    for index, path in enumerate(config_paths):
        entry = {"path": path, "name": short_name(path, folder), "index": index, "excluded": path in exclude,
                 "mode": None, "model": "", "datasets": "", "description": "", "log_path": None,
                 "errors": [], "warnings": [], "status": "ready"}
        entries.append(entry)
        checks, run = _check_config(entry, path, backend, answers, secrets, manager)
        if run:
            runs[path] = run
        for check in checks:
            key = "cache" if check["id"].startswith("cache:") else check["id"]
            if key in SHARED_QUESTIONS and check.get("options"):
                shared.setdefault(key, []).append((entry, check))
            elif key == "validation":
                target = entry["errors"] if check["status"] == "error" else entry["warnings"]
                target += check.get("details") or []
            elif key in ("policy", "capacity"):
                continue  # replaced by the batch's own run-order question
            elif key in ("disk", "disk-cache"):
                entry["warnings"].append(check["message"])
            else:
                if check["status"] == "error":
                    entry["errors"].append(f"{check['title']}: {check['message']}")
                general.setdefault((check["id"], check["status"], check["title"], check["message"]), []).append(
                    entry["name"])

    # Config status: own problems first, then what the batch decides (skip, already running, same outputs).
    skipped = {entry["path"] for entry, _ in shared.get("outputs", [])} if answers.get("outputs") == "skip" else set()
    claimed = {}
    for entry in entries:
        run = runs.get(entry["path"])
        if entry["errors"] or run is None:
            entry["status"] = "blocked"
            continue
        if entry["path"] in skipped:
            entry["status"] = "skipped"
            continue
        owner = manager.active_owner(entry["path"], _folders(run["outputs"])) if manager is not None else None
        if owner:
            entry["status"] = "busy"
            entry["errors"].append(f"Already queued or running in the launcher: {owner['name']} ({owner['status']}).")
            continue
        clash = next((claimed[folder_] for folder_ in _folders(run["outputs"]) if folder_ in claimed), None)
        if clash:
            entry["status"] = "blocked"
            entry["errors"].append(f"Writes to the same output folder as {clash}.")
            continue
        if not entry["excluded"]:
            claimed.update({folder_: entry["name"] for folder_ in _folders(run["outputs"])})
        entry["status"] = "warn" if entry["warnings"] else "ready"
    launchable = [entry for entry in entries if entry["status"] in LAUNCHABLE and not entry["excluded"]]

    checks = preflight.Checks()
    for (id_, status, title, message), names in general.items():
        share = "" if len(names) == total else f" ({len(names)} of {total} configs)"
        checks.add(id_, title, status, message + share, configs=names)
    for key in SHARED_QUESTIONS:
        if shared.get(key):
            checks.items.append(_shared_question(key, shared[key], answers, total))
    _order_question(checks, launchable, runs, backend, answers, prefs, manager)
    if answers.get("wandb") == "key":
        checks.add("wandb_memory", "W&B key kept in memory", "warn",
                   "The key typed here is never written to disk: if the launcher restarts before the last job has "
                   "started, the jobs still queued fail. Logging in once with 'wandb login' (then 'Use the API key "
                   "from your W&B login') or offline mode avoids this.")

    problems = {}
    for entry in entries:
        for level, messages in (("error", entry["errors"]), ("warn", entry["warnings"])):
            for message in messages:
                problems.setdefault((level, message), []).append(entry["name"])
    issues = sorted(({"status": level, "message": message, "configs": names}
                     for (level, message), names in problems.items()),
                    key=lambda issue: (issue["status"] != "error", -len(issue["configs"]), issue["message"]))

    blocking = [check for check in checks.items if check["status"] in ("error", "question")]
    return {
        "checks": checks.items,
        "configs": entries,
        "issues": issues,
        "counts": dict(Counter(entry["status"] for entry in entries)),
        "launchable": len(launchable),
        "can_launch": not blocking and bool(launchable),
        "folder": folder,
        "name": default_name(folder),
        "answers": answers,
        "runs": runs,
    }


def _order_question(checks, launchable, runs, backend, answers, prefs, manager):
    """How the jobs of the batch share the GPU: one after another (default) or several at once."""
    if not launchable:
        return
    run = runs[launchable[0]["path"]]
    gpu = preflight.gpu_index(run["values"], backend, run["answers"])
    limit = prefs["max_parallel_jobs"]
    where = f"GPU {gpu}" if gpu is not None else "the CPU"
    message = f"{len(launchable)} job(s) to queue on {where}, in the listed order."
    busy = manager.jobs_on_gpu(gpu) if manager is not None and gpu is not None else []
    if busy:
        message += f" Already using this GPU: {', '.join(job['name'] for job in busy[:5])}" + \
                   (f" and {len(busy) - 5} more." if len(busy) > 5 else ".")
    options = [{"value": "queue", "label": "One after another",
                "description": f"Each job starts when the previous one on {where} has finished."},
               {"value": "parallel", "label": f"Up to {limit} at once (Settings → Jobs)",
                "description": "Jobs share the GPU memory and can run out of VRAM."}]
    checks.question("policy", "Run order", message, options, answers, default=prefs["gpu_policy"])
    if answers.get("policy") != "parallel":
        return
    owners = {}
    for entry in launchable:
        for cache in runs[entry["path"]]["caches"]:
            owners.setdefault(cache, []).append(entry["name"])
    shared = {cache: names for cache, names in owners.items() if len(names) > 1}
    if shared:
        checks.add("caches", "Shared caches", "info",
                   f"{len(shared)} preprocessed cache folder(s) are filled by several jobs. Jobs that preprocess the "
                   "same cache never preprocess at the same time: each waits until the previous one has finished "
                   "preprocessing.",
                   details=[f"{os.path.basename(cache)}: {', '.join(names)}" for cache, names in shared.items()])


def public(result):
    """The checks as sent to the client (without the effective configs)."""
    return {key: value for key, value in result.items() if key != "runs"}
