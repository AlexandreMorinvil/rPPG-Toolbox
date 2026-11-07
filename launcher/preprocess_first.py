"""Preprocessing first: a batch preprocesses every cache it needs before its first training starts.

The batch's data splits that preprocess are planned once for the whole batch:

- A file list (DataFileLists/<name>.csv) is one task. Splits of several configs that write the same file list with the
  same settings are preprocessed once, by the first config that needs it.
- Tasks are grouped by cache folder: one preprocessing job per folder, because two runs filling one folder at the same
  time would write the same clips (and, for vHRM, the same manifest). The job runs main.py --preprocess_only once per
  config that has tasks in the folder.
- After the preprocessing stage, a training reuses the cache of each split that was preprocessed for it
  (DO_PREPROCESS: false). If the preprocessing of a split did not succeed, the training preprocesses it itself, as
  without this option.

Splits that write the same file list with different settings (for example two vHRM label types: the file list name does
not include them) cannot share it. Incremental vHRM splits are preprocessed in the stage (the videos are decoded once)
and each training rewrites its own list from the cache ("refresh"); other datasets keep their preprocessing in the job.
"""

import copy
import hashlib
import json
import os

from . import schema

# Split settings that do not change what preprocessing writes (worker counts, progress display, decoder choice, the
# tensor layout applied when loading, and the test-only flag config.py derives from the training labels).
RUNTIME_KEYS = frozenset({
    "DO_PREPROCESS", "DATA_FORMAT", "PREPROCESS.NUM_WORKERS", "VHRM.SHOW_VIDEO_PROGRESS", "VHRM.VIDEO_DECODER",
    "VHRM.FFMPEG_PATH", "VHRM.FFPROBE_PATH", "VHRM.PREDICTION_IS_DIFF",
})
# The loaders treat training and validation clips alike, and test clips differently (vHRM keeps the test clips
# without a face), so a test split never shares the file list of a training split.
ROLES = {"TRAIN.DATA": "fit", "VALID.DATA": "fit", "TEST.DATA": "test", "UNSUPERVISED.DATA": "unsupervised"}


def split_identity(values, prefix):
    """Digest of everything in a split's settings that changes its cache or its file list."""
    start = prefix + "."
    settings = {key[len(start):]: value for key, value in values.items()
                if key.startswith(start) and key[len(start):] not in RUNTIME_KEYS}
    settings["_role"] = ROLES[prefix]
    return hashlib.sha1(json.dumps(settings, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def preprocess_splits(effective, derived):
    """The splits of a run that preprocess: {prefix: {cache, file_list, name}} (host paths, as validation derives)."""
    splits = {}
    for prefix in schema.active_splits(effective):
        locations = derived.get(prefix) or {}
        if effective.get(prefix + ".DO_PREPROCESS") and locations.get("cache_dir_host") and \
                locations.get("file_list_host"):
            splits[prefix] = {"cache": locations["cache_dir_host"], "file_list": locations["file_list_host"],
                              "name": locations.get("exp_data_name") or os.path.basename(locations["cache_dir_host"])}
    return splits


def _incremental_vhrm(values, prefix):
    return values.get(prefix + ".DATASET") == "vHRM" and bool(values.get(prefix + ".VHRM.INCREMENTAL_PREPROCESS"))


def plan(entries):
    """Plan the preprocessing stage of a batch.

    entries: the configs to launch, in launch order: dicts with path, name, values (effective) and splits (from
    preprocess_splits).
    Returns caches (one preprocessing job each, in the order the batch first needs them: cache, key, name, steps
    [{config, splits}], lists, configs), splits ({config path: {prefix: {cache, action, part, file_list}}}: what each
    training does after the stage; part is the step of the cache's job that preprocesses it) and counts.
    """
    groups = {}  # file list -> [(entry, prefix)], in launch order
    for entry in entries:
        for prefix, split in entry["splits"].items():
            groups.setdefault(os.path.normcase(split["file_list"]), []).append((entry, prefix))

    caches, assignments = {}, {}
    counts = {"splits": 0, "lists": 0, "duplicates": 0, "in_jobs": 0, "refresh": 0}
    for members in groups.values():
        counts["splits"] += len(members)
        identities = {split_identity(entry["values"], prefix) for entry, prefix in members}
        if len(identities) == 1:
            action = "reuse"
        elif all(_incremental_vhrm(entry["values"], prefix) for entry, prefix in members):
            action = "refresh"
            counts["refresh"] += len(members)
        else:
            counts["in_jobs"] += len(members)
            continue  # each job preprocesses its split itself, as without this option
        counts["lists"] += 1
        counts["duplicates"] += len(members) - 1
        producer, prefix = members[0]
        split = producer["splits"][prefix]
        key = os.path.normcase(split["cache"])
        cache = caches.setdefault(key, {"cache": split["cache"], "key": key, "name": split["name"], "steps": [],
                                        "lists": 0, "configs": [], "first": len(caches)})
        cache["lists"] += 1
        step = next((index for index, item in enumerate(cache["steps"]) if item["config"] == producer["path"]), None)
        if step is None:
            step = len(cache["steps"])
            cache["steps"].append({"config": producer["path"], "name": producer["name"], "splits": []})
        cache["steps"][step]["splits"].append(prefix)
        for entry, member_prefix in members:
            if entry["name"] not in cache["configs"]:
                cache["configs"].append(entry["name"])
            assignments.setdefault(entry["path"], {})[member_prefix] = {
                "cache": key, "action": action, "part": step,
                "file_list": entry["splits"][member_prefix]["file_list"]}
    ordered = sorted(caches.values(), key=lambda cache: cache.pop("first"))
    order = list(schema.SPLITS).index  # main.py preprocesses the splits of a run in this order
    for cache in ordered:
        for step in cache["steps"]:
            step["splits"].sort(key=order)
    assignments = {path: dict(sorted(items.items(), key=lambda item: order(item[0])))
                   for path, items in assignments.items()}
    return {"caches": ordered, "splits": assignments, "counts": counts}


def step_values(values, splits):
    """The config of one preprocessing step: only these splits preprocess, and nothing is logged to W&B."""
    values = copy.deepcopy(values)
    for prefix in schema.SPLITS:
        if prefix + ".DO_PREPROCESS" in values:
            values[prefix + ".DO_PREPROCESS"] = prefix in splits
    if "WANDB.ENABLED" in values:
        values["WANDB.ENABLED"] = False
    return values


def section_names(splits):
    return ", ".join(schema.SPLITS[prefix]["section"] for prefix in splits)
