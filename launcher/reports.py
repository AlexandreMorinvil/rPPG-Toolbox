"""Analysis reports (evaluation/vhrm_report): finding them in experiment folders and preparing report jobs.

A report of a checkpoint lives in <experiment>/report/<checkpoint name>/ (REPORT.md, summary/, mosaics/,
participants/, data/, data_csv/, test_set/). The launcher reads its CSV tables and PNGs; it never imports the
report code (that needs PyTorch), it runs tools/vhrm_report.py as a job.
"""

import csv
import json
import re
from pathlib import Path

from . import configs, paths, preflight, results, settings, validation

REPORT_DIR = results.REPORT_DIR
TEST_SET = "test_set"
HEADLINE = {"hr": ("hr_mae_bpm", "within_5bpm_pct"), "rmssd": ("rmssd_mae_ms", None), "sdnn": ("sdnn_mae_ms", None),
            "ibi": ("ibi_mae_ms", None)}
NAME_ITEM = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._+-]*$")


def slug(value):
    """vHRM name matching (VHRMLoader.slug): lower case, other characters become '-'."""
    return re.sub(r"[^a-z0-9]+", "-", str(value).strip().lower()).strip("-")


def _writable_in_docker(host_path):
    best = None
    for mount in paths.docker_mounts():
        if paths.is_within(host_path, [mount["host"]]) and (best is None or len(mount["host"].parts) > len(best["host"].parts)):
            best = mount
    return best is not None and not best["read_only"]


def report_name(checkpoint, model_file_name):
    """The checkpoint name without the run's checkpoint prefix (Epoch16), as evaluation/vhrm_report.report_folder."""
    name = Path(checkpoint).stem
    prefix = f"{model_file_name}_"
    if model_file_name and name.startswith(prefix) and len(name) > len(prefix):
        return name[len(prefix):]
    return name


def report_location(checkpoint_host, values, backend):
    """Host folder of a checkpoint's report: an evaluation's own folder; for a training, next to the checkpoint's
    PreTrainedModels folder, else in the config's experiment."""
    checkpoint = Path(checkpoint_host)
    name = report_name(checkpoint, values.get("TRAIN.MODEL_FILE_NAME") or "")
    training = values.get("TOOLBOX_MODE") == "train_and_test"
    if training and checkpoint.parent.name == (values.get("MODEL.MODEL_DIR") or "PreTrainedModels"):
        return checkpoint.parent.parent / REPORT_DIR / name
    outputs = validation.output_paths(values, backend)
    folder = outputs.get("model_dir_host" if training else "test_outputs_host")
    if not folder:
        raise ValueError("The report folder of this config is not on this machine (check the Docker mounts).")
    return Path(folder).parent / REPORT_DIR / name


def _number(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def headline(folder):
    """{metric: {mae, within_pct, recordings, samples}} of the 'All recordings' rows of summary/global_metrics.csv."""
    table = Path(folder) / "summary" / "global_metrics.csv"
    if not table.is_file():
        return None
    found = {}
    with table.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row.get("group") == "All recordings" and row.get("metric_key") in HEADLINE:
                found[row["metric_key"]] = {"mae": _number(row.get("mae")), "recordings": _number(row.get("recordings")),
                                            "samples": _number(row.get("samples")),
                                            "within_pct": _number(row.get("within_5bpm_pct"))}
    return found


def list_reports(experiment):
    """Reports of an experiment folder, newest first."""
    reports = []
    for config_file in sorted((Path(experiment) / REPORT_DIR).glob("*/run_config.json")):
        folder = config_file.parent
        try:
            info = json.loads(config_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        test_set = (folder / TEST_SET / "run_config.json").is_file()
        reports.append({
            "name": folder.name, "path": str(folder), "title": info.get("title", folder.name),
            "checkpoint": info.get("checkpoint"), "recordings": len(info.get("recordings") or []),
            "failures": info.get("failures") or {}, "splits": info.get("splits") or {},
            "selection": info.get("selection") or {}, "modified": config_file.stat().st_mtime,
            "test_set": test_set, "summary": headline(folder), "test_summary": headline(folder / TEST_SET) if test_set else None,
        })
    return sorted(reports, key=lambda item: item["modified"], reverse=True)


def _files(folder, pattern):
    return [{"name": path.relative_to(folder).as_posix(), "path": str(path)} for path in sorted(folder.glob(pattern))]


def detail(path):
    """One report (or its test_set/): tables, figures by section, and its main files."""
    folder = results.check_path(path)
    if not (folder / "run_config.json").is_file():
        raise ValueError("Not an analysis report folder")
    tables = {}
    for key, name in (("global", "global_metrics"), ("matrix", "label_split_mae"), ("recordings", "per_recording_metrics")):
        table = folder / "summary" / f"{name}.csv"
        if table.is_file():
            tables[key] = results._read_table(table, 5000)
    participants = {}
    for item in _files(folder, "participants/*/*.png"):
        participants.setdefault(item["name"].split("/")[1], []).append(item)
    return {
        "path": str(folder), "scope": "test" if folder.name == TEST_SET else "all",
        "run_config": json.loads((folder / "run_config.json").read_text(encoding="utf-8")),
        "tables": tables, "summary_figures": _files(folder, "summary/*.png"), "mosaics": _files(folder, "mosaics/*.png"),
        "participants": participants,
        "files": {name: str(folder / name) for name in ("REPORT.md", "METHOD.md", "run_config.json")
                  if (folder / name).is_file()},
        "test_set": str(folder / TEST_SET) if (folder / TEST_SET / "run_config.json").is_file() else None,
    }


def _clean_items(values, label):
    if values is None:
        return []
    if isinstance(values, str):
        values = [item for item in re.split(r"[,\s]+", values) if item]
    cleaned = [str(item).strip() for item in values if str(item).strip()]
    for item in cleaned:
        if not NAME_ITEM.match(item):
            raise ValueError(f"{label}: {item!r} is not a valid name")
    return cleaned


def cache_inventory(values, backend):
    """Labels and participants of the test data's experiments and views found in its preprocessing cache."""
    location = validation.derived_locations(values, "TEST.DATA")["cache_dir"]
    host, _ = paths.config_host_path(location, backend)
    manifest = Path(host) / "preprocessing_manifest.json" if host else None
    experiments = {slug(item) for item in values.get("TEST.DATA.VHRM.EXPERIMENTS") or []}
    views = {slug(item) for item in values.get("TEST.DATA.VHRM.VIEWS") or []}
    labels, participants = {}, set()
    if manifest and manifest.is_file():
        try:
            recordings = json.loads(manifest.read_text(encoding="utf-8")).get("recordings", {})
        except (OSError, ValueError):
            recordings = {}
        for recording_id in recordings:
            parts = recording_id.split("--")
            if len(parts) != 5 or (experiments and parts[0] not in experiments) or (views and parts[4] not in views):
                continue
            labels[parts[3]] = labels.get(parts[3], 0) + 1
            participants.add(parts[1])
    return {"cache": str(host) if host else location, "labels": labels, "participants": sorted(participants)}


def report_request(payload):
    """Validate a report request; returns everything a report job needs (paths for its backend included)."""
    checkpoint_host = results.check_path(payload.get("checkpoint") or "")
    if checkpoint_host.suffix.lower() != ".pth" or not checkpoint_host.is_file():
        raise ValueError("Choose an existing .pth checkpoint.")
    config_rel = payload.get("config") or ""
    config_path = configs.resolve_config_file(config_rel)
    tree = configs.read_yaml_tree(config_path)
    values, errors, _ = configs.normalize(configs.flatten(tree))
    if errors:
        raise ValueError("The config has invalid values: " + "; ".join(f"{k}: {v}" for k, v in errors.items()))
    if values.get("TOOLBOX_MODE") not in ("train_and_test", "only_test"):
        raise ValueError("The report needs the config of a training or of a checkpoint evaluation.")
    if values.get("TEST.DATA.DATASET") != "vHRM":
        raise ValueError("The analysis report needs vHRM test data (TEST.DATA.DATASET: vHRM).")
    if values.get("MODEL.NAME") == "BigSmall":
        raise ValueError("The analysis report does not support BigSmall.")
    backend = payload.get("backend") or configs.guess_backend(configs.relative(config_path), values)
    if backend not in ("local", "docker"):
        raise ValueError("Backend must be local or docker.")
    output_host = report_location(checkpoint_host, values, backend)
    if backend == "docker":
        checkpoint = paths.host_to_container(checkpoint_host)
        output = paths.host_to_container(output_host)
        if not checkpoint or not output:
            raise ValueError("The checkpoint and its report folder must be inside a Docker mount (Settings), "
                             "or use the Local Python backend.")
        if not _writable_in_docker(output_host):
            raise ValueError(f"The report folder {output_host} is on a read-only Docker mount; "
                             "use the Local Python backend.")
    else:
        checkpoint, output = str(checkpoint_host), str(output_host)
        if not Path(settings.load()["local_python"]).is_file():
            raise ValueError("The local Python of Settings was not found.")
    labels = _clean_items(payload.get("labels"), "Labels")
    participants = _clean_items(payload.get("participants"), "Participants")
    return {
        "config_path": config_path, "config_tree": tree, "values": values, "backend": backend,
        "checkpoint_host": str(checkpoint_host), "checkpoint": checkpoint,
        "output_host": str(output_host), "output": output, "labels": labels, "participants": participants,
        "export_csv": bool(payload.get("export_csv", True)), "gpu": preflight.gpu_index(values, backend, {}),
        "exists": (output_host / "run_config.json").is_file(),
    }


def preview(payload):
    """What a report request would do: its folder, whether it replaces a report, the labels in the cache."""
    request = report_request(payload)
    inventory = cache_inventory(request["values"], request["backend"])
    data_path = request["values"].get("TEST.DATA.DATA_PATH") or ""
    data_host, _ = paths.config_host_path(data_path, request["backend"])
    return {
        "output": request["output_host"], "exists": request["exists"], "backend": request["backend"],
        "model": request["values"].get("MODEL.NAME"), "mode": request["values"].get("TOOLBOX_MODE"),
        "default_labels": request["values"].get("REPORT.LABELS") or [],
        "experiments": request["values"].get("TEST.DATA.VHRM.EXPERIMENTS") or [],
        "views": request["values"].get("TEST.DATA.VHRM.VIEWS") or [],
        "data_path": str(data_host) if data_host else data_path,
        "data_available": bool(data_host and Path(data_host).is_dir()), **inventory,
    }


def job_summary(folder):
    """Headline numbers of a finished report, for the job list (all recordings and test participants)."""
    found = headline(folder)
    if not found:
        return None
    summary = {"recordings": (found.get("hr") or {}).get("recordings")}
    for metric, (key, within) in HEADLINE.items():
        summary[key] = (found.get(metric) or {}).get("mae")
        if within:
            summary["hr_" + within] = (found.get(metric) or {}).get("within_pct")
    test = headline(Path(folder) / TEST_SET)
    if test:
        summary["test_recordings"] = (test.get("hr") or {}).get("recordings")
        summary["test_hr_mae_bpm"] = (test.get("hr") or {}).get("mae")
        summary["test_ibi_mae_ms"] = (test.get("ibi") or {}).get("mae")
    return summary
