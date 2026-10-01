"""Pre-launch checks and interactive questions (W&B login, cache reuse, GPU sharing...)."""

import base64
import copy
import hashlib
import json
import os
import re
import threading
import urllib.error
import urllib.request
from datetime import datetime

from . import paths, probes, schema, settings, validation

_wandb_verified = {}
_wandb_lock = threading.Lock()
LOW_DISK_BYTES = 20 * 1024 ** 3


def gpu_index(values, backend, answers):
    if backend == "docker":
        value = answers.get("gpu", paths.docker_env().get("GPU_DEVICE_ID", "0"))
        return int(value) if str(value).isdigit() else 0
    device = values.get("DEVICE", "cuda:0")
    if device.startswith("cuda:") and device[5:].isdigit():
        return int(device[5:])
    return None


def apply_answers(values, answers, stamp=None):
    """Return a copy of values with the user's launch decisions applied."""
    values = copy.deepcopy(values)
    if answers.get("outputs") == "new_folder":
        stamp = stamp or answers.get("_stamp") or datetime.now().strftime("%Y%m%d-%H%M%S")
        base = values["LOG.PATH"].rstrip("/\\")
        sep = "/" if base.startswith("/") or "/" in base else os.sep
        values["LOG.PATH"] = f"{base}{sep}run_{stamp}"
    for prefix in schema.SPLITS:
        if answers.get(f"cache:{prefix}") == "reuse":
            values[prefix + ".DO_PREPROCESS"] = False
    wandb_choice = answers.get("wandb")
    if wandb_choice == "offline":
        values["WANDB.MODE"] = "offline"
    elif wandb_choice == "disable":
        values["WANDB.ENABLED"] = False
    return values


def verify_wandb_key(key):
    digest = hashlib.sha256(key.encode()).hexdigest()
    with _wandb_lock:
        if digest in _wandb_verified:
            return _wandb_verified[digest]
    request = urllib.request.Request(
        "https://api.wandb.ai/graphql",
        data=json.dumps({"query": "query { viewer { username entity } }"}).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Basic " + base64.b64encode(f"api:{key}".encode()).decode(),
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            payload = json.loads(response.read().decode("utf-8"))
        viewer = (payload.get("data") or {}).get("viewer")
        result = {"status": "valid", "user": viewer.get("username") or viewer.get("entity")} if viewer else \
            {"status": "invalid", "message": "W&B did not accept this API key."}
    except urllib.error.HTTPError as error:
        result = {"status": "invalid", "message": f"W&B rejected the API key (HTTP {error.code})."} \
            if error.code in (401, 403) else {"status": "unknown", "message": f"W&B returned HTTP {error.code}."}
    except (urllib.error.URLError, OSError, ValueError) as error:
        result = {"status": "unknown", "message": f"Could not reach W&B to verify the key: {error}"}
    if result["status"] != "unknown":
        with _wandb_lock:
            _wandb_verified[digest] = result
    return result


class Checks:
    def __init__(self):
        self.items = []

    def add(self, id_, title, status, message, **extra):
        self.items.append({"id": id_, "title": title, "status": status, "message": message, **extra})

    def question(self, id_, title, message, options, answers, default=None, **extra):
        chosen = answers.get(id_)
        valid = {option["value"] for option in options}
        status = "answered" if chosen in valid else "question"
        self.items.append({"id": id_, "title": title, "status": status, "message": message, "options": options,
                           "default": default, "answer": chosen if chosen in valid else None, **extra})


def _wandb_checks(values, backend, answers, secrets, checks, local_info):
    original_enabled = values.get("WANDB.ENABLED")
    if not original_enabled:
        checks.add("wandb", "Weights & Biases", "info", "Tracking is disabled for this run.")
        return
    mode = values.get("WANDB.MODE")
    if backend == "local" and local_info.get("ok") and not local_info.get("wandb"):
        checks.question("wandb", "Weights & Biases",
                        "W&B is enabled but the wandb package is not installed in the local Python environment.",
                        [{"value": "disable", "label": "Run without W&B"}], answers, default="disable")
        return
    if mode == "disabled":
        checks.add("wandb", "Weights & Biases", "info", "W&B mode is 'disabled'; nothing will be logged.")
        return
    if mode == "offline" and answers.get("wandb") != "key":
        where = "/runs/wandb (RUNS_PATH on the host)" if backend == "docker" else "the wandb/ folder of the toolbox"
        checks.add("wandb", "Weights & Biases", "ok",
                   f"Offline mode: runs are stored in {where}; upload later with 'wandb sync'. No login needed.")
        return

    key = (secrets or {}).get("wandb_api_key", "").strip()
    sources = []
    if os.environ.get("WANDB_API_KEY"):
        sources.append("the launcher's environment")
    if backend == "docker" and paths.read_env_file().get("WANDB_API_KEY"):
        sources.append("the private .env file")
    if backend == "local" and probes.wandb_netrc():
        sources.append("your W&B login (netrc)")
    options = [
        {"value": "key", "label": "Enter an API key for this run",
         "description": "Kept in memory only and passed to the run's environment; never written to disk."},
        {"value": "offline", "label": "Switch this run to offline mode",
         "description": "Log locally, upload later with 'wandb sync'."},
        {"value": "disable", "label": "Run without W&B"},
    ]
    if sources:
        options.insert(0, {"value": "existing", "label": f"Use the API key from {sources[0]}"})
    message = "Online mode needs a W&B API key (https://wandb.ai/authorize)."
    if sources:
        message = f"An API key is available from {', '.join(sources)}."
    default = "existing" if sources else "key"
    checks.question("wandb", "Weights & Biases login", message, options, answers, default=default,
                    secret={"name": "wandb_api_key", "when": "key", "label": "W&B API key"})
    if answers.get("wandb") == "key":
        item = checks.items[-1]
        if not key:
            item["status"] = "question"
            item["answer"] = "key"
            item["message"] = "Paste your W&B API key below."
        elif len(key) < 20 or re.search(r"\s", key):
            item["status"] = "error"
            item["message"] = "This does not look like a W&B API key."
        else:
            result = verify_wandb_key(key)
            if result["status"] == "valid":
                item["status"] = "answered"
                item["message"] = f"API key verified for W&B user '{result['user']}'."
            elif result["status"] == "invalid":
                item["status"] = "error"
                item["message"] = result["message"]
            else:
                item["status"] = "answered"
                item["message"] = result["message"] + " The key will be used as entered."


def run_checks(values, extras, backend, answers, secrets, manager):
    answers = dict(answers or {})
    checks = Checks()
    effective = apply_answers(values, answers)
    prefs = settings.load()
    local_info = {}

    # Environment
    if backend == "local":
        local_info = probes.local_python(prefs["local_python"])
        if not local_info.get("ok"):
            checks.add("environment", "Local Python", "error", local_info.get("error", "Probe failed"))
        else:
            missing = [name for name in ("torch", "yacs", "yaml", "cv2", "scipy") if not local_info.get(name)]
            device = effective.get("DEVICE", "cpu")
            if missing:
                checks.add("environment", "Local Python", "error",
                           f"{local_info['executable']} lacks: {', '.join(missing)}.")
            elif device.startswith("cuda") and not local_info.get("cuda"):
                checks.add("environment", "Local Python", "error",
                           f"PyTorch {local_info.get('torch_version')} cannot see a CUDA GPU; set DEVICE to cpu "
                           "or fix the CUDA installation.")
            elif device.startswith("cuda:") and int(device[5:] or 0) >= len(local_info.get("gpus", [])):
                checks.add("environment", "Local Python", "error",
                           f"{device} does not exist; visible GPUs: {', '.join(local_info.get('gpus', [])) or 'none'}.")
            else:
                gpus_text = ", ".join(local_info.get("gpus", [])) or "CPU only"
                checks.add("environment", "Local Python", "ok",
                           f"Python {local_info['python']}, PyTorch {local_info.get('torch_version')} ({gpus_text}).")
    else:
        info = probes.docker()
        env = paths.docker_env()
        if not info.get("ok"):
            checks.add("environment", "Docker", "error", info.get("error", "Docker is unavailable."))
        else:
            image = env["TOOLBOX_IMAGE"]
            image_info = probes.docker_image(image)
            if image_info["exists"]:
                checks.add("environment", "Docker", "ok",
                           f"Docker {info['server']}, Compose {info['compose']}, image {image} present.")
            else:
                checks.question("build", f"Docker image {image} not found",
                                "The image must be built (several GB download the first time) or loaded with "
                                "'docker image load'.",
                                [{"value": "build", "label": "Build the image before running"}], answers,
                                default="build")
        if not paths.ENV_FILE.is_file():
            checks.add("dotenv", "Docker settings", "info",
                       "No .env file: Compose defaults are used (./data, ./preprocessed_data, ./runs). "
                       "Edit them in Settings.")
        for mount in paths.docker_mounts(env):
            if mount["read_only"] and mount["key"] and not mount["host"].is_dir():
                checks.add(f"mount:{mount['key']}", "Docker mounts", "error",
                           f"{mount['key']} folder does not exist: {mount['host']} (Compose will refuse to start).")
        gpu_list = probes.gpus()
        options = [{"value": str(g["index"]), "label": f"GPU {g['index']}: {g['name']}",
                    "description": f"{g['memory_used']}/{g['memory_total']} MiB used, {g['utilization']}% busy"}
                   for g in gpu_list]
        default_gpu = env.get("GPU_DEVICE_ID", "0")
        if len(options) > 1:
            checks.question("gpu", "GPU", "Choose the host GPU for this container.", options, answers,
                            default=default_gpu)
        else:
            answers.setdefault("gpu", default_gpu)

    # Static validation of the effective config
    report = validation.validate(effective, extras, backend)
    if report["errors"]:
        checks.add("validation", "Configuration", "error",
                   f"{len(report['errors'])} problem(s) must be fixed in the editor.",
                   details=[e["message"] for e in report["errors"]])
    else:
        checks.add("validation", "Configuration", "warn" if report["warnings"] else "ok",
                   f"Valid ({len(report['warnings'])} warning(s)).",
                   details=[w["message"] for w in report["warnings"]])

    # Existing outputs
    outputs = report["derived"].get("outputs", {})
    original_outputs = validation.output_locations(values)
    original_existing = []
    for key in ("model_dir", "test_outputs", "outputs"):
        if key in original_outputs:
            host, _ = paths.config_host_path(original_outputs[key], backend)
            if host and host.is_dir() and any(host.iterdir()):
                original_existing.append(str(host))
    if original_existing:
        checks.question("outputs", "Existing results",
                        "The output folder already has results: " + "; ".join(original_existing),
                        [{"value": "new_folder", "label": "Write to a new timestamped sub-folder",
                          "description": "LOG.PATH/run_<date-time>"},
                         {"value": "overwrite", "label": "Overwrite / add to the existing folder"}],
                        answers, default="new_folder")

    # Caches
    for prefix in schema.active_splits(values):
        derived = report["derived"].get(prefix) or {}
        if not values.get(prefix + ".DO_PREPROCESS") or not derived.get("file_list_exists"):
            continue
        if values.get(prefix + ".DATASET") == "vHRM" and values.get(prefix + ".VHRM.INCREMENTAL_PREPROCESS"):
            continue
        section = schema.SPLITS[prefix]["section"]
        checks.question(f"cache:{prefix}", f"{section}: existing cache",
                        f"A preprocessed cache already exists ({derived['exp_data_name']}).",
                        [{"value": "reuse", "label": "Reuse it (skip preprocessing)"},
                         {"value": "rebuild", "label": "Preprocess again and overwrite it"}],
                        answers, default="reuse")

    _wandb_checks(values, backend, answers, secrets, checks, local_info)

    # GPU sharing
    gpu = gpu_index(effective, backend, answers)
    if gpu is not None and manager is not None:
        busy = manager.jobs_on_gpu(gpu)
        if busy:
            names = ", ".join(f"{job['name']} ({job['status']})" for job in busy)
            checks.question("policy", f"GPU {gpu} is in use", f"Other jobs use this GPU: {names}.",
                            [{"value": "queue", "label": "Queue: start when the GPU is free"},
                             {"value": "parallel", "label": "Start now in parallel",
                              "description": "Shares GPU memory; can run out of VRAM."}],
                            answers, default=prefs["gpu_policy"])
        running = manager.running_count()
        if running >= prefs["max_parallel_jobs"]:
            checks.add("capacity", "Job limit", "info",
                       f"{running} job(s) running (limit {prefs['max_parallel_jobs']}); this one will wait.")

    # Disk space
    for label, key in (("outputs", "log_path_host"),):
        host = outputs.get(key)
        if host:
            free = probes.disk_free(host)
            if free is not None and free < LOW_DISK_BYTES:
                checks.add("disk", "Disk space", "warn", f"Only {free / 1024 ** 3:.1f} GB free for {label}.")
    for prefix in schema.active_splits(effective):
        derived = report["derived"].get(prefix) or {}
        if effective.get(prefix + ".DO_PREPROCESS") and derived.get("cache_dir_host"):
            free = probes.disk_free(derived["cache_dir_host"])
            if free is not None and free < LOW_DISK_BYTES:
                checks.add("disk-cache", "Disk space", "warn",
                           f"Only {free / 1024 ** 3:.1f} GB free for the preprocessed cache.")
                break

    blocking = [c for c in checks.items if c["status"] in ("error", "question")]
    return {
        "checks": checks.items,
        "can_launch": not blocking,
        "effective_values": effective,
        "validation": report,
        "answers": answers,
    }
