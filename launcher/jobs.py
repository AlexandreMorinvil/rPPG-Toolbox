"""Job queue: creates run folders, starts detached runners and follows their logs."""

import json
import os
import secrets as pysecrets
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from neural_methods.early_stopping import EarlyStopping

from . import batch, configs, crossval, logparse, paths, preflight, probes, reports, schema, settings, validation

ACTIVE = ("queued", "starting", "running", "pausing", "stopping")
RUNNING = ("starting", "running", "pausing", "stopping")
FINISHED = ("succeeded", "failed", "cancelled", "interrupted", "paused")
STOPPING = "TRAIN.EARLY_STOPPING."


def _now():
    return datetime.now().isoformat(timespec="seconds")


def _collective_status(statuses):
    """Status of a cross-validation or a batch from the statuses of its jobs."""
    if any(status in RUNNING for status in statuses):
        return "running"
    if "queued" in statuses:
        return "running" if any(status in FINISHED for status in statuses) else "queued"
    if "paused" in statuses:
        return "paused"
    if all(status == "succeeded" for status in statuses):
        return "succeeded"
    if any(status in ("failed", "interrupted") for status in statuses):
        return "failed"
    return "cancelled"


def _clean_description(text):
    if not isinstance(text, str):
        raise ValueError("The description must be text.")
    return text.replace("\r\n", "\n").strip()[:20000]


def _stopping_values(request):
    """The early stopping values of a continuation: off, or the settings to change (the others stay the job's)."""
    request = request or {}
    values = {STOPPING + "ENABLED": bool(request.get("enabled"))}
    if not values[STOPPING + "ENABLED"]:
        return values
    for name in ("patience", "tolerance", "min_epochs", "patience_after_min_epochs"):
        if request.get(name) in (None, ""):
            continue
        key = STOPPING + name.upper()
        values[key], error = configs.coerce(schema.FIELD_MAP[key], request[name])
        if error:
            raise ValueError(f"{schema.FIELD_MAP[key]['label']}: {error}")
    if len(values) == 1:
        raise ValueError("Enter a new patience or minimum epochs.")
    return values


def _epochs_without_improvement(job, values):
    """Epochs already counted against the patience under these settings, from the validation losses of its log."""
    losses = (job.get("progress") or {}).get("valid_loss") or {}
    monitor = EarlyStopping(values[STOPPING + "PATIENCE"], values[STOPPING + "TOLERANCE"],
                            values[STOPPING + "MIN_EPOCHS"], values[STOPPING + "PATIENCE_AFTER_MIN_EPOCHS"])
    monitor.replay(float("nan") if loss is None else loss for _, loss in sorted(
        losses.items(), key=lambda item: int(item[0])))
    return monitor.bad_epochs, len(losses)


def _fewest_more_epochs(values, used, done):
    """Epochs before early stopping can end a training with done epochs, used of them counted against the patience."""
    patience, minimum = values[STOPPING + "PATIENCE"], values[STOPPING + "MIN_EPOCHS"]
    if values[STOPPING + "PATIENCE_AFTER_MIN_EPOCHS"]:
        return max(minimum - done, 0) + patience - used
    return max(patience - used, minimum - done)


class JobManager:
    def __init__(self):
        paths.JOBS_DIR.mkdir(parents=True, exist_ok=True)
        paths.CV_DIR.mkdir(parents=True, exist_ok=True)
        paths.BATCH_DIR.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.jobs = {}
        self.groups = {}
        self.batches = {}
        self.processes = {}
        self.parsers = {}
        self.offsets = {}
        self.secret_env = {}
        self._load()
        self.thread = threading.Thread(target=self._loop, name="job-monitor", daemon=True)
        self.thread.start()

    # ---------------------------------------------------------------- storage
    def _dir(self, job_id):
        return paths.JOBS_DIR / job_id

    def _save(self, job):
        target = self._dir(job["id"]) / "job.json"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(job, indent=2), encoding="utf-8")
        os.replace(temporary, target)

    def _group_dir(self, group_id):
        return paths.CV_DIR / group_id

    def _save_group(self, group):
        target = self._group_dir(group["id"]) / "group.json"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(group, indent=2), encoding="utf-8")
        os.replace(temporary, target)

    def _batch_dir(self, batch_id):
        return paths.BATCH_DIR / batch_id

    def _save_batch(self, record):
        target = self._batch_dir(record["id"]) / "batch.json"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, indent=2), encoding="utf-8")
        os.replace(temporary, target)

    def _load(self):
        for batch_file in sorted(paths.BATCH_DIR.glob("*/batch.json")):
            try:
                record = json.loads(batch_file.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            self.batches[record["id"]] = record
        for group_file in sorted(paths.CV_DIR.glob("*/group.json")):
            try:
                group = json.loads(group_file.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            self.groups[group["id"]] = group
            if group.get("merge", {}).get("status") == "running":
                group["merge"].update({"status": "failed", "output": "The launcher stopped during the merge."})
                self._save_group(group)
        for job_file in sorted(paths.JOBS_DIR.glob("*/job.json")):
            try:
                job = json.loads(job_file.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            self.jobs[job["id"]] = job
            if job["status"] in ("starting", "running", "pausing", "stopping"):
                self.parsers[job["id"]] = self._new_parser(job)
                self.offsets[job["id"]] = 0
            elif job["status"] == "queued" and job.get("needs_secret"):
                job["status"] = "failed"
                job["error"] = "The W&B API key is kept in memory only and was lost when the launcher restarted. " \
                               "Launch the run again."
                job["finished"] = _now()
                self._save(job)

    # ----------------------------------------------------------------- public
    def list(self):
        with self.lock:
            return sorted((self._public(job) for job in self.jobs.values()), key=lambda j: j["created"],
                          reverse=True)

    def get(self, job_id):
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            return self._public(job, full=True)

    def _public(self, job, full=False):
        data = dict(job)
        checkpoint = job.get("resume_checkpoint_host")
        available = bool(checkpoint and Path(checkpoint).is_file())
        data["can_resume"] = job["status"] in ("paused", "failed", "cancelled", "interrupted") and available
        data["can_pause"] = bool(job.get("training_recovery") and job["status"] == "running"
                                 and job.get("progress", {}).get("phase") not in ("testing", "succeeded"))
        if available:
            try:
                data["recovery"] = json.loads(Path(checkpoint).with_suffix(".json").read_text(encoding="utf-8"))
                if data["recovery"].get("training_complete"):
                    data["can_pause"] = False
            except (OSError, ValueError):
                pass
        data["can_continue"] = self._can_continue(job, data.get("recovery"))
        if not full:
            data.pop("steps", None)
        return data

    def _can_continue(self, job, recovery):
        """Early stopping ended its training before the last epoch, and no later attempt took over the checkpoint."""
        return bool(recovery and recovery.get("early_stopped") and job["status"] in FINISHED
                    and recovery.get("next_epoch", 0) < recovery.get("epochs_total", 0)
                    and not any(other.get("resume_from") == job["id"] for other in self.jobs.values()))

    def running_count(self):
        with self.lock:
            return sum(1 for job in self.jobs.values() if job["status"] in ("starting", "running", "pausing", "stopping"))

    def jobs_on_gpu(self, gpu):
        with self.lock:
            return [self._public(job) for job in self.jobs.values()
                    if job["status"] in ACTIVE and job.get("gpu") == gpu]

    def checkpoint_folder_busy(self, model_dir_host):
        with self.lock:
            return bool(model_dir_host) and any(job["status"] in ACTIVE and job.get("outputs", {}).get(
                "model_dir_host") == model_dir_host for job in self.jobs.values())

    def active_owner(self, source, folders):
        """The queued or running job launched from this config, or writing to one of these output folders."""
        with self.lock:
            for job in self.jobs.values():
                if job["status"] not in ACTIVE:
                    continue
                outputs = job.get("outputs") or {}
                written = {os.path.normcase(str(outputs[key])) for key in batch.OUTPUT_KEYS if outputs.get(key)}
                if (source and job.get("source_config") == source) or written & set(folders):
                    return self._public(job)
        return None

    @staticmethod
    def _launch_request(payload):
        values, type_errors, _ = configs.normalize(payload.get("values") or {})
        if type_errors:
            raise ValueError("Fix the invalid fields before launching.")
        if payload.get("description") is not None:
            values["DESCRIPTION"] = _clean_description(payload["description"])
        answers = dict(payload.get("answers") or {})
        answers["_stamp"] = datetime.now().strftime("%Y%m%d-%H%M%S")
        return values, payload.get("extras") or {}, answers, payload.get("secrets") or {}

    @staticmethod
    def _require_launchable(result):
        if not result["can_launch"]:
            blocking = [c["title"] + ": " + c["message"] for c in result["checks"] if c["status"] in ("error",
                                                                                                    "question")]
            raise ValueError("Launch blocked. " + " | ".join(blocking))

    @staticmethod
    def _new_id():
        return f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{pysecrets.token_hex(3)}"

    def create(self, payload):
        """Re-run preflight with the submitted answers and queue the job when everything is resolved."""
        backend = payload.get("backend")
        values, extras, answers, secrets = self._launch_request(payload)
        if schema.cross_validation_enabled(values):
            raise ValueError("This config runs as a cross-validation; launch it from the editor (one job per fold).")
        result = preflight.run_checks(values, extras, backend, answers, secrets, self)
        self._require_launchable(result)
        # A resumed fold stays part of its cross-validation, a resumed batch job part of its batch.
        original = self.jobs.get(payload.get("resume_from") or "")
        group = dict(original["group"]) if original and original.get("group") else None
        member = dict(original["batch"]) if original and original.get("batch") else None
        job = self._queue(result["effective_values"], extras, backend, result["answers"], secrets,
                          result["validation"]["derived"].get("outputs", {}), payload.get("name"),
                          payload.get("source"), payload.get("resume_from"), group, member)
        return self._public(job)

    def create_batch(self, payload):
        """Queue one job per config of a batch, in the listed order, after one round of checks for all of them.

        payload: configs (every config of the batch), launch (the ones to queue), backend, answers, secrets, name.
        All or nothing: a config that is no longer launchable refuses the whole launch.
        """
        backend = payload.get("backend")
        config_paths = list(dict.fromkeys(payload.get("configs") or []))
        wanted = list(dict.fromkeys(payload.get("launch") or []))
        if not wanted:
            raise ValueError("No config to launch.")
        unknown = [path for path in wanted if path not in config_paths]
        if unknown:
            raise ValueError("Configs to launch must be part of the batch: " + ", ".join(unknown))
        answers = dict(payload.get("answers") or {})
        answers["_stamp"] = datetime.now().strftime("%Y%m%d-%H%M%S")
        secrets = payload.get("secrets") or {}
        result = batch.run_checks(config_paths, backend, answers, secrets, self,
                                  exclude=[path for path in config_paths if path not in wanted])
        blocking = [check["title"] + ": " + check["message"] for check in result["checks"]
                    if check["status"] in ("error", "question")]
        if blocking:
            raise ValueError("Launch blocked. " + " | ".join(blocking))
        entries = {entry["path"]: entry for entry in result["configs"]}
        refused = [entries[path] for path in wanted if entries[path]["status"] not in batch.LAUNCHABLE]
        if refused:
            raise ValueError("Some configs can no longer be launched; check the batch again. " + " | ".join(
                f"{entry['name']}: {(entry['errors'] or [entry['status']])[0]}" for entry in refused))
        batch_id = self._new_id()
        name = (payload.get("name") or "").strip()[:100] or result["name"]
        record = {
            "id": batch_id, "name": name, "created": _now(), "backend": backend, "folder": result["folder"],
            "answers": {key: answers[key] for key in ("outputs", "cache", "wandb", "policy", "gpu") if key in answers},
            "configs": [{"path": path, "name": entries[path]["name"], "index": index}
                        for index, path in enumerate(wanted)],
        }
        self._batch_dir(batch_id).mkdir(parents=True)
        with self.lock:
            self.batches[batch_id] = record
            self._save_batch(record)
            queued = []
            try:
                for config in record["configs"]:
                    run = result["runs"][config["path"]]
                    run_answers = dict(run["answers"])
                    if config["index"]:
                        run_answers.pop("build", None)  # the first job builds a missing Docker image
                    member = {"id": batch_id, "name": name, "index": config["index"], "count": len(wanted)}
                    queued.append(self._queue(run["values"], run["extras"], backend, run_answers, secrets,
                                              run["outputs"], config["name"], config["path"], None, None, member,
                                              run["caches"]))
            except Exception:
                for job in queued:  # all jobs or none
                    self.jobs.pop(job["id"], None)
                    self.secret_env.pop(job["id"], None)
                    shutil.rmtree(self._dir(job["id"]), ignore_errors=True)
                self.batches.pop(batch_id, None)
                shutil.rmtree(self._batch_dir(batch_id), ignore_errors=True)
                raise
        return self.get_batch(batch_id)

    def create_group(self, payload):
        """Queue one job per fold of a cross-validation config; the folds run one after another."""
        backend = payload.get("backend")
        values, extras, answers, secrets = self._launch_request(payload)
        if not schema.cross_validation_enabled(values):
            raise ValueError("Cross-validation is not enabled for this config.")
        group_id = self._new_id()
        result = crossval.run_checks(values, extras, backend, answers, secrets, self, group_id)
        self._require_launchable(result)
        cv, runs, answers = result["cv"], result["fold_runs"], result["answers"]
        name = (payload.get("name") or "").strip()[:100] or \
            f"CV {values.get('MODEL.NAME')} · {len(runs)} folds ({schema.CV_SCHEME_LABELS.get(cv['scheme'], cv['scheme'])})"
        group = {
            "id": group_id, "name": name, "created": _now(), "backend": backend,
            "description": values.get("DESCRIPTION", ""), "source_config": payload.get("source"),
            "root": cv["root"], "root_host": cv["root_host"], "scheme": cv["scheme"],
            "settings": {key: values.get("CROSS_VALIDATION." + key) for key in ("K", "VALID_COUNT", "GROUP_SESSIONS")},
            "pool": cv["pool"], "warnings": cv.get("warnings", []),
            "folds": [{key: run[key] for key in ("name", "test", "valid", "train")} for run in runs],
            "merge": {"status": "none"},
        }
        self._group_dir(group_id).mkdir(parents=True)
        # The editor reopens the config as written, before the launch answers (new sub-folder, W&B mode).
        (self._group_dir(group_id) / "config.yaml").write_text(
            configs.dump_yaml(configs.build_tree(values, extras)), encoding="utf-8")
        with self.lock:
            self.groups[group_id] = group
            self._save_group(group)
            queued = []
            try:
                for index, run in enumerate(runs):
                    fold_answers = dict(answers)
                    if index:
                        fold_answers.pop("build", None)  # the first fold builds a missing Docker image
                    queued.append(self._queue(
                        run["values"], extras, backend, fold_answers, secrets, run["outputs"],
                        f"{name} · fold {run['name']}", payload.get("source"), None,
                        {"id": group_id, "name": name, "fold": run["name"], "index": index, "count": len(runs)}))
            except Exception:
                for job in queued:  # all folds or none
                    self.jobs.pop(job["id"], None)
                    self.secret_env.pop(job["id"], None)
                    shutil.rmtree(self._dir(job["id"]), ignore_errors=True)
                self.groups.pop(group_id, None)
                shutil.rmtree(self._group_dir(group_id), ignore_errors=True)
                raise
        if cv["root_host"]:
            try:
                Path(cv["root_host"]).mkdir(parents=True, exist_ok=True)
                (Path(cv["root_host"]) / "cv_plan.json").write_text(json.dumps(
                    {key: group[key] for key in ("id", "name", "created", "scheme", "settings", "pool", "folds")},
                    indent=2), encoding="utf-8")
            except OSError:
                pass  # informational copy next to the results
        return self.group(group_id)

    @staticmethod
    def _config_fields(effective, backend, answers, outputs):
        """What a job record takes from its config: when it is queued, and again when a queued job is edited."""
        mode = effective["TOOLBOX_MODE"]
        recovery = mode == "train_and_test" and effective.get("TRAIN.SAVE_RESUME", True)
        prefix = effective.get("TRAIN.MODEL_FILE_NAME", "") + "_resume.pt"
        return {
            "mode": mode,
            "model": effective.get("MODEL.NAME") if mode != "unsupervised_method" else
            ",".join(effective.get("UNSUPERVISED.METHOD", [])),
            "datasets": {
                "train": effective.get("TRAIN.DATA.DATASET") if mode == "train_and_test" else None,
                "test": effective.get("UNSUPERVISED.DATA.DATASET") if mode == "unsupervised_method"
                else effective.get("TEST.DATA.DATASET"),
            },
            "epochs_total": effective.get("TRAIN.EPOCHS") if mode == "train_and_test" else None,
            "gpu": preflight.gpu_index(effective, backend, answers),
            "training_recovery": recovery,
            "resume_checkpoint_host": str(Path(outputs["model_dir_host"]) / prefix)
            if recovery and outputs.get("model_dir_host") else None,
            "resume_checkpoint": (outputs.get("model_dir", "").rstrip("/\\") + "/" + prefix) if recovery else None,
        }

    def _queue(self, effective, extras, backend, answers, secrets, outputs, name, source, resume_from, group=None,
               member=None, caches=None):
        job_id = self._new_id()
        job_dir = self._dir(job_id)
        job_dir.mkdir(parents=True)
        config_path = job_dir / "config.yaml"
        config_path.write_text(configs.dump_yaml(configs.build_tree(effective, extras)), encoding="utf-8")
        (job_dir / "output.log").touch()

        fields = self._config_fields(effective, backend, answers, outputs)
        mode, gpu = fields["mode"], fields["gpu"]
        env_additions = {}
        if backend == "docker":
            env_additions["GPU_DEVICE_ID"] = str(gpu if gpu is not None else 0)
        key = (secrets.get("wandb_api_key") or "").strip()
        needs_secret = answers.get("wandb") == "key" and bool(key)
        if needs_secret:
            self.secret_env[job_id] = {"WANDB_API_KEY": key}

        steps = self._steps(job_id, backend, config_path, answers)
        name = (name or "").strip() or self._default_name(effective)
        job = {
            "id": job_id,
            "name": name[:120],
            "description": effective.get("DESCRIPTION", ""),
            "created": _now(),
            "started": None,
            "finished": None,
            "status": "queued",
            "backend": backend,
            **fields,
            "policy": answers.get("policy", "queue"),
            "env": env_additions,
            "needs_secret": needs_secret,
            "wandb": {"enabled": bool(effective.get("WANDB.ENABLED")), "mode": effective.get("WANDB.MODE")},
            "source_config": source,
            "outputs": outputs,
            "steps": steps,
            "pid": None,
            "container": f"rppg-launcher-{job_id}" if backend == "docker" else None,
            "returncode": None,
            "error": None,
            "progress": {},
            "resume_from": resume_from,
        }
        if group:
            job["group"] = group
        if member:
            job["batch"] = member
            job["caches"] = caches or []
        with self.lock:
            if mode == "train_and_test" and any(
                    j["status"] in ACTIVE and j.get("outputs", {}).get("model_dir_host") == outputs.get("model_dir_host")
                    for j in self.jobs.values()):
                self.secret_env.pop(job_id, None)
                shutil.rmtree(job_dir, ignore_errors=True)
                raise ValueError("Another active training job writes to this checkpoint folder. "
                                 "Wait for it to finish or choose a different output folder.")
            self.jobs[job_id] = job
            self._save(job)
        return job

    def create_report(self, payload):
        """Queue the analysis report of a checkpoint (tools/vhrm_report.py) as a job."""
        request = reports.report_request(payload)
        if any(job["status"] in ACTIVE and job.get("outputs", {}).get("report_dir_host") == request["output_host"]
               for job in self.jobs.values()):
            raise ValueError("A report of this checkpoint is already queued or running.")
        values, backend = request["values"], request["backend"]
        job_id = self._new_id()
        job_dir = self._dir(job_id)
        job_dir.mkdir(parents=True)
        config_path = job_dir / "config.yaml"
        config_path.write_text(configs.dump_yaml(request["config_tree"]), encoding="utf-8")
        (job_dir / "output.log").touch()
        options = ["--labels", *request["labels"]]
        if request["participants"]:
            options += ["--participants", *request["participants"]]
        if not request["export_csv"]:
            options.append("--no-csv")
        root = str(paths.TOOLBOX_ROOT)
        env_additions = {}
        if backend == "local":
            steps = [{"title": "Analysis report", "cwd": root, "argv": [
                settings.load()["local_python"], "-u", "tools/vhrm_report.py", "--config", str(config_path),
                "--checkpoint", request["checkpoint"], "--output", request["output"], *options]}]
        else:
            env_additions["GPU_DEVICE_ID"] = str(request["gpu"] if request["gpu"] is not None else 0)
            container_config = f"{paths.JOB_CONFIG_MOUNT}/{job_id}.yaml"
            steps = [{"title": "Analysis report in Docker", "cwd": root, "argv": [
                "docker", "compose", "--project-directory", root, "-f", str(paths.TOOLBOX_ROOT / "compose.yaml"),
                "run", "--rm", "-T", "--name", f"rppg-launcher-{job_id}",
                "-v", f"{config_path.as_posix()}:{container_config}:ro", "--entrypoint", "python", "toolbox",
                "-u", "tools/vhrm_report.py", "--config", container_config, "--checkpoint", request["checkpoint"],
                "--output", request["output"], *options]}]
        checkpoint = Path(request["checkpoint_host"])
        job = {
            "id": job_id,
            "name": (payload.get("name") or "").strip()[:120] or f"Report · {checkpoint.stem}",
            "description": "", "created": _now(), "started": None, "finished": None, "status": "queued",
            "backend": backend, "mode": "report", "model": values.get("MODEL.NAME"),
            "datasets": {"train": values.get("TRAIN.DATA.DATASET") if values.get("TOOLBOX_MODE") == "train_and_test"
                         else None, "test": values.get("TEST.DATA.DATASET")},
            "epochs_total": None, "gpu": request["gpu"], "policy": settings.load()["gpu_policy"],
            "env": env_additions, "needs_secret": False, "wandb": {"enabled": False, "mode": None},
            "source_config": configs.relative(request["config_path"]),
            "outputs": {"report_dir": request["output"], "report_dir_host": request["output_host"]},
            "report": {"checkpoint": request["checkpoint_host"], "labels": request["labels"],
                       "participants": request["participants"], "replaces": request["exists"]},
            "steps": steps, "pid": None, "container": f"rppg-launcher-{job_id}" if backend == "docker" else None,
            "returncode": None, "error": None, "progress": {}, "training_recovery": False,
            "resume_checkpoint_host": None, "resume_checkpoint": None, "resume_from": None,
        }
        with self.lock:
            self.jobs[job_id] = job
            self._save(job)
        return self._public(job)

    @staticmethod
    def _default_name(values):
        mode = values["TOOLBOX_MODE"]
        if mode == "unsupervised_method":
            return f"{'+'.join(values['UNSUPERVISED.METHOD'])} on {values['UNSUPERVISED.DATA.DATASET']}"
        if mode == "only_test":
            checkpoint = Path(values["INFERENCE.MODEL_PATH"].replace("\\", "/")).name
            return f"Evaluate {checkpoint} on {values['TEST.DATA.DATASET']}"
        return f"Train {values['MODEL.NAME']} on {values['TRAIN.DATA.DATASET']}"

    @staticmethod
    def _steps(job_id, backend, config_path, answers):
        root = str(paths.TOOLBOX_ROOT)
        if backend == "local":
            python = settings.load()["local_python"]
            return [{"title": "Run main.py", "cwd": root,
                     "argv": [python, "-u", "main.py", "--config_file", str(config_path)]}]
        compose = ["docker", "compose", "--project-directory", root, "-f", str(paths.TOOLBOX_ROOT / "compose.yaml")]
        steps = []
        if answers.get("build") == "build":
            steps.append({"title": "Build Docker image", "cwd": root, "argv": compose + ["build", "toolbox"]})
        container_config = f"{paths.JOB_CONFIG_MOUNT}/{job_id}.yaml"
        mount = f"{config_path.as_posix()}:{container_config}:ro"
        control_dir = f"{paths.JOB_CONFIG_MOUNT}/{job_id}-control"
        control_mount = f"{config_path.parent.as_posix()}:{control_dir}:ro"
        steps.append({"title": "Run main.py in Docker", "cwd": root, "argv": compose + [
            "run", "--rm", "-T", "--name", f"rppg-launcher-{job_id}", "-v", mount,
            "-v", control_mount, "-e", f"RPPG_TRAINING_CONTROL={control_dir}/training-control.json",
            "toolbox", "--config_file", container_config,
        ]})
        return steps

    def log(self, job_id, offset=None, limit=262144):
        log_path = self._dir(job_id) / "output.log"
        if job_id not in self.jobs or not log_path.is_file():
            raise KeyError(job_id)
        size = log_path.stat().st_size
        if offset is None or offset < 0 or offset > size:
            offset = max(0, size - limit)
        with log_path.open("rb") as handle:
            handle.seek(offset)
            data = handle.read(limit)
        return {"offset": offset, "next": offset + len(data), "size": size,
                "text": data.decode("utf-8", errors="replace")}

    def config_text(self, job_id):
        if job_id not in self.jobs:
            raise KeyError(job_id)
        return (self._dir(job_id) / "config.yaml").read_text(encoding="utf-8")

    def cancel(self, job_id):
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            if job["status"] == "queued":
                job["status"] = "cancelled"
                job["finished"] = _now()
                self.secret_env.pop(job_id, None)
                self._save(job)
                return self._public(job)
            if job["status"] not in ("starting", "running", "pausing"):
                raise ValueError(f"Job is {job['status']}")
            job["status"] = "stopping"
            job["cancel_requested"] = True
            self._save(job)
        threading.Thread(target=self._terminate, args=(dict(job),), daemon=True).start()
        return self._public(job)

    def pause(self, job_id):
        """Cooperative epoch-boundary pause, portable to Windows and Docker."""
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            if not self._public(job)["can_pause"]:
                raise ValueError("Only new, running training jobs with recovery enabled can pause, "
                                 "before plotting/testing. Use Stop for immediate termination.")
            target = self._dir(job_id) / "training-control.json"
            temporary = target.with_suffix(".tmp")
            temporary.write_text(json.dumps({"action": "pause"}), encoding="utf-8")
            os.replace(temporary, target)
            job["status"] = "pausing"
            job["pause_requested"] = True
            self._save(job)
            return self._public(job)

    def _checkpoint_busy(self, job):
        return any(other["status"] in ACTIVE and other.get("outputs", {}).get("model_dir_host") ==
                   job.get("outputs", {}).get("model_dir_host") for other in self.jobs.values())

    def _resume_data(self, job):
        """The job's config snapshot, set to continue from its recovery checkpoint."""
        data = configs.load_config(configs.relative(self._dir(job["id"]) / "config.yaml"))
        values = data["values"]
        values["DESCRIPTION"] = job.get("description", values.get("DESCRIPTION", ""))
        values["MODEL.RESUME"] = job["resume_checkpoint"]
        values["TRAIN.SAVE_RESUME"] = True
        for prefix in ("TRAIN.DATA", "VALID.DATA", "TEST.DATA"):
            values[prefix + ".DO_PREPROCESS"] = False
        return data

    def resume_config(self, job_id):
        """Prepare a new attempt; normal preflight still handles credentials/GPU checks."""
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            if not self._public(job)["can_resume"]:
                raise ValueError("No committed full-state checkpoint is available for this job. "
                                 "Legacy epoch .pth files can only be evaluated, not fully resumed.")
            if self._checkpoint_busy(job):
                raise ValueError("Another active attempt is already using this checkpoint folder.")
            data = self._resume_data(job)
            data.update({"backend": job["backend"], "resume_from": job_id,
                         "name": job["name"] + " (resumed)", "answers": {"outputs": "overwrite", "policy": "queue"}})
            return data

    def continue_jobs(self, payload):
        """Queue trainings ended by early stopping again from their last epoch, with other early stopping settings.

        payload: jobs, early_stopping (enabled, patience, tolerance, min_epochs, patience_after_min_epochs),
        dry_run (only report what would happen).
        All or nothing: a job that cannot continue refuses the whole request.
        """
        stopping = _stopping_values(payload.get("early_stopping"))
        job_ids = list(dict.fromkeys(payload.get("jobs") or []))
        if not job_ids:
            raise ValueError("No job to continue.")
        plans = [self._plan_continuation(job_id, stopping) for job_id in job_ids]
        rows = [row for row, _ in plans]
        if payload.get("dry_run"):
            return {"results": rows}
        refused = [row for row in rows if row["errors"]]
        if refused:
            raise ValueError("Some jobs cannot continue. " + " | ".join(
                f"{row['name']}: {row['errors'][0]}" for row in refused))
        with self.lock:
            queued = []
            try:
                for row, run in plans:
                    queued.append(self._queue(*run))
            except Exception:
                for job in queued:  # all jobs or none
                    self.jobs.pop(job["id"], None)
                    shutil.rmtree(self._dir(job["id"]), ignore_errors=True)
                raise
            return {"results": rows, "jobs": [self._public(job) for job in queued]}

    def _plan_continuation(self, job_id, stopping):
        """(report row, the arguments of _queue) of one job; the row's errors say why it cannot continue."""
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            public = self._public(job)
            busy = self._checkpoint_busy(job)
        recovery = public.get("recovery") or {}
        total, done = recovery.get("epochs_total"), recovery.get("next_epoch")
        row = {"job": job_id, "name": job["name"], "epochs_total": total, "next_epoch": done,
               "best_epoch": (job.get("progress") or {}).get("best_epoch"), "errors": [], "min_epochs": None}
        if not public["can_continue"]:
            row["errors"].append("Only a finished training ended by early stopping before its last epoch can continue.")
            return row, None
        if busy:
            row["errors"].append("Another active attempt is already using this checkpoint folder.")
            return row, None
        data = self._resume_data(job)
        values = data["values"]
        values.update(stopping)
        row["max_epochs"] = total - done
        if values[STOPPING + "ENABLED"]:
            used, known = _epochs_without_improvement(job, values)
            if known == done:
                row["min_epochs"] = min(_fewest_more_epochs(values, used, done), total - done)
                if row["min_epochs"] <= 0:
                    minimum = values[STOPPING + "MIN_EPOCHS"]
                    row["errors"].append(f"A patience of {values[STOPPING + 'PATIENCE']} is already used up: "
                                         f"{used} epochs without improvement so far"
                                         + (f", and the {done} epochs done reach the minimum of {minimum}." if minimum
                                            else "."))
        answers = {"outputs": "overwrite", "policy": job.get("policy", "queue"), "wandb": "existing",
                   "_stamp": datetime.now().strftime("%Y%m%d-%H%M%S")}
        if job["backend"] == "docker":
            answers["gpu"] = str(job["gpu"] if job.get("gpu") is not None else 0)
        result = preflight.run_checks(values, data["extras"], job["backend"], answers, {}, self)
        for check in result["checks"]:
            if check["status"] in ("error", "question"):
                row["errors"].append(f"{check['title']}: " + "; ".join(check.get("details") or [check["message"]]))
        name = job["name"].removesuffix(" (continued)") + " (continued)"
        return row, (result["effective_values"], data["extras"], job["backend"], result["answers"], {},
                     result["validation"]["derived"].get("outputs", {}), name, job.get("source_config"), job_id,
                     job.get("group"), job.get("batch"))

    def edit_queued(self, payload):
        """Change config values of queued jobs before they start, in their config snapshots.

        payload: jobs, changes {key: value}, dry_run (only report what would happen). All or nothing: a job that cannot
        take the change refuses the whole request. Jobs that do not use a parameter are left as they are.
        """
        changes = payload.get("changes") or {}
        job_ids = list(dict.fromkeys(payload.get("jobs") or []))
        if not job_ids or not changes:
            raise ValueError("Choose the jobs and the values to change.")
        for key in changes:
            if key not in schema.FIELD_MAP or schema.locked_when_queued(key):
                raise ValueError(f"{key} cannot be changed on a queued job: cancel it and launch it again.")
        plans = [self._plan_edit(job_id, changes) for job_id in job_ids]
        rows = [row for row, _ in plans]
        if payload.get("dry_run"):
            return {"results": rows}
        refused = [row for row in rows if row["errors"]]
        if refused:
            raise ValueError("Some jobs cannot take this change. " + " | ".join(
                f"{row['name']}: {row['errors'][0]}" for row in refused))
        with self.lock:  # the queue cannot start a job meanwhile
            started = [row["name"] for row in rows if self.jobs[row["job"]]["status"] != "queued"]
            if started:
                raise ValueError("Already started: " + ", ".join(started) + ". Nothing was changed.")
            for row, run in plans:
                if run:
                    self._apply_edit(self.jobs[row["job"]], *run)
            return {"results": rows, "jobs": [self._public(self.jobs[row["job"]]) for row in rows]}

    def _plan_edit(self, job_id, changes):
        """(report row, what _apply_edit needs) of one job; status: change, same, unused or error."""
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            status = job["status"]
        row = {"job": job_id, "name": job["name"], "status": "error", "before": {}, "after": {}, "errors": []}
        if status != "queued" or job["mode"] == "report":
            row["errors"].append("Already started." if status != "queued" else "A report job has no such parameters.")
            return row, None
        data = configs.load_config(configs.relative(self._dir(job_id) / "config.yaml"))
        values = data["values"]
        before = {key: values[key] for key in changes}
        for key, raw in changes.items():
            values[key], error = configs.coerce(schema.FIELD_MAP[key], raw)
            if error:
                row["errors"].append(f"{schema.FIELD_MAP[key]['label']}: {error}")
        if row["errors"]:
            return row, None
        for key in changes:  # after all the changes: one can make another one active
            if schema.is_active(schema.FIELD_MAP[key], values):
                row["before"][key], row["after"][key] = before[key], values[key]
        for key in changes:
            if key not in row["after"]:
                values[key] = before[key]
        if not row["after"]:
            row["status"] = "unused"
            return row, None
        if row["before"] == row["after"]:
            row["status"] = "same"
            return row, None
        if values.get("MODEL.RESUME") and any(not key.startswith(STOPPING) for key in row["after"]):
            row["errors"].append("A queued continuation can only change early stopping: the rest must stay as trained.")
            return row, None
        report = validation.validate(values, data["extras"], job["backend"])
        row["errors"] += [problem["message"] for problem in report["errors"]]
        derived = report["derived"]
        outputs = derived.get("outputs", {})
        if not row["errors"] and batch.folders_of(outputs) != batch.folders_of(job.get("outputs") or {}):
            row["errors"].append("This changes the output folder chosen at launch: cancel the job and launch it again.")
        if row["errors"]:
            return row, None
        row["status"] = "change"
        return row, (values, data["extras"], outputs, batch.caches_of(values, derived))

    def _apply_edit(self, job, values, extras, outputs, caches):
        target = self._dir(job["id"]) / "config.yaml"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(configs.dump_yaml(configs.build_tree(values, extras)), encoding="utf-8")
        os.replace(temporary, target)
        job.update(self._config_fields(values, job["backend"], {"gpu": job.get("gpu")}, outputs))
        if job.get("batch"):
            job["caches"] = caches
        self._save(job)

    def set_description(self, job_id, text):
        """Edit the notes of any job; the config snapshot keeps the description it was launched with."""
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            job["description"] = _clean_description(text)
            self._save(job)
            return self._public(job)

    def _terminate(self, job):
        if job.get("container"):
            probes.run(["docker", "stop", "-t", "20", job["container"]], timeout=60)
        probes.kill_tree(job.get("pid"))

    def remove(self, job_id):
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            if job["status"] in ACTIVE:
                raise ValueError("Stop the job before removing it")
            self.jobs.pop(job_id)
        shutil.rmtree(self._dir(job_id), ignore_errors=True)

    # -------------------------------------------------------- cross-validation
    def _group_jobs(self, group_id):
        return [job for job in self.jobs.values() if (job.get("group") or {}).get("id") == group_id]

    def _group_public(self, group, full=False):
        data = dict(group)
        attempts = self._group_jobs(group["id"])
        folds = []
        for fold in group["folds"]:
            tries = sorted((job for job in attempts if job["group"]["fold"] == fold["name"]), key=lambda j: j["created"])
            latest = tries[-1] if tries else None
            entry = {**fold, "job": None, "attempts": len(tries), "status": "removed"}
            if latest:
                progress = latest.get("progress") or {}
                entry.update({
                    "job": latest["id"], "status": latest["status"], "phase": progress.get("phase"),
                    "epoch": progress.get("epoch"), "epochs_total": latest.get("epochs_total"),
                    "started": latest.get("started"), "finished": latest.get("finished"),
                    "summary": latest.get("summary"), "error": latest.get("error"),
                })
            folds.append(entry)
        data["folds"] = folds
        statuses = [fold["status"] for fold in folds]
        data["status"] = _collective_status(statuses)
        data["active"] = any(status in ACTIVE for status in statuses)
        data["summary"] = crossval.read_summary(group.get("root_host"), tables=full)
        return data

    def list_groups(self):
        with self.lock:
            return sorted((self._group_public(group) for group in self.groups.values()), key=lambda g: g["created"],
                          reverse=True)

    def group(self, group_id):
        with self.lock:
            group = self.groups.get(group_id)
            if group is None:
                raise KeyError(group_id)
            return self._group_public(group, full=True)

    def cancel_group(self, group_id):
        """Cancel the queued folds and stop the running one."""
        with self.lock:
            if group_id not in self.groups:
                raise KeyError(group_id)
            for job in self._group_jobs(group_id):
                if job["status"] in ("queued", "starting", "running", "pausing"):
                    self.cancel(job["id"])
            return self.group(group_id)

    def remove_group(self, group_id):
        """Forget a finished cross-validation and its fold jobs; outputs on disk are kept."""
        with self.lock:
            if group_id not in self.groups:
                raise KeyError(group_id)
            members = self._group_jobs(group_id)
            if any(job["status"] in ACTIVE for job in members):
                raise ValueError("Stop the cross-validation before removing it")
            for job in members:
                self.jobs.pop(job["id"])
            self.groups.pop(group_id)
        for job in members:
            shutil.rmtree(self._dir(job["id"]), ignore_errors=True)
        shutil.rmtree(self._group_dir(group_id), ignore_errors=True)

    def request_merge(self, group_id):
        """Merge the finished folds into <root>/cv_summary in the background (queued if a merge is running)."""
        with self.lock:
            group = self.groups.get(group_id)
            if group is None:
                raise KeyError(group_id)
            merge = group.setdefault("merge", {})
            if merge.get("status") == "running":
                merge["pending"] = True
            else:
                merge.update({"status": "running", "started": _now(), "pending": False})
                threading.Thread(target=self._merge, args=(group_id,), name=f"cv-merge-{group_id}",
                                 daemon=True).start()
            self._save_group(group)
            return self._group_public(group)

    def _merge(self, group_id):
        while True:
            with self.lock:
                group = self.groups.get(group_id)
                if group is None:
                    return
                root_host = group.get("root_host")
            ok, output = crossval.merge(root_host)
            with self.lock:
                if group_id not in self.groups:
                    return
                merge = group["merge"]
                again = merge.pop("pending", False)
                merge.update({"status": "running" if again else ("succeeded" if ok else "failed"),
                              "finished": _now(), "output": output[-4000:]})
                self._save_group(group)
            if not again:
                return

    # ----------------------------------------------------------------- batches
    def _batch_jobs(self, batch_id):
        return [job for job in self.jobs.values() if (job.get("batch") or {}).get("id") == batch_id]

    def _batch_public(self, record):
        data = dict(record)
        attempts = self._batch_jobs(record["id"])
        entries = []
        for config in record["configs"]:
            tries = sorted((job for job in attempts if job["batch"]["index"] == config["index"]),
                           key=lambda j: j["created"])
            latest = tries[-1] if tries else None
            entry = {**config, "job": None, "attempts": len(tries), "status": "removed"}
            if latest:
                entry.update({"job": latest["id"], "status": latest["status"], "started": latest.get("started"),
                              "finished": latest.get("finished")})
            entries.append(entry)
        data["configs"] = entries
        statuses = [entry["status"] for entry in entries]
        data["status"] = _collective_status(statuses)
        data["active"] = any(status in ACTIVE for status in statuses)
        return data

    def list_batches(self):
        with self.lock:
            return sorted((self._batch_public(record) for record in self.batches.values()),
                          key=lambda b: b["created"], reverse=True)

    def get_batch(self, batch_id):
        with self.lock:
            record = self.batches.get(batch_id)
            if record is None:
                raise KeyError(batch_id)
            return self._batch_public(record)

    def cancel_batch(self, batch_id):
        """Cancel the queued jobs of a batch and stop its running ones."""
        with self.lock:
            if batch_id not in self.batches:
                raise KeyError(batch_id)
            for job in self._batch_jobs(batch_id):
                if job["status"] in ("queued", "starting", "running", "pausing"):
                    self.cancel(job["id"])
            return self.get_batch(batch_id)

    def remove_batch(self, batch_id):
        """Forget a finished batch and its jobs; outputs on disk are kept."""
        with self.lock:
            if batch_id not in self.batches:
                raise KeyError(batch_id)
            members = self._batch_jobs(batch_id)
            if any(job["status"] in ACTIVE for job in members):
                raise ValueError("Stop the batch before removing it")
            for job in members:
                self.jobs.pop(job["id"])
            self.batches.pop(batch_id)
        for job in members:
            shutil.rmtree(self._dir(job["id"]), ignore_errors=True)
        shutil.rmtree(self._batch_dir(batch_id), ignore_errors=True)

    def _batch_waits(self, job):
        """Jobs of a batch that preprocess into the same cache folder never preprocess at the same time: a job waits
        while an earlier job of its batch sharing a cache is queued, or while any of them is still preprocessing."""
        member, caches = job.get("batch"), set(job.get("caches") or ())
        if not member or not caches:
            return False
        for other in self.jobs.values():
            if other is job or (other.get("batch") or {}).get("id") != member["id"] or \
                    not caches & set(other.get("caches") or ()):
                continue
            if other["status"] == "queued" and self._position(other) < self._position(job):
                return True
            if other["status"] in RUNNING and (other.get("progress") or {}).get("phase") in (
                    None, "starting", "preprocessing"):
                return True
        return False

    @staticmethod
    def _position(job):
        """Queue order: creation time, then the place in its cross-validation or batch (queued in the same second)."""
        member = job.get("group") or job.get("batch") or {}
        return job["created"], member.get("index", 0)

    def _group_waits(self, job):
        """Folds of one cross-validation run one at a time, in fold order; an unresumed paused fold holds the rest."""
        group = job.get("group")
        if not group:
            return False
        position = (group.get("index", 0), job["created"])
        for other in self._group_jobs(group["id"]):
            if other is job:
                continue
            if other["status"] in RUNNING:
                return True
            if other["status"] == "paused" and not any(j.get("resume_from") == other["id"] for j in self.jobs.values()):
                return True
            if other["status"] == "queued" and (other["group"].get("index", 0), other["created"]) < position:
                return True
        return False

    # ------------------------------------------------------------ monitoring
    def _loop(self):
        while True:
            try:
                self._tick()
            except Exception as error:  # keep the monitor alive
                sys.stderr.write(f"[launcher] monitor error: {error}\n")
            time.sleep(1.0)

    def _tick(self):
        with self.lock:
            jobs = list(self.jobs.values())
        for job in jobs:
            if job["status"] in ("starting", "running", "pausing", "stopping"):
                self._follow(job)
        self._start_queued()

    def _start_queued(self):
        prefs = settings.load()
        with self.lock:
            queued = sorted((j for j in self.jobs.values() if j["status"] == "queued"), key=self._position)
            for job in queued:
                running = [j for j in self.jobs.values() if j["status"] in ("starting", "running", "pausing", "stopping")]
                if len(running) >= prefs["max_parallel_jobs"]:
                    return
                if self._group_waits(job) or self._batch_waits(job):
                    continue
                if job["policy"] != "parallel" and job.get("gpu") is not None:
                    earlier = [j for j in self.jobs.values() if j is not job and j.get("gpu") == job["gpu"] and (
                        j["status"] in ("starting", "running", "pausing", "stopping") or
                        (j["status"] == "queued" and self._position(j) < self._position(job) and
                         j["policy"] != "parallel"))]
                    if earlier:
                        continue
                self._launch(job)

    def _launch(self, job):
        job_dir = self._dir(job["id"])
        (job_dir / "steps.json").write_text(json.dumps(job["steps"], indent=2), encoding="utf-8")
        for stale in ("exit.json", "child.pid"):
            (job_dir / stale).unlink(missing_ok=True)
        env = dict(os.environ)
        env.update(job.get("env") or {})
        env.update(self.secret_env.pop(job["id"], {}))
        env.update({"PYTHONUNBUFFERED": "1", "MPLBACKEND": "Agg", "PYTHONIOENCODING": "utf-8"})
        if job["backend"] == "local":
            env["RPPG_TRAINING_CONTROL"] = str(job_dir / "training-control.json")
        kwargs = {"cwd": str(paths.TOOLBOX_ROOT), "env": env, "stdin": subprocess.DEVNULL,
                  "stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | probes.NO_WINDOW
        else:
            kwargs["start_new_session"] = True
        try:
            process = subprocess.Popen([sys.executable, "-m", "launcher.runner", str(job_dir)], **kwargs)
        except OSError as error:
            job["status"] = "failed"
            job["error"] = f"Could not start the runner: {error}"
            job["finished"] = _now()
            self._save(job)
            return
        self.processes[job["id"]] = process
        self.parsers[job["id"]] = self._new_parser(job)
        self.offsets[job["id"]] = 0
        job.update({"status": "running", "started": _now(), "pid": process.pid})
        job.pop("needs_secret", None)
        self._save(job)

    @staticmethod
    def _summary(job):
        """vHRM evaluations report metrics in summary.json instead of the console; reports in their tables."""
        if job.get("mode") == "report":
            return reports.job_summary(job["outputs"]["report_dir_host"])
        for folder in output_dirs_for(job):
            summary_file = Path(folder) / "summary.json"
            if summary_file.is_file():
                try:
                    return json.loads(summary_file.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    return None
        return None

    def _new_parser(self, job):
        """A resumed attempt carries on the loss curves, best epoch and results of the attempt it resumes."""
        previous = self.jobs.get(job.get("resume_from") or "") or {}
        return logparse.LogParser(job["mode"], job.get("epochs_total"), history=previous.get("progress"))

    def _read_new_output(self, job):
        log_path = self._dir(job["id"]) / "output.log"
        offset = self.offsets.get(job["id"], 0)
        try:
            with log_path.open("rb") as handle:
                handle.seek(offset)
                data = handle.read(4 * 1024 * 1024)
        except OSError:
            return False
        if not data:
            return False
        self.offsets[job["id"]] = offset + len(data)
        parser = self.parsers.setdefault(job["id"], self._new_parser(job))
        parser.feed(data.decode("utf-8", errors="replace"))
        # config.py can rewrite weight prefixes for motion augmentation. Track
        # the actual committed checkpoint rather than assuming the YAML prefix.
        checkpoint = parser.state.get("resume_checkpoint")
        if checkpoint and job.get("training_recovery"):
            host, problem = paths.config_host_path(checkpoint, job["backend"])
            folder = job.get("outputs", {}).get("model_dir_host")
            if (not problem and host is not None and folder
                    and host.parent == Path(folder).resolve()
                    and host.name.endswith("_resume.pt")):
                job["resume_checkpoint"] = checkpoint
                job["resume_checkpoint_host"] = str(host)
        return True

    def _follow(self, job):
        changed = self._read_new_output(job)
        exit_file = self._dir(job["id"]) / "exit.json"
        process = self.processes.get(job["id"])
        finished = exit_file.is_file()
        alive = process.poll() is None if process is not None else probes.pid_alive(job.get("pid"))
        with self.lock:
            parser = self.parsers.get(job["id"])
            if parser is not None and changed:
                job["progress"] = parser.snapshot()
            if finished or not alive:
                while self._read_new_output(job):
                    pass
                if parser is not None:
                    job["progress"] = parser.snapshot()
                result = {}
                if finished:
                    try:
                        result = json.loads(exit_file.read_text(encoding="utf-8"))
                    except (OSError, ValueError):
                        result = {}
                code = result.get("returncode")
                job["returncode"] = code
                job["finished"] = result.get("finished") or _now()
                if job.get("cancel_requested"):
                    job["status"] = "cancelled"
                elif code == 75 and job.get("pause_requested") and job.get("resume_checkpoint_host") and \
                        Path(job["resume_checkpoint_host"]).is_file():
                    job["status"] = "paused"
                elif not finished:
                    job["status"] = "interrupted"
                    job["error"] = "The runner process ended without reporting an exit code."
                elif code == 0:
                    job["status"] = "succeeded"
                else:
                    job["status"] = "failed"
                    progress = job.get("progress") or {}
                    job["error"] = progress.get("error") or f"Exited with code {code}."
                if job["status"] in FINISHED:
                    job.get("progress", {})["phase"] = job["status"]
                    job["summary"] = self._summary(job)
                self.processes.pop(job["id"], None)
                self.parsers.pop(job["id"], None)
                self._save(job)
                group_id = (job.get("group") or {}).get("id")
                if job["status"] == "succeeded" and group_id in getattr(self, "groups", {}):
                    self.request_merge(group_id)
            elif changed:
                self._save(job)


def output_dirs_for(job):
    outputs = job.get("outputs") or {}
    return [outputs[k] for k in ("model_dir_host", "test_outputs_host", "outputs_host", "report_dir_host")
            if outputs.get(k)]
