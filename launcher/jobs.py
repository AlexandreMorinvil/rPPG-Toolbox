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

from . import configs, logparse, paths, preflight, probes, settings

ACTIVE = ("queued", "starting", "running", "stopping")
FINISHED = ("succeeded", "failed", "cancelled", "interrupted")


def _now():
    return datetime.now().isoformat(timespec="seconds")


class JobManager:
    def __init__(self):
        paths.JOBS_DIR.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.jobs = {}
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

    def _load(self):
        for job_file in sorted(paths.JOBS_DIR.glob("*/job.json")):
            try:
                job = json.loads(job_file.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            self.jobs[job["id"]] = job
            if job["status"] in ("starting", "running", "stopping"):
                self.parsers[job["id"]] = logparse.LogParser(job["mode"], job.get("epochs_total"))
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

    @staticmethod
    def _public(job, full=False):
        data = dict(job)
        if not full:
            data.pop("steps", None)
        return data

    def running_count(self):
        with self.lock:
            return sum(1 for job in self.jobs.values() if job["status"] in ("starting", "running", "stopping"))

    def jobs_on_gpu(self, gpu):
        with self.lock:
            return [self._public(job) for job in self.jobs.values()
                    if job["status"] in ACTIVE and job.get("gpu") == gpu]

    def create(self, payload):
        """Re-run preflight with the submitted answers and queue the job when everything is resolved."""
        backend = payload.get("backend")
        values, type_errors, _ = configs.normalize(payload.get("values") or {})
        extras = payload.get("extras") or {}
        if type_errors:
            raise ValueError("Fix the invalid fields before launching.")
        answers = dict(payload.get("answers") or {})
        answers["_stamp"] = datetime.now().strftime("%Y%m%d-%H%M%S")
        secrets = payload.get("secrets") or {}
        result = preflight.run_checks(values, extras, backend, answers, secrets, self)
        if not result["can_launch"]:
            blocking = [c["title"] + ": " + c["message"] for c in result["checks"] if c["status"] in ("error",
                                                                                                    "question")]
            raise ValueError("Launch blocked. " + " | ".join(blocking))
        answers = result["answers"]
        effective = result["effective_values"]

        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        job_id = f"{stamp}-{pysecrets.token_hex(3)}"
        job_dir = self._dir(job_id)
        job_dir.mkdir(parents=True)
        config_path = job_dir / "config.yaml"
        config_path.write_text(configs.dump_yaml(configs.build_tree(effective, extras)), encoding="utf-8")
        (job_dir / "output.log").touch()

        mode = effective["TOOLBOX_MODE"]
        gpu = preflight.gpu_index(effective, backend, answers)
        env_additions = {}
        if backend == "docker":
            env_additions["GPU_DEVICE_ID"] = str(gpu if gpu is not None else 0)
        key = (secrets.get("wandb_api_key") or "").strip()
        needs_secret = answers.get("wandb") == "key" and bool(key)
        if needs_secret:
            self.secret_env[job_id] = {"WANDB_API_KEY": key}

        steps = self._steps(job_id, backend, config_path, answers)
        outputs = result["validation"]["derived"].get("outputs", {})
        name = (payload.get("name") or "").strip() or self._default_name(effective)
        job = {
            "id": job_id,
            "name": name[:120],
            "created": _now(),
            "started": None,
            "finished": None,
            "status": "queued",
            "backend": backend,
            "mode": mode,
            "model": effective.get("MODEL.NAME") if mode != "unsupervised_method" else
            ",".join(effective.get("UNSUPERVISED.METHOD", [])),
            "datasets": {
                "train": effective.get("TRAIN.DATA.DATASET") if mode == "train_and_test" else None,
                "test": effective.get("UNSUPERVISED.DATA.DATASET") if mode == "unsupervised_method"
                else effective.get("TEST.DATA.DATASET"),
            },
            "epochs_total": effective.get("TRAIN.EPOCHS") if mode == "train_and_test" else None,
            "gpu": gpu,
            "policy": answers.get("policy", "queue"),
            "env": env_additions,
            "needs_secret": needs_secret,
            "wandb": {"enabled": bool(effective.get("WANDB.ENABLED")), "mode": effective.get("WANDB.MODE")},
            "source_config": payload.get("source"),
            "outputs": outputs,
            "steps": steps,
            "pid": None,
            "container": f"rppg-launcher-{job_id}" if backend == "docker" else None,
            "returncode": None,
            "error": None,
            "progress": {},
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
        steps.append({"title": "Run main.py in Docker", "cwd": root, "argv": compose + [
            "run", "--rm", "-T", "--name", f"rppg-launcher-{job_id}", "-v", mount,
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
            if job["status"] not in ("starting", "running"):
                raise ValueError(f"Job is {job['status']}")
            job["status"] = "stopping"
            job["cancel_requested"] = True
            self._save(job)
        threading.Thread(target=self._terminate, args=(dict(job),), daemon=True).start()
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
            if job["status"] in ("starting", "running", "stopping"):
                self._follow(job)
        self._start_queued()

    def _start_queued(self):
        prefs = settings.load()
        with self.lock:
            queued = sorted((j for j in self.jobs.values() if j["status"] == "queued"), key=lambda j: j["created"])
            for job in queued:
                running = [j for j in self.jobs.values() if j["status"] in ("starting", "running", "stopping")]
                if len(running) >= prefs["max_parallel_jobs"]:
                    return
                if job["policy"] != "parallel" and job.get("gpu") is not None:
                    earlier = [j for j in self.jobs.values() if j is not job and j.get("gpu") == job["gpu"] and (
                        j["status"] in ("starting", "running", "stopping") or
                        (j["status"] == "queued" and j["created"] < job["created"] and j["policy"] != "parallel"))]
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
        self.parsers[job["id"]] = logparse.LogParser(job["mode"], job.get("epochs_total"))
        self.offsets[job["id"]] = 0
        job.update({"status": "running", "started": _now(), "pid": process.pid})
        job.pop("needs_secret", None)
        self._save(job)

    @staticmethod
    def _summary(job):
        """vHRM evaluations report metrics in summary.json instead of the console."""
        for folder in output_dirs_for(job):
            summary_file = Path(folder) / "summary.json"
            if summary_file.is_file():
                try:
                    return json.loads(summary_file.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    return None
        return None

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
        parser = self.parsers.setdefault(job["id"], logparse.LogParser(job["mode"], job.get("epochs_total")))
        parser.feed(data.decode("utf-8", errors="replace"))
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
            elif changed:
                self._save(job)


def output_dirs_for(job):
    outputs = job.get("outputs") or {}
    return [outputs[k] for k in ("model_dir_host", "test_outputs_host", "outputs_host") if outputs.get(k)]
