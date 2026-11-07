"""Preprocessing first: a batch preprocesses its caches before its trainings (launcher/preprocess_first.py).

The single-run checks are scripted, as in test_launcher_batch: each config has a training and a test split whose cache
folder and file list are chosen per test. No process is started except by the runner test (two tiny Python steps).
"""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from launcher import batch, configs, jobs, logparse, paths, preflight, preprocess_first, resources, runner  # noqa: E402

FOLDER = "configs/launcher/demo"
PREFS = {"max_parallel_jobs": 2, "gpu_policy": "queue", "max_parallel_preprocess": 2, "preprocess_policy": "with_jobs",
         "local_python": sys.executable}


def config_values(name, **overrides):
    flat = {"TOOLBOX_MODE": "train_and_test", "MODEL.NAME": "Physnet", "TEST.USE_LAST_EPOCH": True,
            "TRAIN.DATA.DATASET": "vHRM", "TEST.DATA.DATASET": "vHRM", "TRAIN.DATA.DO_PREPROCESS": True,
            "TEST.DATA.DO_PREPROCESS": True, "TRAIN.DATA.VHRM.PARTICIPANTS": ["p1", "p2"],
            "TEST.DATA.VHRM.PARTICIPANTS": ["p3"], "LOG.PATH": f"C:/Run/{name}", "WANDB.ENABLED": True,
            "DEVICE": "cuda:0"}
    flat.update(overrides)
    values, errors, _ = configs.normalize(flat)
    assert not errors, errors
    return values


class FakeChecks:
    """Scripted single-run checks: every config can launch; splits[(log, prefix)] = (cache folder, file list)."""

    def __init__(self, root):
        self.root = root
        self.splits = {}

    def __call__(self, values, extras, backend, answers, secrets, manager):
        effective = preflight.apply_answers(values, answers)
        log = values["LOG.PATH"]
        derived = {"outputs": {"log_path_host": effective["LOG.PATH"], "model_dir_host": effective["LOG.PATH"] + "/m"}}
        for prefix in ("TRAIN.DATA", "TEST.DATA"):
            cache, listed = self.splits.get((log, prefix), ("shared", prefix.split(".")[0].lower()))
            derived[prefix] = {"cache_dir_host": str(self.root / cache), "exp_data_name": cache,
                               "file_list_host": str(self.root / "DataFileLists" / f"{listed}.csv")}
        checks = [{"id": "validation", "title": "Configuration", "status": "ok", "message": "Valid", "details": []}]
        return {"checks": checks, "can_launch": True, "effective_values": effective,
                "validation": {"derived": derived}, "answers": answers}


class Fixture(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)
        self.fake = FakeChecks(self.folder / "cache")
        self.values = {}
        for target, name, replacement in ((preflight, "run_checks", self.fake), (configs, "load_config", self.load),
                                          (batch.settings, "load", lambda: dict(PREFS)),
                                          (jobs.settings, "load", lambda: dict(PREFS))):
            patcher = patch.object(target, name, replacement)
            patcher.start()
            self.addCleanup(patcher.stop)
        for name in ("JOBS_DIR", "CV_DIR", "BATCH_DIR"):
            patcher = patch.object(paths, name, self.folder / name.lower())
            patcher.start()
            self.addCleanup(patcher.stop)
            (self.folder / name.lower()).mkdir()
        self.manager = self.new_manager()

    @staticmethod
    def new_manager():
        manager = jobs.JobManager.__new__(jobs.JobManager)
        manager.lock = threading.RLock()
        for name in ("jobs", "groups", "batches", "processes", "parsers", "offsets", "secret_env"):
            setattr(manager, name, {})
        return manager

    def add(self, name, **overrides):
        path = f"{FOLDER}/{name}.yaml"
        self.values[path] = config_values(name, **overrides)
        return path

    def load(self, path):
        return {"path": path, "values": self.values[path], "extras": {}, "type_errors": {}, "backend": "local"}

    def split(self, name, prefix, cache, listed):
        self.fake.splits[(f"C:/Run/{name}", prefix)] = (cache, listed)

    def check(self, config_paths, answers):
        return batch.run_checks(config_paths, "local", answers, {}, self.manager)

    def launch(self, config_paths, preprocess="parallel", policy="queue", **answers):
        return self.manager.create_batch({"backend": "local", "configs": config_paths, "launch": config_paths,
                                          "name": "Demo", "answers": {"wandb": "existing", "policy": policy,
                                                                      "preprocess": preprocess, **answers}})

    def members(self, batch_id, stage=None):
        found = [job for job in self.manager._batch_jobs(batch_id) if job["batch"].get("stage") == stage]
        return sorted(found, key=lambda job: job["batch"].get("index", job["batch"].get("task")))

    def start(self, started=None):
        """One pass of the queue; launching marks a job running."""
        started = [] if started is None else started

        def launch(job):
            started.append(job["name"])
            job.update(status="running", progress={"phase": "starting"})

        with patch.object(self.manager, "_launch", side_effect=launch):
            self.manager._start_queued()
        return started

    def finish(self, job, status="succeeded", codes=None):
        """A preprocessing job ends (codes: exit code per runner step)."""
        job["status"] = status
        if codes is None:
            codes = [0] * len(job["steps"])
        self.manager._finish_preprocess(job, {"step_codes": codes})


class PlanTests(Fixture):
    def entries(self, *config_paths):
        result = self.check(list(config_paths), {"wandb": "existing", "policy": "queue", "preprocess": "parallel"})
        return result["plan"]

    def test_same_split_of_several_configs_is_preprocessed_once(self):
        first, second = self.add("a"), self.add("b", **{"MODEL.NAME": "Tscan"})
        plan = self.entries(first, second)
        self.assertEqual(len(plan["caches"]), 1, "both splits of both configs fill one cache folder")
        cache = plan["caches"][0]
        self.assertEqual(cache["steps"], [{"config": first, "name": "a", "splits": ["TRAIN.DATA", "TEST.DATA"]}],
                         "the second config needs nothing more: its splits are the first config's")
        self.assertEqual(cache["configs"], ["a", "b"])
        self.assertEqual(plan["counts"], {"splits": 4, "lists": 2, "duplicates": 2, "in_jobs": 0, "refresh": 0})
        self.assertEqual(plan["splits"][second]["TEST.DATA"]["action"], "reuse")
        self.assertEqual(plan["splits"][second]["TEST.DATA"]["part"], 0)

    def test_tasks_are_grouped_by_cache_folder_in_batch_order(self):
        first, second = self.add("a"), self.add("b", **{"TRAIN.DATA.VHRM.PARTICIPANTS": ["p4"]})
        self.split("a", "TEST.DATA", "uv", "test")
        self.split("b", "TRAIN.DATA", "shared", "train_b")
        self.split("b", "TEST.DATA", "uv", "test")
        plan = self.entries(first, second)
        self.assertEqual([cache["name"] for cache in plan["caches"]], ["shared", "uv"])
        shared, uv = plan["caches"]
        self.assertEqual([(step["name"], step["splits"]) for step in shared["steps"]],
                         [("a", ["TRAIN.DATA"]), ("b", ["TRAIN.DATA"])], "another file list in the same folder")
        self.assertEqual(shared["lists"], 2)
        self.assertEqual([(step["name"], step["splits"]) for step in uv["steps"]], [("a", ["TEST.DATA"])])
        self.assertEqual(plan["splits"][second]["TRAIN.DATA"]["part"], 1)

    def test_one_file_list_with_different_settings(self):
        # vHRM file list names leave out the label type: both splits cannot share the list.
        first = self.add("a")
        second = self.add("b", **{"TRAIN.DATA.PREPROCESS.LABEL_TYPE": "Standardized"})
        plan = self.entries(first, second)
        self.assertEqual(plan["splits"][first]["TRAIN.DATA"]["action"], "refresh",
                         "incremental vHRM: decoded once, then each training writes its own list")
        self.assertEqual(plan["splits"][second]["TRAIN.DATA"]["action"], "refresh")
        self.assertEqual(plan["splits"][first]["TEST.DATA"]["action"], "reuse")
        self.assertEqual(plan["counts"]["refresh"], 2)
        third = self.add("c", **{"TRAIN.DATA.PREPROCESS.LABEL_TYPE": "Standardized",
                                 "TRAIN.DATA.VHRM.INCREMENTAL_PREPROCESS": False})
        plan = self.entries(first, third)
        self.assertNotIn("TRAIN.DATA", plan["splits"][first], "other datasets keep their preprocessing in the job")
        self.assertEqual(plan["counts"]["in_jobs"], 2)

    def test_runtime_settings_and_roles(self):
        values = config_values("a")
        other = dict(values, **{"TRAIN.DATA.PREPROCESS.NUM_WORKERS": 12, "TRAIN.DATA.DATA_FORMAT": "NDCHW",
                                "TRAIN.DATA.DO_PREPROCESS": False})
        self.assertEqual(preprocess_first.split_identity(values, "TRAIN.DATA"),
                         preprocess_first.split_identity(other, "TRAIN.DATA"))
        same = dict(values, **{"TEST.DATA." + key[len("TRAIN.DATA."):]: value for key, value in values.items()
                               if key.startswith("TRAIN.DATA.")})
        self.assertNotEqual(preprocess_first.split_identity(same, "TRAIN.DATA"),
                            preprocess_first.split_identity(same, "TEST.DATA"), "test clips are kept differently")

    def test_step_values(self):
        values = preprocess_first.step_values(config_values("a"), ["TEST.DATA"])
        self.assertFalse(values["TRAIN.DATA.DO_PREPROCESS"])
        self.assertTrue(values["TEST.DATA.DO_PREPROCESS"])
        self.assertFalse(values["WANDB.ENABLED"])


class QuestionTests(Fixture):
    def test_question_counts_the_work(self):
        first, second = self.add("a"), self.add("b")
        result = self.check([first, second], {"wandb": "existing", "policy": "parallel"})
        question = {check["id"]: check for check in result["checks"]}["preprocess"]
        self.assertEqual(question["status"], "question")
        self.assertEqual(question["default"], "with_jobs")
        self.assertEqual([option["value"] for option in question["options"]], ["with_jobs", "queue", "parallel", "auto"])
        self.assertIn("4 data split(s) to preprocess, into 1 cache folder(s): 2 file list(s) (2 split(s) shared",
                      question["message"])
        self.assertIn("Up to 2", question["options"][2]["label"].replace("up to", "Up to"))
        self.assertNotIn("plan", batch.public(result))

    def test_order_question_follows_the_choice(self):
        first, second = self.add("a"), self.add("b")
        with_jobs = self.check([first, second], {"wandb": "existing", "policy": "parallel", "preprocess": "with_jobs"})
        ids = {check["id"]: check for check in with_jobs["checks"]}
        self.assertIn("caches", ids, "jobs filling one cache take turns")
        first_ = self.check([first, second], {"wandb": "existing", "policy": "parallel", "preprocess": "auto"})
        ids = {check["id"]: check for check in first_["checks"]}
        self.assertNotIn("caches", ids, "trainings only read the caches")
        self.assertIn("once the preprocessing is done", ids["policy"]["message"])
        self.assertTrue(first_["can_launch"])

    def test_no_question_without_preprocessing(self):
        only = self.add("a", **{"TRAIN.DATA.DO_PREPROCESS": False, "TEST.DATA.DO_PREPROCESS": False})
        result = self.check([only], {"wandb": "existing", "policy": "queue"})
        self.assertNotIn("preprocess", {check["id"] for check in result["checks"]})
        record = self.launch([only])
        self.assertNotIn("preprocess", record)
        self.assertNotIn("preprocess", record["answers"])
        self.assertEqual(self.members(record["id"], jobs.PREPROCESS), [])


class StageTests(Fixture):
    def setUp(self):
        super().setUp()
        self.paths = [self.add("a"), self.add("b", **{"TRAIN.DATA.VHRM.PARTICIPANTS": ["p4"]}), self.add("c")]
        self.split("a", "TEST.DATA", "uv", "test")
        self.split("b", "TRAIN.DATA", "shared", "train_b")
        self.split("c", "TRAIN.DATA", "third", "train_c")
        self.split("c", "TEST.DATA", "third", "test_c")

    def lists(self):
        """Write the file lists, as the preprocessing jobs would."""
        folder = self.folder / "cache" / "DataFileLists"
        folder.mkdir(parents=True, exist_ok=True)
        for name in ("train", "test", "train_b", "train_c", "test_c"):
            (folder / f"{name}.csv").write_text("input_files\n", encoding="utf-8")

    def config_of(self, job):
        return yaml.safe_load((paths.JOBS_DIR / job["id"] / "config.yaml").read_text(encoding="utf-8"))

    def test_launch_queues_one_preprocessing_job_per_cache(self):
        record = self.launch(self.paths, "auto")
        self.assertEqual(record["answers"]["preprocess"], "auto")
        self.assertEqual(record["preprocess"]["policy"], "auto")
        self.assertEqual(record["preprocess"]["caches"], 3)
        stage = self.members(record["id"], jobs.PREPROCESS)
        self.assertEqual([job["name"] for job in stage], ["Preprocess · shared", "Preprocess · uv", "Preprocess · third"])
        shared = stage[0]
        self.assertEqual([step["title"] for step in shared["steps"]],
                         ["Preprocess a: Training data", "Preprocess b: Training data"])
        self.assertTrue(all(step["continue_on_error"] for step in shared["steps"]))
        self.assertEqual(shared["steps"][0]["argv"][-1], "--preprocess_only")
        snapshot = configs.read_yaml_tree(paths.JOBS_DIR / shared["id"] / "preprocess-1.yaml")
        self.assertTrue(snapshot["TRAIN"]["DATA"]["DO_PREPROCESS"])
        self.assertFalse(snapshot["TEST"]["DATA"]["DO_PREPROCESS"], "the test split is another job's")
        self.assertFalse(snapshot["WANDB"]["ENABLED"])
        self.assertIn("Step 2: b (Training data)", self.manager.config_text(shared["id"]))
        trainings = self.members(record["id"])
        self.assertEqual([job["name"] for job in trainings], ["a", "b", "c"])
        self.assertEqual(trainings[1]["after_preprocess"]["TRAIN.DATA"]["job"], shared["id"])
        self.assertEqual(trainings[1]["after_preprocess"]["TRAIN.DATA"]["part"], 1)
        self.assertEqual(trainings[1]["after_preprocess"]["TEST.DATA"]["job"], stage[1]["id"],
                         "b's test split is a's: preprocessed once")
        public = self.manager.get_batch(record["id"])
        self.assertEqual([entry["name"] for entry in public["configs"]], ["a", "b", "c"],
                         "preprocessing jobs are not configs of the batch")
        self.assertEqual(len(public["preprocess"]["jobs"]), 3)
        with self.assertRaisesRegex(ValueError, "preprocessing job has no such parameters"):
            self.manager.edit_queued({"jobs": [shared["id"]], "changes": {"TRAIN.EARLY_STOPPING.PATIENCE": 3}})

    def test_trainings_wait_for_the_whole_stage_then_reuse_the_caches(self):
        record = self.launch(self.paths, "parallel")
        stage = self.members(record["id"], jobs.PREPROCESS)
        trainings = self.members(record["id"])
        self.assertEqual(self.start(), ["Preprocess · shared", "Preprocess · uv"], "up to 2 preprocessing jobs")
        self.assertEqual(self.manager.get_batch(record["id"])["status"], "running")
        self.lists()
        self.finish(stage[0])
        self.assertEqual(self.start(), ["Preprocess · third"], "no training while the stage runs")
        self.finish(stage[1])
        self.finish(stage[2])
        started = self.start()
        self.assertEqual(started, ["a"], "then the trainings, in the GPU order of the batch")
        for job in trainings:
            self.assertEqual(job["preprocessed"], {"reused": ["TRAIN.DATA", "TEST.DATA"], "kept": []})
            self.assertEqual(job["caches"], [], "they only read the caches: no turns to take")
            tree = self.config_of(job)
            self.assertFalse(tree["TRAIN"]["DATA"]["DO_PREPROCESS"])
            self.assertFalse(tree["TEST"]["DATA"]["DO_PREPROCESS"])
        self.assertFalse(self.manager.get_batch(record["id"])["preprocess"]["active"])

    def test_one_at_a_time_and_the_training_limit_is_separate(self):
        record = self.launch(self.paths, "queue", policy="parallel")
        stage = self.members(record["id"], jobs.PREPROCESS)
        self.assertEqual(self.start(), ["Preprocess · shared"])
        self.assertEqual(self.start(), [])
        self.lists()
        for job in stage:
            self.finish(job)
            job["status"] = "succeeded"
        self.assertEqual(self.start(), ["a", "b"], "max_parallel_jobs applies to the trainings only")

    def test_a_cache_folder_has_one_writer(self):
        first = self.launch(self.paths[:1], "parallel")
        self.split("c", "TRAIN.DATA", "shared", "train_c")
        self.split("c", "TEST.DATA", "shared", "test_c")
        second = self.manager.create_batch({"backend": "local", "configs": self.paths[2:], "launch": self.paths[2:],
                                            "name": "Other", "answers": {"wandb": "existing", "policy": "queue",
                                                                         "preprocess": "parallel"}})
        mine, other = self.members(first["id"], jobs.PREPROCESS)[0], self.members(second["id"], jobs.PREPROCESS)[0]
        self.assertEqual(mine["caches"], other["caches"])
        with patch.object(jobs.settings, "load", return_value=dict(PREFS, max_parallel_preprocess=8)):
            started = self.start()
        self.assertEqual(sorted(started), ["Preprocess · shared", "Preprocess · uv"])
        self.assertEqual((mine["status"], other["status"]), ("running", "queued"),
                         "another batch's job waits for the folder, whatever the limit")
        mine["status"] = "succeeded"
        with patch.object(jobs.settings, "load", return_value=dict(PREFS, max_parallel_preprocess=8)):
            self.assertEqual(self.start(), ["Preprocess · shared"])
        self.assertEqual(other["status"], "running")

    def test_a_split_whose_preprocessing_failed_is_preprocessed_by_its_training(self):
        record = self.launch(self.paths, "parallel")
        stage = self.members(record["id"], jobs.PREPROCESS)
        self.lists()
        self.start()
        self.finish(stage[0], "failed", codes=[0, 1])  # b's training split failed
        self.assertIn("1 of 2 step(s) did not finish (b)", stage[0]["error"])
        self.finish(stage[1])
        stage[2]["status"] = "cancelled"  # c: never ran
        self.start()
        a, b, c = self.members(record["id"])
        self.assertEqual(a["preprocessed"]["reused"], ["TRAIN.DATA", "TEST.DATA"])
        self.assertEqual(b["preprocessed"], {"reused": ["TEST.DATA"], "kept": ["TRAIN.DATA"]})
        self.assertTrue(self.config_of(b)["TRAIN"]["DATA"]["DO_PREPROCESS"])
        self.assertEqual(b["caches"], [os.path.normcase(str(self.folder / "cache" / "shared"))])
        self.assertEqual(c["preprocessed"]["kept"], ["TRAIN.DATA", "TEST.DATA"])

    def test_preprocessing_jobs_never_make_trainings_wait_for_the_gpu(self):
        record = self.launch(self.paths[:1], "parallel")
        single = self.manager._queue(self.values[self.paths[2]], {}, "local", {"policy": "queue"}, {},
                                     {"model_dir_host": "C:/Run/single/m"}, "single", None, None)
        stage = self.members(record["id"], jobs.PREPROCESS)
        started = self.start()
        self.assertIn("single", started, "the running preprocessing jobs share the GPU with it")
        self.assertEqual(self.manager.running_count(), 1)
        self.assertEqual(sorted(job["name"] for job in self.manager.jobs_on_gpu(0)), ["a", "single"],
                         "the queued training of the batch, not its preprocessing jobs")
        self.assertEqual(len([job for job in stage if job["status"] == "running"]), 2)

    def test_step_codes_from_the_log_when_the_runner_was_stopped(self):
        record = self.launch(self.paths, "parallel")
        shared = self.members(record["id"], jobs.PREPROCESS)[0]
        shared["status"] = "cancelled"
        shared["progress"] = {"step_codes": [0]}
        self.manager._finish_preprocess(shared, {})
        self.assertEqual([part["returncode"] for part in shared["preprocess"]["parts"]], [0, None])

    def test_stop_remove_and_reload(self):
        record = self.launch(self.paths, "auto")
        reloaded = self.new_manager()
        reloaded._load()
        self.assertEqual(len(reloaded.get_batch(record["id"])["preprocess"]["jobs"]), 3)
        stopped = self.manager.cancel_batch(record["id"])
        self.assertEqual({job["status"] for job in stopped["preprocess"]["jobs"]}, {"cancelled"})
        self.assertFalse(stopped["active"])
        self.manager.remove_batch(record["id"])
        self.assertEqual(self.manager.jobs, {})

    def test_docker_steps_and_image_build(self):
        record = self.manager.create_batch({"backend": "docker", "configs": self.paths, "launch": self.paths,
                                            "name": "Demo", "answers": {"wandb": "existing", "policy": "queue",
                                                                        "preprocess": "parallel", "build": "build",
                                                                        "gpu": "0"}})
        stage = self.members(record["id"], jobs.PREPROCESS)
        trainings = self.members(record["id"])
        self.assertEqual(stage[0]["steps"][0]["title"], "Build Docker image")
        self.assertEqual(stage[0]["preprocess"]["parts"][0]["step"], 1)
        self.assertEqual(stage[0]["containers"], [f"rppg-launcher-{stage[0]['id']}-1",
                                                  f"rppg-launcher-{stage[0]['id']}-2"])
        self.assertEqual(stage[1]["preprocess"]["after_build"], stage[0]["id"])
        self.assertFalse(any(step["title"] == "Build Docker image" for job in trainings for step in job["steps"]))
        self.assertEqual(self.start(), ["Preprocess · shared"], "the others wait for the image")


class FakeMonitor(resources.Monitor):
    def __init__(self):
        self.now = 1000.0
        self.reading = {"cpu": 20.0, "ram": 40 * resources.GIB, "vram": 2 * resources.GIB, "use": 10}
        super().__init__(clock=lambda: self.now, gpus=self.gpus_, cpu=self, mem=self.mem_)

    def percent(self):
        return self.reading["cpu"]

    def mem_(self):
        return self.reading["ram"], 64 * resources.GIB

    def gpus_(self):
        return [{"index": 0, "memory_used": self.reading["vram"] // resources.MIB, "memory_total": 24576,
                 "utilization": self.reading["use"]}]

    def run(self, seconds, scaler, **reading):
        self.reading.update(reading)
        for _ in range(int(seconds)):
            self.now += 1
            scaler.sample()


class AutoScalerTests(unittest.TestCase):
    def setUp(self):
        self.monitor = FakeMonitor()
        self.scaler = resources.AutoScaler(self.monitor)

    def test_one_more_job_when_the_load_leaves_room(self):
        self.monitor.run(5, self.scaler)
        self.assertTrue(self.scaler.decide([], 0)[0], "the first job starts at once")
        first = {"id": "one", "gpu": 0}
        self.scaler.started(first)
        self.monitor.run(30, self.scaler, cpu=35.0, ram=34 * resources.GIB, vram=3 * resources.GIB)
        allowed, reason = self.scaler.decide([first], 0)
        self.assertFalse(allowed)
        self.assertIn("Measuring the job started", reason)
        self.monitor.run(40, self.scaler)
        allowed, reason = self.scaler.decide([first], 0)
        self.assertTrue(allowed, reason)
        estimate = self.scaler.estimates()
        self.assertAlmostEqual(estimate["ram"] / resources.GIB, 6, delta=0.01, msg="40 GB free before, 34 GB after")
        self.assertAlmostEqual(estimate["vram"] / resources.GIB, 1, delta=0.01)
        self.assertAlmostEqual(estimate["cpu"], 15, delta=0.5)
        status = self.scaler.snapshot()
        self.assertEqual((status["running"], status["can_start"], status["estimate"]["ram_gb"]), (1, True, 6.0))

    def test_it_waits_when_a_resource_runs_short(self):
        first = {"id": "one", "gpu": 0}
        self.monitor.run(5, self.scaler)
        self.scaler.started(first)
        self.monitor.run(100, self.scaler, ram=30 * resources.GIB)  # the job takes 10 GB
        self.monitor.run(20, self.scaler, ram=12 * resources.GIB)  # later, a training starts elsewhere
        self.assertAlmostEqual(self.scaler.estimates()["ram"] / resources.GIB, 10, delta=0.01,
                               msg="the load after the first 90 s is not the job's")
        allowed, reason = self.scaler.decide([first], 0)
        self.assertFalse(allowed)
        self.assertIn("GB of memory free", reason)
        self.monitor.run(20, self.scaler, ram=30 * resources.GIB, cpu=90.0)
        allowed, reason = self.scaler.decide([first], 0)
        self.assertFalse(allowed)
        self.assertIn("CPU is 90% busy", reason)
        self.monitor.run(20, self.scaler, cpu=30.0, use=97)
        self.assertIn("GPU 0 is 97% busy", self.scaler.decide([first], 0)[1])
        self.monitor.run(20, self.scaler, use=40, vram=23 * resources.GIB)
        self.assertIn("GPU 0: 1.0 GB of memory free", self.scaler.decide([first], 0)[1])
        self.monitor.run(20, self.scaler, vram=4 * resources.GIB)
        self.assertTrue(self.scaler.decide([first], 0)[0])

    def test_a_finished_job_does_not_need_to_settle(self):
        first, second = {"id": "one", "gpu": 0}, {"id": "two", "gpu": 0}
        self.monitor.run(70, self.scaler)
        self.scaler.started(first)
        self.monitor.run(70, self.scaler)
        self.scaler.started(second)
        self.monitor.run(20, self.scaler)  # "two" was quick and ended; "one" still runs
        self.assertTrue(self.scaler.decide([first], 0)[0])

    def test_the_monitor_counters(self):
        available, total = resources.memory()
        self.assertGreater(total, available)
        counter = resources.CpuCounter()
        value = counter.percent()
        self.assertTrue(value is None or 0 <= value <= 100)


class RunnerAndLogTests(unittest.TestCase):
    def test_independent_steps_continue_after_a_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            steps = [{"title": f"step {code}", "argv": [sys.executable, "-c", f"raise SystemExit({code})"],
                      "continue_on_error": True} for code in (0, 3, 0)]
            (Path(folder) / "steps.json").write_text(json.dumps(steps), encoding="utf-8")
            self.assertEqual(runner.main(folder), 3)
            result = json.loads((Path(folder) / "exit.json").read_text(encoding="utf-8"))
            self.assertEqual((result["step_codes"], result["failed_step"]), ([0, 3, 0], 1))
            parser = logparse.LogParser("preprocess")
            parser.feed((Path(folder) / "output.log").read_text(encoding="utf-8"))
            state = parser.snapshot()
            self.assertEqual(state["step_codes"], [0, 3, 0])
            self.assertEqual(state["step"], {"index": 3, "total": 3, "title": "step 0"})
            steps[1].pop("continue_on_error")
            (Path(folder) / "steps.json").write_text(json.dumps(steps), encoding="utf-8")
            self.assertEqual(runner.main(folder), 3)
            result = json.loads((Path(folder) / "exit.json").read_text(encoding="utf-8"))
            self.assertEqual(result["step_codes"], [0, 3], "an ordinary job stops at its first failure")

    def test_a_cut_traceback_ends_with_its_step(self):
        parser = logparse.LogParser("preprocess")
        parser.feed("Traceback (most recent call last)\n  File \"main.py\"\n"
                    "[launcher] 2026-11-08T10:00:00 step finished with exit code -9 after 3s\n"
                    "[launcher] 2026-11-08T10:00:01 step 2/2: Preprocess b: Test data\n")
        state = parser.snapshot()
        self.assertEqual(state["step_codes"], [-9])
        self.assertEqual(state["step"]["index"], 2)


class MainFlagTests(unittest.TestCase):
    def test_main_accepts_the_flag(self):
        completed = subprocess.run([sys.executable, "-c", "import argparse, main; p = main.add_args("
                                    "argparse.ArgumentParser()); print(p.parse_args(['--preprocess_only'])"
                                    ".preprocess_only)"], cwd=ROOT, capture_output=True, text=True, timeout=300)
        self.assertEqual(completed.stdout.strip().splitlines()[-1], "True", completed.stderr[-2000:])


if __name__ == "__main__":
    unittest.main()
