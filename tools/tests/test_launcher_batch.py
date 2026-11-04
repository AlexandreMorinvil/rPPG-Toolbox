"""Batches in the launcher: one round of checks for many configs, then one job per config.

The single-run checks (preflight.run_checks) are replaced by a scripted fake, so the tests cover what the batch adds:
shared questions asked once, answers applied only where they were asked, config statuses, grouped problems, and the
jobs. No job is started: the job manager is built without its monitor thread, as in test_launcher_crossval.
"""

import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from launcher import batch, configs, jobs, paths, preflight  # noqa: E402

FOLDER = "configs/launcher/demo"


def config_values(name, **overrides):
    flat = {"TOOLBOX_MODE": "unsupervised_method", "UNSUPERVISED.METHOD": ["POS"],
            "UNSUPERVISED.DATA.DATASET": "UBFC-rPPG", "UNSUPERVISED.DATA.DO_PREPROCESS": True,
            "LOG.PATH": f"C:/Run/{name}", "WANDB.ENABLED": True, "DESCRIPTION": f"Run {name}\nsecond line"}
    flat.update(overrides)
    values, errors, _ = configs.normalize(flat)
    assert not errors, errors
    return values


class FakeChecks:
    """Scripted single-run checks. Per config (by LOG.PATH): existing results, existing cache (question), cache folder,
    errors, warnings."""

    def __init__(self):
        self.results, self.cached, self.caches, self.errors, self.warnings = set(), set(), {}, {}, {}
        self.calls = []

    def __call__(self, values, extras, backend, answers, secrets, manager):
        log = values["LOG.PATH"]
        self.calls.append((log, dict(answers)))
        effective = preflight.apply_answers(values, answers)
        checks = [{"id": "environment", "title": "Local Python", "status": "ok", "message": "Python 3.12"}]

        def question(id_, title, message, options):
            chosen = answers.get(id_) in options
            checks.append({"id": id_, "title": title, "message": message, "default": options[0],
                           "status": "answered" if chosen else "question",
                           "options": [{"value": value, "label": value} for value in options]})

        if log in self.results:
            question("outputs", "Existing results", "The output folder already has results: " + log,
                     ["new_folder", "overwrite"])
        if log in self.cached:
            question("cache:UNSUPERVISED.DATA", "Unsupervised data: existing cache", "A preprocessed cache exists.",
                     ["reuse", "rebuild"])
        if values["WANDB.ENABLED"]:
            question("wandb", "Weights & Biases login", "An API key is available.", ["existing", "offline"])
        if log in self.errors:
            checks.append({"id": "validation", "title": "Configuration", "status": "error", "message": "problems",
                           "details": self.errors[log]})
        else:
            checks.append({"id": "validation", "title": "Configuration",
                           "status": "warn" if log in self.warnings else "ok", "message": "Valid",
                           "details": self.warnings.get(log, [])})
        cache = self.caches.get(log, f"C:/Cache/{Path(log).name}")
        derived = {"outputs": {"log_path_host": effective["LOG.PATH"],
                               "outputs_host": effective["LOG.PATH"] + "/saved_outputs"},
                   "UNSUPERVISED.DATA": {"cache_dir_host": cache}}
        return {"checks": checks, "can_launch": True, "effective_values": effective,
                "validation": {"derived": derived}, "answers": answers}


class BatchFixture(unittest.TestCase):
    def setUp(self):
        self.fake = FakeChecks()
        self.values = {}
        for target, replacement in ((preflight, "run_checks"), (configs, "load_config")):
            patcher = patch.object(target, replacement, self.fake if replacement == "run_checks" else self.load)
            patcher.start()
            self.addCleanup(patcher.stop)

    def add(self, name, **overrides):
        path = f"{FOLDER}/{name}.yaml"
        self.values[path] = config_values(name.replace("/", "_"), **overrides)
        return path

    def load(self, path):
        if path not in self.values:
            raise ValueError(f"Config not found: {path}")
        return {"path": path, "values": self.values[path], "extras": {}, "type_errors": {}, "backend": "local"}

    def check(self, config_paths, answers=None, manager=None, **kwargs):
        return batch.run_checks(config_paths, "local", answers or {}, {}, manager, **kwargs)


class BatchCheckTests(BatchFixture):
    def test_shared_questions_are_asked_once_for_the_batch(self):
        first, second = self.add("a"), self.add("sub/b")
        self.fake.results.add("C:/Run/a")
        self.fake.cached.add("C:/Run/a")
        result = self.check([first, second])
        by_id = {check["id"]: check for check in result["checks"]}
        self.assertEqual(by_id["outputs"]["configs"], ["a"])
        self.assertEqual([o["value"] for o in by_id["outputs"]["options"]], ["new_folder", "overwrite", "skip"])
        self.assertEqual(by_id["cache"]["configs"], ["a"])
        self.assertEqual(by_id["wandb"]["configs"], ["a", "sub/b"])
        self.assertEqual(by_id["environment"]["message"], "Python 3.12", "the environment is shown once")
        self.assertEqual(by_id["policy"]["status"], "question")
        self.assertFalse(result["can_launch"], "open questions block the launch")
        self.assertEqual(result["folder"], FOLDER)
        self.assertEqual(result["name"], "launcher/demo")
        self.assertEqual([entry["description"] for entry in result["configs"]], ["Run a", "Run sub_b"])

    def test_answers_change_only_the_configs_that_asked(self):
        first, second = self.add("a"), self.add("b")
        self.fake.results.add("C:/Run/a")
        self.fake.cached.add("C:/Run/a")
        answers = {"outputs": "new_folder", "cache": "reuse", "wandb": "offline", "policy": "queue", "_stamp": "S"}
        result = self.check([first, second], answers)
        self.assertTrue(result["can_launch"])
        last = {log: answers_ for log, answers_ in self.fake.calls}
        self.assertEqual(last["C:/Run/a"]["outputs"], "new_folder")
        self.assertEqual(last["C:/Run/a"]["cache:UNSUPERVISED.DATA"], "reuse")
        self.assertNotIn("outputs", last["C:/Run/b"], "no new sub-folder for a config without results")
        self.assertFalse(any(key.startswith("cache:") for key in last["C:/Run/b"]), "no reuse without a cache")
        self.assertEqual(last["C:/Run/b"]["wandb"], "offline")
        runs = result["runs"]
        self.assertEqual(runs[first]["values"]["LOG.PATH"], "C:/Run/a/run_S")
        self.assertFalse(runs[first]["values"]["UNSUPERVISED.DATA.DO_PREPROCESS"])
        self.assertEqual(runs[second]["values"]["LOG.PATH"], "C:/Run/b")
        self.assertTrue(runs[second]["values"]["UNSUPERVISED.DATA.DO_PREPROCESS"])
        self.assertEqual(runs[second]["caches"], [os.path.normcase("C:/Cache/b")])

    def test_skip_leaves_out_the_configs_with_results(self):
        first, second = self.add("a"), self.add("b")
        self.fake.results.add("C:/Run/a")
        result = self.check([first, second], {"outputs": "skip", "wandb": "existing", "policy": "queue"})
        self.assertEqual([entry["status"] for entry in result["configs"]], ["skipped", "ready"])
        self.assertEqual(result["launchable"], 1)
        self.assertTrue(result["can_launch"])

    def test_same_outputs_and_running_configs_are_blocked(self):
        first, twin, running = self.add("a"), self.add("twin", **{"LOG.PATH": "C:/Run/a"}), self.add("c")

        class Manager:
            def active_owner(self, source, folders):
                return {"name": "Earlier run", "status": "running"} if source == running else None

            def jobs_on_gpu(self, gpu):
                return []

        result = self.check([first, twin, running], {"wandb": "existing", "policy": "queue"}, Manager())
        statuses = {entry["name"]: (entry["status"], entry["errors"]) for entry in result["configs"]}
        self.assertEqual(statuses["a"], ("ready", []))
        self.assertEqual(statuses["twin"], ("blocked", ["Writes to the same output folder as a."]))
        self.assertEqual(statuses["c"][0], "busy")
        self.assertIn("Earlier run", statuses["c"][1][0])
        # Leaving out the first config frees its output folder.
        result = self.check([first, twin], {"wandb": "existing", "policy": "queue"}, exclude=[first])
        self.assertEqual([entry["status"] for entry in result["configs"]], ["ready", "ready"])
        self.assertEqual(result["launchable"], 1)

    def test_problems_are_grouped_by_message(self):
        paths_ = [self.add(name) for name in ("a", "b", "c")]
        self.fake.errors["C:/Run/a"] = ["Folder of experiment JZC not found", "Only in a"]
        self.fake.errors["C:/Run/b"] = ["Folder of experiment JZC not found"]
        self.fake.warnings["C:/Run/c"] = ["Model usually trained with Standardized"]
        result = self.check(paths_, {"wandb": "existing", "policy": "queue"})
        self.assertEqual([(issue["status"], issue["message"], issue["configs"]) for issue in result["issues"]], [
            ("error", "Folder of experiment JZC not found", ["a", "b"]),
            ("error", "Only in a", ["a"]),
            ("warn", "Model usually trained with Standardized", ["c"])])
        self.assertEqual(result["counts"], {"blocked": 2, "warn": 1})
        self.assertTrue(result["can_launch"], "the configs without errors can still be launched")

    def test_unreadable_and_cross_validation_configs_are_blocked(self):
        cv = self.add("cv", **{"TOOLBOX_MODE": "train_and_test", "TRAIN.DATA.DATASET": "vHRM",
                               "CROSS_VALIDATION.ENABLED": True})
        result = self.check([f"{FOLDER}/missing.yaml", cv], {"wandb": "existing"})
        self.assertEqual([entry["status"] for entry in result["configs"]], ["blocked", "blocked"])
        self.assertIn("Cannot read the config", result["configs"][0]["errors"][0])
        self.assertIn("launch it from the editor", result["configs"][1]["errors"][0])
        self.assertFalse(result["can_launch"])

    def test_parallel_order_lists_the_shared_caches_and_typed_keys_warn(self):
        paths_ = [self.add(name) for name in ("a", "b", "c")]
        self.fake.caches.update({"C:/Run/a": "C:/Cache/lab", "C:/Run/b": "C:/Cache/lab"})
        result = self.check(paths_, {"wandb": "existing", "policy": "parallel"})
        by_id = {check["id"]: check for check in result["checks"]}
        self.assertEqual(by_id["caches"]["details"], ["lab: a, b"])
        self.assertNotIn("wandb_memory", by_id)
        result = self.check(paths_, {"wandb": "key", "policy": "queue"})
        self.assertNotIn("caches", {check["id"] for check in result["checks"]})
        self.assertIn("wandb_memory", {check["id"] for check in result["checks"]})

    def test_input_limits(self):
        with self.assertRaisesRegex(ValueError, "at least one config"):
            self.check([])
        with self.assertRaisesRegex(ValueError, "docker or local"):
            batch.run_checks([self.add("a")], "cloud", {}, {}, None)


class BatchJobTests(BatchFixture):
    def setUp(self):
        super().setUp()
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)
        for name in ("JOBS_DIR", "CV_DIR", "BATCH_DIR"):
            patcher = patch.object(paths, name, self.folder / name.lower())
            patcher.start()
            self.addCleanup(patcher.stop)
            (self.folder / name.lower()).mkdir()
        self.manager = self.new_manager()
        self.manager._load()
        self.paths = [self.add(name) for name in ("x/a", "x/b", "y/c")]
        self.fake.caches.update({"C:/Run/x_a": "C:/Cache/lab", "C:/Run/x_b": "C:/Cache/lab"})

    @staticmethod
    def new_manager():
        manager = jobs.JobManager.__new__(jobs.JobManager)
        manager.lock = threading.RLock()
        for name in ("jobs", "groups", "batches", "processes", "parsers", "offsets", "secret_env"):
            setattr(manager, name, {})
        return manager

    def launch(self, launch=None, answers=None, **payload):
        return self.manager.create_batch({
            "backend": "local", "configs": self.paths, "launch": launch or self.paths, "name": "Demo",
            "answers": answers or {"wandb": "existing", "policy": "queue"}, **payload})

    def members(self, batch_id):
        return sorted(self.manager._batch_jobs(batch_id), key=lambda job: job["batch"]["index"])

    def test_launch_queues_one_job_per_config_in_order(self):
        record = self.launch()
        self.assertEqual(record["status"], "queued")
        members = self.members(record["id"])
        self.assertEqual([job["name"] for job in members], ["x/a", "x/b", "y/c"])
        self.assertEqual([job["source_config"] for job in members], self.paths)
        self.assertEqual([job["batch"]["index"] for job in members], [0, 1, 2])
        self.assertEqual(members[0]["description"], "Run x_a\nsecond line")
        tree = configs.read_yaml_tree(paths.JOBS_DIR / members[0]["id"] / "config.yaml")
        self.assertEqual(tree["LOG"]["PATH"], "C:/Run/x_a")
        saved = json.loads((paths.BATCH_DIR / record["id"] / "batch.json").read_text())
        self.assertEqual([config["name"] for config in saved["configs"]], ["x/a", "x/b", "y/c"])
        self.assertEqual(saved["answers"], {"wandb": "existing", "policy": "queue"})

    def test_only_the_first_job_builds_the_docker_image(self):
        answers = []
        with patch.object(self.manager, "_steps", side_effect=lambda *a: answers.append(a[3].get("build")) or []):
            self.launch(answers={"wandb": "existing", "policy": "queue", "build": "build"})
        self.assertEqual(answers, ["build", None, None])

    def test_launch_is_all_or_nothing(self):
        self.fake.errors["C:/Run/y_c"] = ["Checkpoint not found"]
        with self.assertRaisesRegex(ValueError, "y/c: Checkpoint not found"):
            self.launch()
        with self.assertRaisesRegex(ValueError, "Launch blocked"):
            self.launch(launch=self.paths[:2], answers={"policy": "queue"})  # W&B login not answered
        self.assertEqual(self.manager.jobs, {})
        self.assertEqual(self.manager.batches, {})
        record = self.launch(launch=self.paths[:2])
        self.assertEqual(len(record["configs"]), 2, "the blocked config can be left out")

    def test_jobs_sharing_a_cache_never_preprocess_together(self):
        first, second, third = self.members(self.launch()["id"])
        self.assertFalse(self.manager._batch_waits(first))
        self.assertTrue(self.manager._batch_waits(second), "an earlier job filling the same cache is queued")
        self.assertFalse(self.manager._batch_waits(third), "another cache")
        first.update(status="running", progress={"phase": "preprocessing"})
        self.assertTrue(self.manager._batch_waits(second))
        first["progress"]["phase"] = "training"
        self.assertFalse(self.manager._batch_waits(second), "the cache is complete once training starts")

    def test_queue_follows_the_batch_order(self):
        record = self.launch(answers={"wandb": "existing", "policy": "parallel"})
        started = []

        def launch(job):
            started.append(job["name"])
            job.update(status="running", progress={"phase": "starting"})

        prefs = {"max_parallel_jobs": 3, "gpu_policy": "queue"}
        with patch.object(self.manager, "_launch", side_effect=launch), \
                patch.object(jobs.settings, "load", return_value=prefs):
            self.manager._start_queued()
        self.assertEqual(started, ["x/a", "y/c"], "x/b waits until x/a has filled the shared cache")
        self.assertEqual(self.manager.get_batch(record["id"])["status"], "running")

    def test_one_after_another_on_the_gpu(self):
        self.launch()
        started = []
        with patch.object(self.manager, "_launch", side_effect=lambda job: started.append(job["name"]) or
                          job.update(status="running")), \
                patch.object(jobs.settings, "load", return_value={"max_parallel_jobs": 3, "gpu_policy": "queue"}):
            self.manager._start_queued()
        self.assertEqual(started, ["x/a"])

    def test_outputs_of_queued_jobs_block_a_second_launch(self):
        self.launch(launch=self.paths[:1])
        result = batch.run_checks(self.paths, "local", {"wandb": "existing"}, {}, self.manager)
        self.assertEqual([entry["status"] for entry in result["configs"]], ["busy", "ready", "ready"])

    def test_stop_remove_and_reload(self):
        record = self.launch()
        reloaded = self.new_manager()
        reloaded._load()
        self.assertEqual(reloaded.get_batch(record["id"])["status"], "queued", "batches survive a restart")
        with self.assertRaisesRegex(ValueError, "Stop the batch"):
            self.manager.remove_batch(record["id"])
        stopped = self.manager.cancel_batch(record["id"])
        self.assertEqual({entry["status"] for entry in stopped["configs"]}, {"cancelled"})
        self.assertEqual(stopped["status"], "cancelled")
        self.assertFalse(stopped["active"])
        self.manager.remove_batch(record["id"])
        self.assertEqual(self.manager.jobs, {})
        self.assertFalse((paths.BATCH_DIR / record["id"]).exists())

    def test_a_removed_job_shows_in_its_batch(self):
        record = self.launch()
        first = self.members(record["id"])[0]
        first["status"] = "failed"
        self.manager.remove(first["id"])
        entries = self.manager.get_batch(record["id"])["configs"]
        self.assertEqual([entry["status"] for entry in entries], ["removed", "queued", "queued"])

    def test_resumed_job_stays_in_its_batch(self):
        record = self.launch()
        first = self.members(record["id"])[0]
        first["status"] = "paused"
        result = {"can_launch": True, "checks": [], "answers": {}, "effective_values": self.values[self.paths[0]],
                  "validation": {"derived": {"outputs": {"outputs_host": "C:/Run/x_a/saved_outputs"}}}}
        with patch.object(jobs.preflight, "run_checks", return_value=result):
            job = self.manager.create({"backend": "local", "values": self.values[self.paths[0]], "resume_from": first["id"]})
        self.assertEqual(job["batch"]["id"], record["id"])
        self.assertEqual(self.manager.get_batch(record["id"])["configs"][0]["attempts"], 2)


if __name__ == "__main__":
    unittest.main()
