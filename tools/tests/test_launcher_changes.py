"""Changing jobs in the launcher: continuing trainings ended by early stopping, and editing queued jobs.

The single-run checks and the config validation are replaced by fakes, so the tests cover what the job manager does with
them. No job is started: the job manager is built without its monitor thread, as in test_launcher_recovery.
"""

import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from launcher import configs, jobs, logparse, paths, preflight, server, validation  # noqa: E402


def outputs_of(values):
    root = values["LOG.PATH"]
    return {"log_path_host": root, "model_dir": root + "/models", "model_dir_host": root + "/models",
            "test_outputs_host": root + "/test"}


def fake_checks(problem=None):
    def run_checks(values, extras, backend, answers, secrets, manager):
        checks = [{"id": "validation", "title": "Configuration", "status": "ok", "message": "Valid"}]
        if problem and problem in values["LOG.PATH"]:
            checks.append({"id": "wandb", "title": "W&B", "status": "question", "message": "Needs a key."})
        return {"checks": checks, "can_launch": True, "effective_values": values, "answers": answers,
                "validation": {"derived": {"outputs": outputs_of(values)}}}
    return run_checks


class Fixture(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)
        for name in ("JOBS_DIR", "CV_DIR", "BATCH_DIR"):
            patcher = patch.object(paths, name, self.folder / name.lower())
            patcher.start()
            self.addCleanup(patcher.stop)
        paths.JOBS_DIR.mkdir(parents=True)
        self.manager = jobs.JobManager.__new__(jobs.JobManager)
        self.manager.lock = threading.RLock()
        for name in ("jobs", "groups", "batches", "processes", "parsers", "offsets", "secret_env"):
            setattr(self.manager, name, {})

    def add(self, name, status="succeeded", recovery=None, losses=(), **overrides):
        """A job with its config snapshot; recovery: the metadata beside its checkpoint."""
        log_path = str(self.folder / "runs" / name).replace("\\", "/")
        flat = {"TOOLBOX_MODE": "train_and_test", "MODEL.NAME": "Physnet", "TRAIN.MODEL_FILE_NAME": name,
                "TRAIN.EPOCHS": 20, "TRAIN.EARLY_STOPPING.ENABLED": True, "TRAIN.EARLY_STOPPING.PATIENCE": 4,
                "TEST.USE_LAST_EPOCH": False, "TRAIN.DATA.DO_PREPROCESS": True, "VALID.DATA.DO_PREPROCESS": True,
                "TEST.DATA.DO_PREPROCESS": True, "LOG.PATH": log_path, "TRAIN.LR": 0.001, **overrides}
        values, errors, _ = configs.normalize(flat)
        self.assertFalse(errors)
        job_dir = paths.JOBS_DIR / name
        job_dir.mkdir()
        (job_dir / "config.yaml").write_text(configs.dump_yaml(configs.build_tree(values)), encoding="utf-8")
        outputs = outputs_of(values)
        Path(outputs["model_dir_host"]).mkdir(parents=True)
        checkpoint = Path(outputs["model_dir_host"]) / f"{name}_resume.pt"
        if recovery is not None:
            checkpoint.touch()
            checkpoint.with_suffix(".json").write_text(json.dumps(recovery))
        job = {"id": name, "name": name, "created": "2026-11-05T03:07:50", "status": status, "backend": "local",
               "mode": values["TOOLBOX_MODE"], "policy": "queue", "gpu": 0, "outputs": outputs, "epochs_total": 20,
               "resume_checkpoint": str(checkpoint), "resume_checkpoint_host": str(checkpoint),
               "training_recovery": True, "source_config": f"configs/{name}.yaml", "description": "Notes",
               "batch": {"id": "B1", "name": "batch", "index": len(self.manager.jobs), "count": 9}, "caches": [],
               "progress": {"best_epoch": 1, "epoch": len(losses) - 1, "valid_loss": {
                   str(epoch): loss for epoch, loss in enumerate(losses)}, "train_loss": {"0": 1.5},
                   "metrics": {"test": {"MAE": {"value": 3.0, "se": 0.1}}}}}
        self.manager.jobs[name] = job
        self.manager._save(job)
        return job

    def snapshot(self, job_id):
        return configs.load_config(configs.relative(paths.JOBS_DIR / job_id / "config.yaml"))["values"]


STOPPED = {"format_version": 1, "next_epoch": 6, "epochs_total": 20, "training_complete": True, "early_stopped": True}
LOSSES = [1.0, 0.9, 0.95, 0.96, 0.97, 0.98]


class ContinueTests(Fixture):
    def setUp(self):
        super().setUp()
        patcher = patch.object(preflight, "run_checks", fake_checks())
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_only_finished_trainings_ended_by_early_stopping_can_continue(self):
        self.add("stopped", recovery=STOPPED)
        self.add("failed_test", status="failed", recovery=STOPPED)
        self.add("all_epochs", recovery={**STOPPED, "next_epoch": 20, "early_stopped": False})
        self.add("stopped_on_the_last", recovery={**STOPPED, "next_epoch": 20})
        self.add("running", status="running", recovery=STOPPED)
        self.add("paused", status="paused", recovery={**STOPPED, "training_complete": False, "early_stopped": False})
        self.add("no_checkpoint")
        flags = {job["id"]: job["can_continue"] for job in self.manager.list()}
        self.assertEqual(flags, {"stopped": True, "failed_test": True, "all_epochs": False, "stopped_on_the_last": False,
                                 "running": False, "paused": False, "no_checkpoint": False})

    def test_dry_run_reports_and_queues_nothing(self):
        self.add("a", recovery=STOPPED, losses=LOSSES)
        result = self.manager.continue_jobs({"jobs": ["a"], "early_stopping": {"enabled": False}, "dry_run": True})
        row = result["results"][0]
        self.assertEqual((row["next_epoch"], row["epochs_total"], row["best_epoch"]), (6, 20, 1))
        self.assertEqual((row["min_epochs"], row["max_epochs"], row["errors"]), (None, 14, []))
        self.assertEqual(list(self.manager.jobs), ["a"])

    def test_continuation_is_a_new_attempt_from_the_checkpoint_without_early_stopping(self):
        original = self.add("a", recovery=STOPPED, losses=LOSSES)
        result = self.manager.continue_jobs({"jobs": ["a"], "early_stopping": {"enabled": False}})
        new = self.manager.jobs[result["jobs"][0]["id"]]
        self.assertEqual((new["status"], new["resume_from"], new["name"]), ("queued", "a", "a (continued)"))
        self.assertEqual(new["batch"], original["batch"], "it takes the place of the job in its batch")
        self.assertEqual(new["caches"], [], "it preprocesses nothing")
        self.assertEqual(new["source_config"], "configs/a.yaml")
        values = self.snapshot(new["id"])
        self.assertFalse(values["TRAIN.EARLY_STOPPING.ENABLED"])
        self.assertEqual(values["MODEL.RESUME"], original["resume_checkpoint"])
        self.assertTrue(values["TRAIN.SAVE_RESUME"])
        self.assertEqual(values["TRAIN.LR"], 0.001)
        for prefix in ("TRAIN.DATA", "VALID.DATA", "TEST.DATA"):
            self.assertFalse(values[prefix + ".DO_PREPROCESS"])
        self.assertEqual(original["status"], "succeeded", "each attempt keeps its log and state")
        self.assertFalse(self.manager.get("a")["can_continue"], "the new attempt took over the checkpoint")
        self.assertFalse(self.manager.get(new["id"])["can_continue"], "it is queued")
        with self.assertRaisesRegex(ValueError, "cannot continue"):
            self.manager.continue_jobs({"jobs": ["a"], "early_stopping": {"enabled": False}})

    def test_new_patience_counts_the_epochs_already_without_improvement(self):
        self.add("a", recovery=STOPPED, losses=LOSSES)
        request = {"jobs": ["a"], "early_stopping": {"enabled": True, "patience": "10"}}
        row = self.manager.continue_jobs({**request, "dry_run": True})["results"][0]
        self.assertEqual((row["min_epochs"], row["max_epochs"]), (6, 14), "4 of the 10 epochs are already used")
        new = self.manager.continue_jobs(request)["jobs"][0]
        values = self.snapshot(new["id"])
        self.assertTrue(values["TRAIN.EARLY_STOPPING.ENABLED"])
        self.assertEqual(values["TRAIN.EARLY_STOPPING.PATIENCE"], 10)

    def test_a_minimum_of_epochs_gives_a_used_up_patience_more_epochs(self):
        self.add("a", recovery=STOPPED, losses=LOSSES)
        request = {"jobs": ["a"], "early_stopping": {"enabled": True, "patience": 4, "min_epochs": "10"}}
        row = self.manager.continue_jobs({**request, "dry_run": True})["results"][0]
        self.assertEqual((row["min_epochs"], row["max_epochs"], row["errors"]), (4, 14, []), "6 of the 10 are done")
        new = self.manager.continue_jobs(request)["jobs"][0]
        self.assertEqual(self.snapshot(new["id"])["TRAIN.EARLY_STOPPING.MIN_EPOCHS"], 10)

    def test_a_minimum_of_epochs_alone_keeps_the_patience_of_the_job(self):
        self.add("a", recovery=STOPPED, losses=LOSSES)
        request = {"jobs": ["a"], "early_stopping": {"enabled": True, "patience": "", "tolerance": "", "min_epochs": "10",
                                                     "patience_after_min_epochs": ""}}
        row = self.manager.continue_jobs({**request, "dry_run": True})["results"][0]
        self.assertEqual((row["min_epochs"], row["errors"]), (4, []))
        values = self.snapshot(self.manager.continue_jobs(request)["jobs"][0]["id"])
        self.assertEqual(values["TRAIN.EARLY_STOPPING.PATIENCE"], 4)
        self.assertEqual(values["TRAIN.EARLY_STOPPING.MIN_EPOCHS"], 10)

    def test_a_refusal_says_when_the_minimum_is_already_reached(self):
        self.add("a", recovery=STOPPED, losses=LOSSES)
        def refusal(**stopping):
            request = {"jobs": ["a"], "early_stopping": {"enabled": True, "patience": 4, **stopping}, "dry_run": True}
            return self.manager.continue_jobs(request)["results"][0]["errors"][0]
        self.assertIn("the 6 epochs done reach the minimum of 5", refusal(min_epochs=5))
        self.assertNotIn("minimum", refusal())

    def test_a_minimum_of_epochs_only_counts_when_it_asks_for_more(self):
        self.add("a", recovery=STOPPED, losses=LOSSES, **{"TRAIN.EARLY_STOPPING.MIN_EPOCHS": 9})
        request = {"jobs": ["a"], "early_stopping": {"enabled": True}, "dry_run": True}
        def planned(**stopping):
            request["early_stopping"] = {"enabled": True, **stopping}
            return self.manager.continue_jobs(request)["results"][0]
        self.assertEqual(planned(patience=4)["min_epochs"], 3, "the job's own minimum of 9 stays")
        self.assertEqual(planned(patience=10)["min_epochs"], 6, "the patience asks for more")
        self.assertEqual(planned(patience=4, min_epochs=50)["min_epochs"], 14, "the maximum stays")
        self.assertIn("already used up", planned(patience=4, min_epochs=6)["errors"][0])

    def test_a_patience_after_the_minimum_only_counts_the_epochs_after_it(self):
        self.add("a", recovery=STOPPED, losses=LOSSES)
        def planned(**stopping):
            request = {"jobs": ["a"], "early_stopping": {"enabled": True, "patience": 4,
                                                         "patience_after_min_epochs": "true", **stopping}}
            return self.manager.continue_jobs({**request, "dry_run": True})["results"][0]
        self.assertEqual(planned(min_epochs=4)["min_epochs"], 2, "epochs 5 and 6 of the 4 count")
        self.assertEqual(planned(min_epochs=10)["min_epochs"], 8, "4 to reach the minimum, then the patience")
        self.assertEqual(planned(min_epochs=18)["min_epochs"], 14, "the maximum stays")
        self.assertIn("already used up", planned(min_epochs=1)["errors"][0])
        self.assertIn("already used up", planned(min_epochs=4, patience_after_min_epochs="false")["errors"][0])
        new = self.manager.continue_jobs({"jobs": ["a"], "early_stopping": {
            "enabled": True, "patience": 4, "min_epochs": 10, "patience_after_min_epochs": "true"}})["jobs"][0]
        self.assertTrue(self.snapshot(new["id"])["TRAIN.EARLY_STOPPING.PATIENCE_AFTER_MIN_EPOCHS"])

    def test_a_used_up_patience_or_a_blocked_job_refuses_the_whole_request(self):
        self.add("a", recovery=STOPPED, losses=LOSSES)
        self.add("b", recovery=STOPPED, losses=LOSSES)
        request = {"jobs": ["a", "b"], "early_stopping": {"enabled": True, "patience": 4}}
        rows = self.manager.continue_jobs({**request, "dry_run": True})["results"]
        self.assertIn("already used up", rows[0]["errors"][0])
        with self.assertRaisesRegex(ValueError, "cannot continue"):
            self.manager.continue_jobs(request)
        self.assertEqual(sorted(self.manager.jobs), ["a", "b"])
        with patch.object(preflight, "run_checks", fake_checks(problem="runs/b")):
            with self.assertRaisesRegex(ValueError, "b: W&B: Needs a key"):
                self.manager.continue_jobs({"jobs": ["a", "b"], "early_stopping": {"enabled": False}})
        self.assertEqual(sorted(self.manager.jobs), ["a", "b"])

    def test_patience_is_required_and_must_be_a_number(self):
        self.add("a", recovery=STOPPED, losses=LOSSES)
        with self.assertRaisesRegex(ValueError, "new patience"):
            self.manager.continue_jobs({"jobs": ["a"], "early_stopping": {"enabled": True}, "dry_run": True})
        with self.assertRaisesRegex(ValueError, "whole number"):
            self.manager.continue_jobs({"jobs": ["a"], "early_stopping": {"enabled": True, "patience": "x"}})

    def test_the_new_attempt_carries_on_the_curves_and_results(self):
        self.add("a", recovery=STOPPED, losses=LOSSES)
        new = self.manager.continue_jobs({"jobs": ["a"], "early_stopping": {"enabled": False}})["jobs"][0]
        parser = self.manager._new_parser(self.manager.jobs[new["id"]])
        state = parser.snapshot()
        self.assertEqual(state["valid_loss"]["5"], 0.98)
        self.assertEqual(state["best_epoch"], 1)
        self.assertEqual(state["train_loss"], {"0": 1.5})
        self.assertEqual(state["metrics"]["test"]["MAE"]["value"], 3.0)
        parser.feed("====Training Epoch: 6====\nvalidation loss:  0.5\nUpdate best model! Best epoch: 6\n")
        state = parser.snapshot()
        self.assertEqual((state["epoch"], state["best_epoch"], state["valid_loss"]["6"]), (6, 6, 0.5))
        self.assertEqual(self.manager.jobs["a"]["progress"]["valid_loss"].get("6"), None, "the original stays as it was")

    def test_parser_replays_an_unfinished_epoch_without_mixing_its_losses(self):
        parser = logparse.LogParser("train_and_test", 20, history={"train_loss": {"3": 9.0}, "epoch": 3})
        parser.feed("====Training Epoch: 3====\n")
        self.assertEqual(parser.snapshot()["train_loss"], {})


class EditQueuedTests(Fixture):
    def setUp(self):
        super().setUp()
        patcher = patch.object(validation, "validate", self.validate)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.errors = []

    def validate(self, values, extras=None, backend="docker", **_):
        return {"errors": [{"message": message} for message in self.errors],
                "derived": {"outputs": outputs_of(values)}}

    def test_dry_run_shows_the_change_and_writes_nothing(self):
        self.add("a", status="queued")
        result = self.manager.edit_queued({"jobs": ["a"], "dry_run": True,
                                           "changes": {"TRAIN.EARLY_STOPPING.PATIENCE": "8"}})
        row = result["results"][0]
        self.assertEqual(row["status"], "change")
        self.assertEqual(row["before"], {"TRAIN.EARLY_STOPPING.PATIENCE": 4})
        self.assertEqual(row["after"], {"TRAIN.EARLY_STOPPING.PATIENCE": 8})
        self.assertEqual(self.snapshot("a")["TRAIN.EARLY_STOPPING.PATIENCE"], 4)

    def test_edit_changes_the_snapshot_and_the_job_keeps_its_place(self):
        job = self.add("a", status="queued")
        before = {key: job[key] for key in ("id", "created", "batch", "status", "policy", "source_config")}
        config_before = self.snapshot("a")
        self.manager.edit_queued({"jobs": ["a"], "changes": {"TRAIN.EARLY_STOPPING.PATIENCE": 8, "TRAIN.EPOCHS": "30"}})
        values = self.snapshot("a")
        self.assertEqual((values["TRAIN.EARLY_STOPPING.PATIENCE"], values["TRAIN.EPOCHS"]), (8, 30))
        changed = {key for key in values if values[key] != config_before[key]}
        self.assertEqual(changed, {"TRAIN.EARLY_STOPPING.PATIENCE", "TRAIN.EPOCHS"}, "nothing else moves")
        self.assertEqual(job["epochs_total"], 30)
        self.assertEqual({key: job[key] for key in before}, before)
        self.assertEqual(json.loads((paths.JOBS_DIR / "a" / "job.json").read_text())["epochs_total"], 30)

    def test_jobs_that_do_not_use_a_parameter_are_left_as_they_are(self):
        self.add("a", status="queued")
        self.add("pos", status="queued", **{"TOOLBOX_MODE": "unsupervised_method", "UNSUPERVISED.METHOD": ["POS"]})
        same = self.add("same", status="queued", **{"TRAIN.EARLY_STOPPING.PATIENCE": 8})
        text = (paths.JOBS_DIR / "pos" / "config.yaml").read_text()
        result = self.manager.edit_queued({"jobs": ["a", "pos", "same"], "changes": {"TRAIN.EARLY_STOPPING.PATIENCE": 8}})
        self.assertEqual([row["status"] for row in result["results"]], ["change", "unused", "same"])
        self.assertEqual(self.snapshot("a")["TRAIN.EARLY_STOPPING.PATIENCE"], 8)
        self.assertEqual((paths.JOBS_DIR / "pos" / "config.yaml").read_text(), text)
        self.assertEqual(same["status"], "queued")

    def test_a_field_made_active_by_another_change_is_applied(self):
        self.add("a", status="queued", **{"TRAIN.EARLY_STOPPING.ENABLED": False})
        self.manager.edit_queued({"jobs": ["a"], "changes": {"TRAIN.EARLY_STOPPING.ENABLED": "true",
                                                             "TRAIN.EARLY_STOPPING.PATIENCE": 6}})
        values = self.snapshot("a")
        self.assertTrue(values["TRAIN.EARLY_STOPPING.ENABLED"])
        self.assertEqual(values["TRAIN.EARLY_STOPPING.PATIENCE"], 6)

    def test_all_or_nothing(self):
        self.add("a", status="queued")
        self.add("started", status="running")
        request = {"jobs": ["a", "started"], "changes": {"TRAIN.EARLY_STOPPING.PATIENCE": 8}}
        rows = self.manager.edit_queued({**request, "dry_run": True})["results"]
        self.assertEqual([row["status"] for row in rows], ["change", "error"])
        with self.assertRaisesRegex(ValueError, "started: Already started"):
            self.manager.edit_queued(request)
        self.assertEqual(self.snapshot("a")["TRAIN.EARLY_STOPPING.PATIENCE"], 4)

    def test_a_job_that_starts_during_the_request_stops_it(self):
        job = self.add("a", status="queued")
        other = self.add("b", status="queued")
        plan = self.manager._plan_edit

        def start_b_after_planning(job_id, changes):
            result = plan(job_id, changes)
            if job_id == "b":
                other["status"] = "running"
            return result
        with patch.object(self.manager, "_plan_edit", start_b_after_planning):
            with self.assertRaisesRegex(ValueError, "Already started: b"):
                self.manager.edit_queued({"jobs": ["a", "b"], "changes": {"TRAIN.EARLY_STOPPING.PATIENCE": 8}})
        self.assertEqual(self.snapshot("a")["TRAIN.EARLY_STOPPING.PATIENCE"], 4)
        self.assertEqual(job["status"], "queued")

    def test_refused_values_and_parameters(self):
        self.add("a", status="queued")
        for key in ("TOOLBOX_MODE", "LOG.PATH", "WANDB.MODE", "MODEL.RESUME", "NOT.A.KEY"):
            with self.assertRaisesRegex(ValueError, "cannot be changed"):
                self.manager.edit_queued({"jobs": ["a"], "changes": {key: "x"}})
        row = self.manager.edit_queued({"jobs": ["a"], "dry_run": True,
                                        "changes": {"TRAIN.EARLY_STOPPING.PATIENCE": "many"}})["results"][0]
        self.assertEqual(row["status"], "error")
        self.assertIn("whole number", row["errors"][0])
        self.errors = ["Patience must be at least 1."]
        row = self.manager.edit_queued({"jobs": ["a"], "dry_run": True,
                                        "changes": {"TRAIN.EARLY_STOPPING.PATIENCE": 0}})["results"][0]
        self.assertEqual(row["errors"], ["Patience must be at least 1."])

    def test_a_change_that_moves_the_output_folder_is_refused(self):
        self.add("a", status="queued")
        moved = lambda values, extras=None, backend="docker", **_: {  # noqa: E731
            "errors": [], "derived": {"outputs": outputs_of({"LOG.PATH": values["LOG.PATH"] + "/other"})}}
        with patch.object(validation, "validate", moved):
            row = self.manager.edit_queued({"jobs": ["a"], "dry_run": True, "changes": {"TRAIN.LR": 0.01}})["results"][0]
        self.assertIn("output folder", row["errors"][0])

    def test_a_queued_continuation_can_only_change_early_stopping(self):
        self.add("a", status="queued", **{"MODEL.RESUME": "C:/models/a_resume.pt"})
        rows = self.manager.edit_queued({"jobs": ["a"], "dry_run": True, "changes": {"TRAIN.EPOCHS": 30}})["results"]
        self.assertIn("only change early stopping", rows[0]["errors"][0])
        rows = self.manager.edit_queued({"jobs": ["a"], "dry_run": True,
                                         "changes": {"TRAIN.EARLY_STOPPING.PATIENCE": 9}})["results"]
        self.assertEqual(rows[0]["status"], "change")


class RouteTests(Fixture):
    def test_http_routes(self):
        self.add("stopped", recovery=STOPPED, losses=LOSSES)
        self.add("waiting", status="queued")
        with patch.object(server.jobs, "JobManager", return_value=self.manager):
            http = server.LauncherServer(("127.0.0.1", 0), set())
        http.allowed_hosts = {f"127.0.0.1:{http.server_port}"}
        thread = threading.Thread(target=http.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
        thread.start()

        def post(route, body):
            request = urllib.request.Request(
                f"http://127.0.0.1:{http.server_port}{route}", data=json.dumps(body).encode("utf-8"),
                headers={"Content-Type": "application/json", "X-Launcher-Token": http.token})
            with urllib.request.urlopen(request, timeout=5) as response:
                return json.load(response)
        try:
            with patch.object(preflight, "run_checks", fake_checks()):
                result = post("/api/jobs/continue", {"jobs": ["stopped"], "early_stopping": {"enabled": False},
                                                     "dry_run": True})
            self.assertEqual(result["results"][0]["max_epochs"], 14)
            with patch.object(validation, "validate", lambda values, extras=None, backend="docker", **_: {
                    "errors": [], "derived": {"outputs": outputs_of(values)}}):
                result = post("/api/jobs/edit", {"jobs": ["waiting"], "dry_run": True,
                                                 "changes": {"TRAIN.EARLY_STOPPING.PATIENCE": 8}})
            self.assertEqual(result["results"][0]["status"], "change")
            with self.assertRaises(urllib.error.HTTPError) as refused:
                post("/api/jobs/edit", {"jobs": ["waiting"], "changes": {"LOG.PATH": "C:/elsewhere"}})
            self.assertEqual(refused.exception.code, 400)
        finally:
            http.shutdown()
            thread.join(timeout=5)
            http.server_close()


class WandbLoginTests(unittest.TestCase):
    """A continuation has no field for a key, so it relies on the login that `wandb login` saved."""

    def status_with_netrc(self, content):
        with tempfile.TemporaryDirectory() as home, patch.dict("os.environ"):
            os.environ.pop("WANDB_API_KEY", None)
            if content is not None:
                (Path(home) / "_netrc").write_text(content, encoding="utf-8")
            with patch.object(Path, "home", return_value=Path(home)):
                checks = preflight.Checks()
                preflight._wandb_checks({"WANDB.ENABLED": True, "WANDB.MODE": "online"}, "local",
                                        {"wandb": "existing"}, {}, checks, {"ok": True, "wandb": True})
        return checks.items[-1]["status"]

    def test_the_login_of_recent_wandb_versions_is_found(self):
        self.assertEqual(self.status_with_netrc("machine api.forge.coreweave.com\n  login user\n  password x\n"), "answered")

    def test_the_login_of_older_wandb_versions_is_found(self):
        self.assertEqual(self.status_with_netrc("machine api.wandb.ai\n  login user\n  password x\n"), "answered")

    def test_a_login_for_another_service_or_none_still_asks(self):
        self.assertEqual(self.status_with_netrc("machine example.com\n  login user\n  password x\n"), "question")
        self.assertEqual(self.status_with_netrc(None), "question")


if __name__ == "__main__":
    unittest.main()
