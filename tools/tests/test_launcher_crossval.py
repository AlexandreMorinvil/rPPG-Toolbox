"""Cross-validation in the launcher: shared fold logic, config handling, fold jobs and merged results.

No job is started: the job manager is built without its monitor thread, as in test_launcher_recovery.
"""

import csv
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import cv_folds  # noqa: E402
from launcher import configs, crossval, jobs, paths, results, schema, settings, validation  # noqa: E402

POOL = [f"jzc{index:02d}" for index in range(1, 18)]


def cv_values(root="runs/cv_test", **overrides):
    """A vHRM cross-validation config (train_and_test with validation participants)."""
    flat = {"TOOLBOX_MODE": "train_and_test", "MODEL.NAME": "RhythmFormer", "LOG.PATH": root,
            "TRAIN.MODEL_FILE_NAME": "toy", "TEST.USE_LAST_EPOCH": False,
            "CROSS_VALIDATION.ENABLED": True}
    for prefix in ("TRAIN.DATA", "VALID.DATA", "TEST.DATA"):
        flat.update({prefix + ".DATASET": "vHRM", prefix + ".DATA_PATH": "data", prefix + ".FS": 30,
                     prefix + ".PREPROCESS.DATA_TYPE": ["Standardized"], prefix + ".PREPROCESS.LABEL_TYPE": "Standardized"})
    flat.update(overrides)
    values, errors, _ = configs.normalize(flat)
    assert not errors, errors
    return values


class SyncThread:
    """Runs the target at start(), so background merges finish inside the test."""

    def __init__(self, target, args=(), **_):
        self.target, self.args = target, args

    def start(self):
        self.target(*self.args)


class FoldTests(unittest.TestCase):
    def test_kfold_is_balanced_and_round_robin(self):
        plan = cv_folds.plan(POOL, "kfold", k=5, valid_count=2)
        sizes = [len(fold["test"]) for fold in plan["folds"]]
        self.assertEqual(sizes, [4, 4, 3, 3, 3])
        self.assertEqual(plan["folds"][0]["test"], ["jzc01", "jzc06", "jzc11", "jzc16"])
        tested = sorted(p for fold in plan["folds"] for p in fold["test"])
        self.assertEqual(tested, POOL, "every participant is tested exactly once")
        for fold in plan["folds"]:
            self.assertEqual(len(fold["valid"]), 2)
            self.assertFalse(set(fold["valid"]) & set(fold["test"]))
            self.assertEqual(sorted(fold["train"] + fold["valid"] + fold["test"]), POOL)
        # Validation comes from the next fold, cyclically.
        self.assertEqual(plan["folds"][-1]["valid"], ["jzc01", "jzc06"])

    def test_kfold_caps_k_at_the_participant_count(self):
        plan = cv_folds.plan(POOL[:3], "kfold", k=5)
        self.assertEqual(len(plan["folds"]), 3)
        self.assertTrue(plan["warnings"])
        with self.assertRaises(ValueError):
            cv_folds.plan(POOL[:1], "kfold")
        with self.assertRaises(ValueError):
            cv_folds.plan(POOL, "kfold", k=1)

    def test_leave_one_out_keeps_sessions_of_a_person_together(self):
        pool = POOL + ["jzc02_baseline", "jzc15-baseline", "jzc15-baseline-bl"]
        plan = cv_folds.plan(pool, "loso", valid_count=1)
        self.assertEqual(len(plan["folds"]), 17)
        by_name = {fold["name"]: fold for fold in plan["folds"]}
        self.assertEqual(by_name["jzc02"]["test"], ["jzc02", "jzc02-baseline"])
        self.assertEqual(by_name["jzc15"]["test"], ["jzc15", "jzc15-baseline", "jzc15-baseline-bl"])
        self.assertEqual(by_name["jzc01"]["valid"], ["jzc02", "jzc02-baseline"], "validation counts persons")
        separate = cv_folds.plan(pool, "loso", group_sessions=False)
        self.assertEqual(len(separate["folds"]), 20)

    def test_prefix_grouping_needs_the_shorter_folder(self):
        units = cv_folds.session_units(["p-01", "p-02", "p-02-retest"])
        self.assertEqual(units, {"p-01": ["p-01"], "p-02": ["p-02", "p-02-retest"]})

    def test_custom_groups(self):
        plan = cv_folds.plan(POOL, "groups", groups=[["jzc01", "JZC02"], ["jzc03", "nobody"]])
        self.assertEqual([f["name"] for f in plan["folds"]], ["1", "2"])
        self.assertEqual(plan["folds"][0]["test"], ["jzc01", "jzc02"])
        self.assertIn("jzc17", plan["folds"][0]["train"], "unlisted participants stay in training")
        self.assertTrue(any("nobody" in warning for warning in plan["warnings"]))
        with self.assertRaises(ValueError):
            cv_folds.plan(POOL, "groups", groups=[["jzc01"], ["jzc01"]])

    def test_fold_overrides(self):
        fold = {"name": "2", "test": ["jzc02"], "valid": ["jzc03"], "train": ["jzc01"]}
        values = cv_folds.fold_overrides(fold, "toy", "out/fold_2", "cv_group")
        self.assertEqual(values["TRAIN.MODEL_FILE_NAME"], "toy_fold-2")
        self.assertEqual(values["VALID.DATA.VHRM.PARTICIPANTS"], ["jzc03"])
        self.assertFalse(values["TEST.USE_LAST_EPOCH"])
        self.assertEqual(values["WANDB.GROUP"], "cv_group")
        no_valid = cv_folds.fold_overrides({**fold, "valid": []}, "toy", "out/fold_2")
        self.assertTrue(no_valid["TEST.USE_LAST_EPOCH"])
        self.assertNotIn("VALID.DATA.VHRM.PARTICIPANTS", no_valid)


class ConfigTests(unittest.TestCase):
    def test_enabled_only_for_vhrm_training(self):
        self.assertTrue(schema.cross_validation_enabled(cv_values()))
        self.assertFalse(schema.cross_validation_enabled(cv_values(**{"TRAIN.DATA.DATASET": "PURE"})))
        self.assertFalse(schema.cross_validation_enabled(cv_values(TOOLBOX_MODE="only_test")))

    def test_block_is_written_only_when_enabled(self):
        tree = configs.build_tree(cv_values())
        self.assertEqual(tree["CROSS_VALIDATION"]["SCHEME"], "kfold")
        self.assertNotIn("GROUPS", tree["CROSS_VALIDATION"], "inactive keys are left out")
        disabled = configs.build_tree(cv_values(**{"CROSS_VALIDATION.ENABLED": False, "CROSS_VALIDATION.K": 3}))
        self.assertNotIn("CROSS_VALIDATION", disabled)

    def test_fold_configs_are_single_runs(self):
        runs = crossval.fold_runs(cv_values(root="out"), cv_folds.plan(POOL, "kfold", 5, 2)["folds"], "local", "g")
        self.assertEqual(len(runs), 5)
        tree = configs.build_tree(runs[0]["values"])
        self.assertNotIn("CROSS_VALIDATION", tree)
        self.assertEqual(tree["LOG"]["PATH"], str(Path("out") / "fold_1"))
        self.assertEqual(tree["TEST"]["DATA"]["VHRM"]["PARTICIPANTS"], ["jzc01", "jzc06", "jzc11", "jzc16"])
        self.assertTrue(runs[0]["outputs"]["model_dir"].startswith(str(Path("out") / "fold_1")))


class ValidationTests(unittest.TestCase):
    def check(self, values):
        report = validation.Report()
        validation._check_cross_validation(values, "local", report, {})
        return report

    def test_valid_config_has_no_errors(self):
        report = self.check(cv_values())
        self.assertEqual(report.errors, [])
        self.assertIn("cross_validation", report.derived)

    def test_rules(self):
        cases = {
            "TEST.USE_LAST_EPOCH": cv_values(**{"TEST.USE_LAST_EPOCH": True}),
            "CROSS_VALIDATION.VALID_COUNT": cv_values(**{"CROSS_VALIDATION.VALID_COUNT": 0}),
            "TEST.DATA.DATASET": cv_values(**{"TEST.DATA.DATASET": "PURE"}),
            "MODEL.RESUME": cv_values(**{"MODEL.RESUME": "x_resume.pt"}),
            "CROSS_VALIDATION.GROUPS": cv_values(**{"CROSS_VALIDATION.SCHEME": "groups"}),
        }
        for key, values in cases.items():
            with self.subTest(key=key):
                self.assertIn(key, [error["key"] for error in self.check(values).errors])
        no_validation = cv_values(**{"CROSS_VALIDATION.VALID_COUNT": 0, "TEST.USE_LAST_EPOCH": True})
        self.assertEqual(self.check(no_validation).errors, [])

    def test_excluded_training_participants_are_flagged(self):
        report = self.check(cv_values(**{"TRAIN.DATA.VHRM.EXCLUDE_PARTICIPANTS": ["jzc13"]}))
        self.assertIn("TRAIN.DATA.VHRM.EXCLUDE_PARTICIPANTS", [w["key"] for w in report.warnings])

    def test_split_participant_lists_are_not_leakage(self):
        values = cv_values(**{"TEST.DATA.VHRM.PARTICIPANTS": ["jzc01"], "VALID.DATA.VHRM.PARTICIPANTS": ["jzc01"]})
        report = validation.Report()
        validation._check_split_consistency(values, report, schema.active_splits(values))
        self.assertFalse([w for w in report.warnings if "participants" in w["message"]])
        values["CROSS_VALIDATION.ENABLED"] = False
        report = validation.Report()
        validation._check_split_consistency(values, report, schema.active_splits(values))
        self.assertTrue([w for w in report.warnings if "participants" in w["message"]])


class PlanTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        patcher = patch.object(paths, "CV_DIR", Path(temporary.name) / "cv")
        patcher.start()
        self.addCleanup(patcher.stop)
        crossval._plans.clear()

    def fake_tool(self, calls):
        def run(arguments, timeout):
            calls.append(arguments)
            config = Path(arguments[arguments.index("--config") + 1])
            tree = configs.read_yaml_tree(config)
            self.assertEqual(tree["CROSS_VALIDATION"]["K"], 4)
            plan = cv_folds.plan(POOL, "kfold", 4, 2)
            Path(arguments[arguments.index("--plan-json") + 1]).write_text(json.dumps(plan))
            return 0, "=> Merging a config file"
        return run

    def test_plan_is_cached_and_temporary_files_removed(self):
        calls = []
        values = cv_values(**{"CROSS_VALIDATION.K": 4})
        with patch.object(crossval, "_run_tool", self.fake_tool(calls)):
            first = crossval.plan(values, {}, "local")
            second = crossval.plan(values, {}, "local")
            crossval.plan(values, {}, "local", refresh=True)
        self.assertEqual(len(first["folds"]), 4)
        self.assertEqual(first, second)
        self.assertEqual(len(calls), 2)
        self.assertEqual(list((paths.CV_DIR / "plans").iterdir()), [])

    def test_tool_failure_is_reported(self):
        with patch.object(crossval, "_run_tool", return_value=(1, "Traceback\nValueError: vHRM data path does not exist")):
            with self.assertRaisesRegex(ValueError, "data path does not exist"):
                crossval.plan(cv_values(), {}, "local")


class GroupJobTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)
        for name, value in (("JOBS_DIR", self.folder / "jobs"), ("CV_DIR", self.folder / "cv")):
            patcher = patch.object(paths, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
            value.mkdir()
        self.manager = jobs.JobManager.__new__(jobs.JobManager)
        self.manager.lock = threading.RLock()
        for name in ("jobs", "groups", "processes", "parsers", "offsets", "secret_env"):
            setattr(self.manager, name, {})
        self.root = self.folder / "runs" / "cv"

    def launch(self, folds=3):
        values = cv_values(root=str(self.root))
        plan = cv_folds.plan(POOL[:6], "kfold", folds, 1)
        result = {"can_launch": True, "checks": [], "answers": {"policy": "queue", "build": "build"},
                  "cv": {"root": str(self.root), "root_host": str(self.root), "scheme": "kfold",
                         **{key: plan[key] for key in ("pool", "units", "folds", "warnings")}},
                  "fold_runs": crossval.fold_runs(values, plan["folds"], "local", "cv_test")}
        with patch.object(crossval, "run_checks", return_value=result):
            return self.manager.create_group({"backend": "local", "values": values, "name": "CV toy"})

    def fold_jobs(self, group_id):
        return sorted(self.manager._group_jobs(group_id), key=lambda job: job["group"]["index"])

    def test_launch_queues_one_single_run_per_fold(self):
        group = self.launch()
        self.assertEqual(group["status"], "queued")
        folds = self.fold_jobs(group["id"])
        self.assertEqual([job["name"] for job in folds], ["CV toy · fold 1", "CV toy · fold 2", "CV toy · fold 3"])
        for job in folds:
            tree = configs.read_yaml_tree(paths.JOBS_DIR / job["id"] / "config.yaml")
            self.assertNotIn("CROSS_VALIDATION", tree)
            self.assertEqual(tree["LOG"]["PATH"], str(self.root / f"fold_{job['group']['fold']}"))
        self.assertEqual(json.loads((self.root / "cv_plan.json").read_text())["folds"][0]["test"], ["jzc01", "jzc04"])
        base = configs.read_yaml_tree(paths.CV_DIR / group["id"] / "config.yaml")
        self.assertTrue(base["CROSS_VALIDATION"]["ENABLED"], "the editor reopens the cross-validation config")

    def test_folds_run_one_at_a_time_in_order(self):
        group = self.launch()
        first, second, third = self.fold_jobs(group["id"])
        self.assertFalse(self.manager._group_waits(first))
        self.assertTrue(self.manager._group_waits(second))
        first["status"] = "running"
        self.assertTrue(self.manager._group_waits(second))
        first["status"] = "failed"
        self.assertFalse(self.manager._group_waits(second), "a failed fold does not stop the others")
        self.assertTrue(self.manager._group_waits(third))

    def test_paused_fold_holds_the_rest_until_resumed(self):
        group = self.launch()
        first, second, _ = self.fold_jobs(group["id"])
        first["status"] = "paused"
        self.assertTrue(self.manager._group_waits(second))
        self.manager.jobs["attempt"] = {"id": "attempt", "status": "queued", "created": "9999", "resume_from": first["id"],
                                        "group": dict(first["group"])}
        self.assertFalse(self.manager._group_waits(self.manager.jobs["attempt"]), "the resumed fold goes first")
        self.assertTrue(self.manager._group_waits(second))

    def test_resumed_fold_stays_in_its_group(self):
        group = self.launch()
        first = self.fold_jobs(group["id"])[0]
        first["status"] = "paused"
        result = {"can_launch": True, "checks": [], "answers": {}, "effective_values": cv_values(
            root=str(self.root / "fold_1"), **{"CROSS_VALIDATION.ENABLED": False}),
            "validation": {"derived": {"outputs": {"model_dir_host": str(self.root / "fold_1" / "m")}}}}
        with patch.object(jobs.preflight, "run_checks", return_value=result):
            job = self.manager.create({"backend": "local", "values": result["effective_values"], "resume_from": first["id"]})
        self.assertEqual(job["group"]["fold"], "1")
        self.assertEqual(self.manager.group(group["id"])["folds"][0]["attempts"], 2)

    def test_cross_validation_configs_cannot_launch_as_a_single_job(self):
        with self.assertRaisesRegex(ValueError, "cross-validation"):
            self.manager.create({"backend": "local", "values": cv_values()})

    def test_build_step_only_in_the_first_fold(self):
        answers = []
        with patch.object(self.manager, "_steps", side_effect=lambda *a: answers.append(a[3].get("build")) or []):
            self.launch()
        self.assertEqual(answers, ["build", None, None])

    def test_second_launch_into_active_folds_is_refused_entirely(self):
        self.launch()
        before = dict(self.manager.jobs)
        with self.assertRaisesRegex(ValueError, "checkpoint folder"):
            self.launch()
        self.assertEqual(self.manager.jobs, before, "no fold of the refused launch stays queued")
        self.assertEqual(len(self.manager.groups), 1)

    def test_stop_and_remove(self):
        group = self.launch()
        with patch.object(jobs.probes, "kill_tree"):
            stopped = self.manager.cancel_group(group["id"])
        self.assertEqual({fold["status"] for fold in stopped["folds"]}, {"cancelled"})
        self.assertEqual(stopped["status"], "cancelled")
        self.manager.remove_group(group["id"])
        self.assertEqual(self.manager.jobs, {})
        self.assertFalse((paths.CV_DIR / group["id"]).exists())
        self.assertTrue((self.root / "cv_plan.json").is_file(), "outputs are kept")

    def test_finished_fold_triggers_a_merge(self):
        group = self.launch()
        first = self.fold_jobs(group["id"])[0]
        first.update({"status": "running", "pid": 1})
        (paths.JOBS_DIR / first["id"] / "exit.json").write_text(json.dumps({"returncode": 0}))
        merged = []
        with patch.object(jobs.threading, "Thread", SyncThread), \
                patch.object(jobs.probes, "pid_alive", return_value=False), \
                patch.object(crossval, "merge", side_effect=lambda root: merged.append(root) or (True, "MAE 3")):
            self.manager._follow(first)
        self.assertEqual(first["status"], "succeeded")
        self.assertEqual(merged, [str(self.root)])
        public = self.manager.group(group["id"])
        self.assertEqual(public["merge"]["status"], "succeeded")
        self.assertEqual(public["status"], "running", "two folds are still queued")


class MergeAndResultsTests(unittest.TestCase):
    """Merges real window_results.csv files with tools/vhrm_cross_validation.py and reads them back."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "cv"
        header = ["recording_id", "experiment", "participant", "segment_index", "movement", "view", "window_index",
                  "start_seconds", "end_seconds", "ground_truth_hr_bpm", "predicted_hr_bpm", "error_bpm",
                  "absolute_error_bpm", "label_hr_bpm", "label_error_bpm", "snr_db", "ground_truth_hrv_ms",
                  "predicted_hrv_rmssd_ms", "ground_truth_respiration_rate_bpm", "predicted_mean_ibi_ms"]
        for fold, participant, error in (("1", "jzc01", 2.0), ("2", "jzc02", 4.0)):
            folder = self.root / f"fold_{fold}" / "vHRM_cache" / "saved_test_outputs"
            folder.mkdir(parents=True)
            with (folder / "window_results.csv").open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(header)
                for window in range(3):
                    writer.writerow([f"jzc--{participant}--0001--curl--front", "jzc", participant, "0001", "curl",
                                     "front", window, window * 30, window * 30 + 30, 80, 80 + error, error, error, 80,
                                     error, 1.0, 50, 55, 15, 750])
            (folder.parent / "PreTrainedModels").mkdir()

    def test_merge_and_results_view(self):
        with patch.object(settings, "load", return_value={**settings.DEFAULTS, "local_python": sys.executable}):
            ok, output = crossval.merge(str(self.root))
        self.assertTrue(ok, output)
        summary = crossval.read_summary(str(self.root), tables=True)
        self.assertAlmostEqual(summary["overall"]["mae_bpm"], 3.0)
        self.assertEqual([row["participant"] for row in summary["tables"]["by_participant"]], ["jzc01", "jzc02"])

        with patch.object(results, "_roots", return_value=[self.root.parent]):
            found = {Path(item["path"]).name: item for item in results.scan()}
            self.assertIn("cv", found)
            self.assertEqual(found["cv"]["markers"], ["cv_summary"])
            self.assertAlmostEqual(found["cv"]["summary"]["mae_bpm"], 3.0)
            self.assertIn("vHRM_cache", found, "folds stay experiments of their own")
            detail = results.detail(str(self.root))
        self.assertTrue(detail["cross_validation"])
        self.assertTrue(all(f["name"].startswith("cv_summary/") for f in detail["files"]))
        windows = next(f for f in detail["files"] if f["name"] == "cv_summary/all_windows.csv")
        with patch.object(results, "_roots", return_value=[self.root.parent]):
            table = results.window_csv(windows["path"])
        self.assertEqual(len(table["recordings"]), 2)

    def test_merge_without_folder(self):
        ok, output = crossval.merge(str(self.root / "missing"))
        self.assertFalse(ok)
        self.assertIn("does not exist", output)


if __name__ == "__main__":
    unittest.main()
