"""Analysis reports in the launcher: REPORT fields, report folders in Results, report jobs and their log.

No job is started: the job manager is built without its monitor thread, as in test_launcher_crossval.
"""

import csv
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from launcher import configs, jobs, logparse, paths, reports, results, schema, settings, validation  # noqa: E402


def vhrm_values(**overrides):
    flat = {"TOOLBOX_MODE": "train_and_test", "MODEL.NAME": "RhythmFormer", "LOG.PATH": "runs/toy",
            "TRAIN.MODEL_FILE_NAME": "toy", "TEST.USE_LAST_EPOCH": False}
    for prefix in ("TRAIN.DATA", "VALID.DATA", "TEST.DATA"):
        flat.update({prefix + ".DATASET": "vHRM", prefix + ".DATA_PATH": "data", prefix + ".FS": 30,
                     prefix + ".PREPROCESS.DATA_TYPE": ["Standardized"], prefix + ".PREPROCESS.LABEL_TYPE": "Standardized"})
    flat.update(overrides)
    values, errors, _ = configs.normalize(flat)
    assert not errors, errors
    return values


def write_report(folder, hr_mae, test=True):
    """A minimal report folder, as evaluation/vhrm_report writes it."""
    (folder / "summary").mkdir(parents=True)
    (folder / "mosaics").mkdir()
    (folder / "participants" / "jzc01").mkdir(parents=True)
    for name in ("summary/overview.png", "mosaics/hr_curl1.png", "mosaics/rr_curl1.png", "participants/jzc01/hr.png",
                 "REPORT.md", "METHOD.md"):
        (folder / name).write_bytes(b"x")
    (folder / "run_config.json").write_text(json.dumps({
        "title": "RhythmFormer · toy_Epoch3", "checkpoint": "C:/runs/toy_Epoch3.pth", "recordings": ["jzc01_curl1"],
        "failures": {}, "splits": {"test": ["jzc01"]}, "selection": {"labels": []}}), encoding="utf-8")
    with (folder / "summary" / "global_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "metric_key", "unit", "group", "label", "repetition", "view", "split", "recordings",
                         "samples", "mae", "rmse", "bias", "pearson", "median_recording_mae", "within_5bpm_pct"])
        writer.writerow(["Heart rate", "hr", "bpm", "All recordings", "", "", "", "", 6, 400, hr_mae, 9, -1, 0.5, 7, 60])
        writer.writerow(["RR", "ibi", "ms", "All recordings", "", "", "", "", 6, 500, 95, 120, -3, 0.2, 90, ""])
        writer.writerow(["Heart rate", "hr", "bpm", "Curl", "curl", "", "", "", 3, 200, 12, 14, -2, 0.4, 11, 50])
    (folder / "summary" / "per_recording_metrics.csv").write_text("participant,hr_mae_bpm\njzc01,8\n", encoding="utf-8")
    if test:
        write_report(folder / "test_set", hr_mae + 2, test=False)


class ConfigFieldTests(unittest.TestCase):
    def test_report_block_is_written_only_when_enabled(self):
        self.assertNotIn("REPORT", configs.build_tree(vhrm_values()))
        tree = configs.build_tree(vhrm_values(**{"REPORT.ENABLED": True, "REPORT.LABELS": ["Curl"]}))
        self.assertEqual(tree["REPORT"]["ENABLED"], True)
        self.assertEqual(tree["REPORT"]["LABELS"], ["Curl"])
        other = vhrm_values(**{"REPORT.ENABLED": True, "TEST.DATA.DATASET": "UBFC-rPPG"})
        self.assertFalse(schema.is_active(schema.FIELD_MAP["REPORT.ENABLED"], other))
        self.assertNotIn("REPORT", configs.build_tree(other))

    def test_report_keys_exist_in_config_py(self):
        known = configs.toolbox_defaults()
        if known is None:
            self.skipTest("yacs is not installed")
        self.assertTrue({"REPORT.ENABLED", "REPORT.LABELS", "REPORT.PARTICIPANTS", "REPORT.EXPORT_CSV"} <= set(known))

    def test_validation(self):
        result = validation.validate(vhrm_values(**{"REPORT.ENABLED": True}), backend="local")
        self.assertTrue(any(item["key"] == "REPORT.ENABLED" for item in result["info"]))
        self.assertFalse(any(item["key"] == "REPORT.ENABLED" for item in result["errors"]))
        result = validation.validate(vhrm_values(**{"REPORT.ENABLED": True, "MODEL.NAME": "BigSmall"}), backend="local")
        self.assertTrue(any(item["key"] == "REPORT.ENABLED" for item in result["errors"]))


class ReportFolderTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "runs"
        self.experiment = self.root / "toy" / "vHRM_cache"
        (self.experiment / "PreTrainedModels").mkdir(parents=True)
        (self.experiment / "PreTrainedModels" / "toy_Epoch3.pth").write_bytes(b"0")
        (self.experiment / "saved_test_outputs").mkdir()
        (self.experiment / "saved_test_outputs" / "summary.json").write_text("{}", encoding="utf-8")
        write_report(self.experiment / "report" / "Epoch3", 8.0)
        write_report(self.experiment / "report" / "Epoch9", 10.0, test=False)
        patcher = patch.object(results, "_roots", return_value=[self.root])
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_names_and_locations(self):
        self.assertEqual(reports.report_name("x/toy_Epoch16.pth", "toy"), "Epoch16")
        self.assertEqual(reports.report_name("x/UBFC-rPPG_PhysNet.pth", "toy"), "UBFC-rPPG_PhysNet")
        checkpoint = self.experiment / "PreTrainedModels" / "toy_Epoch3.pth"
        self.assertEqual(reports.report_location(checkpoint, vhrm_values(), "local"), self.experiment / "report" / "Epoch3")

    def test_reports_are_listed_with_their_headline(self):
        found = {item["name"]: item for item in reports.list_reports(self.experiment)}
        self.assertEqual(set(found), {"Epoch3", "Epoch9"})
        self.assertEqual(found["Epoch3"]["summary"]["hr"]["mae"], 8.0)
        self.assertEqual(found["Epoch3"]["test_summary"]["hr"]["mae"], 10.0)
        self.assertIsNone(found["Epoch9"]["test_summary"])
        summary = reports.job_summary(self.experiment / "report" / "Epoch3")
        self.assertEqual((summary["hr_mae_bpm"], summary["ibi_mae_ms"], summary["test_hr_mae_bpm"]), (8.0, 95.0, 10.0))

    def test_detail(self):
        detail = reports.detail(str(self.experiment / "report" / "Epoch3"))
        self.assertEqual(detail["scope"], "all")
        self.assertEqual([f["name"] for f in detail["mosaics"]], ["mosaics/hr_curl1.png", "mosaics/rr_curl1.png"])
        self.assertEqual(list(detail["participants"]), ["jzc01"])
        self.assertIn("REPORT.md", detail["files"])
        self.assertEqual(detail["tables"]["global"]["header"][3], "group")
        self.assertEqual(reports.detail(detail["test_set"])["scope"], "test")
        with self.assertRaises(ValueError):
            reports.detail(str(self.experiment))

    def test_results_keep_reports_out_of_the_experiment_files(self):
        experiments = results.scan()
        self.assertEqual([item["path"] for item in experiments], [str(self.experiment.resolve())])
        self.assertEqual(experiments[0]["reports"], ["Epoch3", "Epoch9"])
        detail = results.detail(str(self.experiment))
        self.assertFalse([f for f in detail["files"] if f["name"].startswith("report/")])
        self.assertEqual(len(detail["reports"]), 2)


class ReportJobTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name).resolve()  # long names: the launcher resolves paths
        for name, value in (("JOBS_DIR", self.folder / "jobs"), ("CV_DIR", self.folder / "cv")):
            patcher = patch.object(paths, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
            value.mkdir()
        self.manager = jobs.JobManager.__new__(jobs.JobManager)
        self.manager.lock = threading.RLock()
        for name in ("jobs", "groups", "processes", "parsers", "offsets", "secret_env"):
            setattr(self.manager, name, {})
        self.runs = self.folder / "runs"
        self.checkpoint = self.runs / "toy" / "vHRM_cache" / "PreTrainedModels" / "toy_Epoch3.pth"
        self.checkpoint.parent.mkdir(parents=True)
        self.checkpoint.write_bytes(b"0")
        snapshot = paths.JOBS_DIR / "train" / "config.yaml"
        snapshot.parent.mkdir()
        snapshot.write_text(configs.dump_yaml(configs.build_tree(vhrm_values())), encoding="utf-8")
        self.config = str(snapshot)
        for target, value in ((results, "_roots"), (settings, "load")):
            replacement = ([self.runs] if value == "_roots" else
                           {**settings.DEFAULTS, "local_python": sys.executable})
            patcher = patch.object(target, value, return_value=replacement)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_local_report_job(self):
        job = self.manager.create_report({"checkpoint": str(self.checkpoint), "config": self.config, "backend": "local",
                                          "labels": "curl, resting", "participants": "", "export_csv": False})
        self.assertEqual((job["mode"], job["status"], job["name"]), ("report", "queued", "Report · toy_Epoch3"))
        output = self.checkpoint.parent.parent / "report" / "Epoch3"
        self.assertEqual(job["outputs"]["report_dir_host"], str(output))
        argv = self.manager.jobs[job["id"]]["steps"][0]["argv"]
        self.assertEqual(argv[:3], [sys.executable, "-u", "tools/vhrm_report.py"])
        self.assertEqual(argv[argv.index("--checkpoint") + 1], str(self.checkpoint))
        self.assertEqual(argv[argv.index("--output") + 1], str(output))
        self.assertEqual(argv[argv.index("--labels") + 1:argv.index("--labels") + 3], ["curl", "resting"])
        self.assertIn("--no-csv", argv)
        self.assertNotIn("--participants", argv)
        self.assertEqual(Path(argv[argv.index("--config") + 1]).parent, paths.JOBS_DIR / job["id"])
        self.assertIn(str(output), jobs.output_dirs_for(job))
        with self.assertRaises(ValueError):  # the same report twice
            self.manager.create_report({"checkpoint": str(self.checkpoint), "config": self.config, "backend": "local"})

    def test_docker_report_job_uses_container_paths(self):
        mounts = [{"container": "/runs", "host": self.runs.resolve(), "key": "RUNS_PATH", "read_only": False},
                  {"container": "/checkpoints", "host": (self.folder / "released").resolve(), "key": "CHECKPOINT_ROOT",
                   "read_only": True}]
        with patch.object(paths, "docker_mounts", return_value=mounts):
            job = self.manager.create_report({"checkpoint": str(self.checkpoint), "config": self.config,
                                              "backend": "docker", "labels": []})
            argv = self.manager.jobs[job["id"]]["steps"][0]["argv"]
            self.assertEqual(argv[argv.index("--entrypoint") + 1:argv.index("--entrypoint") + 5],
                             ["python", "toolbox", "-u", "tools/vhrm_report.py"])
            self.assertEqual(argv[argv.index("--checkpoint") + 1], "/runs/toy/vHRM_cache/PreTrainedModels/toy_Epoch3.pth")
            self.assertEqual(argv[argv.index("--output") + 1], "/runs/toy/vHRM_cache/report/Epoch3")
            self.assertEqual(argv[-1], "--labels")  # an empty list: every label
            released = self.folder / "released" / "PreTrainedModels" / "toy_Epoch3.pth"
            released.parent.mkdir(parents=True)
            released.write_bytes(b"0")
            with patch.object(results, "_roots", return_value=[self.runs, self.folder / "released"]), \
                    self.assertRaisesRegex(ValueError, "read-only"):
                self.manager.create_report({"checkpoint": str(released), "config": self.config, "backend": "docker"})

    def test_requests_are_checked(self):
        with self.assertRaisesRegex(ValueError, "outside"):
            self.manager.create_report({"checkpoint": str(self.folder / "elsewhere.pth"), "config": self.config})
        ubfc = paths.JOBS_DIR / "ubfc" / "config.yaml"
        ubfc.parent.mkdir()
        ubfc.write_text(configs.dump_yaml(configs.build_tree(vhrm_values(**{"TEST.DATA.DATASET": "UBFC-rPPG"}))),
                        encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "vHRM test data"):
            self.manager.create_report({"checkpoint": str(self.checkpoint), "config": str(ubfc), "backend": "local"})
        with self.assertRaisesRegex(ValueError, "not a valid name"):
            self.manager.create_report({"checkpoint": str(self.checkpoint), "config": self.config, "backend": "local",
                                        "labels": ["curl;rm -rf"]})


class ReportLogTests(unittest.TestCase):
    def test_report_phase_path_and_failure(self):
        parser = logparse.LogParser("train_and_test", 30)
        parser.feed("===Testing===\n===Analysis report===\n")
        self.assertEqual(parser.snapshot()["phase"], "report")
        parser.feed("Report:  40%|####      | 4/10 [00:08<00:12,  2.0s/recording]\n")
        self.assertEqual(parser.snapshot()["bar"]["total"], 10)
        parser.feed("[report] Done in 3 min 2 s: C:\\Run\\toy\\cache\\report\\Epoch16\\REPORT.md\n")
        self.assertEqual(parser.snapshot()["report_path"], "C:\\Run\\toy\\cache\\report\\Epoch16")
        failed = logparse.LogParser("train_and_test", 30)
        failed.feed("[report] FAILED: vHRM data path does not exist: F:/data\n")
        state = failed.snapshot()
        self.assertIn("vHRM data path", state["report_error"])
        self.assertTrue(any("analysis report failed" in hint for hint in state["hints"]))


if __name__ == "__main__":
    unittest.main()
