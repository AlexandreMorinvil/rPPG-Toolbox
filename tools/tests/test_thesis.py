"""Thesis experiment support: subject lists, PURE / COHFACE layouts, vHRM experiment folders and face coverage,
per-window results, the paired comparison, the window features and the config generator."""

from pathlib import Path
import argparse
import contextlib
import io
import json
import os
import pickle
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from cache_naming import subject_key, subjects_suffix  # noqa: E402
from config import _C, _vhrm_node, get_config  # noqa: E402
from dataset.data_loader import VHRMLoader as V  # noqa: E402
from dataset.data_loader.COHFACELoader import COHFACELoader  # noqa: E402
from dataset.data_loader.PURELoader import PURELoader  # noqa: E402
from dataset.data_loader.UBFCrPPGLoader import UBFCrPPGLoader  # noqa: E402
from launcher import configs, validation  # noqa: E402
from tools import paired_comparison, thesis_configs, vhrm_window_features  # noqa: E402


def bare(cls, **attributes):
    """A loader without its preprocessing constructor."""
    loader = cls.__new__(cls)
    loader.dataset_name = cls.__name__
    loader.raw_data_path = "root"
    for key, value in attributes.items():
        setattr(loader, key, value)
    return loader


class FakeFaceLoader(V.VHRMLoader):
    """A face 'detector' that finds a face in bright frames (importable by the spawned preprocessing workers)."""

    def _detect_face(self, frame, crop):
        return [0, 0, frame.shape[1], frame.shape[0]] if frame.mean() > 100 else None


class SubjectSelectionTests(unittest.TestCase):
    def test_subject_keys_and_file_list_suffix(self):
        self.assertEqual({subject_key(item) for item in ("subject07", "07", 7, "s7", " 7 ")}, {"7"})
        self.assertEqual(subject_key("abc"), "abc")
        self.assertEqual(subjects_suffix([]), "")
        self.assertEqual(subjects_suffix(["subject1", "38"]), subjects_suffix(["38", "1"]))
        self.assertNotEqual(subjects_suffix(["1"]), subjects_suffix(["2"]))

    def test_split_by_subject_list(self):
        loader = bare(UBFCrPPGLoader, config_data=argparse.Namespace(SUBJECTS=["subject3", "38"]))
        data_dirs = [{"index": f"subject{n}", "path": f"root/subject{n}"} for n in (1, 3, 38)]
        self.assertEqual([d["index"] for d in loader.select_split(data_dirs, 0.0, 1.0)], ["subject3", "subject38"])
        loader.config_data.SUBJECTS = ["40"]
        with self.assertRaises(ValueError):
            loader.select_split(data_dirs, 0.0, 1.0)

    def test_config_and_launcher_name_the_file_lists_alike(self):
        tree = {"TOOLBOX_MODE": "unsupervised_method",
                "UNSUPERVISED": {"METHOD": ["POS"], "DATA": {
                    "DATASET": "PURE", "DATA_PATH": "x", "FS": 30, "SUBJECTS": ["07", "08"],
                    "PREPROCESS": {"DATA_TYPE": ["Raw"], "LABEL_TYPE": "Raw", "DO_CHUNK": False}}}}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "pos.yaml"
            path.write_text(yaml.safe_dump(tree), encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                config = get_config(argparse.Namespace(config_file=str(path)))
            values, errors, _ = configs.normalize(configs.flatten(configs.read_yaml_tree(path)))
        self.assertFalse(errors)
        self.assertTrue(config.UNSUPERVISED.DATA.FILE_LIST_PATH.endswith(subjects_suffix(["7", "8"]) + ".csv"))
        locations = validation.derived_locations(values, "UNSUPERVISED.DATA")
        self.assertEqual(Path(locations["file_list"]).name, Path(config.UNSUPERVISED.DATA.FILE_LIST_PATH).name)


class LiteratureLayoutTests(unittest.TestCase):
    def test_pure_reads_both_layouts(self):
        with tempfile.TemporaryDirectory() as folder:
            flat, nested = Path(folder, "flat"), Path(folder, "nested")
            (flat / "07-01").mkdir(parents=True)
            (flat / "07-01.json").write_text("{}")
            (flat / "07-01_skin_uv.h5").write_bytes(b"")          # a vHRM Studio file is not a recording
            (nested / "08-02" / "08-02").mkdir(parents=True)
            (nested / "08-02" / "08-02.json").write_text("{}")
            loader = bare(PURELoader)
            first, = loader.get_raw_data(str(flat))
            self.assertEqual((first["index"], first["subject"]), (701, 7))
            self.assertEqual(Path(first["frames"]), flat / "07-01")
            self.assertEqual(Path(first["labels"]), flat / "07-01.json")
            second, = loader.get_raw_data(str(nested))
            self.assertEqual(Path(second["frames"]), nested / "08-02" / "08-02")
            self.assertEqual(Path(second["labels"]), nested / "08-02" / "08-02.json")
            self.assertEqual(loader.split_raw_data([first, second], 0.0, 0.5), [first])  # keeps every field

    def test_cohface_sessions_and_subject_split(self):
        with tempfile.TemporaryDirectory() as folder:
            for subject, sessions in ((1, 4), (2, 4), (6, 8)):
                for session in range(sessions):
                    Path(folder, str(subject), str(session)).mkdir(parents=True)
                    Path(folder, str(subject), str(session), "data.hdf5").write_bytes(b"")
            Path(folder, "protocols").mkdir()
            Path(folder, "README.rst").write_text("")
            loader = bare(COHFACELoader)
            dirs = loader.get_raw_data(folder)
            self.assertEqual(len(dirs), 12)                      # sessions 0-3 only
            self.assertEqual(dirs[0]["index"], "01-0")
            self.assertEqual({d["subject"] for d in loader.split_raw_data(dirs, 0.0, 0.5)}, {1})
            loader.config_data = argparse.Namespace(SUBJECTS=["6", "2"])
            self.assertEqual({d["subject"] for d in loader.select_split(dirs, 0.0, 1.0)}, {2, 6})


class ExperimentPathTests(unittest.TestCase):
    def test_experiments_outside_the_data_path(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for experiment_dir, participant in ((root / "lab" / "baseline", "p1_baseline"), (root / "elsewhere" / "rm_uv", "RM01"),
                                                (root / "lab" / "RM", "RM99")):
                segment = experiment_dir / participant / "labelled_segments" / "0000_Rest"
                segment.mkdir(parents=True)
                (segment / "cam2_Camera_Front.mp4").write_bytes(b"")
                (segment / "meta.json").write_text(json.dumps({
                    "index": 0, "label": "Rest", "start_sec": 0.0, "duration_sec": 10.0,
                    "signal_anchor_datetime_utc": "2026-05-27T13:00:00+00:00",
                    "cameras": [{"label": "Camera_Front", "video_file": "cam2_Camera_Front.mp4"}]}))
            segments = V.discover_segments(str(root / "lab"), experiment_paths=[f"RM={root / 'elsewhere' / 'rm_uv'}"])
            self.assertEqual(sorted((s["experiment"], s["participant"]) for s in segments),
                             [("baseline", "p1-baseline"), ("rm", "rm01")])   # replaces lab/RM
            only = V.discover_segments("", ["rm"], experiment_paths=[f"RM={root / 'elsewhere' / 'rm_uv'}"])
            self.assertEqual([s["participant"] for s in only], ["rm01"])
            # A missing folder only matters when its experiment is selected.
            V.discover_segments(str(root / "lab"), ["baseline"], experiment_paths=["JZC=Z:/missing"])
            with self.assertRaises(ValueError):
                V.discover_segments(str(root / "lab"), ["jzc"], experiment_paths=["JZC=Z:/missing"])
            with self.assertRaises(ValueError):
                V.parse_experiment_paths(["RM"])


class FaceCoverageTests(unittest.TestCase):
    """MIN_FACE_FRACTION: training drops the clips without a face; the test keeps them and scores no reading."""

    FS = 30
    CHUNK = 60

    @classmethod
    def setUpClass(cls):
        import cv2
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        segment = cls.root / "data" / "toy" / "p01" / "labelled_segments" / "0000_Rest"
        segment.mkdir(parents=True)
        writer = cv2.VideoWriter(str(segment / "cam2_Camera_Front.avi"), cv2.VideoWriter_fourcc(*"MJPG"), cls.FS, (16, 16))
        for frame in range(20 * cls.FS):                       # a face for 10 s, then none
            writer.write(np.full((16, 16, 3), 200 if frame < 10 * cls.FS else 20, dtype=np.uint8))
        writer.release()
        (segment / "meta.json").write_text(json.dumps({
            "index": 0, "label": "Rest", "start_sec": 0.0, "duration_sec": 20.0,
            "signal_anchor_datetime_utc": "2026-05-27T13:00:00+00:00",
            "cameras": [{"label": "Camera_Front", "video_file": "cam2_Camera_Front.avi", "fps": cls.FS}]}))

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def loader(self, name, split="TEST"):
        data = {"DATASET": "vHRM", "DATA_PATH": str(self.root / "data"), "CACHED_PATH": str(self.root / "cache"),
                "DO_PREPROCESS": True, "FS": self.FS,
                "PREPROCESS": {"DATA_TYPE": ["Raw"], "LABEL_TYPE": "Standardized", "DO_CHUNK": True,
                               "CHUNK_LENGTH": self.CHUNK, "RESIZE": {"W": 8, "H": 8}, "NUM_WORKERS": 1,
                               "CROP_FACE": {"DO_CROP_FACE": True, "BACKEND": "HC",
                                             "DETECTION": {"DO_DYNAMIC_DETECTION": True,
                                                           "DYNAMIC_DETECTION_FREQUENCY": 30}}},
                "VHRM": {"CROP_MODE_BY_VIEW": ["front=face"], "VIDEO_DECODER": "opencv", "SHOW_VIDEO_PROGRESS": False,
                         "REQUIRE_PPG": False, "MIN_FACE_FRACTION": 0.5}}
        path = self.root / f"{name}.yaml"
        path.write_text(yaml.safe_dump({"TOOLBOX_MODE": "only_test", "TEST": {"DATA": data},
                                        "INFERENCE": {"EVALUATION_WINDOW": {"USE_SMALLER_WINDOW": True,
                                                                           "WINDOW_SIZE": 5}}}), encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()):
            config = get_config(argparse.Namespace(config_file=str(path)))
            loader = FakeFaceLoader(name, config.TEST.DATA.DATA_PATH, config.TEST.DATA)
        return config, loader

    def test_training_drops_faceless_clips_and_the_test_scores_them_as_no_reading(self):
        config, test = self.loader("test")
        trace = np.load(next(Path(config.TEST.DATA.CACHED_PATH).rglob("*_face.npy")))
        self.assertEqual(len(trace), 20 * self.FS)
        self.assertTrue(trace[:10 * self.FS].all() and not trace[10 * self.FS:].any())
        self.assertEqual(len(test.inputs), 10)                                  # every clip is tested

        from evaluation.vhrm_metrics import calculate_vhrm_metrics
        file_list = pd.read_csv(config.TEST.DATA.FILE_LIST_PATH)
        times = np.arange(self.CHUNK) / self.FS
        predictions = {row.recording_id: {} for row in file_list.itertuples()}
        for row in file_list.itertuples():
            predictions[row.recording_id][row.chunk] = np.sin(2 * np.pi * 1.2 * (times + row.chunk * self.CHUNK / self.FS))
        output = self.root / "outputs"
        with contextlib.redirect_stdout(io.StringIO()):
            calculate_vhrm_metrics(predictions, None, config, output_dir=str(output), prediction_is_diff=False)
        windows = pd.read_csv(output / "window_results.csv")
        self.assertEqual(windows["has_reading"].tolist(), [True, True, False, False])
        self.assertEqual(windows["start_frame"].tolist(), [0, 150, 300, 450])
        self.assertAlmostEqual(pd.read_csv(output / "overall_summary.csv")["coverage"].iloc[0], 0.5)

        _, train = self.loader("train")  # same file list as the test split here: checked last
        self.assertEqual(len(train.inputs), 5)                                  # only the clips with a face


class FaceSearchBandTests(unittest.TestCase):
    def test_faces_outside_the_central_band_are_not_found(self):
        seen = []

        class Detector(V.VHRMLoader):
            def _detect_face(self, frame, crop):
                seen.append(frame.shape[1])
                return [10, 5, 20, 20] if frame.shape[1] > 0 else None

        loader = Detector.__new__(Detector)
        loader.vhrm = _vhrm_node()
        loader.vhrm.FACE_SEARCH_WIDTH = 0.5
        preprocess = argparse.Namespace(CROP_FACE=argparse.Namespace())
        stats = {"attempts": 0, "failures": 0}
        box = loader._resolve_crop(np.zeros((100, 200, 3), np.uint8), ("face", None), preprocess, None, stats)
        self.assertEqual(seen, [100])                    # the detector only sees the central 100 of 200 pixels
        self.assertEqual(box, [60, 5, 20, 20])           # and its box is moved back to frame coordinates
        self.assertTrue(stats["face_found"])
        loader.vhrm.FACE_SEARCH_WIDTH = 1.0
        loader._resolve_crop(np.zeros((100, 200, 3), np.uint8), ("face", None), preprocess, None, stats)
        self.assertEqual(seen[-1], 200)


class PersonFaceTests(unittest.TestCase):
    def test_the_largest_confident_person_is_kept(self):
        from dataset.data_loader.face_detector.person_detector import largest_box
        boxes = np.array([[0, 0, 10, 10], [5, 5, 105, 205], [0, 0, 300, 300], [20, 20, 60, 60]], dtype=float)
        self.assertEqual(largest_box(boxes, [0.9, 0.8, 0.5, 0.95], [1, 1, 1, 1]), [5, 5, 105, 205])  # 0.5: too unsure
        self.assertIsNone(largest_box(boxes, [0.9, 0.8, 0.9, 0.9], [3, 3, 3, 3]))                    # no person
        self.assertEqual(V.parse_crop_modes(["back-left=person_face"])["back-left"], ("person_face", None))

    def test_the_face_is_searched_inside_the_largest_person(self):
        seen = []

        class Detector(V.VHRMLoader):
            person = [50, 20, 150, 220]

            def _largest_person(self, frame):
                return self.person

            def _detect_face(self, frame, crop):
                seen.append(frame.shape[:2])
                return [10, 5, 20, 20]

        loader = Detector.__new__(Detector)
        loader.vhrm = _vhrm_node()
        loader.vhrm.FACE_SEARCH_WIDTH = 0.5                       # not used by person_face
        preprocess = argparse.Namespace(CROP_FACE=argparse.Namespace())
        stats = {"attempts": 0, "failures": 0}
        frame = np.zeros((300, 400, 3), np.uint8)
        self.assertEqual(loader._resolve_crop(frame, ("person_face", None), preprocess, None, stats), [60, 25, 20, 20])
        self.assertEqual(seen, [(200, 100)])                     # only the person's box was searched
        Detector.person = None                                   # nobody: no face, the previous box is kept
        self.assertEqual(loader._resolve_crop(frame, ("person_face", None), preprocess, [1, 2, 3, 4], stats), [1, 2, 3, 4])
        self.assertFalse(stats["face_found"])
        self.assertEqual((stats["attempts"], stats["failures"]), (2, 1))
        self.assertEqual(len(seen), 1)


class WindowResultsTests(unittest.TestCase):
    def test_generic_datasets_write_one_row_per_window(self):
        import torch
        from evaluation.metrics import calculate_metrics
        config = _C.clone()
        config.defrost()
        config.TOOLBOX_MODE = "only_test"
        config.INFERENCE.MODEL_PATH = "model.pth"
        config.TEST.DATA.DATASET, config.TEST.DATA.FS = "PURE", 30
        config.TEST.DATA.PREPROCESS.LABEL_TYPE = "Standardized"
        config.TEST.METRICS = ["MAE"]
        config.INFERENCE.EVALUATION_WINDOW.USE_SMALLER_WINDOW = True
        config.INFERENCE.EVALUATION_WINDOW.WINDOW_SIZE = 10
        wave = torch.tensor(np.sin(2 * np.pi * 1.2 * np.arange(300) / 30), dtype=torch.float32)
        with tempfile.TemporaryDirectory() as folder:
            config.TEST.OUTPUT_SAVE_DIR = folder
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                calculate_metrics({"701": {0: wave[:150], 1: wave[150:]}}, {"701": {0: wave[:150], 1: wave[150:]}},
                                  config)
            windows = pd.read_csv(Path(folder) / "window_results.csv", dtype={"participant": str})
        self.assertEqual(windows["start_frame"].tolist(), [0])
        self.assertEqual(windows["end_frame"].tolist(), [300])
        self.assertEqual(windows["participant"].tolist(), ["07"])
        self.assertAlmostEqual(windows["predicted_hr_bpm"].iloc[0], 72.0, delta=2.0)


class PairedComparisonTests(unittest.TestCase):
    def windows(self, errors, has_reading=None):
        rows = []
        for (person, error), reading in zip(errors.items(), has_reading or [True] * len(errors)):
            for window in range(3):
                rows.append({"recording_id": f"r-{person}", "start_frame": 900 * window, "person": person,
                             "absolute_error_bpm": error + 0.1 * window, "has_reading": reading})
        return pd.DataFrame(rows)

    def test_verdicts(self):
        reference = self.windows({f"p{i}": 3.0 + i for i in range(10)})
        same = self.windows({f"p{i}": 3.05 + i for i in range(10)})
        better = self.windows({f"p{i}": 1.0 + i for i in range(10)})
        rows, table = paired_comparison.compare({"face": reference, "same": same, "better": better}, "face",
                                                margin=0.5, resamples=2000)
        verdicts = {row["run"]: row.get("verdict") for row in rows}
        self.assertEqual(verdicts, {"face": None, "same": "equivalent", "better": "better"})
        self.assertEqual(len(table), 10)

    def test_coverage_and_common_windows(self):
        reference = self.windows({"p1": 2.0, "p2": 4.0}, has_reading=[True, False])
        other = self.windows({"p1": 3.0, "p2": 1.0})
        rows, _ = paired_comparison.compare({"face": reference, "uv": other}, "face", resamples=200)
        self.assertEqual(rows[0]["coverage"], 0.5)
        self.assertEqual(rows[1]["paired_people"], 1)
        rows, _ = paired_comparison.compare({"face": reference, "uv": other}, "face", resamples=200,
                                            common_windows=True)
        self.assertEqual(rows[1]["people"], 1)

    def test_people_join_their_sessions(self):
        splits = yaml.safe_load((ROOT / "thesis" / "splits.yaml").read_text(encoding="utf-8"))
        lab, literature = paired_comparison.person_table(splits)
        self.assertEqual(lab[("rm", "rm06")], lab[("jzc", "jzc07")])
        self.assertEqual(lab[("baseline", "rm010-baseline-after")], ("RM10", "train"))
        self.assertEqual(literature[("PURE", "7")], "test")


class WindowFeatureTests(unittest.TestCase):
    def test_rotation_speed(self):
        from scipy.spatial.transform import Rotation
        times = np.arange(5) / 30.0
        rotations = Rotation.from_rotvec(np.outer(np.deg2rad(np.arange(5) * 2.0), [0, 0, 1])).as_matrix()
        valid = np.array([True, True, True, False, True])
        speeds = vhrm_window_features.rotation_speeds(rotations, times, valid)
        np.testing.assert_allclose(speeds[1:3], 60.0, rtol=1e-6)               # 2 degrees per frame at 30 fps
        self.assertTrue(np.isnan(speeds[0]) and np.isnan(speeds[3]) and np.isnan(speeds[4]))


class GeneratorTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        _, self.splits = thesis_configs.load_inputs()
        runs = str(self.root / "runs").replace("\\", "/")
        self.setup = {"cache": dict.fromkeys(("batch1", "batch2", "batch3"), "C:/cache"),
                      "runs": dict.fromkeys(("batch1", "batch2", "batch3"), runs),
                      "video": {"UBFC-rPPG": "F:/u", "PURE": "F:/p", "COHFACE": "F:/c", "lab": "F:/lab"},
                      "uv": {"UBFC-rPPG": "F:/uu", "PURE": "F:/up", "COHFACE": "F:/uc",
                             "lab": {"baseline": "F:/uv/baseline", "JZC": "", "RM": "F:/uv/RM"}},
                      "options": {"device": "cuda:0", "wandb": False}}

    def tearDown(self):
        self.directory.cleanup()

    def test_every_config_loads_and_the_splits_never_mix(self):
        runs, waiting = thesis_configs.build_runs(self.setup, self.splits)
        self.assertEqual(len(runs), 188)
        self.assertEqual(len(waiting), 48)                         # batch 2 of baseline, JZC; their fine-tunings
        output = self.root / "configs"
        thesis_configs.write_runs(runs, output)
        for path in sorted(output.rglob("*.yaml")):
            with self.subTest(config=path.name), contextlib.redirect_stdout(io.StringIO()):
                config = get_config(argparse.Namespace(config_file=str(path)))
                values, errors, extras = configs.normalize(configs.flatten(configs.read_yaml_tree(path)))
                self.assertFalse(errors)
                self.assertFalse(extras)
                blocks = [("UNSUPERVISED.DATA", config.UNSUPERVISED.DATA)] if config.TOOLBOX_MODE == "unsupervised_method" \
                    else [("TEST.DATA", config.TEST.DATA)] + ([("TRAIN.DATA", config.TRAIN.DATA), ("VALID.DATA", config.VALID.DATA)]
                                                             if config.TOOLBOX_MODE == "train_and_test" else [])
                for prefix, data in blocks:
                    self.assertEqual(Path(validation.derived_locations(values, prefix)["file_list"]).name,
                                     Path(data.FILE_LIST_PATH).name)
                evaluated = blocks[0][1]
                self.assertEqual(config.REPORT.ENABLED, evaluated.DATASET == "vHRM")    # the report needs vHRM data
                if config.TOOLBOX_MODE != "unsupervised_method" and evaluated.DATASET == "vHRM":
                    # One experiment per config, and only its participant folders.
                    experiment = path.relative_to(output).parts[1]
                    for _, data in blocks:
                        self.assertEqual(list(data.VHRM.EXPERIMENTS), [experiment])
                        folders = set(thesis_configs.lab_folders(self.splits, experiment=experiment))
                        self.assertLessEqual(set(data.VHRM.PARTICIPANTS), folders)
                if config.TOOLBOX_MODE == "train_and_test":
                    if config.TRAIN.DATA.DATASET == "vHRM":
                        chosen = [set(map(V.slug, data.VHRM.PARTICIPANTS)) for _, data in blocks]
                    else:
                        chosen = [set(map(subject_key, data.SUBJECTS)) for _, data in blocks]
                    self.assertFalse(chosen[0] & chosen[1] or chosen[0] & chosen[2] or chosen[1] & chosen[2])
        stale = output / "batch1_literature" / "old.yaml"
        stale.write_text(thesis_configs.HEADER + "BASE: ['']\n", encoding="utf-8")
        self.assertEqual(thesis_configs.write_runs(runs, output), 1)
        self.assertFalse(stale.exists())

    def test_batch3_starts_from_the_tested_batch2_checkpoint(self):
        log = self.root / "runs" / "batch2_lab_front" / "b2_baseline_physnet_face"
        (log / "train_cache" / "PreTrainedModels").mkdir(parents=True)
        (log / "train_cache" / "PreTrainedModels" / "b2_baseline_physnet_face_Epoch7.pth").write_bytes(b"")
        (log / "test_cache" / "saved_test_outputs").mkdir(parents=True)
        with open(log / "test_cache" / "saved_test_outputs" / "b2_baseline_physnet_face_outputs.pickle", "wb") as handle:
            pickle.dump({"meta": {"epoch": 7}}, handle)
        runs, waiting = thesis_configs.build_runs(self.setup, self.splits)
        self.assertNotIn("b2_baseline_physnet_face", waiting)
        self.assertIn("b2_jzc_physnet_face", waiting)              # each experiment starts from its own model
        finetune = runs["batch3_lab_back/baseline/finetune/physnet_face.yaml"]
        self.assertTrue(finetune["MODEL"]["INIT_WEIGHTS"].endswith("b2_baseline_physnet_face_Epoch7.pth"))
        self.assertEqual(runs["batch3_lab_back/baseline/zeroshot/physnet_face.yaml"]["INFERENCE"]["MODEL_PATH"],
                         finetune["MODEL"]["INIT_WEIGHTS"])
        self.assertTrue(runs["batch3_lab_back/JZC/finetune/physnet_face.yaml"]["MODEL"]["INIT_WEIGHTS"].startswith("<"))
        self.assertEqual(finetune["TRAIN"]["DATA"]["VHRM"]["VIEWS"], ["back-left", "back-right"])
        self.assertEqual(finetune["TEST"]["DATA"]["VHRM"]["VIEWS"], ["front", "back-left", "back-right"])

    def make_run(self, batch_folder, name, epoch):
        log = self.root / "runs" / batch_folder / name / "cache"
        (log / "PreTrainedModels").mkdir(parents=True)
        (log / "PreTrainedModels" / f"{name}_Epoch{epoch}.pth").write_bytes(b"")
        (log / "saved_test_outputs").mkdir()
        with open(log / "saved_test_outputs" / f"{name}_outputs.pickle", "wb") as handle:
            pickle.dump({"meta": {"epoch": epoch}}, handle)

    def test_rm_is_tested_with_the_models_of_the_other_experiments(self):
        self.make_run("batch2_lab_front", "b2_baseline_physnet_face", 7)
        self.make_run("batch2_lab_front", "b2_jzc_physnet_face", 5)
        self.make_run("batch3_lab_back", "b3_jzc_ft_physnet_face", 3)
        runs, waiting = thesis_configs.build_runs(self.setup, self.splits)
        model_path = lambda relative: runs[relative]["INFERENCE"]["MODEL_PATH"]
        self.assertTrue(model_path("batch2_lab_front/RM/from_baseline/physnet_face.yaml")
                        .endswith("b2_baseline_physnet_face_Epoch7.pth"))
        self.assertTrue(model_path("batch2_lab_front/RM/from_JZC/physnet_face.yaml")
                        .endswith("b2_jzc_physnet_face_Epoch5.pth"))
        self.assertTrue(model_path("batch3_lab_back/RM/from_JZC/physnet_face.yaml")
                        .endswith("b3_jzc_ft_physnet_face_Epoch3.pth"))          # the fine-tuned model
        self.assertEqual(model_path("batch3_lab_back/RM/from_baseline/physnet_face.yaml"),
                         "<run batch3_lab_back/baseline/finetune/physnet_face.yaml, then tools/thesis_configs.py>")
        self.assertIn("b3_baseline_ft_physnet_face", waiting)
        self.assertEqual(len(waiting), len(set(waiting)))

        tests = {relative: config for relative, config in runs.items() if "/RM/from_" in relative}
        self.assertEqual(len(tests), 48)
        people = {"RM06", "RM12", "RM15", "RM16", "RM17", "RM20"}
        training = set()
        for person in self.splits["lab"]["persons"].values():
            if person["split"] != "test":
                training.update(session.split("/", 1)[1] for session in person["sessions"])
        for relative, config in tests.items():
            with self.subTest(config=relative):
                self.assertEqual(config["TOOLBOX_MODE"], "only_test")
                self.assertNotIn("EPOCHS", config["TRAIN"])
                self.assertNotIn("DATA", config["TRAIN"])
                vhrm = config["TEST"]["DATA"]["VHRM"]
                self.assertEqual(set(vhrm["PARTICIPANTS"]), people)
                self.assertFalse(people & training)                              # nobody the models trained on
                self.assertEqual(vhrm["EXPERIMENTS"], ["RM"])
                self.assertEqual(vhrm["LABELS"], ["Squat"])
                self.assertEqual(vhrm["VIEWS"], ["front"] if relative.startswith("batch2")
                                 else ["front", "back-left", "back-right"])
                self.assertEqual(vhrm["PREDICTION_IS_DIFF"], "/physnet_" in relative)
                self.assertTrue(config["REPORT"]["ENABLED"])
                self.assertEqual(set(config["REPORT"]["PARTICIPANTS"]), people)  # not the people the models saw
        self.assertEqual(runs["batch3_lab_back/baseline/zeroshot/physnet_face.yaml"]["REPORT"]["PARTICIPANTS"], [])
        logs =[config["LOG"]["PATH"] for config in runs.values()]
        self.assertEqual(len(logs), len(set(logs)))                              # no two runs share a folder

    def test_rm_tests_reuse_the_test_cache_of_the_rm_training_configs(self):
        runs, _ = thesis_configs.build_runs(self.setup, self.splits)
        output = self.root / "configs"
        thesis_configs.write_runs(runs, output)
        with contextlib.redirect_stdout(io.StringIO()):
            trained = get_config(argparse.Namespace(config_file=str(output / "batch2_lab_front/RM/physnet_face.yaml")))
            tested = get_config(argparse.Namespace(
                config_file=str(output / "batch2_lab_front/RM/from_baseline/physnet_face.yaml")))
        self.assertEqual(tested.TEST.DATA.CACHED_PATH, trained.TEST.DATA.CACHED_PATH)
        self.assertEqual(tested.TEST.DATA.FILE_LIST_PATH, trained.TEST.DATA.FILE_LIST_PATH)

    def test_tests_of_one_checkpoint_report_in_their_own_folders(self):
        from evaluation.vhrm_report import report_folder
        checkpoint = self.root / "runs/batch2_lab_front/b2_baseline_physnet_face/x/PreTrainedModels/b2_baseline_physnet_face_Epoch7.pth"
        runs, _ = thesis_configs.build_runs(self.setup, self.splits)
        output = self.root / "configs"
        thesis_configs.write_runs(runs, output)
        folders = {}
        for relative in ("batch2_lab_front/RM/from_baseline/physnet_face.yaml",
                         "batch3_lab_back/baseline/zeroshot/physnet_face.yaml"):
            with contextlib.redirect_stdout(io.StringIO()):
                config = get_config(argparse.Namespace(config_file=str(output / relative)))
            folders[relative] = report_folder(config, checkpoint)
            self.assertEqual(folders[relative].parent.parent, Path(config.TEST.OUTPUT_SAVE_DIR).parent)
        self.assertEqual(len(set(folders.values())), 2)
        with contextlib.redirect_stdout(io.StringIO()):
            trained = get_config(argparse.Namespace(config_file=str(output / "batch2_lab_front/baseline/physnet_face.yaml")))
        self.assertEqual(report_folder(trained, checkpoint), checkpoint.parent.parent / "report" / "Epoch7")

    def test_each_batch_keeps_its_own_folders(self):
        self.setup["cache"]["batch1"] = "D:/cache"
        self.setup["runs"]["batch1"] = "D:/runs"
        log = self.root / "runs" / "batch2_lab_front" / "b2_baseline_physnet_face" / "cache"
        (log / "PreTrainedModels").mkdir(parents=True)
        (log / "PreTrainedModels" / "b2_baseline_physnet_face_Epoch7.pth").write_bytes(b"")
        (log / "saved_test_outputs").mkdir()
        with open(log / "saved_test_outputs" / "b2_baseline_physnet_face_outputs.pickle", "wb") as handle:
            pickle.dump({"meta": {"epoch": 7}}, handle)
        runs, _ = thesis_configs.build_runs(self.setup, self.splits)
        literature = runs["batch1_literature/UBFC/physnet_face.yaml"]
        self.assertEqual(literature["LOG"]["PATH"], "D:/runs/batch1_literature/b1_UBFC_physnet_face")
        self.assertEqual(literature["TRAIN"]["DATA"]["CACHED_PATH"], "D:/cache")
        self.assertEqual(literature["TRAIN"]["DATA"]["DATA_PATH"], "F:/u")
        for relative in ("batch2_lab_front/baseline/physnet_face.yaml", "batch3_lab_back/baseline/finetune/physnet_face.yaml"):
            lab = runs[relative]
            self.assertTrue(lab["LOG"]["PATH"].startswith(self.setup["runs"]["batch2"]))
            self.assertEqual(lab["TRAIN"]["DATA"]["CACHED_PATH"], "C:/cache")
            self.assertEqual(lab["TRAIN"]["DATA"]["DATA_PATH"], "F:/lab")
        self.assertTrue(runs["batch3_lab_back/baseline/finetune/physnet_face.yaml"]["MODEL"]["INIT_WEIGHTS"]
                        .endswith("b2_baseline_physnet_face_Epoch7.pth"))

    def test_every_training_has_the_same_early_stopping(self):
        runs, _ = thesis_configs.build_runs(self.setup, self.splits)
        trainings = [config for config in runs.values() if config["TOOLBOX_MODE"] == "train_and_test"]
        self.assertEqual(len(trainings), 87)
        for config in trainings:
            self.assertEqual(config["TRAIN"]["EARLY_STOPPING"],
                             {"ENABLED": True, "PATIENCE": 4, "TOLERANCE": 0.0, "MIN_EPOCHS": 10})

    def test_finding_paths_fills_empty_entries_and_keeps_the_folders_of_other_computers(self):
        self.setup["video"]["PURE"] = ""
        changes = thesis_configs.update_paths(
            self.setup, {("video", "PURE"): ["E:/found/PURE"], ("video", "COHFACE"): ["E:/found/COHFACE"]},
            {"JZC": [(5, "E:/found/JZC")], "RM": [(5, "E:/found/RM")]})
        self.assertEqual(len(changes), 2)
        self.assertEqual(self.setup["video"]["PURE"], "E:/found/PURE")
        self.assertEqual(self.setup["uv"]["lab"]["JZC"], "E:/found/JZC")
        self.assertEqual(self.setup["video"]["COHFACE"], "F:/c")                # missing on this computer: kept
        self.assertEqual(self.setup["uv"]["lab"]["RM"], "F:/uv/RM")
        written = self.root / "paths.yaml"
        self.setup["cache"]["batch1"] = "D:/cache"
        thesis_configs.write_paths(self.setup, written)
        with open(written, encoding="utf-8") as handle:
            self.assertEqual(yaml.safe_load(handle), self.setup)

    def test_the_committed_configs_have_the_folders_of_the_paths_file(self):
        def folders(node, prefix=""):
            found = {}
            for key, value in node.items():
                if isinstance(value, dict):
                    found.update(folders(value, f"{prefix}{key}."))
                elif key in ("DATA_PATH", "CACHED_PATH", "EXPERIMENT_PATHS") or prefix + key == "LOG.PATH":
                    found[prefix + key] = value
            return found

        setup, splits = thesis_configs.load_inputs()
        runs, _ = thesis_configs.build_runs(setup, splits)
        for relative, config in runs.items():
            with self.subTest(config=relative), open(thesis_configs.OUTPUT_DIR / relative, encoding="utf-8") as handle:
                self.assertEqual(folders(yaml.safe_load(handle)), folders(config))

    def test_the_split_file_assigns_each_session_once(self):
        errors, _ = thesis_configs.check_inputs(self.setup, self.splits)
        self.assertEqual(errors, [])
        broken = yaml.safe_load(yaml.safe_dump(self.splits))
        broken["lab"]["persons"]["jzc02"]["sessions"].append("RM/RM06")
        errors, _ = thesis_configs.check_inputs(self.setup, broken)
        self.assertTrue(any("RM/RM06" in error for error in errors))


class LauncherCheckTests(unittest.TestCase):
    def test_subject_overlap_missing_weights_and_experiment_folders(self):
        setup = {"cache": dict.fromkeys(("batch1", "batch2", "batch3"), "C:/cache"),
                 "runs": dict.fromkeys(("batch1", "batch2", "batch3"), "C:/runs"),
                 "video": {"UBFC-rPPG": "F:/u", "PURE": "F:/p", "COHFACE": "F:/c", "lab": "F:/lab"},
                 "uv": {"UBFC-rPPG": "F:/uu", "PURE": "F:/up", "COHFACE": "F:/uc",
                        "lab": {"baseline": "F:/uv/baseline", "JZC": "", "RM": "F:/uv/RM"}}, "options": {}}
        _, splits = thesis_configs.load_inputs()
        runs, _ = thesis_configs.build_runs(setup, splits)
        with tempfile.TemporaryDirectory() as folder:
            thesis_configs.write_runs(runs, folder)
            values, _, _ = configs.normalize(configs.flatten(configs.read_yaml_tree(
                Path(folder) / "batch1_literature" / "PURE" / "physnet_face.yaml")))
            self.assertNotIn("TEST.DATA.SUBJECTS", {w["key"] for w in validation.validate(values, "local")["warnings"]})
            values["TEST.DATA.SUBJECTS"] = ["05", "07"]
            self.assertIn("TEST.DATA.SUBJECTS", {w["key"] for w in validation.validate(values, "local")["warnings"]})
            values, _, _ = configs.normalize(configs.flatten(configs.read_yaml_tree(
                Path(folder) / "batch3_lab_back" / "JZC" / "finetune" / "physnet_uvfull.yaml")))
            errors = {e["key"] for e in validation.validate(values, "local")["errors"]}
        self.assertIn("MODEL.INIT_WEIGHTS", errors)
        self.assertIn("TRAIN.DATA.VHRM.EXPERIMENT_PATHS", errors)


if __name__ == "__main__":
    unittest.main()
