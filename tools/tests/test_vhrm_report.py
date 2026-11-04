"""Analysis report (evaluation/vhrm_report): inference helpers, and a report of a synthetic vHRM dataset.

The synthetic dataset has two participants with a resting and a curl segment each: small flat-colour videos whose
green channel dips on every heartbeat, with the ECG R-peaks, Zephyr HR and PPG files of the vHRM layout. The
network is replaced by the (negated) mean green value, so the report must recover the known HR and beats. POS
runs as it is (UnsupervisedReportTests).
"""

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from config import get_config  # noqa: E402
from evaluation import vhrm_report  # noqa: E402
from evaluation.vhrm_report import data, inference  # noqa: E402

FS = 30
ANCHOR = 1779886800.0  # 2026-05-27T13:00:00Z
SEGMENTS = (("0000_Resting", 0.0), ("0001_Curl", 50.0))
SECONDS = 40.0


class FakeVideo:
    def __init__(self, frames):
        self.frames = frames

    def __len__(self):
        return len(self.frames)

    def clip(self, start, length):
        return self.frames[start:start + length]


class InferenceHelperTests(unittest.TestCase):
    def test_overlap_add_rebuilds_the_signal_and_undoes_sign_flips(self):
        times = np.arange(1000) / FS
        pulse = np.sin(2 * np.pi * 1.3 * times) + 0.3 * np.sin(2 * np.pi * 2.6 * times + 0.5)
        frames = pulse[:, None, None, None].repeat(2, axis=1).repeat(2, axis=2)
        signs = iter(np.where(np.random.default_rng(0).random(100) < 0.5, -1.0, 1.0))
        network = lambda clips: np.stack([next(signs) * 3.0 * clip[:, 0, 0, 0] for clip in clips])  # noqa: E731
        bvp, flips = inference.predict_bvp(network, FakeVideo(frames), 160, 40, 3)
        self.assertEqual(len(bvp), len(pulse))
        self.assertGreater(abs(np.corrcoef(bvp, pulse)[0, 1]), 0.999)
        self.assertGreater(flips, 0)

    def test_chunked_video_reads_clips_across_chunks(self):
        with tempfile.TemporaryDirectory() as folder:
            frames = np.arange(5 * 10 * 2 * 2 * 3, dtype=np.float16).reshape(50, 2, 2, 3)
            paths = []
            for index in range(5):
                paths.append(str(Path(folder) / f"r_input{index}.npy"))
                np.save(paths[-1], frames[index * 10:(index + 1) * 10])
            video = data.ChunkedVideo(paths, keep=2)
            self.assertEqual(len(video), 50)
            for start, length in ((0, 10), (7, 16), (38, 12), (3, 40)):
                np.testing.assert_array_equal(video.clip(start, length), frames[start:start + length])

    def test_frame_wise_models_see_one_batch_of_frames(self):
        class Trainer:
            num_of_gpu, base_len = 1, 10
            model = staticmethod(lambda frames: frames.mean(dim=(1, 2, 3)).unsqueeze(1))  # (frames, 1)
        clips = torch.arange(2 * 20 * 3 * 4 * 4, dtype=torch.float32).reshape(2, 20, 3, 4, 4)
        output = inference.FORWARD["Tscan"](Trainer(), clips)
        self.assertEqual(tuple(output.shape), (2, 20))
        torch.testing.assert_close(output, clips.mean(dim=(2, 3, 4)))
        with self.assertRaises(ValueError):
            inference.FORWARD["Tscan"](Trainer(), clips[:, :15])
        efficient = inference.FORWARD["EfficientPhys"](
            type("T", (Trainer,), {"model": staticmethod(lambda frames: frames[:-1].mean(dim=(1, 2, 3)).unsqueeze(1))})(), clips)
        self.assertEqual(tuple(efficient.shape), (2, 20))

    def test_layouts_follow_the_data_format(self):
        clips = torch.zeros(2, 8, 4, 5, 3)
        self.assertEqual(tuple(inference.to_layout(clips, "NDCHW").shape), (2, 8, 3, 4, 5))
        self.assertEqual(tuple(inference.to_layout(clips, "NCDHW").shape), (2, 3, 8, 4, 5))


def _iso(epochs):
    return [pd.Timestamp(epoch, unit="s", tz="UTC").isoformat() for epoch in epochs]


def _beats(seed, duration):
    """R-peak times: HR 72 +/- 6 bpm (20 s period) with a small beat-to-beat jitter."""
    rng = np.random.default_rng(seed)
    beats = [0.3]
    while beats[-1] < duration:
        hr = 72 + 6 * np.sin(2 * np.pi * beats[-1] / 20)
        beats.append(beats[-1] + 60 / hr + rng.normal(0, 0.01))
    return np.asarray(beats)


def write_dataset(root):
    """vHRM layout: <root>/toy/<participant>/{zephyr_1.csv, synthetic_ppg/*.csv, labelled_segments/*}."""
    beats_of = {}
    for seed, participant in enumerate(("p01", "p02")):
        folder = root / "toy" / participant
        beats = _beats(seed, 100.0)
        beats_of[participant] = beats
        (folder / "synthetic_ppg").mkdir(parents=True)
        seconds = np.arange(0, 100.0, 1.0)
        hr = np.interp(seconds, beats[1:], 60 / np.diff(beats))
        pd.DataFrame({"timestamp_utc": _iso(ANCHOR + seconds), "heart_rate_bpm": hr}).to_csv(folder / "zephyr_1.csv", index=False)
        pd.DataFrame({"timestamp_utc": _iso(ANCHOR + beats)}).to_csv(folder / "synthetic_ppg" / "synthetic_ppg_rr.csv", index=False)
        ppg_times = np.arange(0, 100.0, 0.02)
        ppg = sum(np.exp(-0.5 * ((ppg_times - beat - 0.2) / 0.08) ** 2) for beat in beats)
        pd.DataFrame({"timestamp_utc": _iso(ANCHOR + ppg_times), "synthetic_ppg": ppg}).to_csv(
            folder / "synthetic_ppg" / "synthetic_ppg.csv", index=False)
        for index, (name, start) in enumerate(SEGMENTS):
            segment = folder / "labelled_segments" / name
            segment.mkdir(parents=True)
            times = start + np.arange(int(SECONDS * FS)) / FS
            volume = sum(np.exp(-0.5 * ((times - beat - 0.2) / 0.08) ** 2) for beat in beats)
            writer = cv2.VideoWriter(str(segment / "cam2_Camera_Front.avi"), cv2.VideoWriter_fourcc(*"MJPG"), FS, (24, 24))
            for value in volume:
                writer.write(np.full((24, 24, 3), (90, 140 - 12 * value, 170), dtype=np.uint8))  # BGR
            writer.release()
            meta = {"index": index, "label": name.split("_", 1)[1], "start_sec": start, "duration_sec": SECONDS,
                    "signal_anchor_datetime_utc": "2026-05-27T13:00:00+00:00",
                    "cameras": [{"label": "Camera_Front", "video_file": "cam2_Camera_Front.avi", "fps": FS}]}
            (segment / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return beats_of


def data_block(root, cache, participants, labels):
    return {"DATASET": "vHRM", "DATA_PATH": str(root), "CACHED_PATH": str(cache), "DO_PREPROCESS": True, "FS": FS,
            "DATA_FORMAT": "NCDHW",
            "PREPROCESS": {"DATA_TYPE": ["Raw"], "LABEL_TYPE": "Standardized", "DO_CHUNK": True, "CHUNK_LENGTH": 128,
                           "RESIZE": {"W": 16, "H": 16}, "NUM_WORKERS": 1},
            "VHRM": {"EXPERIMENTS": ["toy"], "PARTICIPANTS": participants, "LABELS": labels, "VIEWS": ["front"],
                     "CROP_MODE_BY_VIEW": ["front=full"], "VIDEO_DECODER": "opencv", "SHOW_VIDEO_PROGRESS": False,
                     "REQUIRE_PPG": False}}


def green(trainer, clips):
    """Stands in for the network: the mean green value of each frame (NCDHW), negated as blood darkens green."""
    return -clips[:, 1].mean(dim=(2, 3))


class SyntheticReportTests(unittest.TestCase):
    @classmethod
    def make_run(cls, log_path, train, test, name):
        """A training config of the synthetic dataset and a checkpoint of its (untrained) network."""
        root = Path(cls.directory.name)
        tree = {"BASE": [""], "TOOLBOX_MODE": "train_and_test", "DEVICE": "cpu",
                "MODEL": {"NAME": "Physnet", "PHYSNET": {"FRAME_NUM": 128}}, "LOG": {"PATH": str(log_path)},
                "TRAIN": {"EPOCHS": 1, "MODEL_FILE_NAME": "toy",
                          "DATA": data_block(root / "data", root / "cache", train, [])},
                "TEST": {"USE_LAST_EPOCH": True, "DATA": data_block(root / "data", root / "cache", test, ["Curl"])},
                "INFERENCE": {"BATCH_SIZE": 4}}
        config_path = root / f"{name}.yaml"
        config_path.write_text(yaml.safe_dump(tree), encoding="utf-8")
        config = get_config(argparse.Namespace(config_file=str(config_path)))
        model_dir = Path(config.MODEL.MODEL_DIR)
        model_dir.mkdir(parents=True)
        checkpoint = model_dir / "toy_Epoch0.pth"
        torch.save(inference.build_trainer(config).model.state_dict(), checkpoint)
        return config, checkpoint

    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        root = Path(cls.directory.name)
        cls.beats = write_dataset(root / "data")
        cls.config, cls.checkpoint = cls.make_run(root / "runs", ["p01"], ["p02"], "toy")
        with patch.dict(inference.FORWARD, {"Physnet": green}):
            cls.output = vhrm_report.generate_report(cls.config, cls.checkpoint)
        cls.table = pd.read_csv(cls.output / "summary" / "per_recording_metrics.csv")

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_report_goes_next_to_the_checkpoint(self):
        self.assertEqual(self.output, self.checkpoint.parent.parent / "report" / "Epoch0")

    def test_every_participant_and_label_with_its_split(self):
        rows = {(r.participant, r.label, r.repetition): r.split for r in self.table.itertuples()}
        self.assertEqual(rows, {("p01", "resting", 1): "training", ("p01", "curl", 1): "training",
                                ("p02", "resting", 1): "test", ("p02", "curl", 1): "test"})
        test_table = pd.read_csv(self.output / "test_set" / "summary" / "per_recording_metrics.csv")
        self.assertEqual(set(test_table.participant), {"p02"})
        self.assertEqual(len(list((self.output / "test_set" / "data").glob("*.npz"))), 2)

    def test_hr_and_beats_are_recovered(self):
        self.assertTrue((self.table.hr_mae_bpm < 2.0).all(), self.table.hr_mae_bpm.tolist())
        self.assertTrue((self.table.ibi_mae_ms < 25).all(), self.table.ibi_mae_ms.tolist())
        self.assertTrue((self.table.ibi_n > 30).all())
        self.assertTrue((self.table.duration_s == 38.4).all(), "9 clips of 128 frames are cached per 40 s segment")

    def test_rpeaks_are_on_the_frame_timeline(self):
        saved = vhrm_report.pipeline.load_result(self.output / "data" / "p02_curl1.npz")
        beats = self.beats["p02"] - SEGMENTS[1][1]
        expected = beats[(beats >= 0) & (beats <= saved["duration_s"])]
        np.testing.assert_allclose(saved["series"]["rpeaks_gt"], expected, atol=1e-3)
        self.assertEqual(saved["split"], "test")

    def test_outputs(self):
        for name in ("REPORT.md", "METHOD.md", "run_config.json", "summary/overview.png", "summary/participant_errors.png",
                     "summary/face_crops.png", "summary/label_split_mae.csv", "mosaics/hr_curl1.png",
                     "mosaics/rr_resting1.png", "participants/p01/pulse.png", "data_csv/README.txt",
                     "data_csv/all_recordings/heart_rate.csv", "test_set/REPORT.md", "test_set/data_csv/recordings.csv"):
            self.assertTrue((self.output / name).is_file(), name)
        groups = set(pd.read_csv(self.output / "summary" / "global_metrics.csv").group)
        self.assertTrue({"All recordings", "Curl", "Resting", "test participants", "Curl · test participants"} <= groups)
        matrix = pd.read_csv(self.output / "summary" / "label_split_mae.csv")
        self.assertEqual(list(matrix.columns), ["metric_key", "unit", "label", "all", "test", "training"])
        run = json.loads((self.output / "run_config.json").read_text(encoding="utf-8"))
        self.assertEqual(run["splits"], {"test": ["p02"], "training": ["p01"]})
        self.assertIn("MAE by movement and split", (self.output / "REPORT.md").read_text(encoding="utf-8"))

    def test_a_second_report_reuses_the_model_output(self):
        calls = []

        def counting(trainer, clips):
            calls.append(len(clips))
            return -clips[:, 1].mean(dim=(2, 3))
        with patch.dict(inference.FORWARD, {"Physnet": counting}):
            vhrm_report.generate_report(self.config, self.checkpoint, labels=["Curl"], export_csv=False)
        self.assertEqual(calls, [], "unchanged checkpoint and input: no inference")
        self.assertEqual(sorted(p.stem for p in (self.output / "data").glob("*.npz")), ["p01_curl1", "p02_curl1"])
        self.assertFalse((self.output / "data_csv").exists())
        # The fuller report again, for the other tests (order-independent).
        with patch.dict(inference.FORWARD, {"Physnet": green}):
            vhrm_report.generate_report(self.config, self.checkpoint)

    def test_cross_validation_pools_the_test_participants_of_every_fold(self):
        import tools.vhrm_cross_validation as cross_validation
        root = Path(self.directory.name) / "cv"
        with patch.dict(inference.FORWARD, {"Physnet": green}):
            for fold, (train, test) in {"1": (["p01"], ["p02"]), "2": (["p02"], ["p01"])}.items():
                config, checkpoint = self.make_run(root / f"fold_{fold}", train, test, f"fold{fold}")
                vhrm_report.generate_report(config, checkpoint, labels=["Curl"], export_csv=False)
        cross_validation.aggregate(str(root))  # what the launcher runs after each fold
        merged = root / "report" / "cross_validation"
        table = pd.read_csv(merged / "summary" / "per_recording_metrics.csv")
        self.assertEqual(sorted(zip(table.participant, table.fold)), [("p01", 2), ("p02", 1)])
        self.assertEqual(set(table.split), {"test"})
        run = json.loads((merged / "run_config.json").read_text(encoding="utf-8"))
        self.assertEqual((run["origin"], [f["fold"] for f in run["folds"]]), ("cross_validation", ["1", "2"]))
        text = (merged / "REPORT.md").read_text(encoding="utf-8")
        self.assertIn("every participant is a test participant of one fold", text)
        self.assertNotIn("MAE by movement and split", text)
        self.assertTrue((merged / "participants" / "p01" / "hr.png").is_file())
        self.assertIn("fold", pd.read_csv(merged / "data_csv" / "all_recordings" / "heart_rate.csv").columns)


class UnsupervisedReportTests(unittest.TestCase):
    """POS on the synthetic dataset, whole recordings (DO_CHUNK off), reported by main.py's hook."""

    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        root = Path(cls.directory.name)
        write_dataset(root / "data")
        block = data_block(root / "data", root / "cache", [], [])
        block["DATA_FORMAT"] = "NDHWC"
        block["PREPROCESS"].update({"LABEL_TYPE": "Raw", "DO_CHUNK": False})
        tree = {"BASE": [""], "TOOLBOX_MODE": "unsupervised_method", "DEVICE": "cpu", "LOG": {"PATH": str(root / "runs")},
                "UNSUPERVISED": {"METHOD": ["POS"], "DATA": block}, "REPORT": {"ENABLED": True}}
        config_path = root / "pos.yaml"
        config_path.write_text(yaml.safe_dump(tree), encoding="utf-8")
        cls.config = get_config(argparse.Namespace(config_file=str(config_path)))
        cls.folders = vhrm_report.report_after_training(cls.config) or []
        cls.output = cls.folders[0] if cls.folders else None

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def setUp(self):
        self.assertEqual(len(self.folders), 1, "the report failed: see its traceback above")
        self.table = pd.read_csv(self.output / "summary" / "per_recording_metrics.csv")

    def test_report_goes_to_the_experiment_folder(self):
        self.assertEqual(self.output, Path(self.config.UNSUPERVISED.OUTPUT_SAVE_DIR).parent / "report" / "POS")

    def test_whole_recordings_of_everyone_without_splits(self):
        self.assertEqual(sorted(zip(self.table.participant, self.table.label)),
                         [("p01", "curl"), ("p01", "resting"), ("p02", "curl"), ("p02", "resting")])
        self.assertTrue(self.table.split.isna().all())
        self.assertTrue((self.table.duration_s == SECONDS).all(), "one uncut clip per recording: nothing is lost")
        self.assertFalse((self.output / "test_set").exists())

    def test_hr_and_beats_are_recovered(self):
        self.assertTrue((self.table.hr_mae_bpm < 2.0).all(), self.table.hr_mae_bpm.tolist())
        self.assertTrue((self.table.ibi_mae_ms < 25).all(), self.table.ibi_mae_ms.tolist())

    def test_outputs_name_the_method(self):
        run = json.loads((self.output / "run_config.json").read_text(encoding="utf-8"))
        self.assertEqual((run["method"], run["checkpoint"], run["config_mode"]), ("POS", None, "unsupervised_method"))
        text = (self.output / "REPORT.md").read_text(encoding="utf-8")
        self.assertIn("none (unsupervised method)", text)
        self.assertIn("POS is an unsupervised method", text)
        for name in ("METHOD.md", "summary/overview.png", "participants/p01/pulse.png", "data_csv/README.txt"):
            self.assertTrue((self.output / name).is_file(), name)

    def test_unknown_methods_fail_before_preprocessing(self):
        with self.assertRaises(ValueError):
            vhrm_report.generate_method_reports(self.config, methods=["XYZ"])


if __name__ == "__main__":
    unittest.main()
