"""Prediction outputs for visualization: signal analysis, saved outputs, highlights, launcher API."""

import csv
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from evaluation import prediction_plots, prediction_store
from evaluation import signal_analysis as sa

FS = 30


def synthetic_recording(seconds=60, hr_start=72, hr_end=90, seed=0, glitch=None):
    """PPG label, differenced noisy prediction and gt[N, 4] (HR, HRV, BR, PPG valid)."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, seconds, 1 / FS)
    hr = np.where(t < seconds / 2, hr_start, hr_end).astype(float)
    ppg = np.sin(np.cumsum(hr / 60 / FS) * 2 * np.pi)
    prediction = ppg + 0.3 * rng.normal(size=t.size)
    if glitch is not None:  # a few seconds locked on the 2x harmonic
        start, end = int(glitch[0] * FS), int(glitch[1] * FS)
        prediction[start:end] = 1.5 * np.sin(np.cumsum(2 * hr[start:end] / 60 / FS) * 2 * np.pi)
    gt = np.column_stack([hr, np.full_like(hr, 40.0), np.full_like(hr, 15.0), np.ones_like(hr)])
    gt[int(10 * FS):int(12 * FS), 3] = 0
    return np.diff(prediction, prepend=prediction[0]), ppg, gt


def chunked(values, chunk=180):
    return {i: values[i * chunk:(i + 1) * chunk] for i in range(len(values) // chunk)}


def write_fixture(path, recordings, with_gt=True):
    predictions, labels, gt = {}, {}, {}
    for name, (prediction, label, truth) in recordings.items():
        predictions[name] = chunked(prediction.astype(np.float32))
        labels[name] = chunked(label.astype(np.float32))
        n = len(predictions[name]) * 180
        gt[name] = truth[:n].astype(np.float32)
    meta = {"split": "test", "name": "fixture", "prediction_is_diff": True, "label_is_diff": False}
    return prediction_store.write_outputs_pickle(str(path), predictions, labels, "Standardized", FS,
                                                 split="test", meta=meta, gt=gt if with_gt else None)


class SignalAnalysisTests(unittest.TestCase):
    def test_beats_spectrogram_and_tracking(self):
        prediction, label, gt = synthetic_recording(glitch=(40, 48))
        wave = sa.pulse_waveform(label, FS, False)
        peaks = sa.beats(wave, FS)
        self.assertAlmostEqual(sa.beat_rate(peaks[peaks < 25 * FS], FS), 72, delta=2)
        times, freqs, power = sa.spectrogram(wave, FS, 10, 1)
        self.assertAlmostEqual(freqs[np.argmax(power[:, 3])], 1.2, delta=0.06)

        rows = sa.recording_windows(prediction, FS, 10, 2, True, label=label, measured_hr=gt[:, 0], track=True)
        raw = sa.window_metrics(rows)
        tracked = sa.window_metrics(rows, "tracked")
        self.assertGreater(raw["max_jump"], 50)          # FFT peak jumps to the harmonic
        self.assertLess(tracked["max_jump"], raw["max_jump"])
        self.assertLess(tracked["mae"], raw["mae"])
        self.assertEqual(sa.invalid_spans(gt[:, 3] > 0.5, FS), [(10.0, 12.0)])

    def test_reference_falls_back_to_label(self):
        prediction, label, gt = synthetic_recording()
        measured = sa.recording_windows(prediction, FS, 10, 0, True, label=label, measured_hr=gt[:, 0])
        from_label = sa.recording_windows(prediction, FS, 10, 0, True, label=label, measured_hr=gt[:, 0],
                                          truth="label")
        self.assertEqual(measured[0]["gt"], 72)
        self.assertEqual(from_label[0]["gt"], from_label[0]["label_hr"])


class PredictionStoreTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_probe_selection_is_stratified_and_deterministic(self):
        ids = [f"{p}--000{i}--{m}--front" for p in ("jzc01", "jzc02") for i, m in
               enumerate(("curl", "resting", "recovery"))]
        first = prediction_store.select_probes(list(reversed(ids)), 3)
        self.assertEqual(first, prediction_store.select_probes(ids, 3))
        self.assertEqual({prediction_store.parse_recording_id(r)["movement"] for r in first},
                         {"curl", "resting", "recovery"})
        self.assertEqual(len({prediction_store.parse_recording_id(r)["participant"] for r in first}), 2)

    def test_probe_history_appends_and_truncates_on_resume(self):
        path = str(self.root / "probe_history.npz")
        probe = {"rec": (np.ones(300), np.zeros(300), None)}
        for epoch in (0, 4, 9):
            prediction_store.update_probe_history(path, epoch, probe)
        history = prediction_store.load_probe_history(path)
        self.assertEqual(history["epochs"].tolist(), [0, 4, 9])
        self.assertEqual(history["pred_0"].shape, (3, 300))
        prediction_store.update_probe_history(path, 4, probe)  # resumed at epoch 4
        self.assertEqual(prediction_store.load_probe_history(path)["epochs"].tolist(), [0, 4])
        prediction_store.update_probe_history(path, 0, {"other": probe["rec"]})  # new run
        history = prediction_store.load_probe_history(path)
        self.assertEqual(history["ids"].tolist(), ["other"])
        self.assertEqual(history["epochs"].tolist(), [0])

    def test_vhrm_ground_truth_follows_chunk_order(self):
        rows = []
        for chunk in (1, 0):
            path = self.root / f"gt{chunk}.npy"
            np.save(path, np.full((4, 4), chunk, dtype=np.float32))
            rows.append({"recording_id": "rec", "chunk": chunk, "gt_files": str(path)})
        file_list = self.root / "list.csv"
        with file_list.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["recording_id", "chunk", "gt_files"])
            writer.writeheader()
            writer.writerows(rows)
        gt = prediction_store.load_vhrm_ground_truth(str(file_list), {"rec": {0: None, 1: None}})
        self.assertEqual(gt["rec"][:, 0].tolist(), [0] * 4 + [1] * 4)

    def test_validation_capture_writes_best_and_probes(self):
        from config import _C

        config = _C.clone()
        config.TOOLBOX_MODE = "train_and_test"
        config.LOG.PATH = str(self.root)
        config.TRAIN.DATA.EXP_DATA_NAME = "exp"
        config.TRAIN.EPOCHS = 10
        config.TRAIN.DATA.PREPROCESS.LABEL_TYPE = "DiffNormalized"
        config.VALID.DATA.FS = FS
        config.VISUALIZE.PROBE_EVERY_N_EPOCHS = 5
        config.VISUALIZE.PROBE_COUNT = 1
        capture = prediction_store.ValidationCapture(config)
        ids, chunks = ["a--0000--curl--front", "b--0000--curl--front"], [0, 0]
        capture.add_batch(ids, chunks, np.zeros((2 * 7, 1)), np.zeros((2 * 7, 1)), frames=8)  # trimmed
        self.assertEqual(capture.predictions, {})
        for epoch, best in ((0, True), (1, False), (4, False)):
            capture.add_batch(ids, chunks, np.random.rand(2, 160), np.random.rand(2, 160))
            capture.commit(epoch, best)
        folder = self.root / "exp" / prediction_store.VALID_DIR_NAME
        best = prediction_store.load_probe_history(str(folder / prediction_store.PROBE_HISTORY_FILE))
        self.assertEqual(best["epochs"].tolist(), [0, 4])
        self.assertEqual(len(best["ids"]), 1)
        import pickle
        with open(folder / prediction_store.BEST_VALID_FILE, "rb") as handle:
            data = pickle.load(handle)
        self.assertEqual(data["meta"]["epoch"], 0)
        self.assertTrue(data["meta"]["prediction_is_diff"])
        self.assertEqual(set(data["predictions"]), set(ids))


class TrainerCaptureTests(unittest.TestCase):
    def test_factorize_validation_loop_saves_outputs(self):
        import importlib
        import torch
        from torch.utils.data import DataLoader, Dataset
        from config import _C

        module = importlib.import_module('neural_methods.trainer.FactorizePhysTrainer')

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.mix = torch.nn.Parameter(torch.tensor([0.4, 0.6, 0.8]))

            def forward(self, data):
                waveform = (data[:, :, :-1].mean(dim=(3, 4)) * self.mix[None, :, None]).sum(dim=1)
                return waveform, waveform

        class Clips(Dataset):
            ids = ["p1--0000--curl--front", "p2--0000--resting--front"]

            def __len__(self):
                return 4

            def __getitem__(self, index):
                return torch.rand(3, 16, 2, 2), torch.rand(16), self.ids[index % 2], index // 2

        with tempfile.TemporaryDirectory() as directory:
            cfg = _C.clone()
            cfg.TOOLBOX_MODE = 'train_and_test'
            cfg.MODEL.NAME = 'FactorizePhys'
            cfg.MODEL.MODEL_DIR = directory
            cfg.LOG.PATH = directory
            cfg.TRAIN.DATA.EXP_DATA_NAME = 'exp'
            cfg.TRAIN.MODEL_FILE_NAME = 'tiny'
            cfg.TRAIN.EPOCHS = 2
            cfg.TRAIN.SAVE_RESUME = False
            cfg.TRAIN.PLOT_LOSSES_AND_LR = False
            cfg.TEST.USE_LAST_EPOCH = False
            cfg.NUM_OF_GPU_TRAIN = 0
            cfg.DEVICE = 'cpu'
            cfg.MODEL.FactorizePhys.MD_FSAM = False
            cfg.VISUALIZE.PROBE_EVERY_N_EPOCHS = 1
            loaders = {'train': DataLoader(Clips(), batch_size=2), 'valid': DataLoader(Clips(), batch_size=2)}
            with (
                patch.object(module, 'FactorizePhys', side_effect=lambda **kwargs: TinyModel()),
                patch.object(module.torch.nn, 'DataParallel', side_effect=lambda model, **kwargs: model),
            ):
                trainer = module.FactorizePhysTrainer(cfg, loaders)
                trainer.train(loaders)
            folder = Path(directory) / 'exp' / prediction_store.VALID_DIR_NAME
            history = prediction_store.load_probe_history(str(folder / prediction_store.PROBE_HISTORY_FILE))
            self.assertEqual(history['epochs'].tolist(), [0, 1])
            self.assertEqual(sorted(history['ids'].tolist()), sorted(Clips.ids))
            self.assertEqual(history['pred_0'].shape, (2, 32))  # two 16-frame chunks per recording
            self.assertTrue((folder / prediction_store.BEST_VALID_FILE).is_file())


class HighlightAndLauncherTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        recordings = {
            "jzc01--0000--resting--front": synthetic_recording(seed=1),
            "jzc01--0001--curl--front": synthetic_recording(seed=2, glitch=(36, 50)),
            "jzc02--0000--resting--front": synthetic_recording(seed=3, hr_start=60, hr_end=65),
        }
        self.folder = self.root / "runs" / "exp" / "saved_test_outputs"
        self.pickle = self.folder / "fixture_outputs.pickle"
        self.outputs = write_fixture(self.pickle, recordings)
        other = self.root / "runs" / "exp2" / "saved_test_outputs" / "other_outputs.pickle"
        write_fixture(other, {k: synthetic_recording(seed=9) for k in recordings})
        self.other = other

    def test_highlights(self):
        index = prediction_plots.render_highlights(self.outputs, str(self.folder), worst=1)
        reasons = [item["reason"] for item in index]
        self.assertEqual(reasons[0], "worst1")
        self.assertEqual(index[0]["recording"], "jzc01--0001--curl--front")
        movements = {prediction_store.parse_recording_id(item["recording"])["movement"] for item in index}
        self.assertEqual(movements, {"curl", "resting"})  # every scenario is represented
        self.assertEqual(len(index), len({item["recording"] for item in index}))
        for item in index:
            self.assertTrue((self.folder / "highlights" / item["file"]).is_file())
        self.assertTrue((self.folder / "highlights" / "index.csv").is_file())

    def test_launcher_analysis(self):
        from launcher import results

        with patch.object(results, "_roots", return_value=[self.root / "runs"]):
            analysis = results.analyse(str(self.pickle), 10, 2, track=True)
            self.assertTrue(analysis["has_gt"])
            self.assertEqual(analysis["truth"], "measured")
            worst = max(analysis["recordings"], key=lambda r: r["mae"])
            self.assertEqual(worst["id"], "jzc01--0001--curl--front")
            self.assertLess(worst["mae_tracked"], worst["mae"])

            detail = results.recording(str(self.pickle), worst["id"], 10, 2, track=True)
            for key in ("prediction", "prediction_raw", "label", "pred_beats", "label_beats", "spectrogram",
                        "gt_channels", "invalid_spans", "metrics_tracked"):
                self.assertIn(key, detail)
            self.assertNotIn("ppg_valid", detail["gt_channels"])
            self.assertEqual(detail["invalid_spans"], [[10.0, 12.0]] if isinstance(detail["invalid_spans"][0], list)
                             else [(10.0, 12.0)])

            overview = results.compare([str(self.pickle), str(self.other)], None, 10, 2)
            self.assertEqual(len(overview["recordings"]), 3)
            self.assertEqual(len(overview["recordings"][0]["mae"]), 2)
            one = results.compare([str(self.pickle), str(self.other)], worst["id"], 10, 2)
            self.assertEqual(len(one["runs"]), 2)

            with patch.object(results, "scan", return_value=[{"path": str(self.root / "runs" / "exp"), "name": "exp"}]):
                self.assertEqual(len(results.sources()), 1)

    def test_launcher_history(self):
        from launcher import results

        folder = self.root / "runs" / "exp" / prediction_store.VALID_DIR_NAME
        prediction, label, gt = synthetic_recording()
        write_fixture(folder / prediction_store.BEST_VALID_FILE, {"rec": (prediction, label, gt)})
        for epoch in (0, 5):
            prediction_store.update_probe_history(str(folder / prediction_store.PROBE_HISTORY_FILE), epoch,
                                                  {"rec": (prediction, label, gt)})
        with patch.object(results, "_roots", return_value=[self.root / "runs"]):
            history = results.history(str(folder / prediction_store.PROBE_HISTORY_FILE), "rec", 10, 2)
        self.assertEqual([e["epoch"] for e in history["epochs"]], [0, 5])
        self.assertLess(history["epochs"][0]["metrics"]["mae"], 5)

    def test_legacy_pickle_without_gt(self):
        from launcher import results

        legacy = self.root / "runs" / "old" / "saved_test_outputs" / "old_outputs.pickle"
        write_fixture(legacy, {"rec": synthetic_recording()}, with_gt=False)
        with patch.object(results, "_roots", return_value=[self.root / "runs"]):
            analysis = results.analyse(str(legacy), 10, 2)
        self.assertFalse(analysis["has_gt"])
        self.assertEqual(analysis["truth"], "label")
        self.assertLess(analysis["overall"]["mae"], 5)


if __name__ == "__main__":
    unittest.main()
