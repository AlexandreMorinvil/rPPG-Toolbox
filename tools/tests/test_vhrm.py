"""vHRM dataset support: label helpers, selection, and config/launcher naming agreement."""

from pathlib import Path
import argparse
import contextlib
import io
import json
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from config import _vhrm_node, get_config  # noqa: E402
from dataset.data_loader import VHRMLoader as V  # noqa: E402
from dataset.data_loader.BaseLoader import BaseLoader  # noqa: E402
from launcher import configs, validation  # noqa: E402

EXAMPLE_CONFIGS = sorted((ROOT / "configs").rglob("*vHRM*.yaml"))


def pulsing_skin(seconds, fs=30, hr_hz=1.2, seed=0):
    """Mean RGB of skin whose blood volume pulses at hr_hz, and that blood volume.

    More blood absorbs more light, so every channel darkens along the pulsatility
    vector of skin (de Haan & van Leest, 2014)."""
    times = np.arange(int(seconds * fs)) / fs
    blood_volume = np.sin(2 * np.pi * hr_hz * times)
    rng = np.random.default_rng(seed)
    rgb = np.array([150.0, 100.0, 80.0]) * (
        1.0 - 0.004 * blood_volume[:, None] * np.array([0.33, 0.77, 0.53])
        + 0.03 * np.sin(2 * np.pi * 0.05 * times)[:, None]       # slow illumination drift
        + 0.0005 * rng.standard_normal((len(times), 3)))
    return rgb, blood_volume


class PosPseudoLabelTests(unittest.TestCase):
    def test_pulse_rate_and_ppg_polarity(self):
        rgb, blood_volume = pulsing_skin(60)
        label = V.pos_pseudo_ppg(rgb, 30)
        self.assertTrue(np.all(np.isfinite(label)))
        self.assertLessEqual(np.nanmax(np.abs(label)), 1.0 + 1e-9)
        spectrum = np.abs(np.fft.rfft(label))
        self.assertAlmostEqual(np.fft.rfftfreq(len(label), 1 / 30)[np.argmax(spectrum)], 1.2, delta=0.05)
        middle = slice(60, -60)
        self.assertGreater(np.corrcoef(label[middle], blood_volume[middle])[0, 1], 0.9)

    def test_matches_the_toolbox_pseudo_labels_up_to_sign(self):
        rgb, _ = pulsing_skin(40)
        reference = BaseLoader.generate_pos_psuedo_labels(None, rgb.reshape(-1, 1, 1, 3), fs=30)
        label = V.pos_pseudo_ppg(rgb, 30)
        middle = slice(90, -90)
        self.assertGreater(np.corrcoef(label[middle], -reference[middle])[0, 1], 0.999)

    def test_frames_without_a_valid_window_are_missing(self):
        rgb, _ = pulsing_skin(60)
        rgb[600:900] = 0.0                                     # black frames
        label = V.pos_pseudo_ppg(rgb, 30)
        self.assertTrue(np.all(np.isnan(label[660:840])))
        self.assertTrue(np.all(np.isfinite(label[:550])))
        self.assertTrue(np.all(np.isnan(V.pos_pseudo_ppg(rgb[:40], 30))))

    def test_build_signals_uses_the_video_trace_without_sync_shift(self):
        rgb, _ = pulsing_skin(20)
        recording = {"participant_dir": "missing", "segment_dir": "missing", "start_epoch": 0.0}
        settings = {"ppg_source": "pos_pseudo", "sync_delay": 2.5, "hr_source": "zephyr",
                    "rpeak_delay": 0.0, "rpeak_sigma": 0.0}
        signals = V.build_signals(recording, 650, 30, settings, rgb)
        np.testing.assert_allclose(signals["ppg"][:600], V.pos_pseudo_ppg(rgb, 30))
        self.assertTrue(np.all(np.isnan(signals["ppg"][600:])))
        self.assertTrue(np.all(np.isnan(signals["hr"])))


class LabelHelperTests(unittest.TestCase):
    def test_sample_series_leaves_gaps_and_edges_missing(self):
        times = np.array([0.0, 1.0, 2.0, 10.0, 11.0])
        values = np.array([0.0, 1.0, 2.0, 10.0, 11.0])
        result = V.sample_series(times, values, np.array([-0.5, 0.5, 1.5, 5.0, 10.5, 12.0]), max_gap=3.0)
        np.testing.assert_allclose(result[[1, 2, 4]], [0.5, 1.5, 10.5])
        self.assertTrue(np.all(np.isnan(result[[0, 3, 5]])))

    def test_rpeak_waveform_peaks_on_delayed_beats(self):
        peaks = np.array([0.0, 1.0, 2.0, 3.0])
        times = np.arange(0, 3, 0.01)
        wave = V.rpeak_waveform(peaks + 0.2, times, "rpeak_sine", 0.15)
        found = times[1:-1][(wave[1:-1] > wave[:-2]) & (wave[1:-1] >= wave[2:])]
        np.testing.assert_allclose(found, [1.2, 2.2], atol=0.011)
        self.assertTrue(np.all(np.isnan(wave[times < 0.2])))

    def test_rpeak_waveform_drops_implausible_intervals(self):
        wave = V.rpeak_waveform(np.array([0.0, 3.5]), np.array([1.0, 2.0]), "rpeak_gaussian", 0.15)
        self.assertTrue(np.all(np.isnan(wave)))

    def test_transform_label_keeps_missing_samples_missing(self):
        ppg = np.sin(np.linspace(0, 20, 200))
        ppg[:10] = np.nan
        for label_type in ("Raw", "DiffNormalized", "Standardized"):
            with self.subTest(label_type=label_type):
                label = V.transform_label(ppg, label_type)
                self.assertEqual(label.shape, ppg.shape)
                self.assertTrue(np.all(np.isnan(label[:9])))
                self.assertTrue(np.all(np.isfinite(label[11:])))

    def test_crop_modes_and_aliases(self):
        modes = V.parse_crop_modes(["front=face", "Back-Left=roi:0.1,0.2,0.3,0.4", "*=full"])
        self.assertEqual(modes["front"], ("face", None))
        self.assertEqual(modes["back-left"], ("roi", (0.1, 0.2, 0.3, 0.4)))
        self.assertEqual(modes["*"], ("full", None))
        with self.assertRaises(ValueError):
            V.parse_crop_modes(["front=blur"])
        self.assertEqual(V.parse_aliases(["Recover=Recovery"]), {"recover": "recovery"})


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        root = Path(self.directory.name)
        for experiment, participant, index, label in (
                ("JZC", "jzc01", 0, "Resting"), ("JZC", "jzc02", 0, "Recover"),
                ("baseline", "jzc01_baseline", 0, "Baseline_part1")):
            segment = root / experiment / participant / "labelled_segments" / f"{index:04d}_{label}"
            segment.mkdir(parents=True)
            cameras = []
            for number, view in enumerate(("Back-left", "Front", "Back-right"), start=1):
                (segment / f"cam{number}_Camera_{view}.mp4").write_bytes(b"")
                cameras.append({"label": f"Camera_{view}", "video_file": f"cam{number}_Camera_{view}.mp4"})
            meta = {"index": index, "label": label, "start_sec": 10.0, "duration_sec": 60.0, "cameras": cameras,
                    "signal_anchor_datetime_utc": "2026-05-27T13:00:00+00:00"}
            (segment / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
        self.segments = V.discover_segments(str(root))

    def tearDown(self):
        self.directory.cleanup()

    def test_filters_and_ids(self):
        node = _vhrm_node()
        node.EXPERIMENTS, node.LABELS, node.VIEWS = ["jzc"], ["recovery"], ["front", "back-right"]
        recordings = V.select_recordings(self.segments, node)
        self.assertEqual(sorted(r["index"] for r in recordings),
                         ["jzc--jzc02--0000--recovery--back-right", "jzc--jzc02--0000--recovery--front"])
        self.assertEqual(recordings[0]["start_epoch"], 1779886800.0 + 10.0)

    def test_sessions_are_separate_participants(self):
        node = _vhrm_node()
        node.EXCLUDE_PARTICIPANTS = ["jzc01"]
        participants = {r["participant"] for r in V.select_recordings(self.segments, node)}
        self.assertEqual(participants, {"jzc02", "jzc01-baseline"})

    def test_labels_are_rebuilt_when_the_segment_timing_changes(self):
        # Applying a video/signal offset in the preprocessing tool moves the anchor in meta.json.
        loader = V.VHRMLoader.__new__(V.VHRMLoader)
        loader.vhrm = _vhrm_node()
        loader.lag_table = {}
        loader.config_data = argparse.Namespace(FS=30)
        preprocess = argparse.Namespace(LABEL_TYPE="DiffNormalized", DO_CHUNK=True, CHUNK_LENGTH=128)
        recording = dict(V.select_recordings(self.segments, _vhrm_node())[0])
        key = loader._label_key(recording, preprocess)
        self.assertEqual(loader._label_key(dict(recording), preprocess), key)
        recording["start_epoch"] -= 1.31
        self.assertNotEqual(loader._label_key(recording, preprocess), key)


class NamingAgreementTests(unittest.TestCase):
    """The launcher re-implements config.py's cache and file-list names."""

    def test_example_configs_agree(self):
        self.assertTrue(EXAMPLE_CONFIGS)
        for path in EXAMPLE_CONFIGS:
            with self.subTest(config=path.name), contextlib.redirect_stdout(io.StringIO()):
                config = get_config(argparse.Namespace(config_file=str(path)))
                values, errors, _ = configs.normalize(configs.flatten(configs.read_yaml_tree(path)))
                self.assertFalse(errors)
                for prefix, split in (("TRAIN.DATA", config.TRAIN.DATA), ("TEST.DATA", config.TEST.DATA),
                                      ("UNSUPERVISED.DATA", config.UNSUPERVISED.DATA)):
                    if split.DATASET != "vHRM":
                        continue
                    locations = validation.derived_locations(values, prefix)
                    self.assertEqual(locations["exp_data_name"], split.EXP_DATA_NAME)
                    self.assertEqual(Path(locations["file_list"]).name, Path(split.FILE_LIST_PATH).name)


class LauncherPseudoLabelTests(unittest.TestCase):
    def test_generic_pseudo_label_switch_is_redirected_and_pos_test_labels_warned(self):
        path = next(path for path in EXAMPLE_CONFIGS if path.parent.name == "launcher")
        values, errors, _ = configs.normalize(configs.flatten(configs.read_yaml_tree(path)))
        self.assertFalse(errors)
        values["TRAIN.DATA.PREPROCESS.USE_PSUEDO_PPG_LABEL"] = True
        for prefix in ("TRAIN.DATA", "VALID.DATA", "TEST.DATA"):
            values[prefix + ".VHRM.PPG_SOURCE"] = "pos_pseudo"
        report = validation.validate(values, backend="local")
        self.assertIn("TRAIN.DATA.PREPROCESS.USE_PSUEDO_PPG_LABEL", {entry["key"] for entry in report["errors"]})
        warned = {entry["key"] for entry in report["warnings"]}
        self.assertIn("TEST.DATA.VHRM.PPG_SOURCE", warned)
        self.assertNotIn("TRAIN.DATA.VHRM.PPG_SOURCE", warned)
        self.assertNotIn("TRAIN.DATA.VHRM.PPG_SOURCE", {entry["key"] for entry in report["errors"]})


if __name__ == "__main__":
    unittest.main()
