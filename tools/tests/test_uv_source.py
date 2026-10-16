"""UV input mode: pooling, filling, file resolution, HDF5 reading, unsupervised averaging and cache naming."""

from pathlib import Path
import argparse
import contextlib
import io
import json
import sys
import tempfile
import unittest

import h5py
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from cache_naming import compact_name, flatten_settings, preprocessing_identity  # noqa: E402
from config import _C, _uv_node, get_config  # noqa: E402
from dataset.data_loader import uv_source as U  # noqa: E402
from dataset.data_loader import VHRMLoader as V  # noqa: E402
from launcher import configs, validation  # noqa: E402
from unsupervised_methods.utils import masked_spatial_mean  # noqa: E402

CHARTS = ["head", "torso"]


def write_h5(path, rgb, status=None, timestamps=None, fps=30.0, stride=1, body=None, chart_map=None):
    """A minimal vHRM Studio file: packed layout with two charts (left half head, right half torso)."""
    count, size = rgb.shape[:2]
    body = np.ones((size, size), np.uint8) if body is None else body
    if chart_map is None:
        chart_map = np.zeros((size, size), np.uint8)
        chart_map[:, size // 2:] = 1
    with h5py.File(path, "w") as h5:
        h5.create_dataset("packed_uv_skin_rgb", data=rgb, chunks=(1,) + rgb.shape[1:], compression="gzip")
        h5.create_dataset("uv_skin_rgb", data=rgb, chunks=(1,) + rgb.shape[1:], compression="gzip")
        h5["packed_uv_body_mask"] = body
        h5["uv_body_mask"] = body
        h5["packed_uv_texel_chart"] = chart_map.reshape(-1)[np.flatnonzero(body)]
        h5["frame_status"] = np.ones(count, np.uint8) if status is None else status
        h5["source_timestamp_s"] = np.arange(count) / fps if timestamps is None else timestamps
        h5.attrs["packed_uv_chart_names"] = CHARTS
        h5.attrs["effective_fps"] = fps
        h5.attrs["frame_stride"] = stride


def settings(**overrides):
    node = _uv_node()
    node.ENABLED = True
    for key, value in overrides.items():
        setattr(node, key, value)
    return node


class PoolingTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.rgb = rng.integers(1, 255, (16, 16, 3)).astype(np.uint8)
        self.skin = rng.random((16, 16)) > 0.4

    def expected(self, factor):
        size = 16 // factor
        out = np.zeros((size, size, 3))
        coverage = np.zeros((size, size))
        for row in range(size):
            for col in range(size):
                block = (slice(row * factor, (row + 1) * factor), slice(col * factor, (col + 1) * factor))
                mask = self.skin[block]
                coverage[row, col] = mask.mean()
                if mask.any():
                    out[row, col] = self.rgb[block][mask].mean(axis=0)
        return out, coverage

    def test_integer_factor_is_the_mean_of_skin_texels(self):
        mean, coverage = U.pool_masked(self.rgb, self.skin, 4)
        expected, expected_coverage = self.expected(4)
        np.testing.assert_allclose(mean, expected, rtol=1e-5)
        np.testing.assert_allclose(coverage, expected_coverage, rtol=1e-6)

    def test_any_size_ignores_non_skin_texels(self):
        rgb = self.rgb.copy()
        rgb[~self.skin] = 0
        reference, _ = U.pool_masked(rgb, self.skin, 6)
        rgb[~self.skin] = 255  # values outside the mask must not leak into the mean
        mean, coverage = U.pool_masked(rgb, self.skin, 6)
        np.testing.assert_allclose(mean, reference, rtol=1e-5)
        self.assertEqual(mean.shape, (6, 6, 3))
        self.assertTrue(np.all((coverage >= 0) & (coverage <= 1 + 1e-6)))
        uniform, _ = U.pool_masked(np.full((16, 16, 3), 100, np.uint8), self.skin, 6)
        np.testing.assert_allclose(uniform[uniform > 0], 100, rtol=1e-5)

    def test_coverage_is_relative_to_the_region(self):
        region = np.zeros((16, 16), bool)
        region[:, :8] = True
        skin = region.copy()
        skin[:, :4] = False
        _, coverage = U.pool_masked(self.rgb, skin, 1, U.pool_region(region, 1))
        self.assertAlmostEqual(float(coverage[0, 0]), 0.5)

    def test_resample_rows_nearest(self):
        rows = U.resample_rows(np.arange(10) / 10.0, 30.0)  # 10 fps stored, 30 fps wanted
        np.testing.assert_array_equal(rows[:6], [0, 0, 1, 1, 1, 2])
        self.assertEqual(len(rows), 28)


class FillTests(unittest.TestCase):
    def setUp(self):
        self.frames = np.zeros((4, 1, 2, 3), np.float32)
        self.valid = np.zeros((4, 1, 2), bool)
        for step, value in ((1, 10.0), (2, 20.0)):
            self.frames[step, 0, 0] = value
            self.valid[step, 0, 0] = True

    def test_none_keeps_zeros(self):
        out = U.fill_invalid(self.frames.copy(), self.valid, "none")
        np.testing.assert_array_equal(out[:, 0, 0, 0], [0, 10, 20, 0])

    def test_temporal_mean(self):
        out = U.fill_invalid(self.frames.copy(), self.valid, "temporal_mean")
        np.testing.assert_array_equal(out[:, 0, 0, 0], [15, 10, 20, 15])
        np.testing.assert_array_equal(out[:, 0, 1], 0)  # never valid

    def test_hold(self):
        out = U.fill_invalid(self.frames.copy(), self.valid, "hold")
        np.testing.assert_array_equal(out[:, 0, 0, 0], [10, 10, 20, 20])
        np.testing.assert_array_equal(out[:, 0, 1], 0)


class FileTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def test_resolution_by_name_and_manifest(self):
        subject = self.root / "subject1"
        subject.mkdir()
        (subject / "vid_skin_uv.h5").write_bytes(b"")
        self.assertEqual(Path(U.resolve_uv_path(subject / "vid.avi", self.root)), subject / "vid_skin_uv.h5")
        (self.root / "01-01_skin_uv.h5").write_bytes(b"")
        self.assertEqual(Path(U.resolve_uv_path(str(self.root / "01-01") + "/", self.root)),
                         self.root / "01-01_skin_uv.h5")
        (subject / "renamed.h5").write_bytes(b"")
        manifest = {"items": [{"source": "subject1/other.mp4", "output": "subject1/renamed.h5"}]}
        (self.root / U.MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
        U._manifest_cache.clear()
        self.assertEqual(Path(U.resolve_uv_path(subject / "other.mp4", self.root)), subject / "renamed.h5")
        with self.assertRaises(FileNotFoundError):
            U.resolve_uv_path(subject / "missing.avi", self.root)

    def test_read_pools_masks_and_keeps_bad_rows(self):
        rgb = np.zeros((5, 32, 32, 3), np.uint8)
        rgb[:, :, :16] = 100           # head chart: skin everywhere
        rgb[:, :4, 16:] = 50           # torso chart: skin in the top rows only
        status = np.array([1, 1, 2, 1, 1], np.uint8)
        path = self.root / "a_skin_uv.h5"
        write_h5(path, rgb, status=status)
        frames, valid, info = U.read_uv_frames(path, U.uv_settings(settings(MIN_COVERAGE=0.2)), 4)
        self.assertEqual(frames.shape, (5, 4, 4, 3))
        self.assertTrue(np.all(frames[2] == 0) and not valid[2].any())  # status 2: kept, invalid
        np.testing.assert_allclose(frames[0, :, :2], 100)
        np.testing.assert_allclose(frames[0, 0, 2:], 50)   # block half skin (coverage 0.5)
        self.assertFalse(valid[0, 1:, 2:].any())
        self.assertEqual(info["bad_rows"], 1)
        head, _, _ = U.read_uv_frames(path, U.uv_settings(settings(CHARTS=["head"])), 4)
        self.assertTrue(np.all(head[0, :, 2:] == 0))
        odd, odd_valid, _ = U.read_uv_frames(path, U.uv_settings(settings()), 6)
        self.assertEqual(odd.shape, (5, 6, 6, 3))
        np.testing.assert_allclose(odd[0][odd_valid[0]].max(), 100, rtol=1e-4)

    def test_read_resamples_to_the_target_rate(self):
        rgb = np.zeros((6, 8, 8, 3), np.uint8)
        for row in range(6):
            rgb[row] = 10 * (row + 1)
        path = self.root / "b_skin_uv.h5"
        write_h5(path, rgb, fps=10.0, stride=3)
        frames, _, info = U.read_uv_frames(path, U.uv_settings(settings()), 2, target_fps=30.0)
        np.testing.assert_allclose(frames[:6, 0, 0, 0], [10, 10, 20, 20, 20, 30])
        self.assertEqual(len(frames), 16)
        self.assertEqual(info["frame_stride"], 3)

    def test_reader_replaces_a_loader_read_video(self):
        subject = self.root / "subject3"
        subject.mkdir()
        write_h5(subject / "vid_skin_uv.h5", np.full((3, 16, 16, 3), 80, np.uint8))
        reader = U.UVVideoReader(settings(), 8, data_root=str(self.root))
        frames = reader(str(subject / "vid.avi"))
        self.assertEqual(frames.shape, (3, 8, 8, 3))
        self.assertEqual(frames.dtype, np.float32)

    def test_vhrm_discovery_uses_the_h5_files(self):
        segment = self.root / "baseline_vhrm" / "jzc01_baseline" / "labelled_segments" / "0000_Baseline_part1"
        segment.mkdir(parents=True)
        (segment / "cam2_Camera_Front_skin_uv.h5").write_bytes(b"")
        meta = {"index": 0, "label": "Baseline_part1", "start_sec": 0.0, "duration_sec": 60.0,
                "signal_anchor_datetime_utc": "2026-05-27T13:00:00+00:00",
                "cameras": [{"label": "Camera_Front", "video_file": "cam2_Camera_Front.mp4"}]}
        (segment / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
        video = V.discover_segments(str(self.root))[0]["cameras"][0]
        uv = V.discover_segments(str(self.root), suffix="_skin_uv.h5")[0]["cameras"][0]
        self.assertFalse(video["exists"])
        self.assertTrue(uv["exists"])
        self.assertTrue(uv["path"].endswith("cam2_Camera_Front_skin_uv.h5"))


class UnsupervisedTests(unittest.TestCase):
    def test_masked_mean_ignores_zero_pixels_and_fills_empty_frames(self):
        frames = np.zeros((3, 2, 2, 3))
        frames[0, 0, 0] = [10, 20, 30]
        frames[2, :, :] = [30, 40, 50]
        rgb = masked_spatial_mean(frames)
        self.assertEqual(rgb.shape, (3, 1, 1, 3))
        np.testing.assert_allclose(rgb[:, 0, 0], [[10, 20, 30], [20, 30, 40], [30, 40, 50]])


class NamingTests(unittest.TestCase):
    def data(self, dataset="UBFC-rPPG"):
        data = _C.TRAIN.DATA.clone()
        data.DATASET, data.FS = dataset, 30
        data.PREPROCESS.DATA_TYPE, data.PREPROCESS.LABEL_TYPE = ["Raw"], "Raw"
        return data

    def test_identity_unchanged_when_off_and_separate_when_on(self):
        data = self.data()
        flat = flatten_settings(data, "TRAIN.DATA")
        off = preprocessing_identity(flat, "TRAIN.DATA")
        self.assertNotIn("uv", off)
        data.UV.ENABLED = True
        on_flat = flatten_settings(data, "TRAIN.DATA")
        self.assertIn("uv", preprocessing_identity(on_flat, "TRAIN.DATA"))
        self.assertIn("_UV_", compact_name(on_flat, "TRAIN.DATA"))
        data.UV.CHARTS = ["head"]
        self.assertNotEqual(compact_name(on_flat, "TRAIN.DATA"),
                            compact_name(flatten_settings(data, "TRAIN.DATA"), "TRAIN.DATA"))

    def test_uv_configs_agree_with_the_launcher(self):
        paths = sorted((ROOT / "configs").rglob("*_UV*.yaml"))
        self.assertTrue(paths)
        for path in paths:
            with self.subTest(config=path.name), contextlib.redirect_stdout(io.StringIO()):
                config = get_config(argparse.Namespace(config_file=str(path)))
                values, errors, _ = configs.normalize(configs.flatten(configs.read_yaml_tree(path)))
                self.assertFalse(errors)
                for prefix, split in (("TRAIN.DATA", config.TRAIN.DATA), ("TEST.DATA", config.TEST.DATA),
                                      ("UNSUPERVISED.DATA", config.UNSUPERVISED.DATA)):
                    if not split.DATASET or not split.UV.ENABLED:
                        continue
                    self.assertFalse(split.PREPROCESS.CROP_FACE.DO_CROP_FACE)
                    locations = validation.derived_locations(values, prefix)
                    self.assertEqual(locations["exp_data_name"], split.EXP_DATA_NAME)
                    self.assertEqual(Path(locations["file_list"]).name, Path(split.FILE_LIST_PATH).name)
                    if split.DATASET == "vHRM":
                        self.assertTrue(split.EXP_DATA_NAME.startswith("vHRMUV_"))

    def test_invalid_settings_are_rejected(self):
        data = self.data()
        with self.assertRaises(ValueError):
            U.validate_settings(settings(LAYOUT="standard", CHARTS=["head"]), data.PREPROCESS)
        data.PREPROCESS.RESIZE.H = 64
        with self.assertRaises(ValueError):
            U.validate_settings(settings(), data.PREPROCESS)
        data.PREPROCESS.RESIZE.H = 128
        with self.assertRaises(ValueError):
            U.validate_settings(settings(), data.PREPROCESS, "SCAMPS")


if __name__ == "__main__":
    unittest.main()
