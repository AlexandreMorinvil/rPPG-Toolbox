"""tools/uv_slim.py: the slimmed file gives the same UV frames and keeps only what the UV mode reads."""

from pathlib import Path
import contextlib
import io
import sys
import tempfile
import unittest

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset.data_loader import uv_source as U  # noqa: E402
from test_uv_source import settings, write_h5  # noqa: E402
import uv_slim  # noqa: E402


def make_file(path):
    rng = np.random.default_rng(1)
    rgb = rng.integers(0, 255, (12, 32, 32, 3)).astype(np.uint8)
    write_h5(path, rgb, status=np.array([1] * 10 + [0, 1], np.uint8))
    with h5py.File(path, "a") as h5:
        h5["body_param/body_pose"] = np.zeros((12, 21, 3))
        h5["pd_cam"] = np.zeros((12, 4, 4))
        h5["packed_uv_skin_rgb"].attrs["note"] = "kept"


def frames(path, **overrides):
    node = settings(**overrides)
    return U.read_uv_frames(path, U.uv_settings(node), 8, target_fps=20.0)


class SlimTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.folder = Path(self.tmp.name)
        self.path = self.folder / "sub" / "cam1_Camera_Front_skin_uv.h5"
        self.path.parent.mkdir()
        make_file(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def run_main(self, *args):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            code = uv_slim.main([str(self.folder), *args])
        return code, out.getvalue()

    def test_same_frames_and_only_kept_items(self):
        expected = [frames(self.path), frames(self.path, CHARTS=["head"])]
        code, _ = self.run_main()
        self.assertEqual(code, 0)
        with h5py.File(self.path, "r") as h5:
            self.assertEqual(sorted(h5.keys()), ["frame_status", "packed_uv_body_mask", "packed_uv_skin_rgb",
                                                 "packed_uv_texel_chart", "source_timestamp_s"])
            self.assertEqual(h5.attrs["effective_fps"], 30.0)
            self.assertEqual(h5["packed_uv_skin_rgb"].attrs["note"], "kept")
            self.assertEqual(h5["packed_uv_skin_rgb"].compression, "gzip")
        for before, after in zip(expected, [frames(self.path), frames(self.path, CHARTS=["head"])]):
            np.testing.assert_array_equal(before[0], after[0])
            np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(list(self.folder.rglob("*" + uv_slim.TMP_SUFFIX)), [])

    def test_dry_run_changes_nothing_and_second_run_is_a_no_op(self):
        size = self.path.stat().st_size
        _, out = self.run_main("--dry-run")
        self.assertIn("would drop", out)
        self.assertEqual(self.path.stat().st_size, size)
        self.run_main()
        _, out = self.run_main()
        self.assertIn("already slim", out)

    def test_keep_motion(self):
        self.run_main("--keep-motion")
        with h5py.File(self.path, "r") as h5:
            self.assertIn("body_param", h5)
            self.assertIn("pd_cam", h5)
            self.assertNotIn("uv_skin_rgb", h5)

    def test_file_without_packed_layout_is_skipped(self):
        with h5py.File(self.path, "a") as h5:
            del h5["packed_uv_skin_rgb"]
        _, out = self.run_main()
        self.assertIn("skipped", out)
        with h5py.File(self.path, "r") as h5:
            self.assertIn("uv_skin_rgb", h5)


if __name__ == "__main__":
    unittest.main()
