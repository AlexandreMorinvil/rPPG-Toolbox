"""Which evaluation windows the unsupervised methods score (generic datasets; vHRM has its own evaluation)."""

from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from evaluation.post_process import calculate_metric_per_video  # noqa: E402
from unsupervised_methods.unsupervised_predictor import window_skip_reason  # noqa: E402

FS = 20
WINDOW = 30 * FS


class WindowSkipTests(unittest.TestCase):
    def test_a_tail_counts_only_from_ten_seconds(self):
        for length, skipped in ((5, True), (9, True), (10, True), (199, True), (200, False), (599, False)):
            self.assertEqual(bool(window_skip_reason(2 * WINDOW, length, WINDOW, FS)), skipped, length)

    def test_full_windows_are_scored(self):
        for start in (0, WINDOW, 2 * WINDOW):
            self.assertIsNone(window_skip_reason(start, WINDOW, WINDOW, FS))

    def test_a_full_window_shorter_than_ten_seconds_is_not_a_tail(self):
        self.assertIsNone(window_skip_reason(100, 100, 100, FS))
        self.assertTrue(window_skip_reason(1200, 50, 100, FS))

    def test_a_recording_shorter_than_ten_seconds_keeps_its_only_window(self):
        self.assertIsNone(window_skip_reason(0, 150, WINDOW, FS))

    def test_the_padding_limit_matches_the_filter(self):
        signal = np.random.default_rng(0).standard_normal(10)
        self.assertTrue(window_skip_reason(0, 9, WINDOW, FS))
        with self.assertRaises(ValueError):
            calculate_metric_per_video(signal[:9], signal[:9], diff_flag=False, fs=FS, hr_method="FFT")
        self.assertIsNone(window_skip_reason(0, 10, WINDOW, FS))
        calculate_metric_per_video(signal, signal, diff_flag=False, fs=FS, hr_method="FFT")


if __name__ == "__main__":
    unittest.main()
