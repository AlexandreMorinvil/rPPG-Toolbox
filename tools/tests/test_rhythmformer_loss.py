"""The RhythmFormer loss on heart rates inside and outside its usual band, on the CPU (the loss itself asks for CUDA)."""

import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from evaluation.post_process import calculate_metric_per_video  # noqa: E402
from neural_methods.loss.RythmFormerLossComputer import (  # noqa: E402
    FULL_HR_BAND, HR_BAND, RhythmFormer_Loss, TorchLossComputer, frequency_band,
)

FS = 30
original_to = torch.Tensor.to


def stay_on_cpu():
    """Patches that keep the loss's explicit CUDA moves on the CPU."""
    def to(self, *args, **kwargs):
        if args and isinstance(args[0], torch.device) and args[0].type == 'cuda':
            return self
        return original_to(self, *args, **kwargs)
    return (patch.object(torch.Tensor, 'cuda', lambda self: self), patch.object(torch.Tensor, 'to', to),
            patch.object(torch.cuda, 'FloatTensor', torch.FloatTensor))


def clip(bpm, frames=160):
    signal = np.sin(2 * math.pi * bpm / 60 * np.arange(frames) / FS).astype(np.float32)
    return torch.from_numpy((signal - signal.mean()) / signal.std())


def label_rate(label):
    return float(calculate_metric_per_video(label, label, diff_flag=0, fs=FS, hr_method='FFT')[0])


class FrequencyBandTests(unittest.TestCase):
    def setUp(self):
        for patcher in stay_on_cpu():
            patcher.start()
            self.addCleanup(patcher.stop)
        torch.manual_seed(0)

    def test_band_of_a_heart_rate(self):
        for rate, band in ((45.0, HR_BAND), (90.0, HR_BAND), (149.9, HR_BAND), (44.9, FULL_HR_BAND),
                           (150.0, FULL_HR_BAND), (36.0, FULL_HR_BAND), (198.0, FULL_HR_BAND)):
            self.assertEqual(frequency_band(rate), band, rate)

    def test_the_full_band_holds_every_heart_rate_the_metric_can_return(self):
        low, high = FULL_HR_BAND
        self.assertEqual((low, high - 1), (round(0.6 * 60), round(3.3 * 60)))

    def fixed_band_cross_entropy(self, prediction, rate, low=45, high=150):
        """The loss before clips outside the band had their own."""
        bpm = torch.arange(low, high, dtype=torch.float)
        spectrum = TorchLossComputer.complex_absolute(prediction.view(1, -1), FS, bpm)
        return F.cross_entropy(spectrum, torch.tensor(rate - low).view(1).long())

    def test_a_clip_inside_the_band_has_exactly_the_loss_it_had(self):
        for bpm in (60, 90, 120):
            label = clip(bpm)
            rate = label_rate(label)
            self.assertEqual(frequency_band(rate), HR_BAND)
            prediction = torch.randn(160)
            value, _ = TorchLossComputer.Frequency_loss(prediction, label, diff_flag=0, Fs=FS, std=3.0)
            self.assertEqual(value.item(), self.fixed_band_cross_entropy(prediction, rate).item())

    def test_clips_outside_the_band_no_longer_break_the_loss(self):
        for bpm in (40, 160, 190):
            label = clip(bpm)
            rate = label_rate(label)
            self.assertFalse(HR_BAND[0] <= rate < HR_BAND[1], f'{bpm} bpm gives {rate}')
            prediction = torch.randn(160, requires_grad=True)
            with self.assertRaises(IndexError):  # the crash this replaces, on the CPU
                self.fixed_band_cross_entropy(prediction, rate)
            loss = RhythmFormer_Loss()(prediction, label, 0, FS, 0)
            self.assertTrue(math.isfinite(loss.item()))
            loss.backward()
            self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_a_clip_outside_the_band_is_rewarded_for_power_at_its_heart_rate(self):
        label = clip(160)
        right, _ = TorchLossComputer.Frequency_loss(label, label, diff_flag=0, Fs=FS, std=3.0)
        wrong, _ = TorchLossComputer.Frequency_loss(clip(80), label, diff_flag=0, Fs=FS, std=3.0)
        self.assertLess(right.item(), wrong.item())


if __name__ == '__main__':
    unittest.main()
