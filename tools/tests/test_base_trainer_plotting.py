"""Regression tests for scheduler learning-rate histories and W&B replay."""

import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
# Load just BaseTrainer, without importing every model through trainer/__init__.
SPEC = importlib.util.spec_from_file_location(
    "base_trainer_plotting_test", ROOT / "neural_methods/trainer/BaseTrainer.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class BaseTrainerPlottingTests(unittest.TestCase):
    def test_scheduler_and_scalar_histories(self):
        histories = (
            [0.001, 0.0001],
            [[0.001], [0.0001]],
            [(0.001,), (0.0001,)],
            [[0.001, 0.002], [0.0001, 0.0002]],
        )
        for history in histories:
            with self.subTest(history=history):
                self._check_plots(history, enabled=True)

    def test_disabled_wandb(self):
        self._check_plots([[0.001], [0.0001]], enabled=False)

    def _check_plots(self, history, enabled):
        trainer = MODULE.BaseTrainer()
        trainer.model_file_name = "regression"
        with tempfile.TemporaryDirectory() as directory:
            config = SimpleNamespace(
                TOOLBOX_MODE="train_and_test",
                LOG=SimpleNamespace(PATH=directory),
                TRAIN=SimpleNamespace(DATA=SimpleNamespace(EXP_DATA_NAME="test")),
            )
            with (
                patch.object(MODULE.wandb_logger, "is_enabled", return_value=enabled),
                patch.object(MODULE.wandb_logger, "log") as log,
                patch.object(MODULE.wandb_logger, "log_image") as log_image,
            ):
                trainer.plot_losses_and_lrs([0.3, 0.2], [0.25, 0.15], history, config)
                if enabled:
                    lr_metrics = [
                        call.args[0] for call in log.call_args_list
                        if "train/lr" in call.args[0]
                    ]
                    self.assertEqual(lr_metrics, [
                        {"train/lr": 0.001, "scheduler_step": 0},
                        {"train/lr": 0.0001, "scheduler_step": 1},
                    ])
                    self.assertEqual(log_image.call_count, 2)
                else:
                    log.assert_not_called()
                    log_image.assert_not_called()
            plot_dir = Path(directory) / "test" / "plots"
            for name in ("losses", "learning_rates"):
                for extension in ("pdf", "png"):
                    self.assertGreater(
                        (plot_dir / f"regression_{name}.{extension}").stat().st_size, 0
                    )


if __name__ == "__main__":
    unittest.main()