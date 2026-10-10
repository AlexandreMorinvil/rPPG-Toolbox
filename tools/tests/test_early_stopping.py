"""Patience semantics and launcher/YAML configuration, without GPU/data access."""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from config import _C, get_config
from launcher import configs, schema, validation
from neural_methods.early_stopping import EarlyStopping, validate_settings


class EarlyStoppingTests(unittest.TestCase):
    def test_worsening_and_flat_losses_stop_after_patience(self):
        for losses in ([1.0, 2.0, 3.0], [1.0, 1.0, 1.0]):
            stopping = EarlyStopping(2, 0.0)
            self.assertEqual([stopping.update(loss) for loss in losses], [False, False, True])

    def test_tolerance_accumulates_and_significant_improvements_reset(self):
        stopping = EarlyStopping(3, 0.125)
        for loss in [1.0, 0.9375, 0.875]:
            self.assertFalse(stopping.update(loss))
        self.assertEqual(stopping.bad_epochs, 2)  # equal-to-tolerance is not enough
        self.assertFalse(stopping.update(0.8125))
        self.assertEqual(stopping.bad_epochs, 0)
        self.assertEqual(stopping.best_loss, 0.8125)

    def test_zero_tolerance_counts_any_strict_improvement(self):
        stopping = EarlyStopping(1, 0.0)
        self.assertFalse(stopping.update(1.0))
        self.assertFalse(stopping.update(0.999999))
        self.assertTrue(stopping.update(0.999999))

    def test_nonfinite_losses_use_patience_and_can_recover(self):
        stopping = EarlyStopping(2, 0.0)
        self.assertFalse(stopping.update(float('nan')))
        self.assertFalse(stopping.update(1.0))
        self.assertEqual(stopping.bad_epochs, 0)
        self.assertFalse(stopping.update(float('inf')))
        self.assertTrue(stopping.update(float('-inf')))
        self.assertEqual(stopping.best_loss, 1.0)

    def test_settings_validation(self):
        cfg = _C.clone()
        validate_settings(cfg.TRAIN.EARLY_STOPPING, True)  # disabled by default
        cfg.TRAIN.EARLY_STOPPING.ENABLED = True
        with self.assertRaisesRegex(ValueError, 'USE_LAST_EPOCH'):
            validate_settings(cfg.TRAIN.EARLY_STOPPING, True)
        for patience in (0, -1, 1.5, True):
            cfg.TRAIN.EARLY_STOPPING.PATIENCE = patience
            with self.assertRaisesRegex(ValueError, 'PATIENCE'):
                validate_settings(cfg.TRAIN.EARLY_STOPPING, False)
        cfg.TRAIN.EARLY_STOPPING.PATIENCE = 1
        for tolerance in (-0.1, float('nan'), float('inf')):
            cfg.TRAIN.EARLY_STOPPING.TOLERANCE = tolerance
            with self.assertRaisesRegex(ValueError, 'TOLERANCE'):
                validate_settings(cfg.TRAIN.EARLY_STOPPING, False)

    def test_launcher_defaults_and_yaml_round_trip(self):
        values, _, _ = configs.normalize({})
        for suffix in ('ENABLED', 'PATIENCE', 'TOLERANCE'):
            key = 'TRAIN.EARLY_STOPPING.' + suffix
            self.assertEqual(values[key], getattr(_C.TRAIN.EARLY_STOPPING, suffix))
            self.assertIn(key, schema.FIELD_MAP)
        values.update({'TOOLBOX_MODE': 'train_and_test', 'TRAIN.EARLY_STOPPING.ENABLED': True,
                       'TRAIN.EARLY_STOPPING.PATIENCE': 3, 'TRAIN.EARLY_STOPPING.TOLERANCE': 0.01,
                       'TEST.USE_LAST_EPOCH': False})
        tree = configs.build_tree(values)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.yaml'
            path.write_text(configs.dump_yaml(tree))
            cfg = get_config(type('Args', (), {'config_file': str(path)})())
            self.assertTrue(cfg.TRAIN.EARLY_STOPPING.ENABLED)
            self.assertEqual(cfg.TRAIN.EARLY_STOPPING.PATIENCE, 3)
            self.assertEqual(cfg.TRAIN.EARLY_STOPPING.TOLERANCE, 0.01)

    def test_launcher_validation_rejects_incompatible_and_invalid_settings(self):
        values, _, _ = configs.normalize({})
        values.update({'TOOLBOX_MODE': 'train_and_test', 'TRAIN.EARLY_STOPPING.ENABLED': True,
                       'TRAIN.EARLY_STOPPING.PATIENCE': 0, 'TRAIN.EARLY_STOPPING.TOLERANCE': float('nan')})
        with patch.object(validation, '_check_split'), patch.object(validation, '_check_model'), \
                patch.object(validation, '_check_split_consistency'):
            report = validation.validate(values, backend='local')
        keys = {entry['key'] for entry in report['errors']}
        self.assertTrue({'TEST.USE_LAST_EPOCH', 'TRAIN.EARLY_STOPPING.PATIENCE',
                         'TRAIN.EARLY_STOPPING.TOLERANCE'}.issubset(keys))


if __name__ == '__main__':
    unittest.main()