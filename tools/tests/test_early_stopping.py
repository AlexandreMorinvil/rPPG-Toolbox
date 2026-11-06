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

    def test_loaded_state_is_judged_by_the_current_patience(self):
        saved = {'best_loss': 1.0, 'bad_epochs': 4, 'stopped': True}
        higher, lower = EarlyStopping(8, 0.0), EarlyStopping(3, 0.0)
        higher.load_state_dict(saved, 6)
        lower.load_state_dict({**saved, 'stopped': False}, 6)
        self.assertFalse(higher.stopped)
        self.assertEqual(higher.bad_epochs, 4)
        self.assertTrue(lower.stopped)

    def test_minimum_epochs_holds_off_a_used_up_patience(self):
        stopping = EarlyStopping(2, 0.0, min_epochs=5)
        self.assertEqual([stopping.update(loss) for loss in [1.0, 2.0, 3.0, 4.0, 5.0]],
                         [False, False, False, False, True])
        self.assertEqual(stopping.bad_epochs, 4)

    def test_minimum_epochs_does_not_hide_an_improvement(self):
        stopping = EarlyStopping(2, 0.0, min_epochs=4)
        self.assertEqual([stopping.update(loss) for loss in [1.0, 2.0, 3.0, 0.5, 0.6, 0.7]],
                         [False, False, False, False, False, True])
        self.assertEqual(stopping.best_loss, 0.5)

    def test_minimum_epochs_is_already_met_when_it_is_low(self):
        stopping = EarlyStopping(2, 0.0, min_epochs=1)
        self.assertEqual([stopping.update(loss) for loss in [1.0, 2.0, 3.0]], [False, False, True])

    def test_patience_can_start_after_the_minimum(self):
        stopping = EarlyStopping(2, 0.0, min_epochs=3, patience_after_min_epochs=True)
        self.assertEqual([stopping.update(loss) for loss in [1.0, 2.0, 3.0, 4.0, 5.0]],
                         [False, False, False, False, True])
        self.assertEqual(stopping.bad_epochs, 2, 'the first three epochs do not count')

    def test_patience_after_the_minimum_compares_with_the_best_loss_of_the_minimum(self):
        stopping = EarlyStopping(2, 0.0, min_epochs=2, patience_after_min_epochs=True)
        self.assertEqual([stopping.update(loss) for loss in [1.0, 0.5, 0.7, 0.8]], [False, False, False, True])
        self.assertEqual(stopping.best_loss, 0.5)

    def test_patience_after_no_minimum_is_the_usual_patience(self):
        stopping = EarlyStopping(2, 0.0, patience_after_min_epochs=True)
        self.assertEqual([stopping.update(loss) for loss in [1.0, 2.0, 3.0]], [False, False, True])

    def test_restored_state_only_counts_the_epochs_after_the_minimum(self):
        saved = {'best_loss': 1.0, 'bad_epochs': 4, 'stopped': True}
        stopping = EarlyStopping(2, 0.0, min_epochs=5, patience_after_min_epochs=True)
        stopping.load_state_dict(saved, 6)
        self.assertEqual((stopping.bad_epochs, stopping.stopped), (1, False))
        stopping.load_state_dict(saved, 3)
        self.assertEqual((stopping.bad_epochs, stopping.stopped), (0, False))
        stopping.load_state_dict(saved, 9)
        self.assertEqual((stopping.bad_epochs, stopping.stopped), (4, True))

    def test_restored_state_is_judged_by_the_current_minimum(self):
        saved = {'best_loss': 1.0, 'bad_epochs': 4, 'stopped': True}
        short, long = EarlyStopping(3, 0.0, min_epochs=6), EarlyStopping(3, 0.0, min_epochs=7)
        short.load_state_dict(saved, 6)
        long.load_state_dict(saved, 6)
        self.assertTrue(short.stopped)
        self.assertFalse(long.stopped)
        self.assertTrue(long.update(2.0), 'the next epoch reaches the minimum')
        replayed = EarlyStopping(2, 0.0, min_epochs=5)
        for _ in range(2):
            replayed.replay([1.0, 2.0, 3.0, 4.0])
            self.assertEqual((replayed.epochs, replayed.stopped), (4, False), 'a replay starts counting again')
        self.assertTrue(replayed.update(5.0))

    def test_replay_gives_the_state_of_these_settings(self):
        losses = [1.0, 0.95, 0.9, 0.85]
        strict = EarlyStopping(5, 0.0)
        strict.replay(losses)
        self.assertEqual((strict.best_loss, strict.bad_epochs), (0.85, 0))
        tolerant = EarlyStopping(2, 0.2)
        tolerant.replay(losses)
        self.assertEqual((tolerant.best_loss, tolerant.bad_epochs, tolerant.stopped), (1.0, 3, True))
        tolerant.replay([])
        self.assertEqual((tolerant.best_loss, tolerant.bad_epochs, tolerant.stopped), (None, 0, False))

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
        cfg.TRAIN.EARLY_STOPPING.TOLERANCE = 0.0
        for min_epochs in (-1, 1.5, True):
            cfg.TRAIN.EARLY_STOPPING.MIN_EPOCHS = min_epochs
            with self.assertRaisesRegex(ValueError, 'MIN_EPOCHS'):
                validate_settings(cfg.TRAIN.EARLY_STOPPING, False)
        cfg.TRAIN.EARLY_STOPPING.MIN_EPOCHS = 0
        validate_settings(cfg.TRAIN.EARLY_STOPPING, False)
        self.assertFalse(cfg.TRAIN.EARLY_STOPPING.PATIENCE_AFTER_MIN_EPOCHS, 'the patience is held back by default')
        for flag in (1, 'yes'):
            cfg.TRAIN.EARLY_STOPPING.PATIENCE_AFTER_MIN_EPOCHS = flag
            with self.assertRaisesRegex(ValueError, 'PATIENCE_AFTER_MIN_EPOCHS'):
                validate_settings(cfg.TRAIN.EARLY_STOPPING, False)

    def test_launcher_defaults_and_yaml_round_trip(self):
        values, _, _ = configs.normalize({})
        for suffix in ('ENABLED', 'PATIENCE', 'TOLERANCE', 'MIN_EPOCHS', 'PATIENCE_AFTER_MIN_EPOCHS'):
            key = 'TRAIN.EARLY_STOPPING.' + suffix
            self.assertEqual(values[key], getattr(_C.TRAIN.EARLY_STOPPING, suffix))
            self.assertIn(key, schema.FIELD_MAP)
        values.update({'TOOLBOX_MODE': 'train_and_test', 'TRAIN.EARLY_STOPPING.ENABLED': True,
                       'TRAIN.EARLY_STOPPING.PATIENCE': 3, 'TRAIN.EARLY_STOPPING.TOLERANCE': 0.01,
                       'TRAIN.EARLY_STOPPING.MIN_EPOCHS': 12, 'TRAIN.EARLY_STOPPING.PATIENCE_AFTER_MIN_EPOCHS': True,
                       'TEST.USE_LAST_EPOCH': False})
        tree = configs.build_tree(values)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.yaml'
            path.write_text(configs.dump_yaml(tree))
            cfg = get_config(type('Args', (), {'config_file': str(path)})())
            self.assertTrue(cfg.TRAIN.EARLY_STOPPING.ENABLED)
            self.assertEqual(cfg.TRAIN.EARLY_STOPPING.PATIENCE, 3)
            self.assertEqual(cfg.TRAIN.EARLY_STOPPING.TOLERANCE, 0.01)
            self.assertEqual(cfg.TRAIN.EARLY_STOPPING.MIN_EPOCHS, 12)
            self.assertTrue(cfg.TRAIN.EARLY_STOPPING.PATIENCE_AFTER_MIN_EPOCHS)

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

    def test_launcher_validation_checks_the_minimum_epochs(self):
        def validated(min_epochs, **stopping):
            values, _, _ = configs.normalize({})
            values.update({'TOOLBOX_MODE': 'train_and_test', 'TRAIN.EARLY_STOPPING.ENABLED': True, 'TRAIN.EPOCHS': 20,
                           'TEST.USE_LAST_EPOCH': False, 'TRAIN.EARLY_STOPPING.MIN_EPOCHS': min_epochs,
                           **{'TRAIN.EARLY_STOPPING.' + name: value for name, value in stopping.items()}})
            with patch.object(validation, '_check_split'), patch.object(validation, '_check_model'), \
                    patch.object(validation, '_check_split_consistency'):
                report = validation.validate(values, backend='local')
            return {kind: {entry['key'] for entry in report[kind]} for kind in ('errors', 'warnings')}
        key = 'TRAIN.EARLY_STOPPING.MIN_EPOCHS'
        self.assertIn(key, validated(-1)['errors'])
        self.assertNotIn(key, validated(19)['warnings'])
        self.assertIn(key, validated(20)['warnings'], 'early stopping could never end the training early')
        after = {'PATIENCE': 7, 'PATIENCE_AFTER_MIN_EPOCHS': True}
        self.assertNotIn(key, validated(12, **after)['warnings'])
        self.assertIn(key, validated(13, **after)['warnings'], 'the minimum plus the patience reach the maximum')


if __name__ == '__main__':
    unittest.main()