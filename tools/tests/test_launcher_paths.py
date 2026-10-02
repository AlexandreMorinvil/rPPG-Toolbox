"""Reject malformed config paths before filesystem resolution or preprocessing."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from launcher import configs, paths, schema, validation


class LauncherPathTests(unittest.TestCase):
    def test_duplicated_drive_is_rejected_before_resolution(self):
        for value in (
            r'C:\C:\Datasets_Preprocessed\UBFC-rPPG\UBFC-rPPG',
            'C:/C:/Datasets_Preprocessed/UBFC-rPPG',
            r'C:\D:\cache',
        ):
            with self.subTest(value=value), patch.object(paths, 'resolve_host') as resolve:
                host, problem = paths.config_host_path(value, 'local')
                self.assertIsNone(host)
                self.assertIn('Invalid Windows path', problem)
                self.assertIn('duplicated drive prefix', problem)
                resolve.assert_not_called()

    def test_valid_host_paths_still_resolve(self):
        for value in (r'C:\Datasets_Preprocessed\UBFC-rPPG',
                      r'\\server\share\cache', 'preprocessed_data'):
            with self.subTest(value=value), patch.object(paths, 'resolve_host', return_value=ROOT) as resolve:
                self.assertEqual(paths.config_host_path(value, 'local'), (ROOT, None))
                resolve.assert_called_once_with(value)

    def test_docker_path_translation_is_unchanged(self):
        with patch.object(paths, 'container_to_host', return_value=ROOT) as translate:
            self.assertEqual(paths.config_host_path('/cache/UBFC-rPPG', 'docker'), (ROOT, None))
            translate.assert_called_once_with('/cache/UBFC-rPPG', None)

    def test_invalid_validation_cache_is_a_field_error(self):
        report = validation.Report()
        key = 'VALID.DATA.CACHED_PATH'
        host = validation._check_path(schema.FIELD_MAP[key], r'C:\C:\cache', 'local', report, {})
        self.assertIsNone(host)
        self.assertFalse(report.as_dict()['ok'])
        self.assertEqual(report.errors[0]['key'], key)

    def test_rerun_config_reuses_matching_cache_roots(self):
        tree = configs.read_yaml_tree(
            ROOT / 'configs/launcher/test/UBFC-rPPG_FactorizePhys_Base_2.train_and_test.yaml')
        self.assertEqual(tree['TOOLBOX_MODE'], 'train_and_test')
        self.assertEqual(tree['TRAIN']['EPOCHS'], 20)
        self.assertTrue(tree['TRAIN']['SAVE_RESUME'])
        for split in ('TRAIN', 'VALID', 'TEST'):
            self.assertEqual(tree[split]['DATA']['CACHED_PATH'], r'C:\Datasets_Preprocessed\UBFC-rPPG')
            self.assertFalse(tree[split]['DATA']['DO_PREPROCESS'])


if __name__ == '__main__':
    unittest.main()