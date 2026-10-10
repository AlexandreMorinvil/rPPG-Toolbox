"""Compact preprocessing names and backwards-compatible cache resolution."""

import argparse
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cache_naming import (SETTINGS_FILENAME, compact_name, flatten_settings,
                          legacy_name, preprocessing_identity, settings_path)
from config import _C, _cache_directory, get_config
from dataset.data_loader.BaseLoader import BaseLoader
from launcher import configs, validation


class CacheNamingTests(unittest.TestCase):
    def setUp(self):
        self.data = _C.TRAIN.DATA.clone()
        self.data.DATASET = 'UBFC-rPPG'
        self.data.FS = 30
        self.data.PREPROCESS.DATA_TYPE = ['DiffNormalized']
        self.data.PREPROCESS.LABEL_TYPE = 'DiffNormalized'
        self.data.PREPROCESS.CHUNK_LENGTH = 128
        self.data.PREPROCESS.RESIZE.W = 72
        self.data.PREPROCESS.RESIZE.H = 72
        self.data.PREPROCESS.CROP_FACE.BACKEND = 'Y5F'

    def name(self, data=None, prefix='TRAIN.DATA'):
        return compact_name(flatten_settings(data or self.data, prefix), prefix)

    def test_short_descriptive_and_deterministic(self):
        self.assertRegex(self.name(), r'^UBFC-rPPG_72x72_C128_DN_LDN_Y5F_F30_[0-9a-f]{12}$')
        self.assertLess(len(self.name()), 65)
        self.assertEqual(self.name(), self.name(prefix='TEST.DATA'))
        self.assertNotEqual(self.name(), self.name(prefix='UNSUPERVISED.DATA'))

    def test_every_preprocessing_setting_changes_digest_except_workers(self):
        base = self.name()
        flat = flatten_settings(self.data, 'TRAIN.DATA')
        for key, value in flat.items():
            if '.PREPROCESS.' not in key or key.endswith('.NUM_WORKERS'):
                continue
            changed = dict(flat)
            if isinstance(value, bool):
                changed[key] = not value
            elif isinstance(value, (int, float)):
                changed[key] = value + 1
            elif isinstance(value, list):
                changed[key] = value + ['different']
            else:
                changed[key] = value + 'different'
            with self.subTest(key=key):
                self.assertNotEqual(base, compact_name(changed, 'TRAIN.DATA'))
        self.data.PREPROCESS.NUM_WORKERS += 1
        self.data.BEGIN = 0.7
        self.data.FOLD.FOLD_NAME = 'different'
        self.assertEqual(base, self.name())
        self.data.FS = 60
        self.assertNotEqual(base, self.name())

    def test_canonical_placeholder_lists_and_numbers(self):
        flat = flatten_settings(self.data, 'TRAIN.DATA')
        normalized = {key: ([] if value == [''] else value) for key, value in flat.items()}
        normalized['TRAIN.DATA.FS'] = 30.0
        self.assertEqual(self.name(), compact_name(normalized, 'TRAIN.DATA'))

    def test_height_is_not_width(self):
        self.data.PREPROCESS.RESIZE.H = 48
        self.assertIn('_72x48_', self.name())
        self.assertIn('SizeH72', legacy_name(flatten_settings(self.data, 'TRAIN.DATA'), 'TRAIN.DATA'))
        self.assertIn('SizeH48', legacy_name(flatten_settings(self.data, 'TEST.DATA'), 'TEST.DATA'))

    def test_legacy_reuse_and_metadata_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            self.data.CACHED_PATH = directory
            self.data.EXP_DATA_NAME = self.name()
            flat = flatten_settings(self.data, 'TRAIN.DATA')
            legacy = Path(directory) / legacy_name(flat, 'TRAIN.DATA')
            legacy.mkdir()
            self.assertEqual(Path(_cache_directory(self.data, 'TRAIN.DATA', True)), legacy)
            self.assertEqual(Path(_cache_directory(self.data, 'TRAIN.DATA', False)).name, self.name())
            settings_path(legacy).write_text('{}', encoding='utf-8')
            self.assertEqual(Path(_cache_directory(self.data, 'TRAIN.DATA', True)).name, self.name())
            settings_path(legacy).write_text(json.dumps(preprocessing_identity(flat, 'TRAIN.DATA')))
            self.assertEqual(Path(_cache_directory(self.data, 'TRAIN.DATA', True)), legacy)
            compact = Path(directory) / self.name()
            compact.mkdir()
            self.assertEqual(Path(_cache_directory(self.data, 'TRAIN.DATA', True)), compact)
            settings_path(legacy).unlink()

    def test_metadata_saved_with_file_list(self):
        with tempfile.TemporaryDirectory() as directory:
            loader = BaseLoader.__new__(BaseLoader)
            loader.config_data = self.data
            loader.cached_path = directory
            loader.file_list_path = str(Path(directory) / 'DataFileLists' / 'clips.csv')
            loader.dataset_name = 'train'
            loader.build_file_list({0: ['subject1_input0.npy']})
            metadata = json.loads((Path(directory) / SETTINGS_FILENAME).read_text())
            self.assertEqual(metadata, preprocessing_identity(flatten_settings(self.data, 'DATA'), 'DATA'))

    def test_all_split_paths_match_launcher_and_explicit_names(self):
        for explicit in ('', 'my_cache'):
            with self.subTest(explicit=explicit), tempfile.TemporaryDirectory() as directory:
                tree = {'TOOLBOX_MODE': 'train_and_test', 'TEST': {'USE_LAST_EPOCH': False}}
                for split in ('TRAIN', 'VALID', 'TEST', 'UNSUPERVISED'):
                    tree.setdefault(split, {})['DATA'] = {
                        'DATASET': 'UBFC-rPPG', 'FS': 30, 'CACHED_PATH': directory,
                        'EXP_DATA_NAME': explicit, 'PREPROCESS': {'DATA_TYPE': ['Raw'], 'LABEL_TYPE': 'Raw'}}
                path = Path(directory) / 'config.yaml'
                path.write_text(yaml.safe_dump(tree))
                with contextlib.redirect_stdout(io.StringIO()):
                    cfg = get_config(argparse.Namespace(config_file=str(path)))
                values, errors, extras = configs.normalize(configs.flatten(tree))
                self.assertFalse(errors)
                values.update(extras)
                for split in ('TRAIN', 'VALID', 'TEST', 'UNSUPERVISED'):
                    data = getattr(cfg, split).DATA
                    preview = validation.derived_locations(values, split + '.DATA')
                    self.assertEqual(data.EXP_DATA_NAME, preview['exp_data_name'])
                    self.assertEqual(Path(data.CACHED_PATH), Path(preview['cache_dir']))
                    self.assertEqual(Path(data.FILE_LIST_PATH), Path(preview['file_list']))


if __name__ == '__main__':
    unittest.main()