"""Preprocessed cache discovery across nested dataset directories."""

import csv
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from launcher import paths, preprocessed


class PreprocessedDiscoveryTests(unittest.TestCase):
    def make_cache(self, directory, experiment):
        lists = directory / 'DataFileLists'
        lists.mkdir(parents=True)
        clips = directory / experiment
        clips.mkdir()
        (clips / 'subject10_input0.npy').touch()
        (clips / 'subject10_label0.npy').touch()
        file_list = lists / f'{experiment}_0.0_1.0.csv'
        relative = clips.relative_to(self.root).as_posix()
        with file_list.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle)
            writer.writerow(['input_files'])
            writer.writerow([f'/cache/{relative}/subject10_input0.npy'])
        return file_list

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.addCleanup(preprocessed._samples.cache_clear)

    def test_parent_file_lists_do_not_hide_nested_cache(self):
        parent_list = self.make_cache(self.root, 'vHRM_existing')
        nested = self.root / 'UBFC_rPPG_v2'
        nested_list = self.make_cache(
            nested, 'UBFC-rPPG_72x72_C128_DN_LDN_Y5F_F30_10a08ad806e6')
        with patch.object(preprocessed, '_roots', return_value=[self.root]):
            sets = preprocessed.scan()
            self.assertEqual({item['path'] for item in sets}, {str(self.root), str(nested)})
            self.assertEqual(
                {item['path'] for cache in sets for item in cache['lists']},
                {str(parent_list), str(nested_list)})
            env = dict(paths.DOCKER_ENV_DEFAULTS, CACHE_PATH=str(self.root))
            with patch.object(paths, 'docker_env', return_value=env):
                result = preprocessed.samples(nested_list)
            self.assertEqual(result['missing'], 0)
            self.assertEqual(len(result['samples']), 1)
            self.assertIsNotNone(result['samples'][0]['label'])

    def test_depth_limit_is_preserved(self):
        self.make_cache(self.root, 'parent')
        self.make_cache(self.root / 'nested', 'child')
        with patch.object(preprocessed, '_roots', return_value=[self.root]):
            self.assertEqual(len(preprocessed.scan(max_depth=0)), 1)
            self.assertEqual(len(preprocessed.scan(max_depth=1)), 2)

    def test_hidden_directories_are_not_scanned(self):
        self.make_cache(self.root / '.hidden', 'experiment')
        with patch.object(preprocessed, '_roots', return_value=[self.root]):
            self.assertEqual(preprocessed.scan(), [])


if __name__ == '__main__':
    unittest.main()