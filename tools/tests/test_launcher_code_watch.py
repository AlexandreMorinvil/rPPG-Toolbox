"""The launcher tells when the code on disk differs from the code it runs."""

import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from launcher.code_watch import CodeWatch

CHANGED = "changed = 2\n"  # another size: a same-size write can share its timestamp with the previous one on Windows


class CodeWatchTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        patcher = patch.dict(sys.modules)
        patcher.start()
        self.addCleanup(patcher.stop)

    def loaded(self, relative, text="code = 1\n", root=None):
        """A source file and its module in memory."""
        path = (root or self.root) / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        module = types.ModuleType("_watched_" + relative.replace("/", "_"))
        module.__file__ = str(path)
        sys.modules[module.__name__] = module
        return path

    def test_nothing_is_reported_until_a_loaded_file_changes(self):
        path = self.loaded("launcher/jobs.py")
        watch = CodeWatch(self.root)
        self.assertEqual(watch.changed_files(), [])
        self.assertIsNone(watch.notice())
        path.write_text(CHANGED)
        self.assertEqual(watch.changed_files(), ["launcher/jobs.py"])

    def test_the_content_decides_not_the_date(self):
        path = self.loaded("launcher/jobs.py")
        watch = CodeWatch(self.root)
        os.utime(path, (1, 1))
        self.assertEqual(watch.changed_files(), [], "a pull that rewrites the same content")
        path.write_text(CHANGED)
        self.assertEqual(watch.changed_files(), ["launcher/jobs.py"])
        path.write_text("code = 1\n")
        self.assertEqual(watch.changed_files(), [], "back to the code in memory")

    def test_a_deleted_file_counts(self):
        path = self.loaded("launcher/jobs.py")
        watch = CodeWatch(self.root)
        path.unlink()
        self.assertEqual(watch.changed_files(), ["launcher/jobs.py"])

    def test_only_toolbox_code_in_memory_counts(self):
        unused = self.root / "launcher" / "unused.py"
        unused.parent.mkdir()
        unused.write_text("code = 1\n")
        library = self.loaded("site-packages/library.py")
        with tempfile.TemporaryDirectory() as elsewhere:
            outside = self.loaded("other.py", root=Path(elsewhere))
            watch = CodeWatch(self.root)
            for path in (unused, library, outside):
                path.write_text(CHANGED)
            self.assertEqual(watch.changed_files(), [])

    def test_config_is_watched_without_a_module_name(self):
        config = self.root / "config.py"
        config.write_text("code = 1\n")
        watch = CodeWatch(self.root)
        config.write_text(CHANGED)
        self.assertEqual(watch.changed_files(), ["config.py"])

    def test_a_module_imported_later_starts_from_when_it_is_first_seen(self):
        watch = CodeWatch(self.root)
        path = self.loaded("launcher/results.py")
        self.assertEqual(watch.changed_files(), [])
        path.write_text(CHANGED)
        self.assertEqual(watch.changed_files(), ["launcher/results.py"])

    def test_the_notice_gives_the_files_and_the_same_command(self):
        path = self.loaded("launcher/jobs.py")
        watch = CodeWatch(self.root)
        path.write_text(CHANGED)
        with patch.object(sys, "argv", ["launcher", "--lan", "--no-browser"]):
            self.assertEqual(watch.notice(), {"files": ["launcher/jobs.py"],
                                              "command": "python -m launcher --lan --no-browser"})


if __name__ == "__main__":
    unittest.main()
