"""Tells when the code this launcher process runs differs from the code on disk.

Python keeps imported modules until a restart, while the page and each job read their code from disk when they start.
A launcher left running can then show a new setting and silently ignore it.
"""

import hashlib
import os
import shlex
import sys
import threading

from . import paths


def _stamp(path):
    try:
        stat = os.stat(path)
    except OSError:
        return None
    return stat.st_mtime_ns, stat.st_size


def _digest(path):
    try:
        with open(path, "rb") as handle:
            return hashlib.sha256(handle.read()).hexdigest()
    except OSError:
        return None


class CodeWatch:
    def __init__(self, root=paths.TOOLBOX_ROOT):
        self.root = os.path.abspath(root)
        self.prefix = os.path.normcase(self.root) + os.sep
        self.lock = threading.Lock()
        self.loaded = {}   # file: digest of the code in memory (a module imported later: when first seen)
        self.current = {}  # file: (stamp, digest) on disk; only read again when its stamp changes
        self.changed_files()

    def _files(self):
        """Source files of the toolbox modules in memory; configs.py loads config.py without a module name."""
        files = {os.path.join(self.root, "config.py")}
        for module in list(sys.modules.values()):
            source = getattr(module, "__file__", None)
            if not isinstance(source, str) or not source.endswith(".py"):
                continue
            source = os.path.abspath(source)
            if os.path.normcase(source).startswith(self.prefix) and "site-packages" not in source:
                files.add(source)
        return files

    def changed_files(self):
        """Files whose content differs from the code in memory, relative to the toolbox root."""
        with self.lock:
            changed = []
            for path in self._files():
                stamp = _stamp(path)
                if path not in self.current or self.current[path][0] != stamp:
                    self.current[path] = (stamp, _digest(path))
                digest = self.current[path][1]
                if self.loaded.setdefault(path, digest) != digest:
                    changed.append(os.path.relpath(path, self.root).replace(os.sep, "/"))
            return sorted(changed)

    def notice(self):
        """What the page needs to ask for a restart, or None when the code in memory is current."""
        changed = self.changed_files()
        if not changed:
            return None
        return {"files": changed, "command": shlex.join(["python", "-m", "launcher", *sys.argv[1:]])}
