"""Persistent launcher preferences (launcher_data/settings.json)."""

import json
import sys
import threading
from pathlib import Path

from . import paths

SETTINGS_FILE = paths.STATE_DIR / "settings.json"
POLICIES = ("queue", "parallel")
_lock = threading.Lock()

DEFAULTS = {
    "local_python": sys.executable,
    "default_backend": "docker",
    "gpu_policy": "queue",
    "max_parallel_jobs": 2,
    "results_roots": [],
    "cache_roots": [],
}


def load():
    with _lock:
        data = dict(DEFAULTS)
        if SETTINGS_FILE.is_file():
            try:
                data.update(json.loads(SETTINGS_FILE.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                pass
        return data


def _clean_roots(values, label):
    if not isinstance(values, list):
        raise ValueError(f"{label} must be a list of folders")
    roots = []
    for value in values:
        text = str(value).strip()
        if not text:
            continue
        path = Path(text).expanduser()
        if not path.is_absolute():
            raise ValueError(f"{label}: use absolute paths ({text})")
        if not path.is_dir():
            raise ValueError(f"{label}: folder does not exist ({text})")
        roots.append(str(path.resolve()))
    return list(dict.fromkeys(roots))


def save(updates):
    current = load()
    if "local_python" in updates:
        python = Path(str(updates["local_python"]).strip()).expanduser()
        if not python.is_file():
            raise ValueError(f"Python executable not found: {python}")
        current["local_python"] = str(python)
    if "default_backend" in updates:
        if updates["default_backend"] not in ("docker", "local"):
            raise ValueError("default_backend must be docker or local")
        current["default_backend"] = updates["default_backend"]
    if "gpu_policy" in updates:
        if updates["gpu_policy"] not in POLICIES:
            raise ValueError("gpu_policy must be queue or parallel")
        current["gpu_policy"] = updates["gpu_policy"]
    if "max_parallel_jobs" in updates:
        try:
            value = int(updates["max_parallel_jobs"])
        except (TypeError, ValueError):
            raise ValueError("max_parallel_jobs must be an integer") from None
        if not 1 <= value <= 16:
            raise ValueError("max_parallel_jobs must be between 1 and 16")
        current["max_parallel_jobs"] = value
    if "results_roots" in updates:
        current["results_roots"] = _clean_roots(updates["results_roots"], "Results folders")
    if "cache_roots" in updates:
        current["cache_roots"] = _clean_roots(updates["cache_roots"], "Preprocessed data folders")
    with _lock:
        paths.STATE_DIR.mkdir(parents=True, exist_ok=True)
        SETTINGS_FILE.write_text(json.dumps(current, indent=2), encoding="utf-8")
    return current


def _existing_unique(candidates):
    result = []
    for candidate in candidates:
        path = Path(candidate).resolve(strict=False)
        if path.is_dir() and path not in result:
            result.append(path)
    return result


def results_roots():
    env = paths.docker_env()
    return _existing_unique([
        paths.resolve_host(env["RUNS_PATH"]),
        paths.TOOLBOX_ROOT / "runs",
        *load()["results_roots"],
    ])


def cache_roots():
    env = paths.docker_env()
    return _existing_unique([
        paths.resolve_host(env["CACHE_PATH"]),
        paths.TOOLBOX_ROOT / "PreprocessedData",
        paths.TOOLBOX_ROOT / "preprocessed_data",
        *load()["cache_roots"],
    ])
