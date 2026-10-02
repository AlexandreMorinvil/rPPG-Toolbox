"""Filesystem locations and Docker mount translation."""

import ntpath
import os
import re
from pathlib import Path, PurePosixPath

LAUNCHER_DIR = Path(__file__).resolve().parent
TOOLBOX_ROOT = LAUNCHER_DIR.parent
STATIC_DIR = LAUNCHER_DIR / "static"
STATE_DIR = TOOLBOX_ROOT / "launcher_data"
JOBS_DIR = STATE_DIR / "jobs"
ENV_FILE = TOOLBOX_ROOT / ".env"
ENV_EXAMPLE = TOOLBOX_ROOT / "docker" / ".env.example"
DOCKER_CONFIG_DIR = TOOLBOX_ROOT / "docker" / "configs"
LOCAL_CONFIG_DIR = TOOLBOX_ROOT / "configs"

# Writable locations for configs saved from the launcher.
CONFIG_SAVE_DIRS = {
    "docker": DOCKER_CONFIG_DIR,
    "local": LOCAL_CONFIG_DIR,
}

DOCKER_ENV_DEFAULTS = {
    "DATA_ROOT": "./data",
    "CHECKPOINT_ROOT": "./final_model_release",
    "CACHE_PATH": "./preprocessed_data",
    "RUNS_PATH": "./runs",
    "GPU_DEVICE_ID": "0",
    "DOCKER_SHM_SIZE": "8gb",
    "CONFIG_FILE": "/opt/vhrm2/configs/train/UBFC-rPPG_DATASET2_DeepPhys.docker.yaml",
    "TOOLBOX_IMAGE": "vhrm2-rppg:toolbox-cu124",
}
EDITABLE_ENV_KEYS = (
    "DATA_ROOT", "CHECKPOINT_ROOT", "CACHE_PATH", "RUNS_PATH",
    "GPU_DEVICE_ID", "DOCKER_SHM_SIZE", "TOOLBOX_IMAGE",
)
# Container prefix -> .env key (None means fixed host directory).
DOCKER_MOUNTS = (
    ("/data", "DATA_ROOT", True),
    ("/checkpoints", "CHECKPOINT_ROOT", True),
    ("/cache", "CACHE_PATH", False),
    ("/runs", "RUNS_PATH", False),
    ("/opt/vhrm2/configs", None, True),
)
JOB_CONFIG_MOUNT = "/opt/vhrm2/launcher"

_ENV_LINE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$")


def _unquote(value):
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
        return value[1:-1]
    if " #" in value:
        value = value.split(" #", 1)[0].rstrip()
    return value


def read_env_file(path=ENV_FILE):
    values = {}
    if not Path(path).is_file():
        return values
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        match = _ENV_LINE.match(line)
        if match:
            values[match.group(1)] = _unquote(match.group(2))
    return values


def _quote(value):
    value = str(value)
    if value == "" or re.search(r"[\s#'\"]", value):
        return '"' + value.replace('"', '\\"') + '"'
    return value


def update_env_file(updates, path=ENV_FILE):
    """Update keys in .env, preserving other lines. Creates it from the example if absent."""
    path = Path(path)
    if path.is_file():
        lines = path.read_text(encoding="utf-8").splitlines()
    elif ENV_EXAMPLE.is_file():
        lines = ENV_EXAMPLE.read_text(encoding="utf-8").splitlines()
    else:
        lines = []
    remaining = dict(updates)
    for index, line in enumerate(lines):
        match = _ENV_LINE.match(line)
        if match and match.group(1) in remaining:
            key = match.group(1)
            lines[index] = f"{key}={_quote(remaining.pop(key))}"
    for key, value in remaining.items():
        lines.append(f"{key}={_quote(value)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def docker_env():
    """Effective Compose variables: defaults < .env < process environment."""
    values = dict(DOCKER_ENV_DEFAULTS)
    values.update({k: v for k, v in read_env_file().items() if k != "WANDB_API_KEY"})
    for key in DOCKER_ENV_DEFAULTS:
        if os.environ.get(key):
            values[key] = os.environ[key]
    return values


def env_has_wandb_key():
    return bool(read_env_file().get("WANDB_API_KEY") or os.environ.get("WANDB_API_KEY"))


def resolve_host(value):
    """Resolve a host path the way Compose does (relative to the repository root)."""
    path = Path(os.path.expandvars(os.path.expanduser(str(value))))
    if not path.is_absolute():
        path = TOOLBOX_ROOT / path
    return path.resolve(strict=False)


def docker_mounts(env=None):
    env = env or docker_env()
    mounts = []
    for container, key, read_only in DOCKER_MOUNTS:
        host = resolve_host(env[key]) if key else DOCKER_CONFIG_DIR
        mounts.append({"container": container, "host": host, "key": key, "read_only": read_only})
    return mounts


def container_to_host(value, env=None):
    """Map a container path (e.g. /data/UBFC) to its host path, or None if not mounted."""
    if not value or not str(value).startswith("/"):
        return None
    posix = PurePosixPath(str(value))
    for mount in docker_mounts(env):
        prefix = PurePosixPath(mount["container"])
        if posix == prefix or prefix in posix.parents:
            relative = posix.relative_to(prefix)
            return mount["host"].joinpath(*relative.parts)
    return None


def host_to_container(path, env=None):
    path = Path(path).resolve(strict=False)
    best = None
    for mount in docker_mounts(env):
        try:
            relative = path.relative_to(mount["host"])
        except ValueError:
            continue
        if best is None or len(mount["host"].parts) > len(best[0].parts):
            best = (mount["host"], mount["container"], relative)
    if best is None:
        return None
    _, container, relative = best
    return str(PurePosixPath(container, *relative.parts))


def looks_like_container_path(value):
    return bool(value) and any(
        str(value) == prefix or str(value).startswith(prefix + "/")
        for prefix, _, _ in DOCKER_MOUNTS
    )


def config_host_path(value, backend, env=None):
    """Host path for a path value written in a config for the given backend.

    Returns (Path or None, problem or None).
    """
    if value in (None, ""):
        return None, None
    value = str(value)
    if backend == "docker":
        if not value.startswith("/"):
            return None, "Docker configs need absolute container paths (/data, /cache, /runs, /checkpoints)."
        host = container_to_host(value, env)
        if host is None:
            return None, "This container path is not on a mounted volume; its contents are lost when the container exits."
        return host, None
    if os.name == "nt" and looks_like_container_path(value):
        return None, "Paths under /data, /cache, /runs or /checkpoints are Docker container (or Linux) paths; " \
                     "local runs need a host path."
    drive, tail = ntpath.splitdrive(value)
    if (os.name == "nt" or drive) and ":" in tail:
        return None, "Invalid Windows path: ':' is only allowed in the leading drive prefix " \
                     "(for example C:\\Datasets_Preprocessed). Remove any duplicated drive prefix."
    return resolve_host(value), None


def is_within(path, roots):
    path = Path(path).resolve(strict=False)
    for root in roots:
        try:
            path.relative_to(Path(root).resolve(strict=False))
            return True
        except ValueError:
            continue
    return False
