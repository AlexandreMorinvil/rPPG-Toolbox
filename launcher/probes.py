"""Environment probes (Python, Docker, GPUs) and process helpers."""

import json
import os
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path

from . import paths

NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
_cache = {}
_cache_lock = threading.Lock()

LOCAL_PROBE = r"""
import json, sys, importlib.util
info = {"python": sys.version.split()[0], "executable": sys.executable}
for name in ("torch", "yacs", "yaml", "cv2", "scipy", "wandb", "tqdm"):
    info[name] = importlib.util.find_spec(name) is not None
if info["torch"]:
    import torch
    info["torch_version"] = torch.__version__
    info["cuda"] = torch.cuda.is_available()
    info["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if info["cuda"] else []
print("@@PROBE@@" + json.dumps(info))
"""


def run(argv, timeout=30, cwd=None):
    try:
        completed = subprocess.run(argv, cwd=cwd, capture_output=True, text=True, timeout=timeout,
                                   creationflags=NO_WINDOW, stdin=subprocess.DEVNULL)
        return completed.returncode, completed.stdout, completed.stderr
    except FileNotFoundError:
        return 127, "", f"{argv[0]} not found"
    except subprocess.TimeoutExpired:
        return 124, "", f"{argv[0]} timed out after {timeout}s"
    except OSError as error:
        return 126, "", str(error)


def _cached(key, ttl, compute):
    with _cache_lock:
        entry = _cache.get(key)
        if entry and time.time() - entry[0] < ttl:
            return entry[1]
    value = compute()
    with _cache_lock:
        _cache[key] = (time.time(), value)
    return value


def clear_cache():
    with _cache_lock:
        _cache.clear()


def local_python(python, refresh=False):
    if refresh:
        _cache.pop(("local", python), None)

    def compute():
        if not Path(python).is_file():
            return {"ok": False, "error": f"Python not found: {python}"}
        code, out, err = run([python, "-c", LOCAL_PROBE], timeout=120, cwd=str(paths.TOOLBOX_ROOT))
        marker = [line for line in out.splitlines() if line.startswith("@@PROBE@@")]
        if code != 0 or not marker:
            return {"ok": False, "error": (err or out).strip()[-800:] or f"exit code {code}"}
        info = json.loads(marker[-1][len("@@PROBE@@"):])
        info["ok"] = True
        return info

    return _cached(("local", python), 300, compute)


def docker(refresh=False):
    if refresh:
        _cache.pop(("docker",), None)

    def compute():
        if not shutil.which("docker"):
            return {"ok": False, "error": "The docker command was not found."}
        code, out, err = run(["docker", "version", "--format", "{{.Server.Version}}"], timeout=20)
        if code != 0:
            return {"ok": False, "error": "Docker is not running or not reachable: " + (err or out).strip()[-400:]}
        info = {"ok": True, "server": out.strip()}
        code, out, err = run(["docker", "compose", "version", "--short"], timeout=20)
        info["compose"] = out.strip() if code == 0 else None
        if code != 0:
            info["ok"] = False
            info["error"] = "Docker Compose v2 is not available."
        return info

    return _cached(("docker",), 60, compute)


def docker_image(image, refresh=False):
    if refresh:
        _cache.pop(("image", image), None)

    def compute():
        code, out, _ = run(["docker", "image", "inspect", image, "--format", "{{.Id}} {{.Created}}"], timeout=20)
        if code != 0:
            return {"exists": False}
        parts = out.strip().split(" ", 1)
        return {"exists": True, "id": parts[0][:19], "created": parts[1] if len(parts) > 1 else ""}

    return _cached(("image", image), 30, compute)


def docker_gpu_check(image, gpu_id):
    """Run the README GPU smoke test through Compose."""
    env = dict(os.environ, GPU_DEVICE_ID=str(gpu_id))
    argv = ["docker", "compose", "--project-directory", str(paths.TOOLBOX_ROOT), "run", "--rm", "-T",
            "--entrypoint", "python", "toolbox", "-c",
            "import torch; assert torch.cuda.is_available(), 'NVIDIA GPU is not visible'; "
            "print(torch.cuda.get_device_name(0))"]
    try:
        completed = subprocess.run(argv, capture_output=True, text=True, timeout=600, env=env,
                                   creationflags=NO_WINDOW, stdin=subprocess.DEVNULL, cwd=str(paths.TOOLBOX_ROOT))
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"ok": False, "output": str(error)}
    output = (completed.stdout + completed.stderr).strip()
    return {"ok": completed.returncode == 0, "output": output[-2000:]}


def gpus():
    def compute():
        if not shutil.which("nvidia-smi"):
            return []
        code, out, _ = run(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                            "--format=csv,noheader,nounits"], timeout=10)
        if code != 0:
            return []
        result = []
        for line in out.strip().splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) == 5:
                try:
                    result.append({"index": int(parts[0]), "name": parts[1], "memory_used": int(parts[2]),
                                   "memory_total": int(parts[3]), "utilization": int(parts[4])})
                except ValueError:
                    continue
        return result

    return _cached(("gpus",), 3, compute)


def wandb_netrc():
    home = Path.home()
    for name in ("_netrc", ".netrc"):
        path = home / name
        try:
            if path.is_file() and "api.wandb.ai" in path.read_text(encoding="utf-8", errors="ignore"):
                return True
        except OSError:
            continue
    return False


def pid_alive(pid):
    if not pid:
        return False
    if os.name == "nt":
        import ctypes
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, int(pid))
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return False
            return code.value == 259
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError):
        return False


def kill_tree(pid):
    if not pid:
        return
    if os.name == "nt":
        run(["taskkill", "/PID", str(pid), "/T", "/F"], timeout=30)
    else:
        try:
            os.killpg(int(pid), signal.SIGTERM)
        except (OSError, ValueError):
            pass


def disk_free(path):
    path = Path(path)
    while not path.exists() and path != path.parent:
        path = path.parent
    try:
        return shutil.disk_usage(path).free
    except OSError:
        return None
