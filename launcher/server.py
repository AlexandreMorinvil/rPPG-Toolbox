"""HTTP server: static UI plus a token-protected JSON API."""

import json
import math
import mimetypes
import os
import secrets
import string
import sys
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import configs, jobs, paths, preflight, preprocessed, probes, results, schema, settings, validation

MAX_BODY = 4 * 1024 * 1024


def _sanitize(value):
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


class ApiError(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


def _browse(path_text, kind):
    if not path_text:
        if os.name == "nt":
            drives = [f"{letter}:\\" for letter in string.ascii_uppercase if Path(f"{letter}:\\").exists()]
            return {"path": "", "parent": None, "entries": [{"name": d, "path": d, "dir": True} for d in drives]}
        path_text = "/"
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = paths.TOOLBOX_ROOT / path
    path = path.resolve(strict=False)
    while not path.is_dir() and path != path.parent:
        path = path.parent
    entries = []
    try:
        for entry in sorted(os.scandir(path), key=lambda e: (not e.is_dir(), e.name.lower())):
            if entry.name.startswith("."):
                continue
            is_dir = entry.is_dir()
            if not is_dir and kind == "dir":
                continue
            entries.append({"name": entry.name, "path": entry.path, "dir": is_dir})
            if len(entries) >= 1000:
                break
    except OSError as error:
        raise ApiError(str(error)) from error
    parent = str(path.parent) if path.parent != path else ("" if os.name == "nt" else None)
    return {"path": str(path), "parent": parent, "entries": entries}


def _convert_paths(values, target):
    converted = dict(values)
    notes = []
    for field in schema.FIELDS:
        if field["type"] != "path" or not values.get(field["key"]):
            continue
        value = str(values[field["key"]])
        if target == "docker":
            if value.startswith("/"):
                continue
            mapped = paths.host_to_container(paths.resolve_host(value))
            if mapped:
                converted[field["key"]] = mapped
            else:
                notes.append(f"{field['key']}: {value} is not inside a Docker mount (see Settings).")
        else:
            if not value.startswith("/"):
                continue
            host = paths.container_to_host(value)
            if host:
                converted[field["key"]] = str(host)
            else:
                notes.append(f"{field['key']}: {value} is not a mounted container path.")
    return converted, notes


class LauncherServer(ThreadingHTTPServer):
    daemon_threads = True
    # On Windows SO_REUSEADDR lets a second launcher silently share the port.
    allow_reuse_address = os.name != "nt"

    def __init__(self, address, allowed_hosts):
        super().__init__(address, Handler)
        self.token = secrets.token_urlsafe(24)
        self.allowed_hosts = allowed_hosts
        self.manager = jobs.JobManager()
        self.env_tests = {}
        self.env_lock = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    server_version = "rPPGLauncher/1.0"

    def log_message(self, fmt, *args):
        if os.environ.get("LAUNCHER_DEBUG"):
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    # ------------------------------------------------------------ helpers
    def _send(self, body, content_type, status=200, headers=None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store")
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload, status=200):
        body = json.dumps(_sanitize(payload), allow_nan=False).encode("utf-8")
        self._send(body, "application/json; charset=utf-8", status)

    def _host_ok(self):
        return self.headers.get("Host", "") in self.server.allowed_hosts

    def _token_ok(self, params):
        supplied = self.headers.get("X-Launcher-Token") or params.get("t", [""])[0]
        return secrets.compare_digest(supplied, self.server.token)

    def _body(self):
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY:
            raise ApiError("Request too large", 413)
        if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
            raise ApiError("Expected JSON", 415)
        try:
            return json.loads(self.rfile.read(length) or b"{}")
        except ValueError as error:
            raise ApiError("Invalid JSON") from error

    def _file(self, path, download=False):
        path = Path(path)
        if not path.is_file():
            raise ApiError("File not found", 404)
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        headers = {}
        if download or content_type not in ("image/png", "image/jpeg", "image/gif", "image/webp", "application/pdf",
                                             "text/plain", "text/csv", "application/json"):
            headers["Content-Disposition"] = f'attachment; filename="{path.name}"'
        if path.suffix.lower() == ".svg":
            headers["Content-Disposition"] = f'attachment; filename="{path.name}"'
        self._send(path.read_bytes(), content_type, headers=headers)

    def _dispatch(self, method):
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)
        route = parsed.path
        try:
            if not self._host_ok():
                raise ApiError("Host not allowed", 403)
            if method == "GET" and route == "/":
                html = (paths.STATIC_DIR / "index.html").read_text(encoding="utf-8")
                html = html.replace("__LAUNCHER_TOKEN__", self.server.token)
                self._send(html.encode("utf-8"), "text/html; charset=utf-8",
                           headers={"Content-Security-Policy": "default-src 'self'; img-src 'self' data: blob:; "
                                                              "style-src 'self' 'unsafe-inline'; "
                                                              "script-src 'self'; frame-ancestors 'none'",
                                    "Referrer-Policy": "no-referrer"})
                return
            if method == "GET" and route.startswith("/static/"):
                candidate = (paths.STATIC_DIR / route[len("/static/"):]).resolve()
                if not paths.is_within(candidate, [paths.STATIC_DIR]):
                    raise ApiError("Not found", 404)
                self._file(candidate)
                return
            if not route.startswith("/api/"):
                raise ApiError("Not found", 404)
            if not self._token_ok(params):
                raise ApiError("Invalid or missing launcher token; reload the page.", 403)
            body = self._body() if method == "POST" else {}
            self._api(method, route, params, body)
        except ApiError as error:
            self._json({"error": str(error)}, error.status)
        except (ValueError, FileExistsError) as error:
            status = 409 if isinstance(error, FileExistsError) else 400
            self._json({"error": str(error)}, status)
        except KeyError as error:
            self._json({"error": f"Not found: {error}"}, 404)
        except Exception as error:  # surface unexpected errors to the UI
            self._json({"error": f"{type(error).__name__}: {error}"}, 500)

    def do_GET(self):
        self._dispatch("GET")

    def do_POST(self):
        self._dispatch("POST")

    # ---------------------------------------------------------------- API
    def _api(self, method, route, params, body):
        q = lambda name, default=None: params.get(name, [default])[0]  # noqa: E731
        manager = self.server.manager

        if method == "GET" and route == "/api/bootstrap":
            env = paths.docker_env()
            self._json({
                "schema": schema.client_schema(),
                "settings": settings.load(),
                "docker_env": {k: env[k] for k in paths.EDITABLE_ENV_KEYS},
                "docker_mounts": [{**m, "host": str(m["host"]), "exists": m["host"].is_dir()}
                                  for m in paths.docker_mounts(env)],
                "env_file_exists": paths.ENV_FILE.is_file(),
                "wandb_key_in_env": paths.env_has_wandb_key(),
                "toolbox_root": str(paths.TOOLBOX_ROOT),
                "platform": os.name,
                "yacs_available": configs.toolbox_defaults() is not None,
                "configs": configs.list_configs(),
                "save_locations": {k: configs.relative(v) for k, v in paths.CONFIG_SAVE_DIRS.items()},
            })
        elif method == "GET" and route == "/api/configs":
            self._json({"configs": configs.list_configs()})
        elif method == "GET" and route == "/api/config":
            self._json(configs.load_config(q("path", "")))
        elif method == "GET" and route == "/api/defaults":
            values, _, _ = configs.normalize({})
            self._json({"values": values})
        elif method == "POST" and route == "/api/validate":
            values, errors, _ = configs.normalize(body.get("values") or {})
            report = validation.validate(values, body.get("extras") or {}, body.get("backend", "docker"), errors)
            self._json(report)
        elif method == "POST" and route == "/api/config/yaml":
            values, _, _ = configs.normalize(body.get("values") or {})
            self._json({"yaml": configs.dump_yaml(configs.build_tree(values, body.get("extras") or {}))})
        elif method == "POST" and route == "/api/config/save":
            values, errors, _ = configs.normalize(body.get("values") or {})
            if errors:
                raise ApiError("Fix invalid fields before saving.")
            saved = configs.save_config(body.get("location"), body.get("name", ""), values, body.get("extras") or {},
                                        bool(body.get("overwrite")))
            self._json({"path": saved, "configs": configs.list_configs()})
        elif method == "POST" and route == "/api/config/convert-paths":
            values, _, _ = configs.normalize(body.get("values") or {})
            converted, notes = _convert_paths(values, body.get("target"))
            self._json({"values": converted, "notes": notes})
        elif method == "POST" and route == "/api/preflight":
            values, errors, _ = configs.normalize(body.get("values") or {})
            if errors:
                raise ApiError("Fix invalid fields first.")
            result = preflight.run_checks(values, body.get("extras") or {}, body.get("backend", "docker"),
                                          body.get("answers") or {}, body.get("secrets") or {}, manager)
            result.pop("effective_values", None)
            self._json(result)
        elif method == "GET" and route == "/api/jobs":
            self._json({"jobs": manager.list(), "gpus": probes.gpus()})
        elif method == "POST" and route == "/api/jobs":
            self._json({"job": manager.create(body)})
        elif route.startswith("/api/jobs/"):
            parts = route.split("/")
            job_id = parts[3] if len(parts) > 3 else ""
            action = parts[4] if len(parts) > 4 else ""
            if method == "GET" and not action:
                self._json({"job": manager.get(job_id)})
            elif method == "GET" and action == "log":
                offset = q("offset")
                self._json(manager.log(job_id, int(offset) if offset not in (None, "") else None))
            elif method == "GET" and action == "config":
                self._send(manager.config_text(job_id).encode("utf-8"), "text/plain; charset=utf-8")
            elif method == "POST" and action == "cancel":
                self._json({"job": manager.cancel(job_id)})
            elif method == "POST" and action == "remove":
                manager.remove(job_id)
                self._json({"ok": True})
            else:
                raise ApiError("Not found", 404)
        elif method == "GET" and route == "/api/browse":
            self._json(_browse(q("path", ""), q("kind", "any")))
        elif method == "GET" and route == "/api/settings":
            env = paths.docker_env()
            self._json({"settings": settings.load(), "docker_env": {k: env[k] for k in paths.EDITABLE_ENV_KEYS},
                        "env_file_exists": paths.ENV_FILE.is_file(), "wandb_key_in_env": paths.env_has_wandb_key()})
        elif method == "POST" and route == "/api/settings":
            saved = settings.save(body.get("settings") or {})
            env_updates = body.get("docker_env") or {}
            if env_updates:
                clean = {}
                for key, value in env_updates.items():
                    if key not in paths.EDITABLE_ENV_KEYS:
                        raise ApiError(f"{key} cannot be edited here")
                    value = str(value).strip()
                    if "\n" in value or "\r" in value:
                        raise ApiError(f"{key} must be a single line")
                    if key == "GPU_DEVICE_ID" and not value.isdigit():
                        raise ApiError("GPU_DEVICE_ID must be a number")
                    clean[key] = value
                paths.update_env_file(clean)
            probes.clear_cache()
            env = paths.docker_env()
            self._json({"settings": saved, "docker_env": {k: env[k] for k in paths.EDITABLE_ENV_KEYS},
                        "docker_mounts": [{**m, "host": str(m["host"]), "exists": m["host"].is_dir()}
                                          for m in paths.docker_mounts(env)],
                        "env_file_exists": paths.ENV_FILE.is_file()})
        elif method == "POST" and route == "/api/env/test":
            target = body.get("target")
            if target == "local":
                self._json(probes.local_python(settings.load()["local_python"], refresh=True))
            elif target == "docker":
                info = probes.docker(refresh=True)
                if info.get("ok"):
                    info["image"] = probes.docker_image(paths.docker_env()["TOOLBOX_IMAGE"], refresh=True)
                self._json(info)
            elif target == "docker_gpu":
                self._json(probes.docker_gpu_check(paths.docker_env()["TOOLBOX_IMAGE"],
                                                   body.get("gpu", paths.docker_env()["GPU_DEVICE_ID"])))
            elif target == "gpus":
                self._json({"gpus": probes.gpus()})
            else:
                raise ApiError("Unknown test")
        elif method == "GET" and route == "/api/results":
            outputs = {job["id"]: jobs.output_dirs_for(job) for job in manager.list()}
            self._json({"experiments": results.scan(outputs),
                        "roots": [str(r) for r in settings.results_roots()]})
        elif method == "GET" and route == "/api/results/detail":
            self._json(results.detail(q("path", "")))
        elif method == "GET" and route == "/api/results/file":
            self._file(results.check_path(q("path", "")), download=q("download") == "1")
        elif method == "GET" and route == "/api/results/predictions":
            path = q("path", "")
            if path.lower().endswith(".csv"):
                self._json(results.window_csv(path))
            else:
                self._json(results.analyse(path, float(q("window", "10") or 0) or None,
                                           float(q("step", "0") or 0) or None, q("diff", "auto")))
        elif method == "GET" and route == "/api/results/recording":
            self._json(results.recording(q("path", ""), q("id", ""), float(q("window", "10") or 0) or None,
                                         float(q("step", "0") or 0) or None, q("diff", "auto")))
        elif method == "GET" and route == "/api/cache":
            self._json({"sets": preprocessed.scan(), "roots": [str(r) for r in settings.cache_roots()]})
        elif method == "GET" and route == "/api/cache/samples":
            self._json(preprocessed.samples(q("list", "")))
        elif method == "GET" and route == "/api/cache/sample":
            self._json(preprocessed.sample_info(q("path", "")))
        elif method == "GET" and route == "/api/cache/frames":
            data, meta = preprocessed.frames(q("path", ""), int(q("group", "0")), int(q("stride", "1")))
            self._send(data, "application/octet-stream", headers={"X-Frames-Meta": json.dumps(meta)})
        elif method == "GET" and route == "/api/cache/label":
            self._json(preprocessed.label(q("path", ""), float(q("fs", "30")), q("diff", "auto")))
        else:
            raise ApiError("Not found", 404)
