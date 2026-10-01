"""Reading, normalising and writing toolbox YAML configs."""

import copy
import importlib.util
import math
import re
from functools import lru_cache
from pathlib import Path

import yaml

from . import paths, schema

CONFIG_ROOTS = (paths.LOCAL_CONFIG_DIR, paths.DOCKER_CONFIG_DIR)
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._ +-]*$")


def flatten(tree, prefix=""):
    flat = {}
    for key, value in (tree or {}).items():
        dotted = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(flatten(value, dotted))
        else:
            flat[dotted] = value
    return flat


def unflatten(flat):
    tree = {}
    for key, value in flat.items():
        node = tree
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return tree


def _deep_merge(base, overlay):
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def read_yaml_tree(path, _depth=0):
    """Load a YAML config, resolving BASE inheritance like config.py."""
    if _depth > 10:
        raise ValueError("BASE inheritance is nested too deeply")
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        tree = yaml.safe_load(handle) or {}
    if not isinstance(tree, dict):
        raise ValueError(f"{path.name} is not a YAML mapping")
    merged = {}
    for base in tree.get("BASE") or [""]:
        if base:
            _deep_merge(merged, read_yaml_tree(path.parent / base, _depth + 1))
    _deep_merge(merged, {k: v for k, v in tree.items() if k != "BASE"})
    return merged


@lru_cache(maxsize=1)
def toolbox_defaults():
    """Flattened config.py defaults, or None when yacs is not installed here."""
    try:
        spec = importlib.util.spec_from_file_location("_launcher_toolbox_config", paths.TOOLBOX_ROOT / "config.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception:
        return None

    def walk(node, prefix=""):
        result = {}
        for key, value in node.items():
            dotted = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                result.update(walk(value, dotted))
            else:
                result[dotted] = list(value) if isinstance(value, tuple) else value
        return result

    return walk(module._C)


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def coerce(field, value):
    """Return (value, error) converted to the field's Python type."""
    kind = field["type"]
    try:
        if kind == "int":
            if isinstance(value, bool):
                raise ValueError
            if isinstance(value, str):
                value = float(value.strip())
            if not _is_number(value) or not float(value).is_integer():
                raise ValueError
            return int(value), None
        if kind == "float":
            if isinstance(value, bool):
                raise ValueError
            number = float(value.strip()) if isinstance(value, str) else float(value)
            if not math.isfinite(number):
                raise ValueError
            return number, None
        if kind == "bool":
            if isinstance(value, bool):
                return value, None
            if isinstance(value, str) and value.strip().lower() in ("true", "false"):
                return value.strip().lower() == "true", None
            raise ValueError
        if kind in ("str", "text", "path", "enum"):
            if value is None:
                return "", None
            if isinstance(value, (dict, list)):
                raise ValueError
            return str(value).strip() if kind != "text" else str(value), None
        if kind in ("multi", "enum_list", "list"):
            if value is None:
                return [], None
            if isinstance(value, str):
                value = [part.strip() for part in value.split(",")]
            if not isinstance(value, (list, tuple)):
                raise ValueError
            items = [str(item).strip() for item in value]
            return [item for item in items if item != ""], None
        if kind == "yamllist":
            if isinstance(value, str):
                value = yaml.safe_load(value) if value.strip() else []
            if not isinstance(value, (list, tuple)):
                raise ValueError
            return list(value), None
        if kind == "floatpair":
            if isinstance(value, str):
                value = [part for part in value.replace("(", "").replace(")", "").split(",") if part.strip()]
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise ValueError
            return [float(item) for item in value], None
    except (TypeError, ValueError, yaml.YAMLError):
        pass
    expected = {
        "int": "a whole number", "float": "a number", "bool": "true or false",
        "multi": "a list", "enum_list": "a list", "list": "a list", "yamllist": "a YAML list",
        "floatpair": "two numbers",
    }.get(kind, "text")
    return field["default"], f"Expected {expected}, got {value!r}"


def normalize(raw_flat):
    """Merge raw values over schema defaults. Returns (values, type_errors, extras)."""
    values = {key: copy.deepcopy(field["default"]) for key, field in schema.FIELD_MAP.items()}
    for key, field in schema.FIELD_MAP.items():
        if field["type"] in ("multi", "list") and values[key] == [""]:
            values[key] = []
    errors = {}
    extras = {}
    for key, value in raw_flat.items():
        if key in schema.DERIVED_KEYS:
            continue
        field = schema.FIELD_MAP.get(key)
        if field is None:
            extras[key] = value
            continue
        coerced, error = coerce(field, value)
        values[key] = coerced
        if error:
            errors[key] = error
    return values, errors, extras


def _same(a, b):
    if isinstance(a, list) and isinstance(b, list):
        return [x for x in a if x != ""] == [x for x in b if x != ""]
    return a == b


def build_tree(values, extras=None):
    """Standalone YAML tree: active fields plus non-default inactive ones."""
    flat = {"BASE": [""]}
    for field in schema.FIELDS:
        key = field["key"]
        if key not in values:
            continue
        value = values[key]
        active = schema.is_active(field, values)
        differs = not _same(value, field["default"])
        if field.get("advanced") or not active:
            if not differs:
                continue
        if key.endswith((".FILE_LIST_PATH", ".EXP_DATA_NAME")) and value == "":
            continue
        if field["type"] == "float":
            value = float(value)
        if key == "INFERENCE.MODEL_PATH":
            # The toolbox names output files with MODEL_PATH.split("/"); backslashes would leak into the name.
            value = value.replace("\\", "/")
        flat[key] = value
    for key, value in (extras or {}).items():
        flat[key] = value
    return unflatten(flat)


class _ConfigDumper(yaml.SafeDumper):
    """Block-style mappings with inline lists, like the shipped configs."""


_ConfigDumper.add_representer(
    list, lambda dumper, data: dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True))


def dump_yaml(tree):
    return yaml.dump(tree, Dumper=_ConfigDumper, sort_keys=False, default_flow_style=False, allow_unicode=True,
                     width=120)


def guess_backend(rel_path, values):
    if rel_path.replace("\\", "/").startswith("docker/"):
        return "docker"
    for key in ("LOG.PATH", "TRAIN.DATA.DATA_PATH", "TEST.DATA.DATA_PATH", "UNSUPERVISED.DATA.DATA_PATH"):
        if paths.looks_like_container_path(values.get(key)):
            return "docker"
    return "local"


def resolve_config_file(rel_path):
    """Resolve a config path inside the repository config folders or launcher job snapshots."""
    candidate = (paths.TOOLBOX_ROOT / rel_path).resolve(strict=False)
    if candidate.suffix.lower() not in (".yaml", ".yml"):
        raise ValueError("Config files must be .yaml or .yml")
    if not paths.is_within(candidate, CONFIG_ROOTS + (paths.JOBS_DIR,)):
        raise ValueError("Configs can only be loaded from configs/, docker/configs/ or launcher jobs")
    if not candidate.is_file():
        raise ValueError(f"Config not found: {rel_path}")
    return candidate


def relative(path):
    try:
        return Path(path).resolve().relative_to(paths.TOOLBOX_ROOT).as_posix()
    except ValueError:
        return str(path)


def list_configs():
    items = []
    for root in CONFIG_ROOTS:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.y*ml")):
            if path.suffix.lower() not in (".yaml", ".yml"):
                continue
            rel = relative(path)
            entry = {"path": rel, "group": "docker" if root == paths.DOCKER_CONFIG_DIR else "local"}
            try:
                flat = flatten(read_yaml_tree(path))
                entry.update({
                    "mode": flat.get("TOOLBOX_MODE", ""),
                    "model": flat.get("MODEL.NAME", ""),
                    "train": flat.get("TRAIN.DATA.DATASET", ""),
                    "test": flat.get("TEST.DATA.DATASET", "") or flat.get("UNSUPERVISED.DATA.DATASET", ""),
                })
            except Exception as error:
                entry["error"] = str(error)
            items.append(entry)
    return items


def load_config(rel_path):
    path = resolve_config_file(rel_path)
    raw = flatten(read_yaml_tree(path))
    values, errors, extras = normalize(raw)
    return {
        "path": relative(path),
        "values": values,
        "extras": extras,
        "type_errors": errors,
        "backend": guess_backend(relative(path), values),
    }


def save_config(location, name, values, extras=None, overwrite=False):
    if location not in paths.CONFIG_SAVE_DIRS:
        raise ValueError("Unknown save location")
    name = str(name).strip()
    if name.lower().endswith((".yaml", ".yml")):
        name = name.rsplit(".", 1)[0]
    parts = [part for part in name.replace("\\", "/").split("/") if part]
    if not parts or any(not SAFE_NAME.match(part) or part in (".", "..") for part in parts):
        raise ValueError("Use letters, digits, '.', '_', '-', '+' and optional sub-folders for the config name")
    target = paths.CONFIG_SAVE_DIRS[location].joinpath(*parts[:-1], parts[-1] + ".yaml")
    if target.exists() and not overwrite:
        raise FileExistsError(relative(target))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(dump_yaml(build_tree(values, extras)), encoding="utf-8")
    return relative(target)
