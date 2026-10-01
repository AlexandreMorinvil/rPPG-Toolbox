"""Static validation of launcher configs (types, ranges, cross-field rules, paths)."""

import csv
import os
import re
from pathlib import Path

from . import configs, paths, schema

RAW_LAYOUTS = {
    "UBFC-rPPG": ("subject*", "subject folders"),
    "PURE": ("*-*", "recording folders"),
    "SCAMPS": ("*.mat", ".mat files"),
    "UBFC-PHYS": ("s*/*.avi", "videos"),
    "iBVP": ("*_*", "recording folders"),
    "LADH": ("p_*", "participant folders"),
    "SUMS": ("0602*", "recording folders"),
    "vHRM": ("*/labelled_segments/*/meta.json", "labelled segments"),
}
MOTION_AUG_DATASETS = {"UBFC-rPPG", "PURE", "iBVP", "PhysDrive"}
MOTION_NAME_PATTERN = re.compile(r"^[^_]+(_[^_]+)?(_[^_]+)?_[^_]+$")


class Report:
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.info = []
        self.paths = {}
        self.derived = {}

    def error(self, message, key=None):
        self.errors.append({"key": key, "message": message})

    def warn(self, message, key=None):
        self.warnings.append({"key": key, "message": message})

    def note(self, message, key=None):
        self.info.append({"key": key, "message": message})

    def as_dict(self):
        return {
            "errors": self.errors,
            "warnings": self.warnings,
            "info": self.info,
            "paths": self.paths,
            "derived": self.derived,
            "ok": not self.errors,
        }


def exp_data_name(values, prefix):
    """Replicates config.py's automatic EXP_DATA_NAME."""
    name = values.get(prefix + ".EXP_DATA_NAME") or ""
    if name:
        return name
    p = prefix + ".PREPROCESS"
    size_h = values[p + ".RESIZE.H"] if prefix == "TEST.DATA" else values[p + ".RESIZE.W"]
    parts = [
        values.get(prefix + ".DATASET", ""),
        f"SizeW{values[p + '.RESIZE.W']}",
        f"SizeH{size_h}",
        f"ClipLength{values[p + '.CHUNK_LENGTH']}",
        "DataType" + "_".join(values[p + ".DATA_TYPE"]),
        "DataAug" + "_".join(values[p + ".DATA_AUG"]),
        f"LabelType{values[p + '.LABEL_TYPE']}",
        f"Crop_face{values[p + '.CROP_FACE.DO_CROP_FACE']}",
        f"Backend{values[p + '.CROP_FACE.BACKEND']}",
        f"Large_box{values[p + '.CROP_FACE.USE_LARGE_FACE_BOX']}",
        f"Large_size{values[p + '.CROP_FACE.LARGE_BOX_COEF']}",
        f"Dyamic_Det{values[p + '.CROP_FACE.DETECTION.DO_DYNAMIC_DETECTION']}",
        f"det_len{values[p + '.CROP_FACE.DETECTION.DYNAMIC_DETECTION_FREQUENCY']}",
        f"Median_face_box{values[p + '.CROP_FACE.DETECTION.USE_MEDIAN_FACE_BOX']}",
    ]
    if prefix == "UNSUPERVISED.DATA":
        parts.append("unsupervised")
    return "_".join(str(part) for part in parts)


def _join(base, *parts):
    """Join config paths the way the target OS would (container paths are POSIX)."""
    if str(base).startswith("/"):
        return "/".join([str(base).rstrip("/")] + [str(p).strip("/") for p in parts])
    return os.path.join(str(base), *[str(p) for p in parts])


def derived_locations(values, prefix):
    exp = exp_data_name(values, prefix)
    cached = values.get(prefix + ".CACHED_PATH") or "PreprocessedData"
    file_list = values.get(prefix + ".FILE_LIST_PATH") or ""
    if not file_list:
        fold = values.get(prefix + ".FOLD.FOLD_NAME") or ""
        name = f"{exp}_{float(values[prefix + '.BEGIN'])}_{float(values[prefix + '.END'])}"
        name += f"_{fold}" if fold else ""
        file_list = _join(cached, "DataFileLists", name + ".csv")
    elif not os.path.splitext(file_list)[1]:
        fold = values.get(prefix + ".FOLD.FOLD_NAME") or ""
        name = f"{exp}_{float(values[prefix + '.BEGIN'])}_{float(values[prefix + '.END'])}"
        name += f"_{fold}" if fold else ""
        file_list = _join(file_list, name + ".csv")
    return {"exp_data_name": exp, "cache_dir": _join(cached, exp), "file_list": file_list}


def output_locations(values):
    log_path = values.get("LOG.PATH") or "runs/exp"
    mode = values.get("TOOLBOX_MODE")
    result = {"log_path": log_path}
    if mode == "train_and_test":
        result["model_dir"] = _join(log_path, exp_data_name(values, "TRAIN.DATA"), values["MODEL.MODEL_DIR"])
    if mode in ("train_and_test", "only_test"):
        result["test_outputs"] = _join(log_path, exp_data_name(values, "TEST.DATA"), "saved_test_outputs")
    if mode == "unsupervised_method":
        result["outputs"] = _join(log_path, exp_data_name(values, "UNSUPERVISED.DATA"), "saved_outputs")
    return result


def _check_field(field, value, report):
    key = field["key"]
    kind = field["type"]
    label = field["label"]
    if field.get("required"):
        empty = value in ("", None) or (isinstance(value, list) and not value)
        if empty:
            report.error(f"{label} is required.", key)
            return
    if kind in ("int", "float"):
        if "min" in field and value < field["min"]:
            report.error(f"{label} must be at least {field['min']}.", key)
        if "max" in field and value > field["max"]:
            report.error(f"{label} must be at most {field['max']}.", key)
    if kind == "floatpair":
        for item in value:
            if not field.get("min", -1e30) <= item <= field.get("max", 1e30):
                report.error(f"{label} values must be between {field['min']} and {field['max']}.", key)
                break
    if kind == "enum" and value not in ("", None) and value not in field["options"]:
        report.error(f"{label}: '{value}' is not one of {', '.join(field['options'])}.", key)
    if kind in ("multi", "enum_list"):
        unknown = [item for item in value if item not in field["options"]]
        if unknown:
            report.error(f"{label}: unsupported value(s) {', '.join(unknown)}.", key)
        if len(set(value)) != len(value):
            report.error(f"{label} contains duplicates.", key)
        if len(value) < field.get("min_items", 0):
            report.error(f"{label} needs at least {field['min_items']} value(s).", key)
    if kind == "list" and len(value) < field.get("min_items", 0):
        report.error(f"{label} needs at least {field['min_items']} value(s).", key)
    if field.get("pattern") and kind in ("str", "enum") and value:
        if not re.match(field["pattern"], str(value)):
            report.error(f"{label}: '{value}' contains unsupported characters or format.", key)


def _check_path(field, value, backend, report, env):
    key = field["key"]
    if not value:
        return None
    spec = field.get("path", {})
    host, problem = paths.config_host_path(value, backend, env)
    entry = {"host": str(host) if host else None, "exists": None}
    report.paths[key] = entry
    if problem:
        report.error(f"{field['label']}: {problem}", key)
        return None
    if backend == "docker":
        expected = {"data": ("/data",), "cache": ("/cache",), "runs": ("/runs",),
                    "checkpoint": ("/checkpoints", "/runs")}.get(spec.get("role"))
        if expected and not any(value == p or value.startswith(p + "/") for p in expected):
            report.error(f"{field['label']} must be under {' or '.join(expected)} in Docker configs.", key)
    exists = host.is_file() if spec.get("kind") == "file" else host.is_dir()
    entry["exists"] = exists
    if spec.get("ext") and not value.lower().endswith(spec["ext"]):
        report.error(f"{field['label']} must be a {spec['ext']} file.", key)
    return host


def _first_cache_entry(file_list_host):
    try:
        with open(file_list_host, newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if "input_files" not in (reader.fieldnames or []):
                return None, "has no input_files column"
            for row in reader:
                return row["input_files"], None
            return None, "is empty"
    except OSError as error:
        return None, str(error)


def _check_split(values, prefix, backend, report, env, active_fields):
    p = prefix + ".PREPROCESS"
    dataset = values.get(prefix + ".DATASET")
    section = schema.SPLITS[prefix]["section"]
    begin, end = values[prefix + ".BEGIN"], values[prefix + ".END"]
    if begin >= end:
        report.error(f"{section}: split begin must be lower than split end.", prefix + ".END")
    if not values[p + ".DO_CHUNK"] and values.get("TOOLBOX_MODE") == "train_and_test" and prefix != "TEST.DATA":
        report.warn(f"{section}: clips are disabled; whole videos of different lengths cannot be batched.",
                    p + ".DO_CHUNK")
    if values[p + ".CROP_FACE.LARGE_BOX_COEF"] != 1.0 and not values[p + ".CROP_FACE.USE_LARGE_FACE_BOX"]:
        report.note(f"{section}: face box scale is ignored because enlarging is off.",
                    p + ".CROP_FACE.LARGE_BOX_COEF")
    if values[p + ".RESIZE.W"] != values[p + ".RESIZE.H"]:
        report.warn(f"{section}: most models expect square inputs.", p + ".RESIZE.W")
    if values[p + ".NUM_WORKERS"] > (os.cpu_count() or 4):
        report.warn(f"{section}: more preprocessing workers than CPU cores.", p + ".NUM_WORKERS")

    aug = values[p + ".DATA_AUG"]
    if len(aug) != 1:
        report.error(f"{section}: choose exactly one augmentation option.", p + ".DATA_AUG")
    elif aug == ["Motion"] and dataset not in MOTION_AUG_DATASETS:
        report.warn(f"{section}: {dataset} ignores motion augmentation.", p + ".DATA_AUG")

    if dataset == "vHRM":
        if values[p + ".LABEL_TYPE"] != "Raw":
            report.error("vHRM heart-rate labels require label transform Raw.", p + ".LABEL_TYPE")
        if values[p + ".CROP_FACE.DETECTION.USE_MEDIAN_FACE_BOX"]:
            report.error("vHRM streaming preprocessing does not support the median face box.",
                         p + ".CROP_FACE.DETECTION.USE_MEDIAN_FACE_BOX")
    if prefix == "UNSUPERVISED.DATA" and values[p + ".USE_PSUEDO_PPG_LABEL"]:
        report.error("Pseudo PPG labels are not supported for unsupervised methods.", p + ".USE_PSUEDO_PPG_LABEL")

    file_list_override = values.get(prefix + ".FILE_LIST_PATH") or ""
    if file_list_override:
        ext = os.path.splitext(file_list_override)[1]
        if ext and ext != ".csv":
            report.error("File list override must be a folder or a .csv file.", prefix + ".FILE_LIST_PATH")
        if ext == ".csv" and values[prefix + ".DO_PREPROCESS"]:
            report.error("A .csv file list override requires preprocessing to be off.", prefix + ".FILE_LIST_PATH")

    locations = derived_locations(values, prefix)
    file_host, _ = paths.config_host_path(locations["file_list"], backend, env)
    cache_host, _ = paths.config_host_path(locations["cache_dir"], backend, env)
    locations["file_list_host"] = str(file_host) if file_host else None
    locations["cache_dir_host"] = str(cache_host) if cache_host else None
    locations["file_list_exists"] = bool(file_host and file_host.is_file())
    report.derived[prefix] = locations

    raw_field = schema.FIELD_MAP[prefix + ".DATA_PATH"]
    raw_host = _check_path(raw_field, values[prefix + ".DATA_PATH"], backend, report, env)
    _check_path(schema.FIELD_MAP[prefix + ".CACHED_PATH"], values[prefix + ".CACHED_PATH"], backend, report, env)

    if values[prefix + ".DO_PREPROCESS"]:
        if raw_host is not None and not raw_host.is_dir():
            report.error(f"{section}: raw data folder not found: {raw_host}", prefix + ".DATA_PATH")
        elif raw_host is not None and dataset in RAW_LAYOUTS:
            pattern, noun = RAW_LAYOUTS[dataset]
            count = sum(1 for _ in raw_host.glob(pattern))
            report.paths[prefix + ".DATA_PATH"]["found"] = f"{count} {noun}"
            if count == 0:
                report.error(f"{section}: no {noun} matching '{pattern}' in {raw_host}. "
                             f"Is this the {dataset} root?", prefix + ".DATA_PATH")
        if locations["file_list_exists"]:
            if dataset == "vHRM" and values.get(prefix + ".VHRM.INCREMENTAL_PREPROCESS"):
                report.note(f"{section}: existing vHRM cache found; only new or changed recordings are processed.",
                            prefix + ".DO_PREPROCESS")
            else:
                report.warn(f"{section}: a preprocessed cache already exists and will be regenerated "
                            f"({locations['exp_data_name']}). Turn preprocessing off to reuse it.",
                            prefix + ".DO_PREPROCESS")
    else:
        if file_host is None:
            pass
        elif not locations["file_list_exists"]:
            report.error(f"{section}: preprocessing is off but no cache file list exists at {file_host}. "
                         "Enable preprocessing or point to an existing cache.", prefix + ".DO_PREPROCESS")
        else:
            first, problem = _first_cache_entry(file_host)
            if problem:
                report.error(f"{section}: cache file list {problem}.", prefix + ".DO_PREPROCESS")
            elif first:
                if backend == "docker":
                    mapped = paths.container_to_host(first, env) if first.startswith("/") else None
                    usable = mapped is not None and mapped.is_file()
                else:
                    usable = Path(first).is_file() if os.path.isabs(first) else (paths.TOOLBOX_ROOT / first).is_file()
                if not usable:
                    origin = "Docker" if first.startswith("/") else "another machine/location"
                    report.error(f"{section}: the cache file list refers to {first}, which is not readable for "
                                 f"a {backend} run (created by {origin}). Re-preprocess or use the matching backend.",
                                 prefix + ".DO_PREPROCESS")
    return locations


def _check_model(values, report, splits):
    model = values.get("MODEL.NAME")
    profile = schema.MODEL_PROFILES.get(model)
    if not profile:
        return
    mode = values["TOOLBOX_MODE"]
    for prefix in splits:
        p = prefix + ".PREPROCESS"
        section = schema.SPLITS[prefix]["section"]
        dataset = values.get(prefix + ".DATASET")
        if profile.get("bigsmall"):
            if dataset != "BP4DPlusBigSmall":
                report.error(f"BigSmall needs the BP4DPlusBigSmall loader ({section}).", prefix + ".DATASET")
            for branch in ("BIG_DATA_TYPE", "SMALL_DATA_TYPE"):
                if not values[f"{p}.BIGSMALL.{branch}"]:
                    report.error(f"BigSmall needs {branch.lower().replace('_', ' ')} transforms ({section}).",
                                 f"{p}.BIGSMALL.{branch}")
            continue
        if not values[p + ".DATA_TYPE"]:
            report.error(f"Choose at least one input transform ({section}).", p + ".DATA_TYPE")
            continue
        if dataset == "BP4DPlusBigSmall":
            report.error(f"The BP4DPlusBigSmall loader is only for the BigSmall model ({section}).",
                         prefix + ".DATASET")
        if values[prefix + ".DATA_FORMAT"] != profile["format"]:
            report.error(f"{model} expects tensor layout {profile['format']} ({section}).", prefix + ".DATA_FORMAT")
        types = values[p + ".DATA_TYPE"]
        channels = 3 * len(types)
        if profile.get("strict_types") and types != profile["types"]:
            report.error(f"{model} needs input transforms {profile['types']} in this order "
                         f"(motion branch, then appearance branch) ({section}).", p + ".DATA_TYPE")
        elif "channels" in profile and channels != profile["channels"]:
            report.error(f"{model} takes {profile['channels']} input channels: choose exactly one input "
                         f"transform ({section}).", p + ".DATA_TYPE")
        elif "channels_key" in profile and channels != values[profile["channels_key"]]:
            report.error(f"{model}: {len(types)} transform(s) give {channels} channels, but the model is "
                         f"configured for {values[profile['channels_key']]} ({section}).", p + ".DATA_TYPE")
        elif types and types != profile["types"] and not profile.get("strict_types"):
            report.warn(f"{model} is usually trained with input transform {profile['types']} ({section}).",
                        p + ".DATA_TYPE")
        if dataset != "vHRM" and values[p + ".LABEL_TYPE"] and values[p + ".LABEL_TYPE"] != profile["label"]:
            report.warn(f"{model} is usually trained with label transform {profile['label']} ({section}).",
                        p + ".LABEL_TYPE")
        chunk = values[p + ".CHUNK_LENGTH"]
        if profile.get("frame_num") and values[p + ".DO_CHUNK"] and chunk != values[profile["frame_num"]]:
            report.error(f"{model}: clip length ({chunk}) must equal the model frame count "
                         f"({values[profile['frame_num']]}) ({section}).", p + ".CHUNK_LENGTH")
        if profile.get("frame_depth"):
            depth = values[profile["frame_depth"]]
            base = depth * max(1, values["NUM_OF_GPU_TRAIN"])
            if chunk % depth:
                report.error(f"{model}: clip length ({chunk}) must be a multiple of the frame depth ({depth}) "
                             f"({section}).", p + ".CHUNK_LENGTH")
            elif chunk % base:
                report.warn(f"{model}: clip length ({chunk}) is not a multiple of frame depth x GPUs ({base}); "
                            f"frames are dropped per batch ({section}).", p + ".CHUNK_LENGTH")
        size = values[p + ".RESIZE.H"]
        expected = profile.get("size")
        if model == "FactorizePhys" and values["MODEL.FactorizePhys.TYPE"].lower() == "big":
            expected = 144
        if expected and size != expected:
            report.warn(f"{model} is normally used with {expected}x{expected} inputs ({section}).",
                        p + ".RESIZE.H")
        if model == "PhysFormer":
            patch = values["MODEL.PHYSFORMER.PATCH_SIZE"]
            if size % patch or chunk % patch:
                report.error(f"PhysFormer: clip length and size must be multiples of the patch size {patch} "
                             f"({section}).", p + ".CHUNK_LENGTH")

    if model == "PhysFormer" and values["MODEL.PHYSFORMER.DIM"] % values["MODEL.PHYSFORMER.NUM_HEADS"]:
        report.error("PhysFormer: embedding dim must be divisible by the number of heads.", "MODEL.PHYSFORMER.DIM")
    if model == "PhysFormer" and mode == "only_test":
        for axis in ("H", "W"):
            if values[f"TRAIN.DATA.PREPROCESS.RESIZE.{axis}"] != values[f"TEST.DATA.PREPROCESS.RESIZE.{axis}"]:
                report.error("PhysFormer evaluation builds the network from TRAIN.DATA.PREPROCESS.RESIZE; it must "
                             "equal the test size. Load a config with matching sizes or edit the YAML.",
                             "TEST.DATA.PREPROCESS.RESIZE.H")
                break


def _check_split_consistency(values, report, splits):
    if values["TOOLBOX_MODE"] != "train_and_test":
        return
    train = "TRAIN.DATA"
    model = values.get("MODEL.NAME")
    profile = schema.MODEL_PROFILES.get(model, {})
    for other in [s for s in splits if s != train]:
        section = schema.SPLITS[other]["section"]
        for suffix, severity in (("DATA_FORMAT", "error"), ("PREPROCESS.DATA_TYPE", "error"),
                                 ("PREPROCESS.RESIZE.H", "error"), ("PREPROCESS.RESIZE.W", "error"),
                                 ("PREPROCESS.LABEL_TYPE", "warn"), ("FS", "warn"),
                                 ("PREPROCESS.CHUNK_LENGTH", "error" if profile.get("frame_num") else "warn")):
            if other == "TEST.DATA" and values.get("TEST.DATA.DATASET") == "vHRM" and suffix in (
                    "PREPROCESS.LABEL_TYPE", "PREPROCESS.CHUNK_LENGTH"):
                continue
            if values[f"{train}.{suffix}"] != values[f"{other}.{suffix}"]:
                label = schema.FIELD_MAP[f"{other}.{suffix}"]["label"]
                message = f"{section}: {label} differs from training data ({values[f'{other}.{suffix}']} vs " \
                          f"{values[f'{train}.{suffix}']})."
                (report.error if severity == "error" else report.warn)(message, f"{other}.{suffix}")
        same_source = values[f"{train}.DATASET"] == values[f"{other}.DATASET"] and \
            values[f"{train}.DATA_PATH"] == values[f"{other}.DATA_PATH"]
        if same_source:
            if max(values[f"{train}.BEGIN"], values[f"{other}.BEGIN"]) < min(values[f"{train}.END"],
                                                                           values[f"{other}.END"]):
                report.warn(f"{section} overlaps the training split of the same dataset (data leakage).",
                            f"{other}.BEGIN")
    if values.get("TEST.DATA.DATASET") == "vHRM":
        diff = values["TRAIN.DATA.PREPROCESS.LABEL_TYPE"] == "DiffNormalized"
        if diff != values["TEST.DATA.VHRM.PREDICTION_IS_DIFF"]:
            report.warn("vHRM: 'model predicts a differenced signal' should match the training label transform "
                        f"({values['TRAIN.DATA.PREPROCESS.LABEL_TYPE']}).", "TEST.DATA.VHRM.PREDICTION_IS_DIFF")


def validate(values, extras=None, backend="docker", type_errors=None, platform=None):
    report = Report()
    env = paths.docker_env()
    platform = platform or os.name
    if backend not in ("docker", "local"):
        report.error("Backend must be docker or local.")
        return report.as_dict()

    for key, message in (type_errors or {}).items():
        report.error(message, key)

    known = configs.toolbox_defaults()
    for key in extras or {}:
        if known is not None and key not in known:
            report.error(f"Unknown config key {key}: config.py would reject this file.")
        else:
            report.note(f"{key} is kept as-is (not editable here).")

    active = [f for f in schema.FIELDS if schema.is_active(f, values)]
    for field in active:
        _check_field(field, values.get(field["key"]), report)

    mode = values.get("TOOLBOX_MODE")
    if mode not in schema.MODES:
        return report.as_dict()

    splits = schema.active_splits(values)
    if mode == "train_and_test" and not values["TEST.USE_LAST_EPOCH"]:
        report.note("The best epoch is selected with the validation split.", "TEST.USE_LAST_EPOCH")
    for prefix in splits:
        _check_split(values, prefix, backend, report, env, active)

    model = values.get("MODEL.NAME")
    if mode in schema.NEURAL_MODES:
        _check_model(values, report, splits)
        _check_split_consistency(values, report, splits)
        if model == "PhysMamba":
            if backend == "docker":
                report.error("PhysMamba needs mamba-ssm/causal-conv1d, which the Docker image does not include.",
                             "MODEL.NAME")
            elif platform == "nt":
                report.error("PhysMamba needs mamba-ssm/causal-conv1d, which are unavailable on Windows.",
                             "MODEL.NAME")
        if not values.get("TEST.METRICS"):
            report.warn("No test metrics selected; the test phase will not report results.", "TEST.METRICS")
    if mode == "train_and_test":
        aug_used = any(values[f"{s}.PREPROCESS.DATA_AUG"] != ["None"] for s in ("TRAIN.DATA", "VALID.DATA",
                                                                                  "TEST.DATA"))
        if aug_used and not MOTION_NAME_PATTERN.match(values["TRAIN.MODEL_FILE_NAME"] or ""):
            report.error("Motion augmentation requires a checkpoint name like TRAIN_VALID_TEST_MODEL "
                         "(config.py rewrites it).", "TRAIN.MODEL_FILE_NAME")
    if mode == "only_test":
        host = _check_path(schema.FIELD_MAP["INFERENCE.MODEL_PATH"], values["INFERENCE.MODEL_PATH"], backend,
                           report, env)
        if host is not None and not host.is_file():
            report.error(f"Checkpoint not found: {host}", "INFERENCE.MODEL_PATH")
        report.note("Make sure preprocessing (size, transforms, clip length) matches how the checkpoint was trained.")
    if mode == "unsupervised_method" and values.get("UNSUPERVISED.DATA.DATASET") == "vHRM":
        report.error("vHRM is only supported for neural evaluation.", "UNSUPERVISED.DATA.DATASET")
    if mode == "unsupervised_method":
        types = values["UNSUPERVISED.DATA.PREPROCESS.DATA_TYPE"]
        if not types:
            report.error("Choose an input transform (Raw for unsupervised methods).",
                         "UNSUPERVISED.DATA.PREPROCESS.DATA_TYPE")
        elif types != ["Raw"]:
            report.warn("Unsupervised methods expect raw RGB input (transform Raw).",
                        "UNSUPERVISED.DATA.PREPROCESS.DATA_TYPE")

    window = values["INFERENCE.EVALUATION_WINDOW.WINDOW_SIZE"]
    fs_key = "UNSUPERVISED.DATA.FS" if mode == "unsupervised_method" else "TEST.DATA.FS"
    if values["INFERENCE.EVALUATION_WINDOW.USE_SMALLER_WINDOW"] and window * max(values.get(fs_key) or 0, 1) < 9:
        report.error("Evaluation windows must contain at least 9 frames.", "INFERENCE.EVALUATION_WINDOW.WINDOW_SIZE")

    device = values.get("DEVICE", "")
    if backend == "docker":
        if device.startswith("cuda:") and device != "cuda:0":
            report.error("In Docker the selected GPU is always cuda:0; choose the host GPU in Settings or at launch.",
                         "DEVICE")
        if device == "cpu":
            report.warn("Docker Compose still requires an NVIDIA GPU even when the config uses the CPU.", "DEVICE")
    elif device == "cpu" and mode == "train_and_test":
        report.warn("Training on the CPU is very slow.", "DEVICE")

    log_field = schema.FIELD_MAP["LOG.PATH"]
    _check_path(log_field, values.get("LOG.PATH"), backend, report, env)
    outputs = output_locations(values)
    for key, value in list(outputs.items()):
        host, _ = paths.config_host_path(value, backend, env)
        outputs[key + "_host"] = str(host) if host else None
    report.derived["outputs"] = outputs
    existing = [outputs.get(k + "_host") for k in ("model_dir", "test_outputs", "outputs")]
    existing = [p for p in existing if p and Path(p).is_dir() and any(Path(p).iterdir())]
    if existing:
        report.warn("The output folder already contains results that may be overwritten: " + ", ".join(existing),
                    "LOG.PATH")

    if values.get("WANDB.ENABLED"):
        if values["WANDB.MODE"] == "online":
            report.note("W&B online mode needs an API key; it is requested before launch if none is found.",
                        "WANDB.MODE")
        elif values["WANDB.MODE"] == "offline":
            report.note("Offline W&B runs are stored with the outputs and can be uploaded later with 'wandb sync'.",
                        "WANDB.MODE")
    return report.as_dict()
