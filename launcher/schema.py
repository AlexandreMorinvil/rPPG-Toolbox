"""Editable rPPG-Toolbox config fields, mirroring config.py defaults."""

MODES = ["train_and_test", "only_test", "unsupervised_method"]
MODE_LABELS = {
    "train_and_test": "Training (train + test)",
    "only_test": "Evaluation of a checkpoint",
    "unsupervised_method": "Unsupervised methods",
}
MODELS = [
    "DeepPhys", "Tscan", "EfficientPhys", "Physnet", "PhysFormer",
    "FactorizePhys", "iBVPNet", "RhythmFormer", "BigSmall", "PhysMamba",
]
TRAIN_DATASETS = [
    "UBFC-rPPG", "PURE", "SCAMPS", "MMPD", "BP4DPlus", "BP4DPlusBigSmall",
    "UBFC-PHYS", "iBVP", "PhysDrive", "LADH", "SUMS",
]
TEST_DATASETS = TRAIN_DATASETS + ["vHRM"]
UNSUPERVISED_DATASETS = ["UBFC-rPPG", "PURE", "SCAMPS", "MMPD", "BP4DPlus", "UBFC-PHYS", "iBVP"]
DATA_TYPES = ["DiffNormalized", "Standardized", "Raw"]
LABEL_TYPES = ["DiffNormalized", "Standardized", "Raw"]
DATA_FORMATS = ["NDCHW", "NCDHW", "NDHWC"]
FACE_BACKENDS = ["HC", "Y5F"]
METRICS = ["MAE", "RMSE", "MAPE", "Pearson", "SNR", "MACC", "BA", "AU_METRICS"]
UNSUPERVISED_METHODS = ["POS", "CHROM", "ICA", "GREEN", "LGI", "PBV", "OMIT"]

# Recommended preprocessing per model, taken from the shipped configs.
MODEL_PROFILES = {
    "DeepPhys": {"format": "NDCHW", "types": ["DiffNormalized", "Standardized"], "label": "DiffNormalized",
                 "chunk": 180, "size": 72, "strict_types": True},
    "Tscan": {"format": "NDCHW", "types": ["DiffNormalized", "Standardized"], "label": "DiffNormalized",
              "chunk": 180, "size": 72, "strict_types": True, "frame_depth": "MODEL.TSCAN.FRAME_DEPTH"},
    "EfficientPhys": {"format": "NDCHW", "types": ["Standardized"], "label": "DiffNormalized",
                      "chunk": 180, "size": 72, "channels": 3, "frame_depth": "MODEL.EFFICIENTPHYS.FRAME_DEPTH"},
    "Physnet": {"format": "NCDHW", "types": ["DiffNormalized"], "label": "DiffNormalized",
                "chunk": 128, "size": 72, "channels": 3, "frame_num": "MODEL.PHYSNET.FRAME_NUM"},
    "PhysFormer": {"format": "NCDHW", "types": ["DiffNormalized"], "label": "DiffNormalized",
                   "chunk": 160, "size": 128, "channels": 3},
    "FactorizePhys": {"format": "NCDHW", "types": ["Raw"], "label": "Standardized", "chunk": 160, "size": 72,
                      "frame_num": "MODEL.FactorizePhys.FRAME_NUM", "channels_key": "MODEL.FactorizePhys.CHANNELS"},
    "iBVPNet": {"format": "NCDHW", "types": ["Raw"], "label": "Standardized", "chunk": 160, "size": 72,
                "frame_num": "MODEL.iBVPNet.FRAME_NUM", "channels_key": "MODEL.iBVPNet.CHANNELS"},
    "RhythmFormer": {"format": "NDCHW", "types": ["Standardized"], "label": "Standardized",
                     "chunk": 160, "size": 128, "channels": 3},
    "PhysMamba": {"format": "NCDHW", "types": ["DiffNormalized"], "label": "DiffNormalized",
                  "chunk": 128, "size": 128, "channels": 3},
    "BigSmall": {"format": "NDCHW", "label": "DiffNormalized", "chunk": 3, "bigsmall": True},
}

SPLITS = {
    "TRAIN.DATA": {"section": "Training data", "modes": ["train_and_test"], "datasets": TRAIN_DATASETS},
    "VALID.DATA": {"section": "Validation data", "modes": ["train_and_test"], "datasets": TRAIN_DATASETS,
                   "needs_valid": True},
    "TEST.DATA": {"section": "Test data", "modes": ["train_and_test", "only_test"], "datasets": TEST_DATASETS},
    "UNSUPERVISED.DATA": {"section": "Unsupervised data", "modes": ["unsupervised_method"],
                          "datasets": UNSUPERVISED_DATASETS},
}
NEURAL_MODES = ["train_and_test", "only_test"]

FIELDS = []


def _add(key, type_, default, label, section, help_text="", **extra):
    field = {"key": key, "type": type_, "default": default, "label": label, "section": section,
             "help": help_text}
    field.update(extra)
    FIELDS.append(field)


def _experiment_fields():
    s = "Experiment"
    _add("TOOLBOX_MODE", "enum", "", "Run type", s, "What main.py should do.",
         options=MODES, option_labels=MODE_LABELS, required=True)
    _add("MODEL.NAME", "enum", "", "Model", s, "Neural architecture to train or evaluate.",
         options=MODELS, required=True, when={"modes": NEURAL_MODES})
    _add("LOG.PATH", "path", "runs/exp", "Output folder (LOG.PATH)", s,
         "Checkpoints, plots and test outputs are written below this folder. Use a distinct folder per experiment.",
         path={"kind": "dir", "role": "runs", "must_exist": False}, required=True)
    _add("DEVICE", "str", "cuda:0", "Device", s,
         "cuda:N or cpu. In Docker the selected GPU always appears as cuda:0.", pattern=r"^(cpu|cuda:\d+)$")
    _add("NUM_OF_GPU_TRAIN", "int", 1, "Number of GPUs", s, "The toolbox effectively uses one GPU.",
         min=0, max=16, when={"modes": NEURAL_MODES}, advanced=True)


def _training_fields():
    s = "Training"
    when = {"modes": ["train_and_test"]}
    _add("TRAIN.EPOCHS", "int", 50, "Epochs", s, min=1, max=10000, when=when)
    _add("TRAIN.BATCH_SIZE", "int", 4, "Batch size", s, "Also used for validation.", min=1, max=4096, when=when)
    _add("TRAIN.LR", "float", 1e-4, "Learning rate", s, "Peak learning rate (OneCycle for most trainers).",
         min=1e-8, max=1.0, when=when)
    _add("TRAIN.MODEL_FILE_NAME", "str", "", "Checkpoint name prefix", s,
         "Checkpoints are saved as <prefix>_Epoch<N>.pth.", pattern=r"^[A-Za-z0-9][A-Za-z0-9._+-]*$",
         required=True, when=when)
    _add("TRAIN.PLOT_LOSSES_AND_LR", "bool", True, "Save loss/LR plots", s, when=when)
    _add("TRAIN.OPTIMIZER.EPS", "float", 1e-4, "Optimizer epsilon", s, "Used by some trainers only.",
         min=0.0, max=1.0, when=when, advanced=True)
    _add("TRAIN.OPTIMIZER.BETAS", "floatpair", [0.9, 0.999], "Optimizer betas", s, "Used by some trainers only.",
         min=0.0, max=0.99999, when=when, advanced=True)
    _add("TRAIN.OPTIMIZER.MOMENTUM", "float", 0.9, "SGD momentum", s, "Used by some trainers only.",
         min=0.0, max=1.0, when=when, advanced=True)


def _split_fields(prefix, spec):
    s = spec["section"]
    when = {"modes": spec["modes"]}
    if spec.get("needs_valid"):
        when["needs_valid"] = True

    def add(suffix, type_, default, label, help_text="", group="Source", **extra):
        extra_when = extra.pop("when_extra", {})
        _add(f"{prefix}.{suffix}", type_, default, label, s, help_text, group=group,
             when={**when, **extra_when}, **extra)

    add("DATASET", "enum", "", "Dataset", "Selects the toolbox data loader.", options=spec["datasets"],
        required=True)
    add("DATA_PATH", "path", "", "Raw data folder", "Root of the raw recordings for this dataset.",
        path={"kind": "dir", "role": "data", "must_exist": True}, required=True)
    add("CACHED_PATH", "path", "PreprocessedData", "Preprocessed cache folder",
        "Preprocessed clips are stored in <cache folder>/<cache name>.",
        path={"kind": "dir", "role": "cache", "must_exist": False}, required=True)
    add("EXP_DATA_NAME", "str", "", "Cache name (EXP_DATA_NAME)",
        "Leave empty to derive a name from the preprocessing settings. Use distinct names when preprocessing differs.",
        pattern=r"^[A-Za-z0-9._+-]*$")
    add("DO_PREPROCESS", "bool", False, "Preprocess raw data",
        "Off reuses an existing cache (its file list must exist).")
    add("FS", "int", 0, "Frame rate (Hz)", "Sampling rate of the video/labels after preprocessing.",
        min=1, max=1000, required=True)
    add("BEGIN", "float", 0.0, "Split begin", "Fraction of the subjects where this split starts.", min=0.0, max=1.0)
    add("END", "float", 1.0, "Split end", "Fraction of the subjects where this split ends.", min=0.0, max=1.0)
    add("DATA_FORMAT", "enum", "NDCHW", "Tensor layout", "Must match the model.", options=DATA_FORMATS,
        group="Preprocessing")
    add("PREPROCESS.DATA_TYPE", "multi", [""], "Input transforms",
        "Each transform contributes 3 channels, in the selected order.", options=DATA_TYPES,
        group="Preprocessing")
    add("PREPROCESS.LABEL_TYPE", "enum", "", "Label transform", options=LABEL_TYPES, group="Preprocessing",
        required=True)
    add("PREPROCESS.DATA_AUG", "enum_list", ["None"], "Augmentation",
        "Motion augmentation needs pre-augmented data and a [TRAIN]_[VALID]_[TEST] checkpoint name.",
        options=["None", "Motion"], group="Preprocessing")
    add("PREPROCESS.DO_CHUNK", "bool", True, "Split into clips", group="Preprocessing")
    add("PREPROCESS.CHUNK_LENGTH", "int", 180, "Clip length (frames)", min=1, max=100000, group="Preprocessing")
    add("PREPROCESS.RESIZE.W", "int", 128, "Width (px)", min=4, max=2048, group="Preprocessing")
    add("PREPROCESS.RESIZE.H", "int", 128, "Height (px)", min=4, max=2048, group="Preprocessing")
    add("PREPROCESS.NUM_WORKERS", "int", 4, "Preprocessing workers",
        "Each worker loads a full video and its own face detector. Lower it if RAM/VRAM runs out.",
        min=1, max=64, group="Preprocessing")
    add("PREPROCESS.USE_PSUEDO_PPG_LABEL", "bool", False, "Use POS pseudo-PPG labels",
        group="Preprocessing", advanced=True)
    add("PREPROCESS.CROP_FACE.DO_CROP_FACE", "bool", True, "Crop face", group="Face crop")
    add("PREPROCESS.CROP_FACE.BACKEND", "enum", "HC", "Face detector",
        "HC: Haar cascade (CPU). Y5F: YOLO5Face (GPU, more robust).", options=FACE_BACKENDS,
        option_labels={"HC": "HC - Haar cascade", "Y5F": "Y5F - YOLO5Face"}, group="Face crop")
    add("PREPROCESS.CROP_FACE.USE_LARGE_FACE_BOX", "bool", True, "Enlarge face box", group="Face crop")
    add("PREPROCESS.CROP_FACE.LARGE_BOX_COEF", "float", 1.5, "Face box scale", min=1.0, max=5.0,
        group="Face crop")
    add("PREPROCESS.CROP_FACE.DETECTION.DO_DYNAMIC_DETECTION", "bool", False, "Re-detect periodically",
        group="Face crop")
    add("PREPROCESS.CROP_FACE.DETECTION.DYNAMIC_DETECTION_FREQUENCY", "int", 30, "Re-detection interval (frames)",
        min=1, max=100000, group="Face crop")
    add("PREPROCESS.CROP_FACE.DETECTION.USE_MEDIAN_FACE_BOX", "bool", False, "Use median face box",
        group="Face crop")
    add("PREPROCESS.IBVP.DATA_MODE", "enum", "RGB", "iBVP modality", options=["RGB", "T", "RGBT"],
        group="Preprocessing", when_extra={"dataset": [prefix, ["iBVP"]]})
    if prefix != "UNSUPERVISED.DATA":
        bigsmall = {"models": ["BigSmall"]}
        add("PREPROCESS.BIGSMALL.BIG_DATA_TYPE", "multi", [""], "BigSmall big-branch transforms",
            options=DATA_TYPES, group="BigSmall", when_extra=bigsmall)
        add("PREPROCESS.BIGSMALL.SMALL_DATA_TYPE", "multi", [""], "BigSmall small-branch transforms",
            options=DATA_TYPES, group="BigSmall", when_extra=bigsmall)
        add("PREPROCESS.BIGSMALL.RESIZE.BIG_W", "int", 144, "Big width", min=4, max=2048, group="BigSmall",
            when_extra=bigsmall)
        add("PREPROCESS.BIGSMALL.RESIZE.BIG_H", "int", 144, "Big height", min=4, max=2048, group="BigSmall",
            when_extra=bigsmall)
        add("PREPROCESS.BIGSMALL.RESIZE.SMALL_W", "int", 9, "Small width", min=1, max=2048, group="BigSmall",
            when_extra=bigsmall)
        add("PREPROCESS.BIGSMALL.RESIZE.SMALL_H", "int", 9, "Small height", min=1, max=2048, group="BigSmall",
            when_extra=bigsmall)
    if prefix == "TEST.DATA":
        vhrm = {"dataset": [prefix, ["vHRM"]]}
        add("VHRM.VIEWS", "list", ["front"], "Camera views", "Camera labels without the Camera_ prefix, e.g. Front.",
            group="vHRM", when_extra=vhrm, min_items=1)
        add("VHRM.SIGNAL_COLUMN", "enum", "heart_rate_bpm", "Ground-truth column", options=["heart_rate_bpm"],
            group="vHRM", when_extra=vhrm)
        add("VHRM.PREDICTION_IS_DIFF", "bool", True, "Model predicts a differenced signal",
            "True for DiffNormalized-trained models (DeepPhys, Tscan, PhysNet...).", group="vHRM",
            when_extra=vhrm)
        add("VHRM.VIDEO_DECODER", "enum", "auto", "Video decoder", options=["auto", "nvdec", "opencv"],
            group="vHRM", when_extra=vhrm)
        add("VHRM.FFMPEG_PATH", "str", "ffmpeg", "ffmpeg executable", group="vHRM", when_extra=vhrm,
            advanced=True)
        add("VHRM.FFPROBE_PATH", "str", "ffprobe", "ffprobe executable", group="vHRM", when_extra=vhrm,
            advanced=True)
        add("VHRM.INCREMENTAL_PREPROCESS", "bool", True, "Incremental preprocessing",
            "Only new or changed recordings are preprocessed.", group="vHRM", when_extra=vhrm)
        add("VHRM.ADOPT_LEGACY_CACHE", "bool", True, "Adopt legacy cache", group="vHRM", when_extra=vhrm,
            advanced=True)
        add("VHRM.MANIFEST_FILENAME", "str", "preprocessing_manifest.json", "Manifest file name", group="vHRM",
            when_extra=vhrm, advanced=True, pattern=r"^[A-Za-z0-9._-]+\.json$")
        add("VHRM.CACHE_DTYPE", "enum", "float16", "Cache precision", options=["float16", "float32"],
            group="vHRM", when_extra=vhrm)
        add("VHRM.SHOW_VIDEO_PROGRESS", "bool", True, "Show per-video progress", group="vHRM",
            when_extra=vhrm)
    add("FILE_LIST_PATH", "str", "", "File list override",
        "Leave empty to use <cache folder>/DataFileLists. A .csv file requires preprocessing off.",
        group="Advanced", advanced=True)
    add("FOLD.FOLD_NAME", "str", "", "Fold name", group="Advanced", advanced=True)
    add("FOLD.FOLD_PATH", "str", "", "Fold file", group="Advanced", advanced=True)
    add("FILTERING.USE_EXCLUSION_LIST", "bool", False, "Use exclusion list", group="Advanced", advanced=True)
    add("FILTERING.EXCLUSION_LIST", "list", [""], "Excluded subjects", group="Advanced", advanced=True)
    add("FILTERING.SELECT_TASKS", "bool", False, "Select tasks", group="Advanced", advanced=True)
    add("FILTERING.TASK_LIST", "list", [""], "Tasks", group="Advanced", advanced=True)
    for name, default in (("LIGHT", [""]), ("MOTION", [""]), ("EXERCISE", [True]), ("SKIN_COLOR", [1]),
                          ("GENDER", [""]), ("GLASSER", [True]), ("HAIR_COVER", [True]), ("MAKEUP", [True])):
        add(f"INFO.{name}", "yamllist", default, f"MMPD filter {name.lower()}", "YAML list, e.g. [1, 2].",
            group="Advanced", advanced=True)


def _evaluation_fields():
    s = "Evaluation"
    _add("TEST.METRICS", "multi", [], "Test metrics", s, options=METRICS, when={"modes": NEURAL_MODES})
    _add("TEST.USE_LAST_EPOCH", "bool", True, "Test the last epoch",
         s, "Off: select the epoch with the lowest validation loss (requires validation data).",
         when={"modes": ["train_and_test"]})
    _add("INFERENCE.MODEL_PATH", "path", "", "Checkpoint to evaluate (.pth)", s,
         path={"kind": "file", "role": "checkpoint", "must_exist": True, "ext": ".pth"},
         required=True, when={"modes": ["only_test"]})
    _add("INFERENCE.BATCH_SIZE", "int", 4, "Test batch size", s, min=1, max=4096, when={"modes": NEURAL_MODES})
    _add("INFERENCE.DATA_LOADER_WORKERS", "int", 0, "Test loader workers", s, min=0, max=64,
         when={"modes": NEURAL_MODES})
    _add("INFERENCE.FRAME_BATCH_SIZE", "int", 0, "Frames per forward pass", s,
         "0 processes a whole batch at once. Lower values save VRAM (DeepPhys).", min=0, max=1000000,
         when={"modes": NEURAL_MODES}, advanced=True)
    _add("INFERENCE.EVALUATION_METHOD", "enum", "FFT", "Heart-rate estimation", s,
         options=["FFT", "peak detection"])
    _add("INFERENCE.EVALUATION_WINDOW.USE_SMALLER_WINDOW", "bool", False, "Evaluate in windows", s,
         "Off evaluates each whole recording.")
    _add("INFERENCE.EVALUATION_WINDOW.WINDOW_SIZE", "int", 10, "Window length (s)", s, min=1, max=36000)
    _add("UNSUPERVISED.METHOD", "multi", [], "Unsupervised methods", "Unsupervised",
         options=UNSUPERVISED_METHODS, min_items=1, when={"modes": ["unsupervised_method"]})
    _add("UNSUPERVISED.METRICS", "multi", [], "Metrics", "Unsupervised", options=METRICS,
         when={"modes": ["unsupervised_method"]})


def _model_fields():
    s = "Model"
    neural = {"modes": NEURAL_MODES}

    def model_when(name):
        return {"modes": NEURAL_MODES, "models": [name]}

    _add("MODEL.DROP_RATE", "float", 0.0, "Dropout", s, min=0.0, max=0.95, when=neural)
    _add("MODEL.MODEL_DIR", "str", "PreTrainedModels", "Checkpoint subfolder", s,
         "Relative to <output folder>/<training cache name>.", pattern=r"^[A-Za-z0-9._-]+$", when=neural,
         advanced=True)
    _add("MODEL.RESUME", "str", "", "Resume checkpoint", s, "Not used by the current trainers.", when=neural,
         advanced=True)
    _add("MODEL.PHYSNET.FRAME_NUM", "int", 64, "PhysNet frames", s, "Must equal the clip length.",
         min=1, max=100000, when=model_when("Physnet"))
    _add("MODEL.iBVPNet.FRAME_NUM", "int", 160, "iBVPNet frames", s, "Must equal the clip length.",
         min=1, max=100000, when=model_when("iBVPNet"))
    _add("MODEL.iBVPNet.CHANNELS", "int", 3, "iBVPNet input channels", s, min=1, max=12,
         when=model_when("iBVPNet"))
    fp = model_when("FactorizePhys")
    _add("MODEL.FactorizePhys.FRAME_NUM", "int", 160, "Frames", s, "Must equal the clip length.",
         min=1, max=100000, when=fp)
    _add("MODEL.FactorizePhys.CHANNELS", "int", 3, "Input channels", s, min=1, max=12, when=fp)
    _add("MODEL.FactorizePhys.TYPE", "enum", "Standard", "Variant", s, "Standard: 72x72 input. Big: 144x144.",
         options=["Standard", "Big"], when=fp)
    _add("MODEL.FactorizePhys.MD_FSAM", "bool", False, "Use FSAM", s, when=fp)
    _add("MODEL.FactorizePhys.MD_TYPE", "enum", "NMF", "Decomposition", s, options=["NMF", "VQ"], when=fp)
    _add("MODEL.FactorizePhys.MD_TRANSFORM", "enum", "T_KAB", "Transform", s,
         options=["T_KAB", "TK_AB", "K_TAB"], when=fp, advanced=True)
    _add("MODEL.FactorizePhys.MD_R", "int", 1, "Rank (R)", s, min=1, max=64, when=fp, advanced=True)
    _add("MODEL.FactorizePhys.MD_S", "int", 1, "S", s, min=1, max=64, when=fp, advanced=True)
    _add("MODEL.FactorizePhys.MD_STEPS", "int", 4, "Steps", s, min=1, max=100, when=fp, advanced=True)
    _add("MODEL.FactorizePhys.MD_INFERENCE", "bool", True, "Use FSAM at inference", s, when=fp, advanced=True)
    _add("MODEL.FactorizePhys.MD_RESIDUAL", "bool", True, "Residual FSAM", s, when=fp, advanced=True)
    _add("MODEL.TSCAN.FRAME_DEPTH", "int", 10, "TS-CAN frame depth", s, "Clip length must be a multiple.",
         min=1, max=1000, when=model_when("Tscan"))
    _add("MODEL.EFFICIENTPHYS.FRAME_DEPTH", "int", 10, "EfficientPhys frame depth", s,
         "Clip length must be a multiple.", min=1, max=1000, when=model_when("EfficientPhys"))
    _add("MODEL.BIGSMALL.FRAME_DEPTH", "int", 3, "BigSmall frame depth", s, min=1, max=1000,
         when=model_when("BigSmall"))
    pf = model_when("PhysFormer")
    _add("MODEL.PHYSFORMER.PATCH_SIZE", "int", 4, "Patch size", s, min=1, max=64, when=pf)
    _add("MODEL.PHYSFORMER.DIM", "int", 96, "Embedding dim", s, min=8, max=4096, when=pf)
    _add("MODEL.PHYSFORMER.FF_DIM", "int", 144, "Feed-forward dim", s, min=8, max=16384, when=pf)
    _add("MODEL.PHYSFORMER.NUM_HEADS", "int", 4, "Attention heads", s, min=1, max=64, when=pf)
    _add("MODEL.PHYSFORMER.NUM_LAYERS", "int", 12, "Layers", s, min=1, max=64, when=pf)
    _add("MODEL.PHYSFORMER.THETA", "float", 0.7, "Theta", s, min=0.0, max=1.0, when=pf)


def _wandb_fields():
    s = "Weights & Biases"
    on = {"wandb": True}
    _add("WANDB.ENABLED", "bool", False, "Enable W&B tracking", s)
    _add("WANDB.MODE", "enum", "online", "Mode", s,
         "online syncs live (needs an API key); offline stores runs locally for a later 'wandb sync'.",
         options=["online", "offline", "disabled"], when=on)
    _add("WANDB.PROJECT", "str", "rPPG-Toolbox", "Project", s, pattern=r"^[A-Za-z0-9._-]+$", required=True,
         when=on)
    _add("WANDB.ENTITY", "str", "", "Entity (team/user)", s, "Empty uses your default entity.",
         pattern=r"^[A-Za-z0-9._-]*$", when=on)
    _add("WANDB.RUN_NAME", "str", "", "Run name", s, "Empty derives it from the checkpoint name.", when=on)
    _add("WANDB.GROUP", "str", "", "Group", s, when=on)
    _add("WANDB.JOB_TYPE", "str", "", "Job type", s, when=on)
    _add("WANDB.TAGS", "list", [], "Tags", s, when=on)
    _add("WANDB.NOTES", "text", "", "Notes", s, when=on)
    _add("WANDB.WATCH_MODEL", "bool", False, "Watch gradients", s, when=on, advanced=True)
    _add("WANDB.LOG_BATCH_LOSS", "bool", True, "Log batch loss", s, when=on)
    _add("WANDB.LOG_FREQ", "int", 50, "Batch log interval", s, min=1, max=100000, when=on)


_experiment_fields()
_training_fields()
for _prefix, _spec in SPLITS.items():
    _split_fields(_prefix, _spec)
_evaluation_fields()
_model_fields()
_wandb_fields()

FIELD_MAP = {field["key"]: field for field in FIELDS}
# Keys accepted in YAML but not edited (derived by config.py at runtime).
DERIVED_KEYS = {"BASE", "TEST.OUTPUT_SAVE_DIR", "UNSUPERVISED.OUTPUT_SAVE_DIR"}


def split_of(key):
    for prefix in SPLITS:
        if key.startswith(prefix + "."):
            return prefix
    return None


def is_active(field, values):
    when = field.get("when") or {}
    mode = values.get("TOOLBOX_MODE")
    if "modes" in when and mode not in when["modes"]:
        return False
    if when.get("needs_valid") and values.get("TEST.USE_LAST_EPOCH", True):
        return False
    if "models" in when and values.get("MODEL.NAME") not in when["models"]:
        return False
    if "dataset" in when:
        prefix, datasets = when["dataset"]
        if values.get(prefix + ".DATASET") not in datasets:
            return False
    if when.get("wandb") and not values.get("WANDB.ENABLED"):
        return False
    return True


def active_splits(values):
    mode = values.get("TOOLBOX_MODE")
    if mode == "train_and_test":
        splits = ["TRAIN.DATA"]
        if not values.get("TEST.USE_LAST_EPOCH", True):
            splits.append("VALID.DATA")
        return splits + ["TEST.DATA"]
    if mode == "only_test":
        return ["TEST.DATA"]
    if mode == "unsupervised_method":
        return ["UNSUPERVISED.DATA"]
    return []


def defaults():
    return {field["key"]: field["default"] for field in FIELDS}


def client_schema():
    return {
        "fields": FIELDS,
        "splits": {key: {k: v for k, v in spec.items() if k != "datasets"} for key, spec in SPLITS.items()},
        "model_profiles": MODEL_PROFILES,
        "modes": MODES,
        "mode_labels": MODE_LABELS,
    }
