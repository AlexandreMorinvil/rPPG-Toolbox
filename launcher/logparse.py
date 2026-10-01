"""Incremental parsing of toolbox console output into progress and metrics."""

import re

TQDM = re.compile(
    r"(?P<desc>[^\r\n|]*?):?\s*(?P<pct>\d{1,3})%\|[^|]*\|\s*(?P<n>\d+)/(?P<total>\d+)\s*"
    r"\[(?P<elapsed>[\d:]+)(?:<(?P<eta>[\d:?]+))?(?P<rest>[^\]]*)\]"
)
LOSS_IN_BAR = re.compile(r"loss=([-+\d.eE]+|nan)")
EPOCH = re.compile(r"^====Training Epoch: (\d+)====")
BATCH_LOSS = re.compile(r"^\[(\d+),\s*(\d+)\] loss: ([-+\d.eE]+|nan)")
VALID_LOSS = re.compile(r"^validation loss:\s*([-+\d.eE]+|nan)")
BEST = re.compile(r"Best epoch: (\d+)")
BEST_FINAL = re.compile(r"^best trained epoch: (\d+), min_val_loss: ([-+\d.eE]+|nan)")
METRIC = re.compile(
    r"^(?P<method>FFT|PEAK|Peak)\s+(?P<name>MAE|RMSE|MAPE|Pearson|SNR|MACC)\s*\([^)]*\):\s*"
    r"(?P<value>[-+\w.]+)\s*\+/-\s*(?P<se>[-+\w.]+)"
)
MACC_AVG = re.compile(r"^MACC \(avg\):\s*([-+\w.]+)\s*\+/-\s*([-+\w.]+)")
UNSUPERVISED = re.compile(r"^===Unsupervised Method \( (\w+) \) Predicting ===")
WANDB_URL = re.compile(r"(https://wandb\.ai/\S+/runs/\S+)")
OUTPUT_PICKLE = re.compile(r"^Saving outputs to:\s*(.+)$")
SAVED_MODEL = re.compile(r"^Saved Model Path:\s*(.+)$")
EXCEPTION = re.compile(r"^([A-Za-z_][\w.]*(?:Error|Exception|Interrupt|Exit)|Killed)(?::\s*(.*))?$")

HINTS = (
    ("CUDA out of memory", "GPU memory exhausted: lower the batch size, INFERENCE.FRAME_BATCH_SIZE or the input size."),
    ("could not select device driver", "Docker cannot access the GPU: check the NVIDIA driver and Docker GPU support."),
    ("Bus error", "Shared memory exhausted: increase DOCKER_SHM_SIZE or lower data loader workers."),
    ("shared memory", "Shared memory exhausted: increase DOCKER_SHM_SIZE or lower data loader workers."),
    ("No such file or directory", "A file or folder was not found: check data, cache and checkpoint paths."),
    ("Non-existent config key", "The YAML contains a key config.py does not know."),
    ("api_key not configured", "W&B could not log in: provide an API key or use offline mode."),
    ("Cannot connect to the Docker daemon", "Docker is not running."),
    ("pull access denied", "The Docker image is not available locally: build it first."),
    ("No Face Detected", "Some frames had no detected face; consider the Y5F detector or dynamic detection."),
)


def _number(text):
    try:
        return float(text)
    except ValueError:
        return None


class LogParser:
    def __init__(self, mode=None, epochs_total=None, state=None):
        self.partial = ""
        self.in_traceback = False
        self.state = state or {
            "phase": "starting",
            "epoch": None,
            "epochs_total": epochs_total,
            "bar": None,
            "train_loss": {},
            "valid_loss": {},
            "best_epoch": None,
            "metrics": {},
            "method": None,
            "wandb_url": None,
            "outputs_pickle": None,
            "checkpoints": 0,
            "error": None,
            "traceback": [],
            "hints": [],
            "warnings": 0,
            "last_line": "",
        }
        self.mode = mode
        self._loss_sum = {}

    def feed(self, text):
        data = self.partial + text
        lines = data.split("\n")
        self.partial = lines.pop()
        for line in lines:
            self._line(line)
        if self.partial:
            # Interim tqdm updates arrive without a newline.
            segments = [s for s in self.partial.split("\r") if s.strip()]
            for segment in segments[-3:]:
                self._bar(segment)

    def snapshot(self):
        state = dict(self.state)
        state["train_loss"] = {k: v["sum"] / v["count"] for k, v in self.state["train_loss"].items() if v["count"]}
        return state

    def _bar(self, segment):
        match = TQDM.search(segment)
        if not match:
            return False
        desc = match.group("desc").strip()
        bar = {
            "desc": desc,
            "n": int(match.group("n")),
            "total": int(match.group("total")),
            "elapsed": match.group("elapsed"),
            "eta": match.group("eta"),
        }
        self.state["bar"] = bar
        lowered = desc.lower()
        if lowered.startswith("train epoch"):
            self.state["phase"] = "training"
            loss = LOSS_IN_BAR.search(match.group("rest") or "")
            epoch = self.state["epoch"]
            if loss and epoch is not None and _number(loss.group(1)) is not None:
                entry = self.state["train_loss"].setdefault(str(epoch), {"sum": 0.0, "count": 0})
                entry["sum"] += float(loss.group(1))
                entry["count"] += 1
        elif lowered.startswith("validation"):
            self.state["phase"] = "validating"
        return True

    def _line(self, line):
        segments = [s for s in line.split("\r") if s.strip()]
        if not segments:
            return
        for segment in segments[:-1]:
            self._bar(segment)
        text = segments[-1].rstrip()
        if self._bar(text) and "%|" in text:
            return
        stripped = text.strip()
        self.state["last_line"] = stripped[:300]

        if self.in_traceback:
            self.state["traceback"].append(text[:500])
            self.state["traceback"] = self.state["traceback"][-40:]
            match = EXCEPTION.match(stripped)
            if match and not text.startswith((" ", "\t")):
                self.state["error"] = stripped[:500]
                self.in_traceback = False
            return
        if stripped.startswith("Traceback (most recent call last)"):
            self.in_traceback = True
            self.state["traceback"] = [stripped]
            return

        for needle, hint in HINTS:
            if needle in stripped and hint not in self.state["hints"]:
                self.state["hints"].append(hint)
        if "Warning" in stripped or "WARNING" in stripped:
            self.state["warnings"] += 1

        if stripped.startswith("Preprocessing dataset") or "| decode" in stripped:
            self.state["phase"] = "preprocessing"
        elif stripped.startswith("File list does not exist"):
            self.state["phase"] = "preprocessing"
        match = EPOCH.match(stripped)
        if match:
            self.state["epoch"] = int(match.group(1))
            self.state["phase"] = "training"
            return
        if stripped.startswith("===Validating==="):
            self.state["phase"] = "validating"
            return
        if stripped.startswith("===Testing==="):
            self.state["phase"] = "testing"
            return
        match = UNSUPERVISED.match(stripped)
        if match:
            self.state["phase"] = "unsupervised"
            self.state["method"] = match.group(1)
            return
        match = VALID_LOSS.match(stripped)
        if match and self.state["epoch"] is not None:
            self.state["valid_loss"][str(self.state["epoch"])] = _number(match.group(1))
            return
        match = BEST_FINAL.match(stripped) or BEST.search(stripped)
        if match:
            self.state["best_epoch"] = int(match.group(1))
            return
        match = METRIC.match(stripped)
        if match:
            group = self.state["method"] or "test"
            self.state["metrics"].setdefault(group, {})[match.group("name")] = {
                "value": _number(match.group("value")), "se": _number(match.group("se")),
            }
            return
        match = MACC_AVG.match(stripped)
        if match:
            group = self.state["method"] or "test"
            self.state["metrics"].setdefault(group, {})["MACC"] = {
                "value": _number(match.group(1)), "se": _number(match.group(2)),
            }
            return
        match = OUTPUT_PICKLE.match(stripped)
        if match:
            self.state["outputs_pickle"] = match.group(1).strip()
            return
        if SAVED_MODEL.match(stripped):
            self.state["checkpoints"] += 1
            return
        match = WANDB_URL.search(stripped)
        if match:
            self.state["wandb_url"] = match.group(1)
            return
        match = EXCEPTION.match(stripped)
        if match and not text.startswith((" ", "\t")) and match.group(1) != "Warning":
            self.state["error"] = stripped[:500]
