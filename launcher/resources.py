"""How many preprocessing jobs this computer can run at once (the automatic mode of a batch's preprocessing stage).

The launcher measures the load while the jobs run: CPU use, free memory and the memory and use of the GPU. It starts
the first job at once, then one more each time the last one has run for a minute (enough for its workers to load
their videos) if the measured load leaves room for another job of the size the jobs it started so far reached, plus a
reserve. It never stops a running job: when the load grows (a training that starts, another user's work), it only
waits before starting the next one.

The readings cover the whole computer, so the jobs of another batch or another user count too. Docker jobs run in
Docker Desktop's virtual machine, whose own memory limit can be lower than the computer's free memory.
"""

import ctypes
import os
import threading
import time
from collections import deque, namedtuple
from datetime import datetime

from . import probes

try:
    import psutil
except ImportError:  # the launcher only needs PyYAML and NumPy; fall back to the system counters
    psutil = None

GIB = 1024 ** 3
MIB = 1024 ** 2
SETTLE_SECONDS = 60      # a job's footprint is judged over its first minute before another job starts
TRACK_SECONDS = 90       # how long after its start a job's peak use is attributed to it (later load is not its own)
WINDOW_SECONDS = 15      # load: mean CPU and GPU use, least free memory over this window
GPU_PERIOD = 5           # nvidia-smi at most every 5 s
CPU_BUSY = 85            # % of all cores
GPU_BUSY = 90            # %
MAX_JOBS = 16
RAM_FLOOR = 1.5 * GIB    # smallest footprint assumed per job: each preprocessing worker holds a whole video
VRAM_FLOOR = 0.5 * GIB   # each worker loads its own face detector
CPU_FLOOR = 5.0          # %
REMEMBERED = 6           # footprint estimate: the largest of the last jobs started

Sample = namedtuple("Sample", "time cpu ram ram_total gpus")  # gpus: {index: (used bytes, total bytes, use %)}


def ram_reserve(total):
    return max(4 * GIB, 0.10 * total)


def vram_reserve(total):
    return max(1 * GIB, 0.10 * total)


class _FILETIME(ctypes.Structure):
    _fields_ = [("low", ctypes.c_uint32), ("high", ctypes.c_uint32)]


class _MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_uint32), ("dwMemoryLoad", ctypes.c_uint32),
                ("ullTotalPhys", ctypes.c_uint64), ("ullAvailPhys", ctypes.c_uint64),
                ("ullTotalPageFile", ctypes.c_uint64), ("ullAvailPageFile", ctypes.c_uint64),
                ("ullTotalVirtual", ctypes.c_uint64), ("ullAvailVirtual", ctypes.c_uint64),
                ("ullAvailExtendedVirtual", ctypes.c_uint64)]


def _cpu_times():
    """(idle, total) CPU time counters of the whole computer, or None."""
    if os.name == "nt":
        idle, kernel, user = _FILETIME(), _FILETIME(), _FILETIME()
        if not ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)):
            return None
        value = lambda t: (t.high << 32) | t.low  # noqa: E731
        return value(idle), value(kernel) + value(user)  # kernel time includes the idle time
    try:
        with open("/proc/stat", encoding="ascii") as handle:
            fields = [int(part) for part in handle.readline().split()[1:]]
        return fields[3] + (fields[4] if len(fields) > 4 else 0), sum(fields[:8])
    except (OSError, ValueError, IndexError):
        return None


def memory():
    """(available, total) physical memory in bytes, or (None, None)."""
    if psutil is not None:
        info = psutil.virtual_memory()
        return info.available, info.total
    if os.name == "nt":
        status = _MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return status.ullAvailPhys, status.ullTotalPhys
        return None, None
    try:
        with open("/proc/meminfo", encoding="ascii") as handle:
            info = {line.split(":")[0]: int(line.split()[1]) * 1024 for line in handle if ":" in line}
        return info["MemAvailable"], info["MemTotal"]
    except (OSError, ValueError, KeyError, IndexError):
        return None, None


class CpuCounter:
    """CPU use of the whole computer (%) since the previous call."""

    def __init__(self):
        self.last = None if psutil is not None else _cpu_times()
        if psutil is not None:
            psutil.cpu_percent(interval=None)  # the first reading only starts the measurement

    def percent(self):
        if psutil is not None:
            return psutil.cpu_percent(interval=None)
        now = _cpu_times()
        if now is None or self.last is None:
            self.last = now
            return None
        idle, total = now[0] - self.last[0], now[1] - self.last[1]
        self.last = now
        return 100.0 * (1 - idle / total) if total > 0 else None


class Monitor:
    """Readings of the load, taken by the job monitor every second while an automatic preprocessing job waits."""

    def __init__(self, clock=time.monotonic, gpus=probes.gpus, cpu=None, mem=memory):
        self.clock, self.gpus, self.mem = clock, gpus, mem
        self.cpu = cpu or CpuCounter()
        self.samples = deque(maxlen=600)
        self._gpu_time = None
        self._gpu = {}

    def sample(self):
        now = self.clock()
        cpu = self.cpu.percent()
        available, total = self.mem()
        if self._gpu_time is None or now - self._gpu_time >= GPU_PERIOD:
            self._gpu_time = now
            self._gpu = {gpu["index"]: (gpu["memory_used"] * MIB, gpu["memory_total"] * MIB, gpu["utilization"])
                         for gpu in self.gpus()}
        self.samples.append(Sample(now, cpu, available, total, dict(self._gpu)))
        return self.samples[-1]

    def window(self, seconds=WINDOW_SECONDS):
        now = self.clock()
        return [sample for sample in self.samples if sample.time >= now - seconds]

    def reset(self):
        self.samples.clear()


def _mean(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def _least(values):
    values = [value for value in values if value is not None]
    return min(values) if values else None


def load(samples, gpu=None):
    """The load over these samples: mean CPU and GPU use, least free memory (GPU: most memory used)."""
    if not samples:
        return None
    result = {"cpu": _mean(s.cpu for s in samples), "ram_free": _least(s.ram for s in samples),
              "ram_total": samples[-1].ram_total, "gpu": None}
    readings = [s.gpus[gpu] for s in samples if gpu is not None and gpu in s.gpus]
    if readings:
        result["gpu"] = {"index": gpu, "used": max(r[0] for r in readings), "total": readings[-1][1],
                         "use": _mean(r[2] for r in readings)}
    return result


class AutoScaler:
    """Decides whether one more automatic preprocessing job may start, from the readings of a Monitor."""

    def __init__(self, monitor=None):
        self.monitor = monitor or Monitor()
        self.lock = threading.Lock()
        self.starts = deque(maxlen=REMEMBERED)  # the jobs it started: load before, peak use after
        self.status = None

    # ------------------------------------------------------------ readings
    def sample(self):
        """One reading; the peak use after the last start is attributed to the job started then."""
        with self.lock:
            sample = self.monitor.sample()
            start = self.starts[-1] if self.starts else None
            if start and sample.time - start["time"] <= TRACK_SECONDS:
                recent = load(self.monitor.window(10), start["gpu"])
                if sample.ram is not None:
                    start["least_ram"] = _least([start["least_ram"], sample.ram])
                if recent and recent["cpu"] is not None:
                    start["most_cpu"] = max(start["most_cpu"] or 0.0, recent["cpu"])
                if start["gpu"] in sample.gpus:
                    start["most_vram"] = max(start["most_vram"] or 0, sample.gpus[start["gpu"]][0])

    def idle(self):
        """No automatic job waits: forget the readings (the next batch measures again) but keep the estimates."""
        with self.lock:
            self.monitor.reset()
            self.status = None

    def started(self, job):
        with self.lock:
            before = load(self.monitor.window(10), job.get("gpu")) or {}
            gpu = before.get("gpu") or {}
            self.starts.append({"job": job["id"], "gpu": job.get("gpu"), "time": self.monitor.clock(),
                                "ram": before.get("ram_free"), "least_ram": before.get("ram_free"),
                                "cpu": before.get("cpu"), "most_cpu": before.get("cpu"),
                                "vram": gpu.get("used"), "most_vram": gpu.get("used")})

    # ----------------------------------------------------------- estimates
    def estimates(self):
        """Footprint of one more job: the largest the jobs started so far reached, at least the floors."""
        ram, vram, cpu = RAM_FLOOR, VRAM_FLOOR, CPU_FLOOR
        for start in self.starts:
            if start["ram"] is not None and start["least_ram"] is not None:
                ram = max(ram, start["ram"] - start["least_ram"])
            if start["vram"] is not None and start["most_vram"] is not None:
                vram = max(vram, start["most_vram"] - start["vram"])
            if start["cpu"] is not None and start["most_cpu"] is not None:
                cpu = max(cpu, start["most_cpu"] - start["cpu"])
        return {"ram": ram, "vram": vram, "cpu": cpu}

    def decide(self, running, gpu):
        """(may start, reason) for a job on this GPU, with these preprocessing jobs running."""
        with self.lock:
            allowed, reason, current = self._decide(running, gpu)
            estimate = self.estimates()
            self.status = {
                "running": len(running), "can_start": allowed, "reason": reason, "checked": datetime.now()
                .isoformat(timespec="seconds"), "load": current,
                "estimate": {"ram_gb": round(estimate["ram"] / GIB, 1), "vram_gb": round(estimate["vram"] / GIB, 1),
                             "cpu": round(estimate["cpu"])},
            }
            return allowed, reason

    def _decide(self, running, gpu):
        if not running:
            return True, "No preprocessing job is running.", None
        if len(running) >= MAX_JOBS:
            return False, f"{MAX_JOBS} jobs are running, the most it starts.", None
        now = self.monitor.clock()
        ids = {job["id"] for job in running}
        last = self.starts[-1] if self.starts else None
        if last and last["job"] in ids and now - last["time"] < SETTLE_SECONDS:
            return False, f"Measuring the job started {now - last['time']:.0f} s ago.", None
        current = load(self.monitor.window(), gpu)
        if not current or current["ram_free"] is None:
            return False, "Measuring the load.", current
        estimate = self.estimates()
        reasons = []
        cpu = current["cpu"]
        if cpu is not None and (cpu > CPU_BUSY or cpu + estimate["cpu"] > 98):
            reasons.append(f"the CPU is {cpu:.0f}% busy")
        free, total = current["ram_free"], current["ram_total"] or current["ram_free"]
        if free - estimate["ram"] < ram_reserve(total):
            reasons.append(f"{free / GIB:.1f} GB of memory free, ~{estimate['ram'] / GIB:.1f} GB more per job and "
                           f"{ram_reserve(total) / GIB:.0f} GB kept free")
        card = current["gpu"]
        if card:
            vram_free = card["total"] - card["used"]
            if vram_free - estimate["vram"] < vram_reserve(card["total"]):
                reasons.append(f"GPU {gpu}: {vram_free / GIB:.1f} GB of memory free, ~{estimate['vram'] / GIB:.1f} GB "
                               f"more per job and {vram_reserve(card['total']) / GIB:.0f} GB kept free")
            if card["use"] is not None and card["use"] > GPU_BUSY:
                reasons.append(f"GPU {gpu} is {card['use']:.0f}% busy")
        if reasons:
            return False, "Waiting: " + "; ".join(reasons) + ".", current
        return True, "The load leaves room for one more job.", current

    def snapshot(self):
        with self.lock:
            return dict(self.status) if self.status else None
