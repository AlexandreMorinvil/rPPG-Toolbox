"""Detached job runner: executes a job's steps, appends output to its log, records the exit code.

Usage: python -m launcher.runner <job_dir>
Running separately from the web server lets jobs survive a launcher restart.
"""

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def _now():
    return datetime.now().isoformat(timespec="seconds")


def main(job_dir):
    job_dir = Path(job_dir)
    steps = json.loads((job_dir / "steps.json").read_text(encoding="utf-8"))
    log_path = job_dir / "output.log"
    returncode = 0
    failed_step = None
    with log_path.open("ab", buffering=0) as log:
        for index, step in enumerate(steps):
            log.write(f"\n[launcher] {_now()} step {index + 1}/{len(steps)}: {step['title']}\n".encode())
            log.write(("[launcher] $ " + subprocess.list2cmdline(step["argv"]) + "\n").encode())
            started = time.time()
            try:
                process = subprocess.Popen(step["argv"], cwd=step.get("cwd"), stdout=log, stderr=subprocess.STDOUT,
                                           stdin=subprocess.DEVNULL)
                (job_dir / "child.pid").write_text(str(process.pid), encoding="utf-8")
                returncode = process.wait()
            except OSError as error:
                log.write(f"[launcher] Could not start command: {error}\n".encode())
                returncode = 127
            log.write(f"[launcher] {_now()} step finished with exit code {returncode} "
                      f"after {time.time() - started:.0f}s\n".encode())
            if returncode != 0:
                failed_step = index
                break
    (job_dir / "exit.json").write_text(json.dumps({
        "returncode": returncode, "failed_step": failed_step, "finished": _now(),
    }), encoding="utf-8")
    return returncode


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
