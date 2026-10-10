# rPPG-Toolbox Launcher

A local web application to prepare, validate, launch and follow toolbox runs, then browse preprocessed data and results.

## Start

From the `rPPG-Toolbox` folder, using the toolbox Python environment (it needs PyYAML and NumPy; SciPy, yacs and PyTorch are used when available):

```powershell
conda activate vHRM_development
python -m launcher
```

The browser opens `http://127.0.0.1:8790/`. Options: `--port 8800`, `--no-browser`, `--host` (only on trusted networks: the launcher can start processes).

### Access from another computer on the network

```powershell
python -m launcher --lan --no-browser
```

`--lan` listens on all interfaces and prints the URLs to use, e.g. `http://192.168.1.20:8790/` or `http://<machine-name>:8790/`. The machine name (or `<machine-name>.local`) is usually more convenient than the IP because it does not change with DHCP. Anyone who can reach the page can start processes on this machine, so use it only on trusted networks. Windows may need a firewall rule (administrator PowerShell):

```powershell
New-NetFirewallRule -DisplayName "rPPG launcher" -Direction Inbound -Protocol TCP -LocalPort 8790 -Profile Private -Action Allow
```

Jobs run in a detached runner process, so they keep running when the launcher is closed; reopening the launcher re-attaches to them.

## Tabs

- **New run** — start from any YAML in `configs/` or `docker/configs/`, or from a blank config (training, checkpoint evaluation, or unsupervised methods). Choose the **Docker** or **Local Python** backend per run.
  - Only the fields relevant to the selected run type, model and datasets are shown. Dropdowns and toggles limit values to supported options.
  - Every change is validated: types and ranges, model/preprocessing compatibility (tensor layout, input transforms, clip length vs. frame count or frame depth), split consistency and overlap, vHRM rules, paths (existence, raw-data layout, Docker mounts), and whether an existing cache can be reused by this backend.
  - *Convert paths* translates data/cache/output/checkpoint paths between host paths and Docker container paths using the `.env` mounts.
  - **Training sync (optional)** — the *Validation/test follow training preprocessing* checkbox links all preprocessing parameters, frame rate and tensor layout to both splits. *Sync settings…* lets you link individual parameters to validation, test, or both, including selected vHRM operational settings. Both controls share the same links; the checkbox is partially checked for a custom selection. Enabling links opens a preview before any values are overwritten. Destination fields display *Linked to training* and stay read-only until you click *Unlink*. Turning off the quick checkbox removes its preprocessing links while preserving any other custom links and all current values; *Clear all* in the custom panel removes everything after preview/apply. Links also update hidden validation/model-specific fields. Dataset identities, participants, split boundaries, paths and cache/output identifiers are excluded. Links start off and reset when loading/creating a config (including reopening a job) or leaving training mode; saved YAML contains only ordinary values, not link settings.
  - *Save config* writes a standalone YAML; *Launch* runs the pre-launch checks.
- **Pre-launch checks** — environment (local Python/PyTorch/CUDA or Docker/Compose/image), configuration, and questions that need your decision: overwrite or use a new output sub-folder, reuse or rebuild an existing cache, Weights & Biases login (use an existing key, enter one for this run — verified with W&B and kept in memory only — switch to offline, or disable), building a missing Docker image, which GPU to use, and whether to queue or run in parallel when the GPU is busy.
- **Jobs** — queued/running/paused/finished runs with phase, epoch, current progress bar and ETA, loss per epoch, parsed test metrics, errors with hints, the live console output, and actions to pause, stop, resume, view the config, reopen it in the editor, or open the results.
- **Preprocessed data** — browse caches (`DataFileLists/*.csv`), play clips per 3-channel group, inspect array statistics and labels (pulse waveform with spectrum and heart rate, or vHRM HR/HRV/respiration channels). File lists written in Docker or another location are remapped when possible.
- **Results** — experiment folders found under the results folders: summaries, CSV reports, plots, checkpoints (with *Evaluate…* to prefill an evaluation run), and prediction analysis from `*_outputs.pickle` (or vHRM `window_results.csv`): per-window heart rate, MAE/RMSE/MAPE/Pearson/SNR, scatter and Bland-Altman plots, per-recording HR over time, waveforms and spectra.
- **Settings** — local Python executable, Docker `.env` values (mount folders, default GPU, shared memory, image), environment tests, default GPU policy (queue or parallel), maximum simultaneous jobs, and extra results/preprocessed-data folders to browse.

## Early stopping

In **Training → Early stopping**, enable early stopping and set **Patience (epochs)**
and **Improvement tolerance**. Turn off **Test the last epoch** and configure the
validation split. Patience counts consecutive epochs without a loss reduction
strictly greater than the absolute tolerance; zero tolerance counts any reduction.
Defaults are disabled, patience 10, tolerance 0.0. The best actual validation-loss
checkpoint is selected independently of the tolerance, and **Epochs** stays an upper
limit. The patience state survives pause/resume. An early-stopped run is complete;
recovery retries plots/testing without training further.

## Pause, stop and resume training

New neural training jobs save full recovery state by default (`TRAIN.SAVE_RESUME: true`).
The latest committed checkpoint is named `<checkpoint prefix>_resume.pt` in the model
folder, alongside the existing inference-only epoch weight files. It is written before
the first epoch and after every **completed training epoch and validation**. Writes
use a temporary file and atomic replacement, so a crash during writing leaves the
previous committed checkpoint intact.

- **Pause after epoch** requests a cooperative pause. The current epoch and validation
  finish, state is committed, and the process exits with the job marked **paused**.
  The GPU and job slot are released. A request during preprocessing waits until the
  initial training checkpoint is ready. This is not an instantaneous in-memory freeze.
- **Stop now** terminates the process immediately. **Resume training** is available
  when a full checkpoint exists; any unfinished epoch is replayed from its beginning.
  The same recovery works for **failed** and **interrupted** jobs, including after
  restarting the launcher or rebooting the machine.
- **Resume training** opens the original config snapshot and normal pre-launch checks.
  It selects the recovery checkpoint, disables preprocessing and defaults to adding
  to the original outputs. Confirm the W&B/GPU choices and click **Launch**. A new job
  attempt keeps its own log and links to the original; the original log is preserved.
- **Retry plots / test** appears when all training epochs were already committed.
  It restores histories/model selection, skips all training epochs, then reruns the
  normal post-training plotting and testing. Fix the cause of a crash before retrying.

State includes model weights, optimizer momentum, scheduler, best epoch and validation
loss, best-model weights, plot histories, Python/NumPy/PyTorch/CUDA RNG states,
DataLoader generators and the W&B optimizer-step axis. Resumed attempts create a new
W&B run; API keys are never embedded in checkpoints. The original inference weight
format remains unchanged. BigSmall's fixed learning rate and PhysFormer's epoch-wise
scheduler use the same mechanism as the batch-wise OneCycle trainers.

Keep the same model, optimizer settings, **original total epoch budget**, batch size,
data splits, clip file lists and runtime/device setup. Incompatible resumes are rejected;
this is continuation, not fine-tuning with a new scheduler. Seed restoration is tested
for deterministic CPU training; arbitrary CUDA kernels or changed runtimes are not
guaranteed bit-for-bit reproducible. Do not modify cached clips while a run is paused.

Outside the launcher, set `MODEL.RESUME` to the trusted recovery checkpoint in the
training YAML and run normally; preprocessing is automatically skipped. Setting
`TRAIN.SAVE_RESUME: false` disables full-state saving and cooperative pause.
Only load checkpoints created by a trusted source: full recovery files use PyTorch's
pickle-based loader for optimizer and RNG state.

**Existing runs:** old epoch `.pth` files contain weights only. They cannot recover
optimizer/scheduler/progress retrospectively; use checkpoint evaluation for those runs.
Restart the launcher to load these controls (detached jobs keep running). Docker users
must rebuild the toolbox image before starting new jobs; rebuilding does not upgrade
containers that are already running.

## Files

- Launcher state, job logs and config snapshots: `launcher_data/` (Git-ignored).
- Each job's exact config is `launcher_data/jobs/<job>/config.yaml`; Docker jobs mount it read-only into the container.
- The W&B API key is never written to disk by the launcher.
