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
  - *Launch a batch…* runs many saved configs at once (see *Batches* below).
- **Pre-launch checks** — environment (local Python/PyTorch/CUDA or Docker/Compose/image), configuration, and questions that need your decision: overwrite or use a new output sub-folder, reuse or rebuild an existing cache, Weights & Biases login (use an existing key, enter one for this run — verified with W&B and kept in memory only — switch to offline, or disable), building a missing Docker image, which GPU to use, and whether to queue or run in parallel when the GPU is busy.
- **Jobs** — queued/running/paused/finished runs with phase, epoch, current progress bar and ETA, loss per epoch, parsed test metrics, errors with hints, the live console output, and actions to pause, stop, resume, view the config, reopen it in the editor, or open the results.
- **Run descriptions** — `DESCRIPTION` (Experiment section, or a top-level key in YAML) holds free-form notes about the run. The pre-launch dialog shows it next to the job name for a last edit. In **Jobs**, *Edit* on the Description card changes it at any time, including for running or finished jobs. The job list shows its first line, the Results overview shows it next to the launcher job, and the Results filter searches it. Edits are stored with the job (`job.json`); the config snapshot keeps the description the job was launched with, while *Open in editor* and *Resume* use the edited one. An empty description is left out of the YAML; Docker users must rebuild the toolbox image before launching with a non-empty one.
- **Preprocessed data** — browse caches (`DataFileLists/*.csv`), play clips per 3-channel group, inspect array statistics and labels (pulse waveform with spectrum and heart rate, or vHRM HR/HRV/respiration channels). File lists written in Docker or another location are remapped when possible.
- **Results** — experiment folders found under the results folders: summaries, CSV reports, plots, checkpoints (with *Evaluate…* to prefill an evaluation run), and prediction analysis from `*_outputs.pickle` (or vHRM `window_results.csv`). See *Prediction viewer* below.

## Prediction viewer (Results → Predictions)

Runs save compact per-frame signals; every plot is drawn on demand, so nothing is rendered for recordings you never open.

- **Sources:** test outputs (`saved_test_outputs`), unsupervised outputs (`saved_outputs/<method>`), and the best-epoch validation predictions saved during training (`saved_valid_outputs`).
- **Settings:** window and step for the sliding HR estimate (default 10 s / 2 s), and the ground truth: the measured vHRM HR when the file has it (newer runs), otherwise the HR of the PPG label. *Tracked HR* adds a causal Viterbi-tracked line that limits jumps to about 8 bpm/s.
- **Overview:** MAE/RMSE/MAPE/Pearson/bias/% within ±5 bpm/SNR, scatter and Bland-Altman plots, and an MAE heatmap by movement × view (or participant). Click a cell to filter.
- **Recordings table:** sortable, with MAE, max jump and % within ±5 bpm. *Worst*/*Best* jump to extremes; ↑/↓ step through rows.
- **Recording detail**, on one time axis:
  - HR over time: measured, label-derived, predicted and tracked. Click it to move the waveform window.
  - Error per window.
  - Pulse waveform: label vs prediction, optional unfiltered output, detected beats with the beat rate in view, and shaded spans where the PPG label is invalid.
  - Spectrogram with the measured HR and its 2× harmonic.
  - HRV and respiration channels.
  - Every chart has a *PNG* button.
- **Compare with…:** pick other runs that share recordings. You get an MAE matrix (recordings × runs), with their HR and waveforms overlaid on the selected recording.
- **Validation probes:** for a training run, open the best-validation source. Probe recordings get an epoch selector and an MAE-by-epoch chart from `probe_history.npz`.

What runs save is set in the run editor under **Visualization outputs** (`VISUALIZE.*`): the highlight PNGs, the best-epoch validation set, and the probe interval and selection. Docker users must rebuild the toolbox image before using configs that contain these keys.
- **Settings** — local Python executable, Docker `.env` values (mount folders, default GPU, shared memory, image), environment tests, default GPU policy (queue or parallel), maximum simultaneous jobs, and extra results/preprocessed-data folders to browse.

## Batches

**New run → Launch a batch…** launches many saved configs at once, for example every config of
`configs/launcher/thesis/batch1_literature`. Each config becomes one ordinary job.

1. **Choose the configs.** Tick configs or whole folders in the tree; a folder ticks every config below it. The search
   keeps the selection. The backend follows the selected configs (container paths → Docker) and can be changed.
2. **Check them together.** The usual pre-launch checks run for every config (a few seconds for 140 configs). Questions
   that several configs share are asked **once for the batch**:
   - *Existing results*: new sub-folder, overwrite, or **skip** the configs that already have results (useful to
     relaunch a folder after some runs finished).
   - *Existing caches*: reuse or preprocess again.
   - *Weights & Biases login*, *GPU* and *Docker image*.
   - *Run order*: one after another on the GPU (default), or up to the job limit of Settings at once.

   An answer only changes the configs that asked the question, as if each config were launched alone: a config without
   results keeps its output folder, and a split without a cache still preprocesses.
3. **Read the results of the checks.**
   - *Problems shared by several configs* lists each problem once, with the number of configs it affects (e.g. a data
     folder not set in `thesis/paths.yaml`).
   - The table gives every config's status:
     - *Ready* (possibly with warnings).
     - *Blocked*: errors, or the same output folder as an earlier config of the batch.
     - *Skipped*.
     - *Already running*: the config, or its output folder, belongs to a queued or running job.
   - Untick ready configs to leave them out; *Open* loads a config in the editor. Click a count at the top to show only
     those configs.
4. **Launch.** The launch checks everything again and queues all the ready configs, or none of them if something
   changed since the check.

**In Jobs:**

- **Batch row:** a batch is one row with its jobs nested below; a batch of more than 12 jobs starts collapsed. The row
  shows the jobs done and an estimate of the time left from the finished ones.
- **Detail:** the detail lists every job (status, phase, epoch, MAE, duration; click a row to open the job) and the
  launch answers.
- **Actions:** *Stop remaining jobs*; *Relaunch N unfinished…*, which opens the batch dialog with the failed, stopped
  or removed configs (to continue a training from its last epoch, use *Resume training* on its job instead); *Remove*,
  which keeps the outputs.
- **Each job:** each job is named after its config path, keeps its config's description, and links back to its batch.
  It has the usual features (pause, resume, results).

**Running a batch:**

- **Order:** jobs start in the listed order (config paths sorted).
- **Shared caches:** jobs that preprocess into the same cache folder never preprocess at the same time, even in
  parallel. For example, the thesis lab configs of one model and input share a vHRM cache.
- **The launcher must be open:** jobs start only while the launcher runs. Keep it open, or reopen it, until the last
  job has started. Queued jobs wait across a restart, except with a W&B key typed at launch: that key is kept in memory
  only, so prefer `wandb login` or offline mode for long batches.
- **Change queued jobs:** **Edit N queued…** on the batch (or **Edit parameters…** on a job) changes parameters of jobs
  that have not started, such as the patience, with a preview per job. It changes their config snapshots, not the config
  files, so a later relaunch from those files uses the old values. Changes that would move the output folder (for
  example preprocessing settings), the output path, W&B settings and the mode are refused: cancel the job and launch it
  again for those.
- **Continue finished jobs:** see [Continue a training that early stopping ended](#continue-a-training-that-early-stopping-ended).

## Cross-validation (vHRM)

A training config on vHRM data can run as a participant-wise cross-validation: one training per fold, so every
participant is tested once. Turn on **Cross-validation → Run as a cross-validation** in the editor.

- **Folds:** *K balanced groups* (default, K = 5: participants are dealt round-robin in sorted order, so fold sizes
  differ by at most one), *Leave one participant out*, or *Custom test groups* (a YAML list such as
  `[[jzc01, jzc02], [jzc03]]`; unlisted participants stay in training).
- **Participants:** the pool is what the training data selects (experiments, participants, labels, views). Clear
  *Excluded participants* of the training data, or they are left out of every fold. In each fold the participant lists
  of the training, validation and test data are replaced; everything else in those sections (labels, views,
  preprocessing) applies as set. For example, test labels `[Curl]` test every fold on its Curl segments only.
- **Validation:** *Validation participants per fold* (default 2) are taken from the next fold. Keep *Test the last
  epoch* off and set up the validation data. With 0, turn *Test the last epoch* on.
- **Sessions:** *Keep the sessions of a person together* (default on) puts `jzc02` and `jzc02_baseline` in the same
  fold: a folder joins the shorter folder name it extends with `_` or `-`. Counts then refer to persons.
- **Preview folds…** (in the section) lists the participants of every fold. Listing reads the dataset folders with the
  local Python of **Settings**, which needs the toolbox environment. This also applies to Docker runs.

**Launch cross-validation…** runs the usual pre-launch checks once for the whole group and shows the folds. It then
queues one ordinary job per fold, named `<name> · fold <N>`.

- **Order:** the folds run one after another, in fold order, and preprocessing is shared after the first one.
- **Failures and pauses:** a failed fold does not stop the others. A paused fold holds the remaining folds until you
  resume it. Removing the paused fold releases them.
- **Fold jobs:** each fold has the usual job features: progress, pause/resume, stop, retry and results. Its config
  snapshot is a plain single run without the `CROSS_VALIDATION` block.

**Outputs:** `LOG.PATH` is the cross-validation folder.

- **Per fold:** each fold writes to `LOG.PATH/fold_<name>`, and the participants of every fold are recorded in
  `cv_plan.json`.
- **Merged summary:** after each finished fold, the launcher merges the evaluation windows of all folds into
  `LOG.PATH/cv_summary` (`overall_summary.csv`, `by_fold.csv`, `by_participant.csv`, `by_movement.csv`,
  `all_windows.csv`). It uses `tools/vhrm_cross_validation.py --aggregate-only` with the local Python; *Merge results
  now* repeats it.
- **Relaunching:** if the folder already has fold results, the pre-launch checks offer a new timestamped sub-folder,
  because the merge reads every `fold_*` folder.

**Jobs:**

- **Group row:** a cross-validation is one row with its folds nested below. The row shows the folds done and an
  estimate of the time left, based on the finished folds.
- **Detail:** the detail shows the fold table (participants, status, epoch, test MAE), the merged results, and the
  actions *Stop remaining folds*, *Merge results now*, *Open in editor* (the cross-validation config) and *Remove*
  (keeps the outputs).

**Results:**

- **Cross-validation folder:** listed as *cross-validation*. Its overview shows the merged tables and the folds.
- **Predictions:** *Predictions → cv_summary/all_windows.csv* plots every window of every fold, for example the
  participant × movement MAE heatmap.
- **Folds:** each fold's folder remains a result of its own.

The command line runs the same folds:
`python tools/vhrm_cross_validation.py --config <config> --output-root <folder>` uses the `CROSS_VALIDATION` block of
the config unless options override it.

## Analysis report (vHRM)

A report analyses one checkpoint on every participant of its test data's experiments and views: heart rate,
beat-to-beat RR intervals and HRV against the chest sensors (Zephyr HR, ECG R-peaks), per movement (resting, curl,
recovery, …), per repetition and per split, with figures, tables and CSV files. It is the JZC curl analysis
(`jzc_rppg_analysis`) built into the toolbox; the method is in `evaluation/vhrm_report/METHOD.md`, copied into each report.

- **For any checkpoint:** Results → an experiment → *Checkpoints* → **Report…**. Choose the run config (the launcher job
  that trained the checkpoint is preselected: it gives the model, the input preprocessing and which participants were
  training, validation and test), the backend, and optionally the segment labels (the labels found in the cache are
  shown) and participants. The report runs as a job in **Jobs**. The checkpoint the run tested is marked *tested*.
- **At the end of a training:** in the editor, **Analysis report → Generate the analysis report** (`REPORT.ENABLED`).
  After the test, main.py analyses the tested checkpoint (the best validation epoch, or the last epoch). A failed report
  is logged and shown as a warning; it never fails the training. For a cross-validation, every fold writes its own
  report. Docker users must rebuild the toolbox image first.
- **For an unsupervised run (POS, …):** the same `REPORT.ENABLED` on a vHRM unsupervised config. After the run, each
  method of `UNSUPERVISED.METHOD` is analysed on whole recordings, in `<experiment>/report/<method>/`, without splits
  (the method learns nothing). From a terminal: `python tools/vhrm_report.py --config <config> [--methods POS]`.
- **Reading it:** the experiment gets a **Report** tab: headline numbers, the MAE matrix movement × split (only the
  *Test* column measures unseen participants), pooled results per group, the summary figures, the recordings table
  (click a row for that participant's HR, HRV, RR and pulse figures) and the mosaics of every participant per recording.
  *Test participants only* switches every table and figure to the test participants. With several reports (e.g. epochs)
  a table compares them. `REPORT.md`, `METHOD.md` and `run_config.json` open from the tab.
- **Where:** `<experiment>/report/<checkpoint>/`, for example `…/PreTrainedModels/name_Epoch16.pth` →
  `…/report/Epoch16/`. Generating it again replaces it; the model output of unchanged recordings is reused.
- **Input:** the toolbox's preprocessed cache, i.e. exactly what the checkpoint was trained and tested on. Recordings
  missing from it (e.g. the resting segments of test participants tested on Curl only) are preprocessed first. The
  dataset folder must be reachable: the report reads the ECG R-peaks and the segment timing from it.

The command line does the same: `python tools/vhrm_report.py --config <run config> [--checkpoint <.pth>] [--labels …]`.

## Early stopping

In **Training → Early stopping**, enable early stopping and set **Patience (epochs)**
and **Improvement tolerance**. Turn off **Test the last epoch** and configure the
validation split. Patience counts consecutive epochs without a loss reduction
strictly greater than the absolute tolerance; zero tolerance counts any reduction.
**Minimum epochs** holds off the stop until that many epochs are done; the epochs before
it still count against the patience, so a patience already used up stops training as
soon as the minimum is reached. With **Start patience after the minimum**, only the epochs
after it count, compared with the best loss so far, so training lasts at least the minimum
plus the patience. Defaults are disabled, patience 10, tolerance 0.0, no minimum, patience
counted from the first epoch. The best actual validation-loss checkpoint is selected independently of the
tolerance, and **Epochs** stays an upper limit. The patience state survives pause/resume.
**Retry plots / test** on an early-stopped run does not train further; **Continue
training…** does.

### Continue a training that early stopping ended

**Continue training…** on a job (or **Continue N trainings…** on its batch) queues it
again from its last epoch, either without early stopping or with other early stopping
settings; a setting left empty keeps the job's value. A new attempt with its own log links
to the original and takes its place in the batch.

- **As if it had not stopped:** weights, optimizer, learning-rate schedule, RNG and
  histories come from the recovery checkpoint, so it continues the same training. Epochs
  without improvement before the stop still count against a new patience: patience 8
  after a stop at patience 4 gives at least 4 more epochs. A patience that is already used
  up is refused, unless a higher **minimum epochs** still requires more epochs.
- **The maximum stays:** **Epochs** cannot be raised here, because the OneCycle schedule
  was planned for that number and has no epochs after it.
- **Results:** if the best epoch changes, the test and the analysis report run with the new
  weights: the test results are replaced and the new report is written to its own
  `report/Epoch<N>` folder, next to the previous one. If it does not change, the test and
  the report are skipped and the results stay. The test also runs if the saved results
  cannot be shown to come from this checkpoint (missing, or older than it).
- **Later steps:** batch 3 of the thesis configs reads the tested checkpoint when
  `tools/thesis_configs.py` runs, so run it again after continuing batch 2.

Runs from before this feature can continue too, as long as they have their
`<prefix>_resume.pt` file. Outside the launcher, set `MODEL.RESUME` to it and change
`TRAIN.EARLY_STOPPING` in the YAML.

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

- Launcher state, job logs and config snapshots: `launcher_data/` (Git-ignored); batches in `launcher_data/batches/`.
- Each job's exact config is `launcher_data/jobs/<job>/config.yaml`; Docker jobs mount it read-only into the container.
- The W&B API key is never written to disk by the launcher.
