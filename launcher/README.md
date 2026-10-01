# rPPG-Toolbox Launcher

A local web application to prepare, validate, launch and follow toolbox runs, then browse preprocessed data and results.

## Start

From the `rPPG-Toolbox` folder, using the toolbox Python environment (it needs PyYAML and NumPy; SciPy, yacs and PyTorch are used when available):

```powershell
conda activate vHRM_development
python -m launcher
```

The browser opens `http://127.0.0.1:8790/`. Options: `--port 8800`, `--no-browser`, `--host` (only on trusted networks: the launcher can start processes).

Jobs run in a detached runner process, so they keep running when the launcher is closed; reopening the launcher re-attaches to them.

## Tabs

- **New run** — start from any YAML in `configs/` or `docker/configs/`, or from a blank config (training, checkpoint evaluation, or unsupervised methods). Choose the **Docker** or **Local Python** backend per run.
  - Only the fields relevant to the selected run type, model and datasets are shown. Dropdowns and toggles limit values to supported options.
  - Every change is validated: types and ranges, model/preprocessing compatibility (tensor layout, input transforms, clip length vs. frame count or frame depth), split consistency and overlap, vHRM rules, paths (existence, raw-data layout, Docker mounts), and whether an existing cache can be reused by this backend.
  - *Convert paths* translates data/cache/output/checkpoint paths between host paths and Docker container paths using the `.env` mounts.
  - *Save config* writes a standalone YAML; *Launch* runs the pre-launch checks.
- **Pre-launch checks** — environment (local Python/PyTorch/CUDA or Docker/Compose/image), configuration, and questions that need your decision: overwrite or use a new output sub-folder, reuse or rebuild an existing cache, Weights & Biases login (use an existing key, enter one for this run — verified with W&B and kept in memory only — switch to offline, or disable), building a missing Docker image, which GPU to use, and whether to queue or run in parallel when the GPU is busy.
- **Jobs** — queued/running/finished runs with phase, epoch, current progress bar and ETA, loss per epoch, parsed test metrics, errors with hints, the live console output, and actions to stop, view the config, reopen it in the editor, or open the results.
- **Preprocessed data** — browse caches (`DataFileLists/*.csv`), play clips per 3-channel group, inspect array statistics and labels (pulse waveform with spectrum and heart rate, or vHRM HR/HRV/respiration channels). File lists written in Docker or another location are remapped when possible.
- **Results** — experiment folders found under the results folders: summaries, CSV reports, plots, checkpoints (with *Evaluate…* to prefill an evaluation run), and prediction analysis from `*_outputs.pickle` (or vHRM `window_results.csv`): per-window heart rate, MAE/RMSE/MAPE/Pearson/SNR, scatter and Bland-Altman plots, per-recording HR over time, waveforms and spectra.
- **Settings** — local Python executable, Docker `.env` values (mount folders, default GPU, shared memory, image), environment tests, default GPU policy (queue or parallel), maximum simultaneous jobs, and extra results/preprocessed-data folders to browse.

## Files

- Launcher state, job logs and config snapshots: `launcher_data/` (Git-ignored).
- Each job's exact config is `launcher_data/jobs/<job>/config.yaml`; Docker jobs mount it read-only into the container.
- The W&B API key is never written to disk by the launcher.
