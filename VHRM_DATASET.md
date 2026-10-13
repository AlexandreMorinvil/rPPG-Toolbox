# Training on the vHRM lab dataset

`DATASET: vHRM` works for TRAIN, VALID, TEST and UNSUPERVISED.

## Expected data layout

`DATA_PATH` is the folder that contains the experiments (for example `F:/Datasets preparation`):

```
<DATA_PATH>/<experiment>/<participant>/
    zephyr_1.csv, polar_1.csv, synthetic_ppg/*.csv            full-session signals (preferred)
    labelled_segments/<NNNN>_<Label>/
        meta.json, cam<N>_Camera_<View>.mp4, zephyr_1.csv, synthetic_ppg/*.csv
```

- **Recordings:** one recording is one segment seen from one camera. Its id is
  `experiment--participant--segment--label--view`, for example `jzc--jzc05--0001--curl--front`.
- **Names:** names are matched case-insensitively. Any character other than a letter or digit becomes `-`
  (`Back-left` → `back-left`, `jzc02_baseline` → `jzc02-baseline`).
- **Sessions:** a participant is a participant folder. `jzc02` and `jzc02_baseline` are therefore different participants.
- **Time alignment:** frame *i* is placed at `signal_anchor_datetime_utc + start_sec + i / FS`.
  All signals are sampled at those absolute times.

## Config keys (`<SPLIT>.DATA.VHRM`)

| Key | Meaning |
|---|---|
| `EXPERIMENTS`, `PARTICIPANTS`, `EXCLUDE_PARTICIPANTS`, `LABELS`, `EXCLUDE_LABELS` | Which data a split uses. An empty list means all. |
| `LABEL_ALIASES`, `VIEW_ALIASES` | `old=new` renames. The default `recover=recovery` merges jzc14's spelling. |
| `VIEWS` | `front`, `back-left`, `back-right`, or any combination. |
| `CROP_MODE_BY_VIEW` | Image region per view. `face` keeps the last face box when detection fails. `face_or_full` uses the full frame when no face is found. `full` uses the whole frame. `roi:x,y,w,h` is a rectangle in pixels, or in fractions when every value is ≤ 1. |
| `PPG_SOURCE` | Training waveform. `synthetic` uses `synthetic_ppg.csv`. `rpeak_sine` and `rpeak_gaussian` are rebuilt from the ECG R-peaks and shifted by `RPEAK_PULSE_DELAY_SEC`. |
| `PPG_DELAY_SEC`, `LAG_TABLE_PATH` | Video/signal sync correction: one global value, plus optional per-segment values from `tools/vhrm_sync_check.py`. A positive value shifts the signals later. These are added to the timing in `meta.json`: leave them at 0 / empty for participants whose offset was applied in the preprocessing tool (PPG alignment step, `video_signal_offset_sec` in `meta.json`), or the correction is counted twice. |
| `HR_SOURCE` | Evaluation ground truth: `zephyr` (default), `polar` or `synthetic_hr`. |
| `REQUIRE_PPG` | Drops clips without a valid training waveform. Use True for TRAIN/VALID and False for TEST (e.g. HH, which has no ECG). |

Other new keys:
- `TRAIN.DATA_LOADER_WORKERS` sets the batch-loader worker count. The default is 16; use 0–4 on Windows.
- `MODEL.INIT_WEIGHTS` starts training from released weights for fine-tuning.

## Workflow

```
# 1. What is in the dataset, and what does a config select? (No videos are opened.)
python tools/vhrm_inventory.py --data-path "F:/Datasets preparation"
python tools/vhrm_inventory.py --config configs/train_configs/vHRM_vHRM_vHRM_PHYSNET_FRONT.yaml

# 2. Check the video/label sync on calm front-view segments (writes a lag table).
#    Preferred: measure and apply the offset in the preprocessing tool (PPG alignment step), which
#    writes the corrected timing into each segment's meta.json; the cached labels follow it.
python tools/vhrm_sync_check.py --data-path "F:/Datasets preparation" --experiments JZC baseline \
    --labels Resting Baseline_part1 --views front --output runs/sync/front.csv

# 3. Train / test.
python main.py --config_file configs/train_configs/vHRM_vHRM_vHRM_PHYSNET_FRONT.yaml

# 4. Leave-one-participant-out cross-validation from the same config.
python tools/vhrm_cross_validation.py --config configs/train_configs/vHRM_vHRM_vHRM_PHYSNET_FRONT.yaml \
    --folds loso --valid-count 2 --output-root runs/cv_physnet_front
```

Other example configs:
- `vHRM_vHRM_vHRM_DEEPPHYS_FRONT.yaml`
- `vHRM_vHRM_vHRM_PHYSNET_FRONT_FINETUNE_PURE.yaml`
- `configs/infer_configs/vHRM_UNSUPERVISED_FRONT.yaml` (POS, CHROM, …)

## Cache

- **Folder:** `<CACHED_PATH>/vHRM_<W>x<H>_C<clip>_<types>_FS<fs>_<hash>/<experiment>/<participant>/`.
  The name is kept short because Windows paths are limited to 260 characters.
- **Contents:** video clips (`*_input<k>.npy`) are shared by every split and every fold that uses the same
  preprocessing. Labels (`*_label<k>.npy`, the PPG) and ground truth (`*_gt<k>.npy`: HR, HRV, BR, PPG valid)
  are stored in `labels_<hash>/` folders, one per label setting.
- **What triggers work:**
  - Changing `PPG_SOURCE`, a delay, the lag table or a segment's timing in `meta.json` (an offset applied
    in the preprocessing tool) only rewrites labels.
  - Changing the image region, the size or the input transform decodes the videos again.
  - Only new or changed videos are decoded (`preprocessing_manifest.json`).
- **File lists:** `DataFileLists/<cache>_<begin>_<end>_vHRM<hash>.csv`. The hash covers the selection and label
  settings, so splits never overwrite each other's lists.

## Evaluation output (`<LOG.PATH>/<cache>/saved_test_outputs/`)

- `window_results.csv` holds one row per evaluation window:
  - predicted HR vs the measured HR (`HR_SOURCE`, the primary metric);
  - predicted HR vs the HR of the PPG label (`label_hr_bpm`), comparable with the usual toolbox numbers.
- Summaries: `overall_summary.csv`, `summary.json`, `by_experiment` / `by_movement` / `by_participant` / `by_view` CSVs, plus plots.
- `*_outputs.pickle` keeps the per-frame prediction and PPG label of every recording, plus the measured
  ground truth (`gt`: HR, HRV, respiration, PPG-valid mask). Unsupervised runs write one per method in
  `saved_outputs/<method>/`. Browse them in the launcher (Results → Predictions).
- `highlights/` holds a few PNGs (worst, best, median, and the worst of each movement × view):
  HR over time, a BVP excerpt with detected beats, and a spectrogram (`VISUALIZE.HIGHLIGHTS`).
- During training, `<LOG.PATH>/<train cache>/saved_valid_outputs/` keeps the validation predictions of
  the best epoch and a `probe_history.npz` with a few probe recordings every
  `VISUALIZE.PROBE_EVERY_N_EPOCHS` epochs (see `VISUALIZE` in `config.py`).
- The cross-validation runner merges every fold into `<output-root>/cv_summary/`.

## Caveats

- **Sync:** video start times come from file names with 1 s resolution, and segment cuts snap to keyframes.
  Check the sync before trusting waveform-level training.
- **Synthetic PPG:** beat timing comes from the ECG, but the shape and the pulse delay are modelled, not measured.
- **Back views:** without visible skin, expect poor results; treat them as experimental.
- **Breaking change:** the previous test-only vHRM layout (`separated_signals/final_signal.csv`, participants
  directly under `DATA_PATH`) is no longer read.
