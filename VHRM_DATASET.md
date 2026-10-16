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
| `PPG_SOURCE` | Training waveform. `synthetic` uses `synthetic_ppg.csv`. `rpeak_sine` and `rpeak_gaussian` are rebuilt from the ECG R-peaks and shifted by `RPEAK_PULSE_DELAY_SEC`. `pos_pseudo` is weak supervision: a POS pulse computed from the cropped video itself (band-passed 0.7–3 Hz, Hilbert-envelope normalized, flipped to PPG polarity), as the toolbox's `USE_PSUEDO_PPG_LABEL` does for other datasets. It is in sync by construction and needs no pulse sensor, but its morphology is poor and the model learns to imitate POS. `PPG_DELAY_SEC` and the lag table still shift the measured HR/HRV/BR. vHRM rejects `USE_PSUEDO_PPG_LABEL`. |
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
- Body UV maps instead of the face crop: `vHRM_vHRM_vHRM_PHYSNET_UV128.yaml`, `vHRM_vHRM_vHRM_TSCAN_UV72.yaml`,
  `configs/infer_configs/vHRM_UNSUPERVISED_UV.yaml` (see [UV input mode](#uv-input-mode-body-uv-maps-from-vhrm-studio))

## UV input mode (body UV maps from vHRM Studio)

`<SPLIT>.DATA.UV.ENABLED: True` replaces the video with the **compact body UV map** of the vHRM Studio HDF5 files
(`packed_uv_skin_rgb`): every frame's visible skin mapped onto the SMPL-X surface, repacked into 22 upright body
charts (head, torso, arms, legs, palms, backs of the hands, fingers, feet) in one 1024x1024 square. The map is the
model input in place of the face crop, for training, testing and the unsupervised methods. It works for `vHRM` and for
the other loaders (UBFC-rPPG, PURE, iBVP RGB, UBFC-PHYS, COHFACE, LADH, SUMS), but not for SCAMPS, MMPD or BP4D+.

- **Data:** convert the dataset with vHRM Studio in *mirror* mode and set `DATA_PATH` to the converted root. The
  ground-truth files are next to the `.h5` files there, so the loaders find their labels unchanged. Each source
  (`dir/x.mp4`, or an image folder `dir/F/`) is read from `dir/x_skin_uv.h5` (or `dir/F_skin_uv.h5`), or from the
  output listed in `vhrm_manifest.json`. For vHRM, `DATA_PATH` is the folder that holds the converted experiments
  (for example `data_example`, with `EXPERIMENTS: ['baseline_vhrm']`), and each camera comes from
  `cam<N>_Camera_<View>_skin_uv.h5`.
- **Size:** the map is pooled to `PREPROCESS.RESIZE` (square). Each output pixel is the mean of the **skin texels** of
  its block: 8x8 texels at 128, 16x16 at 64, and an area-weighted ~14x14 at 72. Pooling averages many skin samples
  instead of point-sampling the video once, so a smaller map loses spatial detail but not pulse signal.
- **Model sizes:** 128 suits PhysNet, PhysFormer, RhythmFormer, iBVPNet and PhysMamba. 72 suits TS-CAN, DeepPhys,
  EfficientPhys and FactorizePhys. Pick the size with `tools/uv_preview.py`.
- **Invalid pixels:** a pixel is valid when at least `MIN_COVERAGE` of its block's region texels are skin; otherwise it
  is 0. Frames the studio could not process (`frame_status != 1`) are kept as empty frames, so the timing does not
  change.

| `UV` key | Meaning |
|---|---|
| `LAYOUT` | `packed` (the compact atlas, default) or `standard` (the SMPL-X UV layout, 74 % filled). |
| `CHARTS`, `EXCLUDE_CHARTS` | Packed charts to keep or drop (`[]` = all), e.g. `['head']` for the face/head only, `['arm_l', 'arm_r']`. |
| `MIN_COVERAGE` | Skin fraction of a block for its pixel to be valid (default 0.25). |
| `FILL` | Invalid pixels: `none` keeps 0 (**unsupervised**: the methods then average only the valid pixels of each frame); `temporal_mean` (default, **neural models**) uses the pixel's mean over its valid frames, so DiffNormalized sees no 0/value jumps when skin appears or disappears; `hold` repeats the last valid value. |
| `MIN_PIXEL_VALID_FRACTION` | Pixels valid in fewer frames than this are 0 for the whole recording. |
| `RESAMPLE_TO_FS` | Other datasets: resample the stored rows to `FS` using their timestamps. Leave it off when the loader matches labels to frames one to one (UBFC) or resamples them to the frame count. vHRM always resamples to `FS`. |
| `H5_SUFFIX` | `_skin_uv.h5`. |

```
# Which part of the map is valid at a given size and coverage threshold (writes a CSV and a coverage PNG):
python tools/uv_preview.py <folder>/cam2_Camera_Front_skin_uv.h5 --sizes 36 72 128 --output runs/uv_preview

# Unsupervised, PhysNet at 128 px, TS-CAN at 72 px; UBFC sanity check (head chart vs. the face-crop results):
python main.py --config_file configs/infer_configs/vHRM_UNSUPERVISED_UV.yaml
python main.py --config_file configs/train_configs/vHRM_vHRM_vHRM_PHYSNET_UV128.yaml
python main.py --config_file configs/train_configs/vHRM_vHRM_vHRM_TSCAN_UV72.yaml
python main.py --config_file configs/infer_configs/UBFC-rPPG_UNSUPERVISED_UV.yaml
python tools/vhrm_inventory.py --config configs/infer_configs/vHRM_UNSUPERVISED_UV.yaml   # selection dry-run
python tools/vhrm_inventory.py --data-path <converted experiments root> --uv                 # every segment
```

- **Cache:** UV caches are named `vHRMUV_...` for vHRM and `<dataset>_..._UV_...` for the other datasets. Their hash
  covers the UV settings, so they never mix with face-crop caches. Changing a `UV` key reads the `.h5` files again.
- **Speed:** reading a file costs about 14 ms per stored frame, so a 78 s camera file takes about 35 s.
- **Experiments it enables:**
  - face crop vs. `CHARTS: ['head']` vs. the whole body;
  - map size (36 / 72 / 128) and `MIN_COVERAGE`;
  - each camera view (`VIEWS`).
- **Limits:**
  - Each camera is its own recording; the cameras are not fused into one map.
  - At small sizes, some blocks straddle two neighbouring charts (the gutter between charts is 1 texel). Only skin from
    both sides is averaged.
  - Spatial position is fixed in a UV map, so do not use flip or rotation augmentation.
  - The packed layout stores no skin mask: a texel is skin when its colour is not 0. Adding `'skin_mask'` to vHRM
    Studio's `packed_uv_outputs` stores an explicit mask, which the reader then uses.
  - The UV RGB contains the face texture, so treat cached clips as identifying data.

## Cache

- **Folder:** `<CACHED_PATH>/vHRM_<W>x<H>_C<clip>_<types>_FS<fs>_<hash>/<experiment>/<participant>/`.
  The name is kept short because Windows paths are limited to 260 characters.
- **Contents:** video clips (`*_input<k>.npy`) are shared by every split and every fold that uses the same
  preprocessing, next to the mean RGB of every decoded frame (`*_rgb.npy`, the input of `pos_pseudo`).
  Labels (`*_label<k>.npy`, the PPG) and ground truth (`*_gt<k>.npy`: HR, HRV, BR, PPG valid)
  are stored in `labels_<hash>/` folders, one per label setting.
- **What triggers work:**
  - Changing `PPG_SOURCE`, a delay, the lag table or a segment's timing in `meta.json` (an offset applied
    in the preprocessing tool) only rewrites labels. Exception: the first `pos_pseudo` run decodes again the
    recordings cached before `*_rgb.npy` existed.
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
