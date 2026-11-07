# Thesis experiments

The configs are in `configs/launcher/thesis/` (launcher config tree: `launcher/thesis`). They are generated: change
`thesis/paths.yaml` (where the data is) or `thesis/splits.yaml` (who is in which split), then run
`python tools/thesis_configs.py`. Do not edit the generated files: the next generation overwrites them.

| File | Role |
|---|---|
| `thesis/paths.yaml` | Dataset folders (videos and vHRM Studio conversions), cache and run folders, device, wandb |
| `thesis/splits.yaml` | Train / validation / test people (lab) and subjects (UBFC-rPPG, PURE, COHFACE) |
| `tools/thesis_configs.py` | Writes the 188 configs; `--check` checks the inputs; `--find-paths` fills the empty entries of `paths.yaml` |
| `tools/paired_comparison.py` | Participant-level paired comparison of runs (bootstrap CI, equivalence) |
| `tools/vhrm_window_features.py` | Movement and skin visibility of every evaluation window |

## Before running

1. Plug in the drives, then `python tools/thesis_configs.py --find-paths`. It fills the empty entries of `paths.yaml`
   (it searches every drive, about 10 s per drive) and keeps the others, even when their folder is missing here: the
   batches run on different computers, so `paths.yaml` holds the folders of the computer that runs each config. The
   cache and run folders are per batch; `--check` warns about folders this computer does not have.
2. `uv.lab.JZC` must point to the folder of the JZC conversions (`jzc01/`, `jzc02/`, ... with
   `labelled_segments/*/cam*_skin_uv.h5`). Until then the UV lab configs show an error in the launcher.
3. `python tools/thesis_configs.py` writes the configs and lists what is missing. `--check` lists it without writing.

## The runs

Inputs: `face` = face crop, `uvhead` = UV map of the head chart, `uvbody` = every chart except the head,
`uvfull` = the whole body. Models: PhysNet, FactorizePhys, RhythmFormer (and POS, which is not trained).

| Folder | Runs | What |
|---|---|---|
| `batch1_literature/<UBFC, PURE, COHFACE>/` | 27 + 9 POS | One training per dataset, model and input (face, uvhead, uvfull) |
| `batch2_lab_front/<baseline, JZC, RM>/` | 36 + 12 POS | Front camera, one training per experiment, model and input (face, uvhead, uvbody, uvfull) |
| `batch3_lab_back/<baseline, JZC>/finetune/` | 24 | The experiment's batch 2 model fine-tuned on its back-left + back-right videos, tested on the back views |
| `batch3_lab_back/<baseline, JZC>/zeroshot/` | 24 | The experiment's batch 2 model tested on the back views as it is (the reference for the fine-tuning) |
| `batch3_lab_back/<baseline, JZC>/` | 8 POS | POS on the back views (batch 2 has the front) |
| `batch2_lab_front/RM/from_<baseline, JZC>/` | 24 tests | No training on RM: the batch 2 model of baseline or JZC tested on the RM test people, front camera |
| `batch3_lab_back/RM/from_<baseline, JZC>/` | 24 tests | No training on RM: the fine-tuned batch 3 model of baseline or JZC tested on the RM test people, back views |

Each experiment uses its own people of the split (the same person is in the same split in every experiment). The
JZC UV configs wait for `uv.lab.JZC`; everything else in batches 1 and 2 can run now.

The `RM/from_*` tests replace the RM trainings (`batch2_lab_front/RM/*.yaml`, kept but not used: their results are
poor). They use the 6 RM test people, Squat segments only, who are also test people in baseline and JZC, so the models
never trained on them. Their report covers the same 6 people with every label (like `test_set/` of the RM trainings)
and is written in the test's own run folder. To launch only them, tick the `from_*` folders: ticking `RM/` also ticks
the trainings.

Order: batch 1 and batch 2 in any order. When batch 2 is done, run `python tools/thesis_configs.py` again: it writes
each batch 2 run's tested checkpoint into the batch 3 configs of the same experiment (`MODEL.INIT_WEIGHTS`,
`INFERENCE.MODEL_PATH`) and into `batch2_lab_front/RM/from_*`. Then batch 3, then the script again for
`batch3_lab_back/RM/from_*`.

To launch a whole folder at once, use the launcher's **New run → Launch a batch…** and tick the folder (e.g.
`thesis/batch1_literature`): one round of pre-launch checks for all its configs, then one job per config, run one after
another (see `launcher/README.md`, *Batches*).

Runs write to `<runs>/<batch folder>/<run name>` (`runs` of the batch in `paths.yaml`). Each model and input preprocesses its
data once: batch 2 and batch 3 share the lab caches (same crop modes and sizes), and the first lab run of each model
and input decodes the videos (the face crop is the slow one).

## Settings shared by every run

- Neural models: the toolbox settings of each model (PhysNet 72 px DiffNormalized clips of 128 frames; FactorizePhys
  72 px Raw, 160 frames; RhythmFormer 128 px Standardized, 160 frames), 10 to 20 epochs (early stopping after 4
  epochs without a lower validation loss, not before epoch 10), best epoch on the validation loss.
  Batch 3 fine-tuning: 10 epochs (the same minimum, so early stopping never ends it) at a tenth of the learning rate.
- RM: Squat segments only, in training, validation, test and POS (`VHRM.LABELS: [Squat]`, from `LAB_LABELS` in
  `tools/thesis_configs.py`). The baseline sessions of the RM people (`baseline/RMNN_baseline_*`) keep every label.
- Face crop: YOLO5Face, re-detected every second, 1.5x box, the same on every view. Lab data: faces are searched in
  the central 70 % of the frame width only (`VHRM.FACE_SEARCH_WIDTH: 0.7`; the participant's head stays inside it in
  97-100 % of the frames of every view); a clip or window needs a face in at least half of its frames
  (`VHRM.MIN_FACE_FRACTION: 0.5`). Training and validation drop the other clips; tests keep them and count their
  windows as "no reading" (`coverage` in the summaries).
- Back cameras (`back-left=person_face`, `back-right=person_face`): a person detector finds the largest person (the
  participant) and the face is searched inside that person's box only; otherwise no face, as on the front camera.
  The experimenter, who faces those cameras, is ignored unless their face falls inside the participant's box (see the
  plan, open point 4). Every lab config lists these modes, so batches 2 and 3 share their caches.
- UV map: packed layout, a pixel is valid with 25 % skin coverage; neural models fill invalid pixels with their
  temporal mean, POS averages the valid pixels only. UV windows always have a reading.
- Lab data: first 120 s of each recording (`VHRM.MAX_DURATION_SEC`), Zephyr HR as the ground truth.
- Evaluation: HR by FFT in 30 s windows for every dataset (the literature usually scores whole videos; the
  `window_results.csv` of a run is enough to recompute that).
- Unsupervised methods (POS) on UBFC-rPPG, PURE and COHFACE: the incomplete window at the end of a recording is scored
  only if it lasts at least 10 s (`MIN_TAIL_SECONDS` in `unsupervised_methods/unsupervised_predictor.py`). COHFACE
  needed this: its recordings last about 60 s at 20 fps, which leaves tails of a few frames. A 9-frame tail crashed the
  evaluation, and a 10-16 frame tail can only give 75 or 150 bpm.
  Not applied to the runs of 2026-11-05: the POS runs on UBFC-rPPG and PURE and every neural run on these datasets still
  count their shorter tails (4-9 s). Their `window_results.csv` has `start_seconds` and `end_seconds`, so dropping those
  rows gives the same rule without a new run. State which rule each reported number uses. Lab data (vHRM) is not
  affected.

## Analysis

Every run writes `window_results.csv` (one row per window, with `has_reading`). Compare runs per person:

```
# Batch 1, PhysNet on UBFC-rPPG: is the UV map equivalent to the face crop (margin 1 bpm)?
python tools/paired_comparison.py --reference face --margin 1.0 \
    --run face=D:/Run/thesis/batch1_literature/b1_UBFC_physnet_face \
    --run uvhead=D:/Run/thesis/batch1_literature/b1_UBFC_physnet_uvhead \
    --run uvfull=D:/Run/thesis/batch1_literature/b1_UBFC_physnet_uvfull \
    --output D:/Run/thesis/analysis/b1_UBFC_physnet

# POS: every participant (--split all), or the test people only (default)
python tools/paired_comparison.py --split all --reference face --run face=.../b1_UBFC_pos_face --run uvfull=...

# Batch 2, one experiment; batch 3, per view (coverage shows how often the face crop had a reading)
python tools/paired_comparison.py --reference face --run face=.../batch2_lab_front/b2_jzc_rhythmformer_face \
    --run uvhead=.../b2_jzc_rhythmformer_uvhead --run uvbody=.../b2_jzc_rhythmformer_uvbody --run uvfull=...
python tools/paired_comparison.py --group-by view --reference face --run face=.../b3_baseline_ft_rhythmformer_face ...

# Movement and face visibility: features of each window, then the comparison in bins of a feature
python tools/vhrm_window_features.py --windows C:/Run/thesis/batch2_lab_front/b2_rm_physnet_face \
    --output C:/Run/thesis/analysis/features_b2_rm.csv
python tools/paired_comparison.py --features C:/Run/thesis/analysis/features_b2_rm.csv \
    --bin-by body_speed_deg_s --bins 3 --reference face --run face=... --run uvfull=...
```

`--common-windows` scores only the windows every run read (the same windows for the face crop and the UV map).
Verdicts: `equivalent` (90 % interval inside the margin), `better` / `worse` (95 % interval below / above 0),
`inconclusive`.
