# How the analysis report is computed

The report measures how well a checkpoint estimates **heart rate (HR)**, **beat-to-beat R-R intervals (RR)** and
**heart-rate variability (HRV)** from video, against the chest sensors of the vHRM dataset. It is the method of
`jzc_rppg_analysis` (the JZC curl study), built into the toolbox and extended to every vHRM experiment, movement
and camera view. An unsupervised method (POS, CHROM, …) is reported the same way: only how the pulse signal is
obtained differs (section 4, step 1).

## 1. Recordings

- **Scope.** The experiments and views of the config's test data (its unsupervised data for an unsupervised
  method), **every participant** and **every segment label** (resting, curl, recovery, …), unless `REPORT.LABELS` /
  `REPORT.PARTICIPANTS` narrow it. One recording is one labelled segment seen by one camera, as everywhere in the
  toolbox.
- **Repetitions.** A label that occurs several times for a participant is numbered in segment order: `curl 1`,
  `curl 2`. File names follow: `jzc17_curl1`, `mosaics/hr_curl1.png`.
- **Splits.** For a checkpoint trained on vHRM, every participant is tagged with its split of the run, read from the
  config's training, validation and test selections:
  - **test**: never seen. These are the only fair numbers; `test_set/` repeats the whole report on them alone.
  - **validation**: not trained on, but they chose the epoch.
  - **training**: their recordings were learned from.
  - **unused**: in the experiments but in none of the run's selections.

  For a cross-validation fold, these are the fold's participants. When the config did not train on vHRM, the
  training participants are unknown and no split is shown. An unsupervised method learns nothing from the data:
  no split is shown and every participant is unseen.
- **Timing.** Frame *i* is at `signal_anchor_datetime_utc + start_sec + i / FS`, shifted by the run's sync
  correction (`PPG_DELAY_SEC`, lag table), exactly as for the training labels (`VHRM_DATASET.md`).

## 2. Model input

The report reads the **toolbox's preprocessed cache**: the exact clips this checkpoint was trained and tested on
(same crop, size, input transform and layout). Recordings missing from the cache are preprocessed first, like any
test split, and stay cached. The clips of a recording are contiguous slices of one recording-wide transform, so
they are read back as one continuous video. The last incomplete clip of a recording is not cached, so up to one clip
length (5.3 s at 160 frames) is missing at the end of each segment. Unsupervised runs usually cache whole recordings
(`DO_CHUNK: false`), which then lose nothing.

The original JZC study decoded the videos itself (smoothed square face box). On the RhythmFormer fold-2 test
participants, switching between the two crops changed the HR MAE from 11.2 to 10.9 bpm.

## 3. Ground truth

| Quantity | Source | How it is compared |
|---|---|---|
| **HR** | the measured HR of `HR_SOURCE` (Zephyr 1 Hz summary by default), sampled at every frame by the loader (gaps > 3 s left empty) | taken at the centre of each rPPG HR window |
| **RR** | ECG R-peaks (`synthetic_ppg/synthetic_ppg_rr.csv`, from the Zephyr ECG) | R-R interval = time between consecutive R-peaks |
| **HRV** | the same R-R intervals | RMSSD and SDNN computed **with exactly the same windows, artefact rule and code as the rPPG beats** |

The full-session R-peak file of the participant is preferred, so beats up to 5 s outside the segment form its
first and last intervals. Recordings without an ECG (e.g. HH) get HR results only. The Zephyr `HRV` column is not
used: it behaves like an SDNN over the last ~300 beats, so during a short segment it mostly describes what came
before.

## 4. Method

```
cached clips ─► network (clips of the training length, 75 % overlap) ─► phase-aligned overlap-add pulse signal
             ─► detrend + 0.7–3.5 Hz band-pass ─┬─► Welch PSD per 8 s window ─► tracked HR
                                                └─► HR-guided peak detection ─► beats ─► RR intervals ─► RMSSD / SDNN
```

1. **Inference.** The network is built by its own trainer from the config and called as in its test (input layout,
   appended last frame, frame-wise models). Clips of the training clip length start every quarter clip (75 %
   overlap; the last clip is anchored to the end). Each predicted clip is z-scored, sign-flipped if it
   anti-correlates with the signal already reconstructed where they overlap, then Hann-weighted (floor 10⁻³) and
   added. Models trained on DiffNormalized labels output a derivative, which is integrated.
   An **unsupervised method** runs once over the whole recording (the cached clips joined), with the toolbox's own
   implementation, as in its unsupervised run: on UV maps only the skin pixels of each frame are averaged. No clips
   or overlap-add are involved. POS follows the reflected light, which falls when the blood volume rises:
   its output is negated so that its peaks are pulse peaks, as in the PPG labels (the loader's POS pseudo-labels are
   flipped the same way). Other methods are used with the sign of their implementation.
2. **Pulse signal.** Linear detrend, then a zero-phase 4th-order Butterworth band-pass of 0.70–3.50 Hz (42–210 bpm).
3. **Heart rate.** Dominant frequency of the pulse signal in **8 s windows every 1 s** (Welch PSD, ≤ 256-sample
   segments, 2048-point FFT). The peak is followed with the toolbox's **causal Viterbi tracker**
   (`evaluation/signal_analysis.track_hr`): changes faster than 8 bpm/s are penalised after an acquisition phase. The
   untracked peak is kept (grey line in `participants/*/hr.png`, `hr_mae_untracked_bpm`).
4. **Beats.** Pulse peaks with a prominence ≥ 0.1 × the signal's SD, accepted in order of prominence; a candidate
   closer than **0.6 × the expected period** (60 / tracked HR) to an accepted beat is dropped. Peak times are refined
   by parabolic interpolation (30 FPS alone resolves 33 ms).
5. **Intervals.** RR (rPPG: inter-beat interval, IBI) = time between consecutive beats, stamped at the second beat.
   **Artefact rule** (same for rPPG and ECG): rejected outside 0.3–2.0 s or more than **25 %** away from the running
   median of 11 neighbours. Each valid ECG interval is compared with the rPPG tachogram (valid intervals)
   interpolated at the R-peak time, where the two neighbouring rPPG intervals are less than 2 s apart.
6. **HRV.** **RMSSD** (short-term vagal HRV) and **SDNN** over **30 s windows every 1 s** from the valid intervals
   (≥ 15 per window; a successive difference needs two consecutive valid intervals), and over the whole segment.

Beat-timing noise adds to the successive differences, so **RMSSD from rPPG is biased upward even when every beat is
found** (a synthetic pulse with a true RMSSD of 19 ms gave 29 ms). SDNN is much less affected.

## 5. Metrics

For each metric, *error = estimate − ground truth*:

| Metric | Meaning |
|---|---|
| **MAE** | mean absolute error, the main number |
| **RMSE** | root mean square error, penalises large errors more |
| **Bias** | mean error: positive = rPPG overestimates |
| **r** | Pearson correlation: does the estimate follow the ground truth's changes? |
| **% within 5 bpm** | share of HR windows with \|error\| ≤ 5 bpm |
| **Bland-Altman** (`overview.png`) | bias line and 95 % limits of agreement (bias ± 1.96 SD) |

A sample is an HR window, an HRV window or an ECG beat. Pooled values (`global_metrics.csv`) use every sample of
the group, so longer recordings weigh more; the **median recording MAE** weights each recording equally. Groups:
all recordings, each label, each label × repetition, each view (if several), each split and each label × split.
`label_split_mae.csv` is the MAE matrix label × split. Consecutive windows overlap and are not independent, so no
p-values are computed. Heart-rate meters are usually accepted within ±10 % or ±5 bpm, whichever is greater (ANSI/AAMI
EC13).

## 6. Outputs

```
report/<checkpoint>/         report/<method>/ for an unsupervised method (POS, …)
├─ REPORT.md                 the numbers of this report
├─ METHOD.md                 this file
├─ summary/
│  ├─ global_metrics.csv          pooled metrics per group
│  ├─ label_split_mae.csv         MAE per label × split
│  ├─ per_recording_metrics.csv   one row per recording, every metric
│  ├─ overview.png                estimate vs ground truth + Bland-Altman (HR, RMSSD, RR), coloured by label
│  ├─ participant_errors.png      MAE heatmaps: participant × recording
│  └─ face_crops.png              the model input at mid-segment, per recording
├─ mosaics/{hr,hrv,rr}_<label><n>.png   every participant on one page, per recording condition
├─ participants/<id>/{hr,hrv,rr,pulse}.png, metrics.csv
├─ data/<recording>.npz      every time series behind the figures + the raw model output
├─ data_csv/                 the same data as CSV files, with a README for non-programmers
├─ run_config.json           checkpoint, cache, selection, splits, settings, software versions
└─ test_set/                 everything above on the test participants only (when the run has some)
```

In the time-series figures, **black is the ground truth and blue is the rPPG estimate**. Times are seconds since
the start of the segment. The model output of each recording is kept in `data/`: generating the report again with
the same checkpoint (or method) and cache skips the network.

## 7. Limitations

- **Ground-truth HR is smoothed by the sensor.** The Zephyr 1 Hz HR averages several beats and lags fast changes.
  The ECG R-R intervals are the beat-exact reference.
- **Sync.** Video start times have 1 s resolution; the per-participant offset was estimated from the video. A few
  hundred ms barely affect HR or HRV but shift the RR tachograms.
- **Short segments.** Ultra-short RMSSD (30 s windows) is noisier than the usual 5-minute value, for the ECG too.
- **Few unseen participants.** A single training has 2–4 test participants; a cross-validation tests everyone once.
- **Offline processing.** Zero-phase filtering and overlapping clips use future samples.
