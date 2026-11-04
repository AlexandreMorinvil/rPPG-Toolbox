"""The .npz files of a report's data/ folder as CSV files anyone can open (Excel, R, ...).

<report>/data_csv/ (and <report>/test_set/data_csv/) hold:
    README.txt                  what every file and column means
    recordings.csv              one row per recording: duration, fps, beat counts, whole-segment HRV, ...
    all_recordings/<table>.csv  each table below for every recording stacked, with recording columns in front
    <participant>/<recording>/  the tables of one recording, plus face_thumbnail.png
"""

import json
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# file name -> (description, [(column, npz field), ...]); the first column is the time axis they share.
TABLES = {
    "pulse_signal": ("rPPG pulse wave, one row per video frame", [
        ("time_s", "time"), ("bvp_raw", "bvp_raw"), ("bvp_filtered", "bvp")]),
    "heart_rate": ("HR over sliding windows; time_s is the window centre", [
        ("time_s", "hr_times"), ("hr_rppg_bpm", "hr"), ("hr_rppg_untracked_bpm", "hr_untracked"),
        ("hr_measured_bpm", "hr_gt")]),
    "hrv": ("RMSSD / SDNN over sliding windows; time_s is the window centre", [
        ("time_s", "hrv_times"), ("rmssd_rppg_ms", "rmssd"), ("sdnn_rppg_ms", "sdnn"),
        ("rmssd_ecg_ms", "rmssd_gt"), ("sdnn_ecg_ms", "sdnn_gt")]),
    "beats_rppg": ("pulse peaks detected in the rPPG signal", [("time_s", "beats_est")]),
    "rpeaks_ecg": ("R-peaks of the ECG within the video", [("time_s", "rpeaks_gt")]),
    "rr_intervals_rppg": ("rPPG inter-beat intervals; time_s is the beat closing the interval", [
        ("time_s", "ibi_est_times"), ("ibi_ms", "ibi_est"), ("valid", "ibi_est_valid")]),
    "rr_intervals_ecg": ("ECG R-R intervals; time_s is the R-peak closing the interval", [
        ("time_s", "ibi_gt_times"), ("ibi_ecg_ms", "ibi_gt"), ("ibi_rppg_at_ecg_ms", "ibi_est_at_gt")]),
}
STORED_ELSEWHERE = {"info", "thumbnail", "model_fingerprint", "polarity_flips"}
RECORDING_COLUMNS = ("recording", "participant", "split", "label", "repetition", "view")

README = """\
{title}
Exported from {source}

QUICK GUIDE: OUR CAMERA ESTIMATE vs GROUND TRUTH
================================================
We estimate the pulse from the video (rPPG = "camera"). Ground truth comes from the chest sensors: the measured
heart rate ({hr_source}) and the R-peaks of the Zephyr ECG. Each recording is one labelled segment (for example a curl)
of one participant, seen by one camera. All times are in seconds from the start of the segment. An empty cell means
"no value there".

To compare the camera with the ground truth, open the files in all_recordings/ (every recording in one file, with
participant, split, label, repetition and view columns to filter on). On each row, the two columns below are at the
same moment, so you can subtract them, plot one against the other, etc.

    What                         File                   Camera column         Ground-truth column
    Heart rate (bpm)             heart_rate.csv         hr_rppg_bpm           hr_measured_bpm
    HRV, RMSSD (ms)              hrv.csv                rmssd_rppg_ms         rmssd_ecg_ms
    HRV, SDNN (ms)               hrv.csv                sdnn_rppg_ms          sdnn_ecg_ms
    Peak-to-peak interval (ms)   rr_intervals_ecg.csv   ibi_rppg_at_ecg_ms    ibi_ecg_ms
    HRV over the whole segment   recordings.csv         segment_rmssd         segment_rmssd_gt
                                                        segment_sdnn          segment_sdnn_gt

- Heart rate: one value per second, each computed over the {hr_window} s around that time.
- HRV: one value per second, each computed over the {hrv_window} s around that time (so it starts
  {hrv_half} s into the video).
- Peak-to-peak (R-R / inter-beat interval): one row per ECG heartbeat. The camera's interval at that same moment is
  next to it.
- The raw beat times are also there if you want them (beats_rppg.csv for the camera, rpeaks_ecg.csv for the ECG),
  but the camera and the ECG don't find the same number of beats, so their rows don't line up one to one.
- split tells whether the model saw the participant: test participants were never seen; validation participants
  chose the training epoch; training participants were learned from.

You can ignore: hr_rppg_untracked_bpm, bvp_raw, the valid column, and the per-participant folders (they hold the same
tables, one recording each).


DETAILS
=======

recordings.csv
    One row per recording: id (toolbox recording id), key, participant, split, label, repetition, view, video
    path, duration_s, fs, face_found_pct (face detections that succeeded), polarity_flips, beats_gt / beats_est
    (ECG R-peaks / rPPG beats in the video), ibi_rejected_pct (rPPG intervals rejected as artefacts), segment_*
    (RMSSD / SDNN in ms over the whole segment; *_gt from the ECG), model_fingerprint (model weights and settings).

<participant>/<label><repetition>/ holds one recording; all_recordings/ holds the same tables for every recording
stacked, with the recording columns in front.

pulse_signal.csv        rPPG pulse wave, one row per video frame. bvp_raw is the model output, bvp_filtered
                        the band-pass filtered signal ({band} Hz) used for everything below.
heart_rate.csv          HR in bpm over {hr_window} s windows every {hr_step} s (time_s = window centre).
                        hr_rppg_bpm is the tracked estimate; hr_rppg_untracked_bpm the raw spectral peak of each
                        window; hr_measured_bpm the measured HR ({hr_source}) at the same time.
hrv.csv                 RMSSD and SDNN in ms over {hrv_window} s windows every {hrv_step} s (time_s = window
                        centre), from rPPG intervals and from ECG R-R intervals. Empty when a window has fewer
                        than {hrv_min} valid intervals.
beats_rppg.csv          Time of every pulse peak found in the rPPG signal.
rpeaks_ecg.csv          Time of every ECG R-peak within the video.
rr_intervals_rppg.csv   Interval between consecutive rPPG beats in ms (time_s = second beat). valid = 0 marks
                        intervals rejected as artefacts (outside {ibi_range} s, or more than {ibi_dev:.0%} away
                        from the local median); they are left out of HRV.
rr_intervals_ecg.csv    ECG R-R intervals in ms (time_s = second R-peak); empty where rejected by the same rule or
                        outside the video. ibi_rppg_at_ecg_ms is the valid rPPG interval interpolated at the same
                        time, for a beat-by-beat comparison.
face_thumbnail.png      The model input at the middle of the video.
"""


def read_npz(path):
    with np.load(path) as npz:
        return {name: npz[name] for name in npz.files}


def recording_tables(data):
    tables = {name: pd.DataFrame({column: data[field] for column, field in columns})
              for name, (_, columns) in TABLES.items() if all(field in data for _, field in columns)}
    for table in tables.values():
        table[table.select_dtypes(bool).columns] = table.select_dtypes(bool).astype(int)
    used = STORED_ELSEWHERE | {field for name in tables for _, field in TABLES[name][1]}
    for field in sorted(set(data) - used):  # fields added to the pipeline after this module was written
        tables[f"extra_{field}"] = pd.DataFrame(np.atleast_2d(data[field].T).T).add_prefix(f"{field}_")
    return tables


def write_readme(output, title, data_dir, settings, hr_source):
    text = README.format(
        title=title, source=data_dir.resolve(), hr_source=hr_source,
        band="–".join(str(f) for f in settings["signal"]["bandpass_hz"]),
        hr_window=settings["hr"]["window_sec"], hr_step=settings["hr"]["step_sec"],
        hrv_window=settings["hrv"]["window_sec"], hrv_step=settings["hrv"]["step_sec"],
        hrv_half=settings["hrv"]["window_sec"] / 2,
        hrv_min=settings["hrv"]["min_intervals"],
        ibi_range="–".join(str(s) for s in settings["ibi"]["range_sec"]),
        ibi_dev=settings["ibi"]["max_relative_deviation"])
    (output / "README.txt").write_text(text, encoding="utf-8")


def export(data_dir, output, title, settings, hr_source):
    """Write output/ (replaced if it exists) from the .npz files of data_dir. Returns the recording count."""
    files = sorted(data_dir.glob("*.npz"))
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    infos, stacked = [], {}
    for path in files:
        data = read_npz(path)
        info = json.loads(str(data["info"]))
        info["model_fingerprint"] = str(data["model_fingerprint"])
        participant = info["participant"]
        folder = output / participant / info["key"].split(f"{participant}_", 1)[-1]
        folder.mkdir(parents=True, exist_ok=True)
        columns = {"recording": info["key"], **{name: info.get(name) for name in RECORDING_COLUMNS[1:]},
                   **({"fold": info["fold"]} if info.get("fold") else {})}
        for name, table in recording_tables(data).items():
            table.to_csv(folder / f"{name}.csv", index=False)
            stacked.setdefault(name, []).append(table.assign(**columns)[[*columns, *table.columns]])
        plt.imsave(folder / "face_thumbnail.png", data["thumbnail"])
        infos.append({key: value for key, value in info.items() if key != "fold" or value})
    pd.DataFrame(infos).to_csv(output / "recordings.csv", index=False)
    (output / "all_recordings").mkdir()
    for name, tables in stacked.items():
        pd.concat(tables, ignore_index=True).to_csv(output / "all_recordings" / f"{name}.csv", index=False)
    write_readme(output, title, data_dir, settings, hr_source)
    return len(files)
