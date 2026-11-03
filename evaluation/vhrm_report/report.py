"""REPORT.md of one report: the numbers, regenerated with every run (the method is in METHOD.md)."""

from datetime import datetime

import numpy as np

from .data import label_name
from .evaluate import METRICS, SPLIT_NAMES


def _cell(value, digits=1):
    if isinstance(value, str):
        return value
    if value is None or not np.isfinite(value):
        return "–"
    return f"{value:.{digits}f}"


def _table(header, rows, numeric=True):
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join("---:" if i and numeric else "---" for i in range(len(header))) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def _scope_note(info, recordings, scope):
    """Which participants the checkpoint saw, and the link to the other set of results."""
    if scope == "cv":
        return ["", "Cross-validation: every participant is a test participant of one fold, and each recording is "
                    "analysed by the checkpoint of that fold, which never saw the participant. These are the test "
                    "participants of each fold's report (`" + info["checkpoint"] + "`)."]
    if scope == "test":
        return ["", f"Only the recordings of the test participants ({', '.join(info['test_participants'])}): the "
                    "checkpoint never saw them, neither in training nor to choose the epoch (validation). The same "
                    "analysis on every recording: [../REPORT.md](../REPORT.md)."]
    if not info["splits"]:
        return ["", "The training participants of this checkpoint are not known (the config did not train on vHRM), "
                    "so the recordings are not split into seen and unseen participants."]
    note = ("The checkpoint was trained on vHRM: only the *test participants* rows are measured on people it never "
            "saw. Validation participants chose the training epoch; training participants were learned from.")
    if info["test_participants"]:
        note += " Every table and figure on the test participants alone: [test_set/REPORT.md](test_set/REPORT.md)."
    return ["", note]


def _metric_tables(global_metrics, within_bpm):
    within_column = f"within_{within_bpm:g}bpm_pct"
    lines = []
    for metric, (label, unit, _, _) in METRICS.items():
        rows = global_metrics[global_metrics.metric_key == metric]
        if not len(rows):
            continue
        header = ["Group", "Recordings", "Samples", "MAE", "RMSE", "Bias", "r", "Median recording MAE"]
        if metric == "hr":
            header.append(f"% within {within_bpm:g} bpm")
        body = []
        for row in rows.itertuples():
            cells = [row.group, str(row.recordings), str(row.samples), _cell(row.mae), _cell(row.rmse),
                     _cell(row.bias), _cell(row.pearson, 2), _cell(row.median_recording_mae)]
            if metric == "hr":
                cells.append(_cell(getattr(row, within_column), 0))
            body.append(cells)
        lines += [f"### {label} ({unit})", "", _table(header, body), ""]
    return lines


def _matrix(matrix, splits):
    if not len(splits):
        return []
    lines = ["## MAE by movement and split", "",
             "Pooled MAE of each segment label (rows) for each group of participants (columns).", ""]
    for metric, (label, unit, _, _) in METRICS.items():
        rows = matrix[matrix.metric_key == metric]
        header = [f"{label} ({unit})", "All", *[SPLIT_NAMES[split] for split in splits]]
        body = [["All recordings" if row.label == "all" else label_name(row.label),
                 _cell(getattr(row, "all")), *[_cell(getattr(row, split)) for split in splits]]
                for row in rows.itertuples()]
        lines += [_table(header, body), ""]
    return lines


def write_report(path, info, global_metrics, matrix, recordings, failures, within_bpm, scope="all"):
    test_only = scope == "test"
    by_fold = "fold" in recordings
    has_splits = by_fold or bool(recordings.split.astype(bool).any())
    splits = [split for split in ("test", "validation", "training", "unused") if split in set(recordings.split)]
    labels = recordings.groupby("label").size().to_dict()
    recording_rows = [[row.participant, *([str(row.fold if by_fold else row.split)] if has_splits else []), row.condition,
                       _cell(row.duration_s, 0), _cell(row.face_found_pct, 0),
                       _cell(row.hr_mean_gt_bpm, 0), _cell(row.hr_mae_bpm), _cell(row.hr_bias_bpm),
                       _cell(row.hr_pearson, 2),
                       f"{_cell(row.rmssd_segment_gt_ms, 0)} / {_cell(row.rmssd_segment_est_ms, 0)}",
                       _cell(row.rmssd_mae_ms), _cell(row.ibi_mae_ms, 0), _cell(row.ibi_rejected_pct, 0)]
                      for row in recordings.itertuples()]
    lines = [
        f"# rPPG analysis report · {info['title']}{' · test participants only' if test_only else ''}",
        "",
        f"Generated {datetime.now():%Y-%m-%d %H:%M} by `evaluation/vhrm_report`. How these numbers are computed "
        "and what they mean: [METHOD.md](" + ("../" if test_only else "") + "METHOD.md).",
        "",
        _table(["Setting", "Value"], [
            ["Model", info["model"]],
            ["Checkpoint", f"`{info['checkpoint']}`"],
            ["Data", f"`{info['data_path']}` · experiments {', '.join(info['experiments']) or 'all'} · "
                     f"views {', '.join(info['views']) or 'all'}"],
            ["Input", f"`{info['cache']}` ({info['input']})"],
            ["Recordings", f"{len(recordings)} from {recordings.participant.nunique()} participants ("
                           + ", ".join(f"{label_name(label)}: {count}" for label, count in labels.items()) + ")"],
            ["Splits", "; ".join(f"{SPLIT_NAMES[split]}: {', '.join(members)}"
                                 for split, members in info["split_members"].items()) or "unknown"],
            ["Failed recordings", ", ".join(failures) if failures else "none"],
        ], numeric=False),
        "",
        "## Global results",
        "",
        "Pooled over every HR window (8 s, 1 s step), HRV window (30 s, 1 s step) or ECG beat of the group. "
        "MAE, RMSE and bias (estimate − ground truth) are in the metric's unit; r is Pearson's correlation; "
        "the median recording MAE weights every recording equally.",
        *_scope_note(info, recordings, scope),
        "",
        *_metric_tables(global_metrics, within_bpm),
        *(_matrix(matrix, splits if has_splits else []) if scope == "all" else []),
        "## Per recording",
        "",
        "HR in bpm, RMSSD and IBI in ms. *RMSSD segment* is computed once over the whole segment "
        "(ground truth / rPPG). *IBIs rejected* is the share of rPPG inter-beat intervals removed by the artefact rule.",
        "",
        _table(["Participant", *([("Fold" if by_fold else "Split")] if has_splits else []), "Recording",
                "Duration (s)", "Face found %",
                "Mean GT HR", "HR MAE", "HR bias", "HR r",
                "RMSSD segment GT / rPPG", "RMSSD MAE", "IBI MAE", "IBIs rejected %"], recording_rows),
        "",
        "## Files",
        "",
        "- `summary/overview.png` – estimate vs ground truth and Bland-Altman, pooled (HR, RMSSD, IBI)",
        "- `summary/participant_errors.png` – MAE per participant and recording",
        "- `summary/face_crops.png` – the model input at mid-segment, per recording",
        "- `summary/global_metrics.csv`, `summary/per_recording_metrics.csv`, `summary/label_split_mae.csv` – the "
        "tables above, all columns",
        "- `mosaics/{hr,hrv,rr}_<label><repetition>.png` – every participant on one page, per metric and recording",
        "- `participants/<id>/{hr,hrv,rr,pulse}.png` and `metrics.csv` – one participant",
        "- `data/<recording>.npz` – every time series behind the figures (and the raw model output)"
        + (", copies of the test recordings of `../data/`" if test_only else "")
        + (", copied from the test set of each fold's report" if scope == "cv" else ""),
        "- `data_csv/` – the same data as CSV files, with a README",
        "- `run_config.json` – the exact settings and software versions of this report",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
