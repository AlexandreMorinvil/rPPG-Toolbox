"""Agreement metrics between the rPPG estimates and the ground truth, per recording and pooled per group."""

import numpy as np
import pandas as pd

from .data import SPLIT_ORDER, label_name

# metric key -> (label, unit, estimate field, ground-truth field) in a recording result
METRICS = {
    "hr": ("Heart rate", "bpm", "hr", "hr_gt"),
    "rmssd": ("HRV (RMSSD)", "ms", "rmssd", "rmssd_gt"),
    "sdnn": ("HRV (SDNN)", "ms", "sdnn", "sdnn_gt"),
    "ibi": ("RR / inter-beat interval", "ms", "ibi_est_at_gt", "ibi_gt"),
}
SPLIT_NAMES = {"test": "test participants", "validation": "validation participants",
               "training": "training participants", "unused": "participants outside the run's splits"}


def paired(estimate, truth, within=None):
    estimate = np.asarray(estimate, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    keep = np.isfinite(estimate) & np.isfinite(truth)
    error = estimate[keep] - truth[keep]
    result = {"n": int(keep.sum()), "mae": np.nan, "rmse": np.nan, "bias": np.nan, "pearson": np.nan}
    if within is not None:
        result["within_pct"] = np.nan
    if error.size:
        result.update(mae=float(np.mean(np.abs(error))), rmse=float(np.sqrt(np.mean(error ** 2))),
                      bias=float(np.mean(error)))
        if within is not None:
            result["within_pct"] = float(np.mean(np.abs(error) <= within) * 100)
    if error.size >= 3 and np.std(estimate[keep]) > 0 and np.std(truth[keep]) > 0:
        result["pearson"] = float(np.corrcoef(estimate[keep], truth[keep])[0, 1])
    return result


def metric_pairs(result, metric):
    _, _, estimate, truth = METRICS[metric]
    return result["series"][estimate], result["series"][truth]


def label_order(results):
    """Labels in protocol order (their first segment), e.g. resting, curl, recovery."""
    first = {}
    for result in results:
        first[result["label"]] = min(first.get(result["label"], 10 ** 9), int(result["segment_index"] or 0))
    return sorted(first, key=lambda label: (first[label], label))


def conditions(results):
    """(label, repetition, view) of the results, in protocol order."""
    order = {label: index for index, label in enumerate(label_order(results))}
    found = {(r["label"], r["repetition"], r["view"]) for r in results}
    return sorted(found, key=lambda c: (order[c[0]], c[1], c[2]))


def split_order(results):
    present = {result.get("split") for result in results}
    return [split for split in (*SPLIT_ORDER, "unused") if split in present]


def recording_order(results):
    order = {condition: index for index, condition in enumerate(conditions(results))}
    return sorted(results, key=lambda r: (r["experiment"], r["participant"], order[(r["label"], r["repetition"],
                                                                                     r["view"])]))


def recording_row(result, within_bpm):
    series = result["series"]
    row = {
        "experiment": result["experiment"],
        "participant": result["participant"],
        "split": result.get("split") or "",
        **({"fold": result["fold"]} if result.get("fold") else {}),
        "label": result["label"],
        "repetition": result["repetition"],
        "view": result["view"],
        "condition": result["condition"],
        "recording": result["key"],
        "recording_id": result["id"],
        "duration_s": round(result["duration_s"], 1),
        "face_found_pct": result["face_found_pct"],
        "polarity_flips": result["polarity_flips"],
    }
    for metric, (_, unit, _, _) in METRICS.items():
        stats = paired(*metric_pairs(result, metric), within=within_bpm if metric == "hr" else None)
        row[f"{metric}_n"] = stats["n"]
        row[f"{metric}_mae_{unit}"] = stats["mae"]
        row[f"{metric}_rmse_{unit}"] = stats["rmse"]
        row[f"{metric}_bias_{unit}"] = stats["bias"]
        row[f"{metric}_pearson"] = stats["pearson"]
        if metric == "hr":
            row[f"hr_within{within_bpm:g}_pct"] = stats["within_pct"]
            row["hr_mae_untracked_bpm"] = paired(series["hr_untracked"], series["hr_gt"])["mae"]
            row["hr_mean_gt_bpm"] = float(np.nanmean(series["hr_gt"])) if np.isfinite(series["hr_gt"]).any() else np.nan
    row.update({
        "rmssd_segment_gt_ms": result["segment_hrv"]["rmssd_gt"],
        "rmssd_segment_est_ms": result["segment_hrv"]["rmssd"],
        "sdnn_segment_gt_ms": result["segment_hrv"]["sdnn_gt"],
        "sdnn_segment_est_ms": result["segment_hrv"]["sdnn"],
        "beats_gt": result["beats_gt"],
        "beats_est": result["beats_est"],
        "ibi_rejected_pct": result["ibi_rejected_pct"],
    })
    return row


def groups(results, by_split=True):
    """[(name, {label, repetition, view, split}, members)]: everything, each label, label x repetition,
    each view, and (by_split) each split and label x split."""
    labels = label_order(results)
    views = sorted({r["view"] for r in results})
    splits = split_order(results) if by_split else []
    found = [("All recordings", {}, results)]
    if len(labels) > 1:
        found += [(label_name(label), {"label": label}, [r for r in results if r["label"] == label])
                  for label in labels]
    if any(r["repetition"] > 1 for r in results):
        found += [(f"{label_name(label)} {repetition}", {"label": label, "repetition": repetition},
                   [r for r in results if r["label"] == label and r["repetition"] == repetition])
                  for label in labels for repetition in sorted({r["repetition"] for r in results if r["label"] == label})]
    if len(views) > 1:
        found += [(f"{view} view", {"view": view}, [r for r in results if r["view"] == view]) for view in views]
    for split in splits:
        members = [r for r in results if r.get("split") == split]
        found.append((SPLIT_NAMES[split], {"split": split}, members))
        if len(labels) > 1:
            found += [(f"{label_name(label)} · {SPLIT_NAMES[split]}", {"label": label, "split": split},
                       [r for r in members if r["label"] == label]) for label in labels]
    return [(name, keys, members) for name, keys, members in found if members]


def global_table(results, within_bpm, by_split=True):
    """Pooled agreement (every window / beat of every recording of the group)."""
    rows = []
    within_column = f"within_{within_bpm:g}bpm_pct"
    for metric, (label, unit, _, _) in METRICS.items():
        for group, keys, members in groups(results, by_split):
            estimates, truths, per_recording = [], [], []
            for result in members:
                estimate, truth = metric_pairs(result, metric)
                estimates.append(estimate)
                truths.append(truth)
                per_recording.append(paired(estimate, truth)["mae"])
            stats = paired(np.concatenate(estimates), np.concatenate(truths),
                           within=within_bpm if metric == "hr" else None)
            rows.append({
                "metric": label, "metric_key": metric, "unit": unit, "group": group,
                "label": keys.get("label", ""), "repetition": keys.get("repetition", ""),
                "view": keys.get("view", ""), "split": keys.get("split", ""),
                "recordings": len(members), "samples": stats["n"], "mae": stats["mae"], "rmse": stats["rmse"],
                "bias": stats["bias"], "pearson": stats["pearson"],
                "median_recording_mae": float(np.nanmedian(per_recording)) if np.isfinite(per_recording).any() else np.nan,
                within_column: stats.get("within_pct", np.nan),
            })
    return pd.DataFrame(rows)


def label_split_table(global_metrics, labels, splits):
    """MAE matrix of each metric: rows = labels (+ all), columns = splits (+ all recordings)."""
    rows = []
    labels = labels if len(labels) > 1 else []
    for metric in METRICS:
        part = global_metrics[global_metrics.metric_key == metric]
        for label in ["", *labels]:
            row = {"metric_key": metric, "unit": METRICS[metric][1], "label": label or "all"}
            for split in ["", *splits]:
                match = part[(part.label == label) & (part.split == split) & (part.repetition == "")
                             & (part.view == "")]
                row[split or "all"] = float(match.mae.iloc[0]) if len(match) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)
