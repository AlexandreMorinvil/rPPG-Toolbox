"""Paired comparison of runs at the participant level (thesis statistics).

Every run writes window_results.csv (one row per evaluation window). Windows of one person are not independent, so the
unit here is the PERSON: each person's MAE per run, then the paired difference to the reference run over the people
both runs scored, with bootstrap confidence intervals over people.

    python tools/paired_comparison.py --run face=<run folder> --run uvfull=<run folder> --reference face \\
        --margin 1.0 --output runs/compare/b1_ubfc_physnet
    python tools/paired_comparison.py --run face=... --run uvfull=... --reference face --group-by view \\
        --filter experiment=jzc --features runs/features/windows.csv --bin-by body_speed_deg_s --bins 3

A run is a window_results.csv file or a folder holding exactly one (POS runs: saved_outputs/POS/). People come from
thesis/splits.yaml (lab: jzc07 and RM06 are one person; literature: the subject); --split keeps one split (default:
test; "all" keeps everyone, e.g. for POS). Windows without a reading (face crop, MIN_FACE_FRACTION) are left out of
the MAE and counted in the coverage; --common-windows scores only the windows every run read.

Verdicts per run against the reference (difference = run - reference, in bpm of MAE):
  equivalent   the 90 % interval lies inside +-margin (two one-sided tests at 5 %; needs --margin)
  better/worse the 95 % interval lies entirely below / above 0
  inconclusive otherwise
"""

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cache_naming import subject_key  # noqa: E402
from dataset.data_loader.VHRMLoader import slug  # noqa: E402

KEY = ["recording_id", "start_frame"]


def find_windows(path):
    path = Path(path)
    if path.is_file():
        return path
    found = sorted(glob.glob(str(path / "**" / "window_results.csv"), recursive=True))
    if len(found) != 1:
        raise ValueError(f"{path}: expected one window_results.csv, found {len(found)}: {found[:5]}")
    return Path(found[0])


def person_table(splits):
    """{(experiment, participant) slugs: (person, split)} and {(dataset, subject): split}."""
    lab = {}
    for person, entry in (splits.get("lab") or {}).get("persons", {}).items():
        for session in entry["sessions"]:
            experiment, folder = session.split("/", 1)
            lab[(slug(experiment), slug(folder))] = (person, entry["split"])
    literature = {}
    for dataset, lists in (splits.get("literature") or {}).items():
        for split, subjects in lists.items():
            for subject in subjects:
                literature[(dataset, subject_key(subject))] = split
    return lab, literature


def load_run(name, path, lab, literature):
    windows = pd.read_csv(find_windows(path), keep_default_na=True)
    if "start_frame" not in windows:  # vHRM runs written before start_frame existed
        raise ValueError(f"{name}: window_results.csv has no start_frame column; evaluate the run again")
    windows["recording_id"] = windows["recording_id"].astype(str)
    windows["has_reading"] = windows["has_reading"].astype(bool) if "has_reading" in windows else True
    if "experiment" in windows and windows["experiment"].fillna("").astype(str).str.len().gt(0).all():
        owners = [lab.get((str(row.experiment), str(row.participant)), (None, None))
                  for row in windows[["experiment", "participant"]].itertuples()]
    else:
        dataset = windows["dataset"].astype(str) if "dataset" in windows else ""
        owners = [(f"{d}:{subject_key(p)}", literature.get((d, subject_key(p))))
                  for d, p in zip(dataset, windows["participant"].astype(str))]
    windows["person"] = [owner[0] for owner in owners]
    windows["split"] = [owner[1] for owner in owners]
    windows["absolute_error_bpm"] = (windows["predicted_hr_bpm"] - windows["ground_truth_hr_bpm"]).abs()
    return windows


def bootstrap(differences, resamples, seed):
    rng = np.random.default_rng(seed)
    draws = rng.choice(differences, size=(resamples, len(differences)), replace=True).mean(axis=1)
    return {level: tuple(np.percentile(draws, [(100 - level) / 2, 100 - (100 - level) / 2])) for level in (90, 95)}


def verdict(ci90, ci95, margin):
    if margin and -margin < ci90[0] and ci90[1] < margin:
        return "equivalent"
    if ci95[1] < 0:
        return "better"
    if ci95[0] > 0:
        return "worse"
    return "inconclusive"


def compare(runs, reference, margin=None, resamples=10000, seed=0, common_windows=False):
    """runs: {name: windows of one group}. Returns (summary rows, per-person table)."""
    if common_windows:
        keys = None
        for windows in runs.values():
            read = set(map(tuple, windows.loc[windows["has_reading"], KEY].itertuples(index=False)))
            keys = read if keys is None else keys & read
        runs = {name: windows[[tuple(key) in keys for key in windows[KEY].itertuples(index=False)]]
                for name, windows in runs.items()}
    persons = {}
    rows = []
    for name, windows in runs.items():
        windows = windows.assign(read_error=windows["absolute_error_bpm"].where(windows["has_reading"]))
        per_person = windows.groupby("person").agg(
            windows=("has_reading", "size"), read=("has_reading", "sum"), mae=("read_error", "mean"))
        per_person["coverage"] = per_person["read"] / per_person["windows"]
        persons[name] = per_person
        read = windows[windows["has_reading"]]
        rows.append({"run": name, "people": int(per_person["mae"].notna().sum()), "windows": len(windows),
                     "coverage": per_person["read"].sum() / max(1, per_person["windows"].sum()),
                     "window_mae_bpm": read["absolute_error_bpm"].mean(),
                     "person_mae_bpm": per_person["mae"].mean()})
    table = pd.concat({name: frame for name, frame in persons.items()}, axis=1)
    for row in rows:
        if row["run"] == reference or reference not in persons:
            continue
        pair = pd.concat([persons[row["run"]]["mae"], persons[reference]["mae"]], axis=1, keys=["run", "ref"]).dropna()
        row["paired_people"] = len(pair)
        if len(pair) < 2:
            row["verdict"] = "too few people"
            continue
        differences = (pair["run"] - pair["ref"]).to_numpy()
        intervals = bootstrap(differences, resamples, seed)
        row.update({"difference_bpm": differences.mean(), "ci95_low": intervals[95][0], "ci95_high": intervals[95][1],
                    "ci90_low": intervals[90][0], "ci90_high": intervals[90][1],
                    "verdict": verdict(intervals[90], intervals[95], margin)})
        try:
            from scipy.stats import wilcoxon
            row["wilcoxon_p"] = float(wilcoxon(differences).pvalue) if np.any(differences) else 1.0
        except (ImportError, ValueError):
            row["wilcoxon_p"] = np.nan
    return rows, table


def markdown_table(frame):
    def cell(value):
        if isinstance(value, (float, np.floating)):
            return "" if np.isnan(value) else f"{value:.3g}"
        return str(value)

    lines = ["| " + " | ".join(frame.columns) + " |", "|" + "---|" * len(frame.columns)]
    lines += ["| " + " | ".join(cell(value) for value in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="append", required=True, metavar="NAME=PATH",
                        help="A run: its window_results.csv or a folder holding one. Repeat for each run.")
    parser.add_argument("--reference", help="Run the others are compared with (default: the first).")
    parser.add_argument("--margin", type=float, help="Equivalence margin in bpm (e.g. 1.0).")
    parser.add_argument("--splits", default=str(ROOT / "thesis" / "splits.yaml"), help="People and their splits.")
    parser.add_argument("--split", default="test", help="train | valid | test | all (default: test).")
    parser.add_argument("--filter", action="append", default=[], metavar="COLUMN=VALUE",
                        help="Keep windows whose column equals one of the comma-separated values, e.g. view=front.")
    parser.add_argument("--group-by", nargs="*", default=[], help="Analyse each group separately, e.g. experiment view.")
    parser.add_argument("--common-windows", action="store_true", help="Score only the windows every run read.")
    parser.add_argument("--features", help="Window features CSV (tools/vhrm_window_features.py) to bin by.")
    parser.add_argument("--bin-by", help="Feature column to bin windows by (quantiles of all windows).")
    parser.add_argument("--bins", type=int, default=3, help="Number of quantile bins for --bin-by.")
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", help="Folder for summary.csv, people.csv and summary.md.")
    args = parser.parse_args(argv)

    with open(args.splits, encoding="utf-8") as handle:
        lab, literature = person_table(yaml.safe_load(handle))
    runs = {}
    for item in args.run:
        name, separator, path = item.partition("=")
        if not separator:
            parser.error(f"--run must look like NAME=PATH: {item}")
        windows = load_run(name, path, lab, literature)
        unknown = windows["person"].isna()
        if unknown.any():
            print(f"WARNING: {name}: {unknown.sum()} windows of people missing from {args.splits} are left out")
        windows = windows[~unknown]
        if args.split != "all":
            windows = windows[windows["split"] == args.split]
        for condition in args.filter:
            column, _, values = condition.partition("=")
            windows = windows[windows[column].astype(str).isin(values.split(","))]
        runs[name] = windows
    reference = args.reference or next(iter(runs))
    if reference not in runs:
        parser.error(f"--reference {reference} is not one of the runs")

    group_columns = list(args.group_by)
    if args.features:
        features = pd.read_csv(args.features)
        features["recording_id"] = features["recording_id"].astype(str)
        if not args.bin_by:
            parser.error("--features needs --bin-by")
        runs = {name: windows.merge(features[KEY + [args.bin_by]], on=KEY, how="left") for name, windows in runs.items()}
        values = pd.concat([windows[args.bin_by] for windows in runs.values()]).dropna()
        edges = np.unique(np.quantile(values, np.linspace(0, 1, args.bins + 1)))
        for windows in runs.values():
            windows[f"{args.bin_by}_bin"] = pd.cut(windows[args.bin_by], edges, include_lowest=True).astype(str)
        group_columns.append(f"{args.bin_by}_bin")

    groups = sorted({tuple(row) for windows in runs.values()
                     for row in windows[group_columns].astype(str).itertuples(index=False)}) if group_columns else [()]
    summary, people = [], []
    for group in groups:
        subset = {name: windows[(windows[group_columns].astype(str) == list(group)).all(axis=1)] if group_columns
                  else windows for name, windows in runs.items()}
        rows, table = compare(subset, reference, args.margin, args.resamples, args.seed, args.common_windows)
        labels = dict(zip(group_columns, group))
        summary += [{**labels, **row} for row in rows]
        table.columns = [f"{run}_{column}" for run, column in table.columns]
        people.append(table.reset_index().assign(**labels))
    summary = pd.DataFrame(summary)
    columns = group_columns + [column for column in (
        "run", "people", "windows", "coverage", "window_mae_bpm", "person_mae_bpm", "paired_people", "difference_bpm",
        "ci95_low", "ci95_high", "ci90_low", "ci90_high", "verdict", "wilcoxon_p") if column in summary]
    summary = summary[columns]
    text = (f"Reference: {reference}; split: {args.split}; margin: {args.margin if args.margin else 'none'}; "
            f"{'common windows only' if args.common_windows else 'each run its own windows'}\n\n"
            + markdown_table(summary))
    print(text)
    if args.output:
        os.makedirs(args.output, exist_ok=True)
        summary.to_csv(os.path.join(args.output, "summary.csv"), index=False)
        pd.concat(people).to_csv(os.path.join(args.output, "people.csv"), index=False)
        Path(args.output, "summary.md").write_text(text + "\n", encoding="utf-8")
        print(f"\nSaved summary.csv, people.csv and summary.md to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
