"""Participant folds for vHRM cross-validation.

Shared by tools/vhrm_cross_validation.py and the launcher so both build the same folds. Pure Python on
purpose (no dataset, torch or pandas import). Participants are the slugged folder names the vHRM loader
selects: jzc02_baseline -> jzc02-baseline.

Schemes:
  kfold   K balanced groups: participants are dealt round-robin in sorted order, so fold sizes differ by
          at most one.
  loso    one fold per participant (leave one out).
  groups  explicit test groups; participants not listed in any group stay in training.

With group_sessions, the sessions of one person stay together: a folder joins the shortest folder of the
pool that prefixes its name followed by '-' (jzc02-baseline and jzc02-baseline-bl join jzc02). Fold sizes
and the validation count then count persons, not folders.

Validation participants of a fold are the first members of the next folds (cyclic), never the test group.
"""

import re
from collections import OrderedDict

SCHEMES = ("kfold", "loso", "groups")


def slug(value):
    """Same normalisation as the vHRM loader (VHRMLoader.slug)."""
    return re.sub(r"[^a-z0-9]+", "-", str(value).strip().lower()).strip("-")


def session_units(pool, group_sessions=True):
    """OrderedDict unit -> participants of that unit (one person, or one folder without grouping)."""
    pool = sorted({slug(participant) for participant in pool})
    units = OrderedDict()
    for participant in pool:
        unit = participant
        if group_sessions:
            # Sorted order puts a prefix before its extensions, so the first match is the shortest one.
            unit = next((candidate for candidate in pool if participant.startswith(candidate + "-")), participant)
        units.setdefault(unit, []).append(participant)
    return units


def _group_items(groups):
    if isinstance(groups, dict):
        return [(str(name), members) for name, members in groups.items()]
    width = len(str(len(groups)))
    return [(f"{index + 1:0{width}d}", members) for index, members in enumerate(groups)]


def build_folds(pool, scheme="kfold", k=5, groups=(), group_sessions=True):
    """(folds: OrderedDict fold name -> unit names, units: OrderedDict unit -> participants, warnings)."""
    if scheme not in SCHEMES:
        raise ValueError(f"Unknown cross-validation scheme {scheme!r}; use one of {', '.join(SCHEMES)}")
    units = session_units(pool, group_sessions)
    names = list(units)
    warnings = []
    if len(names) < 2:
        raise ValueError("Cross-validation needs at least 2 participants in the training selection.")
    folds = OrderedDict()
    if scheme == "kfold":
        k = int(k)
        if k < 2:
            raise ValueError("K must be at least 2.")
        count = min(k, len(names))
        if count < k:
            warnings.append(f"Only {len(names)} participants: {count} folds instead of {k}.")
        width = len(str(count))
        fold_names = [f"{index + 1:0{width}d}" for index in range(count)]
        for name in fold_names:
            folds[name] = []
        for index, unit in enumerate(names):
            folds[fold_names[index % count]].append(unit)
    elif scheme == "loso":
        for unit in names:
            folds[unit] = [unit]
    else:
        unit_of = {participant: unit for unit, members in units.items() for participant in members}
        unknown, seen = [], {}
        for name, members in _group_items(groups or []):
            name = slug(name) or name
            members = [members] if isinstance(members, str) else list(members or [])
            fold_units = []
            for member in members:
                unit = unit_of.get(slug(member))
                if unit is None:
                    unknown.append(str(member))
                elif unit in seen and seen[unit] != name:
                    raise ValueError(f"{unit} is in test groups {seen[unit]} and {name}; a participant can only "
                                     "be tested once.")
                elif unit not in fold_units:
                    fold_units.append(unit)
                    seen[unit] = name
            if fold_units:
                folds[name] = fold_units
            else:
                warnings.append(f"Test group {name} has no participant of the training selection; skipped.")
        if unknown:
            warnings.append("Not in the training selection: " + ", ".join(sorted(set(unknown))) + ".")
        if not folds:
            raise ValueError("No test group contains a participant of the training selection.")
    return folds, units, warnings


def plan(pool, scheme="kfold", k=5, valid_count=0, groups=(), group_sessions=True):
    """Test / validation / training participants of every fold."""
    folds, units, warnings = build_folds(pool, scheme, k, groups, group_sessions)
    valid_count = max(0, int(valid_count or 0))
    names = list(folds)
    everyone = [participant for members in units.values() for participant in members]
    result = []
    for position, name in enumerate(names):
        valid_units = []
        if valid_count:
            for offset in range(1, len(names)):
                for unit in folds[names[(position + offset) % len(names)]]:
                    if unit not in valid_units:
                        valid_units.append(unit)
                if len(valid_units) >= valid_count:
                    break
            valid_units = valid_units[:valid_count]
            if len(valid_units) < valid_count:
                warnings.append(f"Fold {name}: only {len(valid_units)} validation participant(s) available.")
        test = [participant for unit in folds[name] for participant in units[unit]]
        valid = [participant for unit in valid_units for participant in units[unit]]
        train = [participant for participant in everyone if participant not in test and participant not in valid]
        result.append({"name": name, "test": test, "valid": valid, "train": train})
    return {
        "scheme": scheme, "pool": everyone, "units": dict(units), "folds": result, "warnings": warnings,
    }


def fold_overrides(fold, model_file_name, log_path, wandb_group=None):
    """Dotted config keys that turn the base config into the config of one fold."""
    model = model_file_name or "vHRM"
    run_name = f"{model}_fold-{fold['name']}"
    values = {
        "LOG.PATH": log_path,
        "TRAIN.MODEL_FILE_NAME": run_name,
        "TRAIN.DATA.VHRM.PARTICIPANTS": list(fold["train"]),
        "TRAIN.DATA.VHRM.EXCLUDE_PARTICIPANTS": [],
        "TEST.DATA.VHRM.PARTICIPANTS": list(fold["test"]),
        "TEST.DATA.VHRM.EXCLUDE_PARTICIPANTS": [],
        # Without validation participants the last epoch is tested.
        "TEST.USE_LAST_EPOCH": not fold["valid"],
    }
    if fold["valid"]:
        values["VALID.DATA.VHRM.PARTICIPANTS"] = list(fold["valid"])
        values["VALID.DATA.VHRM.EXCLUDE_PARTICIPANTS"] = []
    if wandb_group:
        values["WANDB.GROUP"] = wandb_group
        values["WANDB.RUN_NAME"] = run_name
    return values
