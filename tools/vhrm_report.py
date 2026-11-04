"""Analysis report of a checkpoint on vHRM: HR, beat-to-beat RR intervals and HRV against the chest sensors.

    python tools/vhrm_report.py --config <run config> --checkpoint <run>/<cache>/PreTrainedModels/<name>_Epoch16.pth
    python tools/vhrm_report.py --config <run config>                  # the checkpoint the run tested
    python tools/vhrm_report.py --config <run config> --labels Curl --participants jzc16 jzc17

The config is the run's own (for a launcher job: launcher_data/jobs/<job>/config.yaml). It gives the model, the
preprocessing of the input, the data and the participant splits. The report goes to
<experiment folder>/report/<checkpoint>/ unless --output is given. Setting REPORT.ENABLED in a config generates
the same report at the end of main.py. Method and outputs: evaluation/vhrm_report/METHOD.md.
"""

import argparse
import os
import pickle
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from config import get_config  # noqa: E402


def tested_checkpoint(config):
    """The checkpoint the run's own test used, from the epoch saved with its test outputs."""
    if config.TOOLBOX_MODE == "only_test":
        return config.INFERENCE.MODEL_PATH
    if config.TEST.USE_LAST_EPOCH:
        epoch = config.TRAIN.EPOCHS - 1
    else:
        outputs = os.path.join(config.TEST.OUTPUT_SAVE_DIR, f"{config.TRAIN.MODEL_FILE_NAME}_outputs.pickle")
        if not os.path.isfile(outputs):
            raise SystemExit(f"No test outputs at {outputs}: the run has not been tested yet. Pass --checkpoint.")
        with open(outputs, "rb") as handle:
            epoch = (pickle.load(handle).get("meta") or {}).get("epoch")
        if epoch is None:
            raise SystemExit(f"{outputs} does not record the tested epoch. Pass --checkpoint.")
    return os.path.join(config.MODEL.MODEL_DIR, f"{config.TRAIN.MODEL_FILE_NAME}_Epoch{epoch}.pth")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="the run's YAML config")
    parser.add_argument("--checkpoint", help="the .pth to analyse (default: the checkpoint the run tested)")
    parser.add_argument("--output", help="report folder (default: <experiment folder>/report/<checkpoint>)")
    parser.add_argument("--labels", nargs="*", help="segment labels to analyse (default: REPORT.LABELS, empty = all)")
    parser.add_argument("--participants", nargs="*",
                        help="participants to analyse (default: REPORT.PARTICIPANTS, empty = all)")
    parser.add_argument("--no-csv", action="store_true", help="do not write data_csv/")
    args = parser.parse_args()

    paths = {name: os.path.abspath(getattr(args, name)) for name in ("config", "checkpoint", "output")
             if getattr(args, name)}
    os.chdir(ROOT)  # as main.py runs: relative paths in the config are relative to the toolbox
    config = get_config(argparse.Namespace(config_file=paths["config"]))
    from evaluation.vhrm_report import generate_report
    generate_report(config, paths.get("checkpoint") or tested_checkpoint(config), output=paths.get("output"),
                    labels=args.labels, participants=args.participants,
                    export_csv=False if args.no_csv else None)


if __name__ == "__main__":
    main()
