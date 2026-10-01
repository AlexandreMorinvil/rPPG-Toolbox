# rPPG-Toolbox training and evaluation in Docker

All Docker build files, Compose settings, and portable experiment configs live
inside this repository. No parent vHRM2 workspace or external config files are
required. One shared image runs compatible training, evaluation, and
unsupervised experiments using Python 3.12 and PyTorch 2.5.1 with CUDA 12.4.

Included standalone configs:

- [DeepPhys training on UBFC-rPPG DATASET_2](configs/train/UBFC-rPPG_DATASET2_DeepPhys.docker.yaml)
- [DeepPhys evaluation on vHRM](configs/infer/UBFC-rPPG_DATASET2_DeepPhys.docker.yaml)
- [FactorizePhys evaluation on vHRM](configs/infer/SCAMPS_vHRM_FactorizePhys_FSAM_Res.docker.yaml)

These preserve the previous experiment settings, with container paths. They do
not inherit configs from outside the repository.

To edit configs with validation, launch and follow runs, and browse caches and
results from a browser, see the [launcher](../launcher/README.md)
(`python -m launcher`).

## Target computer requirements

- An x86-64 computer with an NVIDIA GPU and a current driver supporting CUDA
  12.4. The host does not need Python, PyTorch, or the CUDA toolkit.
- Linux: Docker Engine, Compose v2, and NVIDIA Container Toolkit configured
  for Docker. Follow [NVIDIA's installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
- Windows: Docker Desktop using Linux containers and the WSL2 backend, with
  WSL integration enabled and a current NVIDIA Windows driver. See
  [Docker Desktop GPU support](https://docs.docker.com/desktop/features/gpu/).
- Docker must be running. Allocate enough RAM and disk space for the CUDA
  image, raw data, cache, and results. For the default training, 16 GB host RAM
  is a starting point: each of four preprocessing workers loads a full video
  and its own face detector. Allocate enough memory to WSL2 on Windows.
- Obtain the required datasets under their terms. The default training needs
  the complete UBFC-rPPG DATASET_2, not just one sample subject.

## Clone and configure

After committing and pushing the Docker files to your own repository, clone it
on the target computer:

```powershell
git clone YOUR_REPOSITORY_URL rPPG-Toolbox
cd rPPG-Toolbox
Copy-Item docker/.env.example .env
```

On Linux use `cp docker/.env.example .env` for the last command. Run all Docker
commands below from the rPPG-Toolbox repository root, not from `docker/` or a
parent workspace.

Edit your private `.env` for the target computer:

```dotenv
DATA_ROOT=/mnt/datasets
CHECKPOINT_ROOT=./final_model_release
CACHE_PATH=/mnt/training/preprocessed_data
RUNS_PATH=/mnt/training/runs
GPU_DEVICE_ID=0
DOCKER_SHM_SIZE=8gb
CONFIG_FILE=/opt/vhrm2/configs/train/UBFC-rPPG_DATASET2_DeepPhys.docker.yaml
TOOLBOX_IMAGE=vhrm2-rppg:toolbox-cu124
```

Windows paths may use forward slashes, for example `DATA_ROOT=D:/datasets`.
Quote values containing spaces. Relative host paths are resolved from the
repository root. Dataset and checkpoint roots must already exist; Compose
refuses to silently create empty input folders. Cache and output folders are
created if absent. Training does not need supplied checkpoints, but the
checkpoint root must still exist; an empty directory is sufficient.

For the supplied configs, organize the dataset root as follows:

```text
DATA_ROOT/
  UBFC_DATASET/
    DATASET_2/
      subject1/vid.avi
      subject1/ground_truth.txt
      ...
  vHRM/
    ... existing prepared vHRM recording structure ...
```

Keep the vHRM recording layout intact beneath `vHRM/`. Other experiments may
use other subdirectories. Cross-dataset experiments need both datasets beneath
`/data`, or additional read-only mounts in a Compose override.

| Host setting | Container location | Access |
| --- | --- | --- |
| `DATA_ROOT` (default `./data`) | `/data` | Read-only |
| `CHECKPOINT_ROOT` (default `./final_model_release`) | `/checkpoints` | Read-only |
| `CACHE_PATH` (default `./preprocessed_data`) | `/cache` | Persistent, writable |
| `RUNS_PATH` (default `./runs`) | `/runs` | Persistent, writable |
| Repository `docker/configs/` | `/opt/vhrm2/configs` | Read-only |

`GPU_DEVICE_ID` selects one host GPU, which appears as `cuda:0` inside the
container. The `/opt/vhrm2/...` paths are internal container paths, not host
workspace requirements.

### Migrating from the parent-workspace setup

Docker commands now run from this repository. Docker configs moved to
`docker/configs/`, but their container paths and cache layout are unchanged.
If you have a private `.env` in the parent workspace, transfer it privately to
this repository root without overwriting another `.env`, and update its
relative host paths. Use absolute paths to retain existing caches and outputs.
The default `DATA_ROOT` is now `./data`, not `./dataset`; the latter is the
toolbox's Python loader package, not the raw-data directory.

For the earlier DeepPhys-only version, also replace the `train` service name
with `toolbox`, `TRAIN_CONFIG` with `CONFIG_FILE`, and `UBFC_DATASET_PATH` with
the parent `DATA_ROOT`. The image tag is `vhrm2-rppg:toolbox-cu124`.

## Build and run

```powershell
docker compose config --quiet
docker compose build toolbox
docker compose run --rm --entrypoint python toolbox -c "import torch; assert torch.cuda.is_available(), 'NVIDIA GPU is not visible'; print(torch.cuda.get_device_name(0)); print(torch.ones(2, device='cuda').sum().item())"
docker compose run --rm toolbox
```

The initial build downloads several GB. It checks dependency consistency,
imports the experiment CLI, and loads the tracked YOLO5Face weights on CPU.
The GPU check above must pass before starting an experiment. The build context
includes only toolbox code and configs, not Git metadata, raw data, caches,
previous runs, credentials, or released rPPG checkpoints. The face-detector
checkpoint is included because preprocessing needs it. Plots use a
non-interactive backend.

### Select training or evaluation

Set `CONFIG_FILE` in `.env` for your usual experiment. For a one-off run,
override the configured command:

```powershell
docker compose run --rm toolbox --config_file /opt/vhrm2/configs/train/UBFC-rPPG_DATASET2_DeepPhys.docker.yaml
docker compose run --rm toolbox --config_file /opt/vhrm2/configs/infer/UBFC-rPPG_DATASET2_DeepPhys.docker.yaml
docker compose run --rm toolbox --config_file /opt/vhrm2/configs/infer/SCAMPS_vHRM_FactorizePhys_FSAM_Res.docker.yaml
```

The inference examples both evaluate on `/data/vHRM`. Their released weights,
`UBFC-rPPG_DeepPhys.pth` and `SCAMPS_FactorizePhys_FSAM_Res.pth`, are tracked in
`final_model_release/` and available through the default checkpoint mount.
For custom weights, use your own `CHECKPOINT_ROOT`. To evaluate a newly trained
model, set `INFERENCE.MODEL_PATH` in an inference YAML to its
`/runs/.../PreTrainedModels/...pth` path. Match architecture, input resolution,
channels, normalization, and clip length to the checkpoint.

### Add another experiment

Add YAMLs under `docker/configs/` so they are included in Git and mounted in the
container. Use standalone configs, or inherit other configs within this
directory using relative `BASE` paths. Every inherited file must travel with
the repository and be visible in the same mount; do not inherit a parent
workspace's configs. Edits and new configs need no image rebuild.

Keep the experiment's `TOOLBOX_MODE` (`train_and_test`, `only_test`, or
`unsupervised_method`) and change machine-specific paths to container paths:

- Training: `TRAIN.DATA`, `VALID.DATA`, and `TEST.DATA` raw/cache paths.
- Evaluation: active `TEST.DATA` paths and `INFERENCE.MODEL_PATH`.
- Unsupervised: `UNSUPERVISED.DATA` raw/cache paths.
- Outputs: a distinct `LOG.PATH` under `/runs` and relative
  `MODEL.MODEL_DIR: PreTrainedModels`.
- Auxiliary inputs: any explicit `FILE_LIST_PATH`, fold paths, or other inputs.

For example, a cross-dataset experiment can use `/data/PURE` for training and
validation and `/data/UBFC_DATASET/DATASET_2` for testing. Environment variables
are expanded by Compose, not the experiment YAML loader; YAMLs must contain
actual container paths, not `${...}`.

Use distinct log paths per experiment/run to avoid overwriting checkpoints and
plots. Use distinct `EXP_DATA_NAME` cache names when preprocessing differs;
share caches only when preprocessing is identical.

### Software environment limits

This image covers models available with the standard toolbox requirements.
PhysMamba needs a separate image variant with a CUDA compiler and the optional
`mamba-ssm`/`causal-conv1d` extensions. PEAR and pyVHR are separate software
stacks, not config variants. Compose still requires an NVIDIA GPU even if a
YAML selects `DEVICE: cpu`. Use another image when dependencies change, not
for every experiment. Select compatible versioned images with `TOOLBOX_IMAGE`.

### Detached runs

```powershell
docker compose up -d toolbox
docker compose logs -f toolbox
```

Use `docker compose stop toolbox` to stop and `docker compose down` to remove
the container. Bind-mounted results remain. Restarting does not automatically
resume training from a saved checkpoint.

## Git and image transfer

Commit the Dockerfile, Compose file, Docker ignore file, Git ignore changes,
and `docker/` directory to your fork alongside the README update. No dataset,
cache, output, image archive, or credential needs to be committed. Private
`.env` files and default generated directories are Git-ignored; the environment
template is deliberately not ignored. The source and required detector weights
already belong to this repository. Cloning your updated fork retrieves the
whole setup; datasets still need to be obtained separately.

To avoid rebuilding on the target computer, export the built image:

```powershell
docker image save --output vhrm2-rppg.tar vhrm2-rppg:toolbox-cu124
```

Clone the repository on the target, transfer the archive and datasets/custom
checkpoints separately, then from the repository root:

```powershell
docker image load --input vhrm2-rppg.tar
Copy-Item docker/.env.example .env
```

On Linux use `cp` for the last command. Edit `.env`, run the GPU check, then
`docker compose run --rm toolbox`. Compose uses the loaded image; do not add
`--build`. Ensure `TOOLBOX_IMAGE` matches the loaded tag. The archive excludes
datasets, evaluation checkpoints, cache, outputs, and W&B credentials.
Saving/loading the same image preserves installed dependencies exactly;
rebuilding later can select newer versions within the toolbox's requirements.
Installed versions are recorded in `/opt/vhrm2/installed-requirements.txt`.

## Outputs and cached preprocessing

With the defaults, training checkpoints appear on the host under
`runs/UBFC-rPPG_DATASET2_DeepPhys/Initial_Run/PreTrainedModels`, evaluation
outputs under `runs/UBFC-rPPG_DATASET2_DeepPhys/Initial_Run/saved_test_outputs`,
and cache files under `preprocessed_data/UBFC-rPPG_DATASET2/Initial_Run`.

The first training run preprocesses all three splits. To reuse its cache, set
`DO_PREPROCESS: False` under each split's `DATA` in the training YAML. Do not
reuse Windows-generated file-list CSVs unchanged: they contain absolute Windows
paths. Rebuild in Docker or deliberately migrate those paths. Docker-generated
caches can move between machines because `/cache` is stable, provided arrays
and split CSVs are copied together.

The raw UBFC loader uses filesystem enumeration order for subject splits. The
same fractions need not select the same subjects on another machine. For exact
membership, carry over the Docker-generated cache and split CSVs, disable
preprocessing, and verify subjects before comparing experiments. Different GPU
hardware can also affect numerical reproducibility.

## Weights & Biases

The training config uses offline tracking; inference configs disable tracking
unless you enable it. With `WANDB.ENABLED: True` and `MODE: offline`, files stay
under the mounted `runs/` folder without login. For live syncing, set
`WANDB.ENABLED: True` and `WANDB.MODE: online` in the experiment YAML and supply
`WANDB_API_KEY` through your private `.env` or shell environment. Compose
forwards it at runtime, never during the build. Never commit the API key.

Offline runs can be uploaded later with the key supplied privately:

```powershell
docker compose run --rm --entrypoint wandb toolbox sync /runs/wandb/offline-run-YOUR_RUN_DIRECTORY
```

## Resource and permission issues

- GPU unavailable: check the driver, selected GPU, WSL2 backend or NVIDIA
  Container Toolkit, then rerun the GPU check. macOS/AMD GPUs are not supported
  by this CUDA setup.
- Shared-memory/bus errors: increase `DOCKER_SHM_SIZE` and available Docker/WSL2
  RAM. Training uses 16 data-loader workers, independently of preprocessing.
- Preprocessing RAM/VRAM exhaustion: set `PREPROCESS.NUM_WORKERS: 1` inside
  each active split's `DATA`. Reduce `TRAIN.BATCH_SIZE` and
  `INFERENCE.BATCH_SIZE` for training/testing VRAM exhaustion. Such changes can
  affect experiment results.
- Linux root-owned outputs: precreate writable cache/output directories and
  use `docker compose run --rm --user "$(id -u):$(id -g)" toolbox` to run as
  your UID. The container otherwise runs as root.