# CHANGELOG — Python 3.12 Modernization

**Target environment:** conda env `vHRM_development`, Python **3.12.13**, Windows 11, PyTorch **2.5.1+cu124**.

**Scope:** make rPPG-Toolbox import and run cleanly on Python 3.12 with current pinned dependencies, while preserving research behavior. PhysMamba is soft-disabled on Windows (mamba-ssm / causal-conv1d have no Windows wheels).

---

## 1. Dependency updates

### `requirements.txt` (full rewrite)

| Package | Before | After | Rationale |
|---|---|---|---|
| `numpy` | `==1.22.0` | `>=1.26,<2.0` | 1.22 has no cp312 wheels. 1.26 line keeps NumPy 1.x ABI compat with most ML libs. |
| `scipy` | `==1.5.2` | `>=1.13,<1.15` | First scipy line with cp312 wheels and stable signal API. |
| `pandas` | `==1.1.5` | `>=2.2,<2.3` | 1.1 uses removed Py3.12 APIs; 2.2 is current LTS. |
| `h5py` | `==2.10.0` | `>=3.11` | 2.x will not build on Py3.12. |
| `scikit_image` | `==0.17.2` | `>=0.24` (renamed `scikit-image`) | cp312 wheels. |
| `scikit_learn` | `==1.0.2` | `>=1.4` (renamed `scikit-learn`) | cp312 wheels + sklearn 1.x APIs preserved. |
| `matplotlib` | `==3.1.2` | `>=3.8` | cp312 wheels. |
| `opencv_python` | `==4.5.2.54` | `>=4.9` (renamed `opencv-python`) | cp312 wheels; 4.5 has no Py3.12 build. |
| `protobuf` | `==3.20.3` | `>=4.25,<6` | 3.20 has no cp312 wheels. |
| `tensorboardX` | `==2.4.1` | `>=2.6.2.2` | Compat with protobuf 4/5. |
| `mat73` | `==0.59` | `>=0.65` | Modern release; reads MATLAB v7.3 as before. |
| `PyYAML` | `==6.0` | `>=6.0.1` | 6.0 had a bad sdist on newer pip; 6.0.1+ fixes it. |
| `tqdm` | `==4.64.0` | `>=4.66` | Latest, no API changes. |
| `ipykernel` | `==6.26.0` | `>=6.29` | Notebook tooling parity. |
| `ipywidgets` | `==8.1.1` | `>=8.1` | Same. |
| `fsspec` | `==2024.10.0` | `>=2024.10.0` | Unchanged minimum. |
| `timm` | `==1.0.11` | `>=1.0.11` | Unchanged minimum. |
| `neurokit2` | `==0.2.10` | `>=0.2.10` | Unchanged minimum. |
| `thop` | `==0.1.1.post2209072238` | `>=0.1.1.post2209072238` | Unmaintained, but works on 3.12 (emits a benign distutils `DeprecationWarning` from inside the package — see Known issues). |
| `causal-conv1d` | `==1.0.0` | **removed** | No Windows wheels; required only by PhysMamba (now lazy). Install manually on Linux. |
| `mamba-ssm` | `==2.2.2` | **removed** | Same. |
| `torch / torchvision / torchaudio` | inlined as `2.1.2+cu121` in `setup.sh` | moved fully out of `requirements.txt`; install separately via the PyTorch index (recommended `2.5.1+cu124`) | Avoids accidental CPU-only fallbacks; gives users explicit CUDA flavor control. |

The new file additionally documents the Linux-only PhysMamba install path inline.

### `setup.sh`

- Bumped target Python from `3.8` → `3.12` (both `conda` and `uv` paths).
- Bumped pinned PyTorch to `torch==2.5.1+cu124 / torchvision==0.20.1+cu124 / torchaudio==2.5.1+cu124` (was `2.1.2+cu121`).
- Replaced the unconditional `cd tools/mamba && python setup.py install` (which fails outside Linux + CUDA toolkit) with a guarded `pip install causal-conv1d mamba-ssm` block that runs only when `uname -s == Linux`, with a clear warning on other platforms that PhysMamba will be disabled.
- Added explicit `pip install --upgrade pip setuptools wheel` step (Py3.12 needs a recent pip for some metadata edge cases).
- Added a top-of-file note that on Windows users should follow the same steps manually or use WSL.

### Installed environment (verified)

`vHRM_development` now contains, among others:

```
torch                   2.5.1+cu124    (CUDA 12.4, CUDA available: True)
numpy                   1.26.4
scipy                   1.14.1
pandas                  2.2.3
h5py                    3.16.0
matplotlib              3.10.9
opencv-python           4.11.0.86
scikit-image            0.26.0
scikit-learn            1.8.0
yacs                    0.1.8
PyYAML                  6.0.3
tqdm                    4.67.3
tensorboardX            2.6.5
mat73                   0.65
fsspec                  2026.2.0
neurokit2               0.2.12
thop                    0.1.1.post2209072238
timm                    1.0.26
protobuf                5.29.6
ipykernel               7.2.0
ipywidgets              8.1.8
```

---

## 2. Source code changes

### Type: **Optional-dependency soft-disable (PhysMamba / mamba-ssm / causal-conv1d)**

#### [neural_methods/trainer/__init__.py](code/rPPG-Toolbox/neural_methods/trainer/__init__.py)
- Wrapped `import neural_methods.trainer.PhysMambaTrainer` in `try/except ImportError`.
- Added module-level flags `PHYSMAMBA_AVAILABLE: bool` and `PHYSMAMBA_IMPORT_ERROR: Exception | None` so callers can branch cleanly.

#### [main.py](code/rPPG-Toolbox/main.py)
- In both `train_and_test()` and `test()`, the `'PhysMamba'` dispatch branch now checks `trainer.PHYSMAMBA_AVAILABLE` first and raises a descriptive `ImportError` quoting the original failure (instead of an opaque `AttributeError`) if the user requests PhysMamba on a system without `mamba-ssm`.

### Type: **PyTorch ≥2.4 FutureWarning fix (`weights_only`)**

PyTorch 2.4 emits a `FutureWarning` for every `torch.load(...)` call without an explicit `weights_only=` kwarg, and PyTorch 2.6 will flip the default to `True` (which would break loading of optimizer state / full checkpoints). Set `weights_only=False` explicitly on every project-owned checkpoint load. These checkpoints are produced by the toolbox itself, so trusting them is appropriate.

Files changed (24 call sites total):

- [neural_methods/trainer/TscanTrainer.py](code/rPPG-Toolbox/neural_methods/trainer/TscanTrainer.py) — 3 sites (lines 161, 169, 175)
- [neural_methods/trainer/PhysnetTrainer.py](code/rPPG-Toolbox/neural_methods/trainer/PhysnetTrainer.py) — 3 sites (152, 161, 167)
- [neural_methods/trainer/DeepPhysTrainer.py](code/rPPG-Toolbox/neural_methods/trainer/DeepPhysTrainer.py) — 3 sites (155, 163, 169)
- [neural_methods/trainer/EfficientPhysTrainer.py](code/rPPG-Toolbox/neural_methods/trainer/EfficientPhysTrainer.py) — 3 sites (170, 178, 184)
- [neural_methods/trainer/iBVPNetTrainer.py](code/rPPG-Toolbox/neural_methods/trainer/iBVPNetTrainer.py) — 3 sites (164, 173, 179)
- [neural_methods/trainer/RhythmFormerTrainer.py](code/rPPG-Toolbox/neural_methods/trainer/RhythmFormerTrainer.py) — 3 sites (154, 162, 168)
- [neural_methods/trainer/PhysFormerTrainer.py](code/rPPG-Toolbox/neural_methods/trainer/PhysFormerTrainer.py) — 3 sites (222, 231, 237)
- [neural_methods/trainer/FactorizePhysTrainer.py](code/rPPG-Toolbox/neural_methods/trainer/FactorizePhysTrainer.py) — 3 sites (243, 252, 258), preserving existing `map_location=` and `strict=False`
- [neural_methods/trainer/BigSmallTrainer.py](code/rPPG-Toolbox/neural_methods/trainer/BigSmallTrainer.py) — 1 site (397)
- [neural_methods/model/FactorizePhys/test_FactorizePhys.py](code/rPPG-Toolbox/neural_methods/model/FactorizePhys/test_FactorizePhys.py) — 1 site (99)
- [neural_methods/model/FactorizePhys/test_FactorizePhysBig.py](code/rPPG-Toolbox/neural_methods/model/FactorizePhys/test_FactorizePhysBig.py) — 1 site (105)
- [dataset/data_loader/face_detector/YOLO5Face.py](code/rPPG-Toolbox/dataset/data_loader/face_detector/YOLO5Face.py) — 1 site (37)

`PhysMambaTrainer.py` was intentionally **not** modified — that file is only loaded when `mamba-ssm` is installed (Linux + CUDA), not in the active Windows env. If you later run on Linux with PhysMamba, apply the same fix there.

### Type: **Python 3.12 `SyntaxWarning` fix (raw string regex)**

#### [dataset/data_loader/UBFCrPPGLoader.py](code/rPPG-Toolbox/dataset/data_loader/UBFCrPPGLoader.py#L50)
Line 50: changed `'subject(\d+)'` → `r'subject(\d+)'`. Python 3.12 promotes invalid escape sequences in non-raw string literals from a deprecation note to a `SyntaxWarning` (and Python 3.14 will make it a `SyntaxError`). Verified by compiling every `.py` in the repo with `SyntaxWarning` recording — this was the only such site in the project source.

---

## 3. Verified behavior

Smoke test executed in the modernized env (`vHRM_development`, Python 3.12.13, Torch 2.5.1+cu124):

- `import torch` → `2.5.1+cu124`, **CUDA available: True**
- All scientific deps (`numpy scipy pandas h5py matplotlib cv2 sklearn skimage yaml yacs tqdm tensorboardX mat73 fsspec neurokit2 thop timm google.protobuf`) import cleanly.
- `from config import get_config` → OK
- `from dataset import data_loader` → OK (loads all 12 dataset loaders)
- `from neural_methods import trainer` → OK; `trainer.PHYSMAMBA_AVAILABLE == False`, `trainer.PHYSMAMBA_IMPORT_ERROR == ModuleNotFoundError("No module named 'mamba_ssm'")` (expected on Windows).
- `from unsupervised_methods.unsupervised_predictor import unsupervised_predict` → OK
- Repo-wide `SyntaxWarning` scan after fixes: **0 hits**.

---

## 4. Tradeoff decisions explained

| Decision | Options considered | Choice | Why |
|---|---|---|---|
| PhysMamba on Windows | (a) full removal, (b) lazy/soft-disable, (c) document WSL build only | **(b) lazy/soft-disable** | Preserves all source files and configs; everything except the PhysMamba dispatch branch keeps working; clear runtime error tells the user exactly what is missing. Non-destructive. |
| PyTorch CUDA flavor | cu121 vs cu124 vs CPU | **2.5.1+cu124** | CUDA 12.4 is the current stable for recent NVIDIA drivers; 2.5.1 is the most recent torch with broad ecosystem support and predates the 2.6 `weights_only` default flip. |
| NumPy major version | 1.26.x vs 2.x | **1.26.4** | NumPy 2.0 broke a lot of dtype/string APIs that older sub-deps (e.g., legacy timm/thop paths, some scipy interactions) still rely on. 1.26 is the last 1.x line and ships cp312 wheels. Easy to bump later when the ecosystem catches up. |
| `torch.load(weights_only=...)` value | `True` (safe but breaks optimizer-state loads) vs `False` (preserves behavior) | **`False`** | Toolbox checkpoints contain optimizer state and trainer metadata, not just `state_dict`; `True` would break checkpoint resume. Threat model: these files come from the user's own training runs, not untrusted sources. |
| Drop or keep `tools/mamba` source tree | drop vs keep | **keep, do not build** | Kept untouched on disk so PhysMamba works on Linux when deps are installed; never invoked from `setup.sh` on non-Linux; `requirements.txt` no longer pins anything from it. |

---

## 5. Known issues / future work (not fixed here)

These were identified during the audit but intentionally left alone — none block Py3.12 operation, all are pre-existing or upstream:

1. **`thop` emits `DeprecationWarning: distutils Version classes are deprecated`** at import time (its `profile.py` lines 12 and 68 use `distutils.version.LooseVersion`). `thop` is unmaintained. Options for later: pin `thop` (current behavior), vendor a tiny patched copy, or migrate to `fvcore`/`calflops`. No functional impact today.
2. **`eval()` use in `dataset/data_loader/face_detector/model/yolo.py` (lines 268, 271)** — pre-existing arbitrary-code-execution risk in YOLO5Face's model parser. Required by the YOLO5Face checkpoint format; replacing with `ast.literal_eval` would need broader refactoring of the model spec parser. Out of scope for the 3.12 modernization.
3. **`tools/mamba/` internal files** (`selective_scan_interface.py` lines 9–10) have an unconditional `import causal_conv1d`. Not relevant on Windows (we never build mamba). On Linux without `causal-conv1d`, the existing try/except in `mamba_simple.py` is sufficient for the higher-level `Mamba` class, but importing `selective_scan_interface` directly will still fail. Recommend adding the same try/except guard there if you later use the toolbox on a Linux machine without `causal-conv1d`.
4. **`np.double(...)` calls in 6 files** (`evaluation/post_process.py`, `evaluation/bigsmall_multitask_metrics.py`, `unsupervised_methods/methods/POS_WANG.py`, `unsupervised_methods/methods/ICA_POH.py`, `dataset/data_loader/BP4DPlusBigSmallLoader.py`, `dataset/data_loader/BaseLoader.py`). `np.double` is still a valid alias for `np.float64` in both NumPy 1.26 and 2.x — *not* deprecated, *not* a 3.12 issue. Left as-is to keep the diff focused. Could be modernized to `np.asarray(x, dtype=np.float64)` later as a style preference.
5. **`from collections import OrderedDict` imports in 9 files** — still valid; only flagged as "redundant since 3.7" (`dict` preserves order). Not a Py3.12 issue.

---

## 6. Files changed (summary)

```
modified: requirements.txt                                                (rewrite)
modified: setup.sh                                                        (Py3.8→3.12, torch 2.1→2.5, conditional mamba)
modified: main.py                                                         (PhysMamba dispatch guard ×2)
modified: neural_methods/trainer/__init__.py                              (lazy PhysMamba import)
modified: neural_methods/trainer/TscanTrainer.py                          (torch.load weights_only ×3)
modified: neural_methods/trainer/PhysnetTrainer.py                        (torch.load weights_only ×3)
modified: neural_methods/trainer/DeepPhysTrainer.py                       (torch.load weights_only ×3)
modified: neural_methods/trainer/EfficientPhysTrainer.py                  (torch.load weights_only ×3)
modified: neural_methods/trainer/iBVPNetTrainer.py                        (torch.load weights_only ×3)
modified: neural_methods/trainer/RhythmFormerTrainer.py                   (torch.load weights_only ×3)
modified: neural_methods/trainer/PhysFormerTrainer.py                     (torch.load weights_only ×3)
modified: neural_methods/trainer/FactorizePhysTrainer.py                  (torch.load weights_only ×3)
modified: neural_methods/trainer/BigSmallTrainer.py                       (torch.load weights_only ×1)
modified: neural_methods/model/FactorizePhys/test_FactorizePhys.py        (torch.load weights_only ×1)
modified: neural_methods/model/FactorizePhys/test_FactorizePhysBig.py     (torch.load weights_only ×1)
modified: dataset/data_loader/face_detector/YOLO5Face.py                  (torch.load weights_only ×1)
modified: dataset/data_loader/UBFCrPPGLoader.py                           (raw-string regex, line 50)
added:    CHANGELOG_PY312.md                                              (this file)
```

Total: **15 source files modified**, **1 doc added**, **0 files deleted**.
