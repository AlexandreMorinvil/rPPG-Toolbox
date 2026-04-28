import neural_methods.trainer.BaseTrainer
import neural_methods.trainer.PhysnetTrainer
import neural_methods.trainer.iBVPNetTrainer
import neural_methods.trainer.TscanTrainer
import neural_methods.trainer.DeepPhysTrainer
import neural_methods.trainer.EfficientPhysTrainer
import neural_methods.trainer.BigSmallTrainer
import neural_methods.trainer.PhysFormerTrainer
import neural_methods.trainer.RhythmFormerTrainer
import neural_methods.trainer.FactorizePhysTrainer

# PhysMamba depends on `mamba-ssm` and `causal-conv1d`, which have no Windows
# wheels and require Linux + CUDA toolkit + nvcc to build from source. Import
# lazily so the rest of the toolbox keeps working when those deps are missing.
try:
    import neural_methods.trainer.PhysMambaTrainer  # noqa: F401
    PHYSMAMBA_AVAILABLE = True
    PHYSMAMBA_IMPORT_ERROR = None
except ImportError as _physmamba_err:  # pragma: no cover - environment-dependent
    PHYSMAMBA_AVAILABLE = False
    PHYSMAMBA_IMPORT_ERROR = _physmamba_err
