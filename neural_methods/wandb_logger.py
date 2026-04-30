"""Lightweight Weights & Biases (wandb) integration for rPPG-Toolbox.

This module exposes a tiny, process-global facade so the rest of the codebase
(trainers, evaluation routines, plotting helpers) can call short functions like
``wandb_logger.log({...})`` without caring whether wandb is installed,
configured, or disabled in the current run.

Design goals
------------
* **Zero overhead when disabled.** If ``config.WANDB.ENABLED`` is False (the
  default), every public function is a no-op and ``wandb`` is not even
  imported.
* **Single init point.** ``init(config, ...)`` is called once from ``main.py``
  before training/testing/inference dispatch. ``finish()`` is called once at
  the very end. Re-calling ``init`` is safe.
* **Robust to missing dependency.** If the user enables wandb but the package
  is not installed, we print a single warning and silently disable logging
  rather than crashing the training run.
* **Robust to network failures.** Logging exceptions are swallowed (with a
  one-time warning) so a wandb outage cannot kill a long-running experiment.
"""

from __future__ import annotations

import os
import warnings
from typing import Any, Dict, Iterable, Optional

# These are populated lazily inside ``init`` so we don't import wandb when the
# integration is disabled.
_wandb = None              # the ``wandb`` module, or None if disabled/unavailable
_run = None                # the active wandb Run object, or None
_enabled = False           # mirrors config.WANDB.ENABLED & dependency presence
_warned_log_failure = False
_global_step = 0           # monotonically-increasing optimiser-step counter


def is_enabled() -> bool:
    """Return True if wandb is initialised and ready to receive log calls."""
    return _enabled and _run is not None


def _cfg_to_dict(cfg) -> Dict[str, Any]:
    """Convert a yacs CfgNode (or any mapping-ish object) to a plain dict."""
    try:
        # yacs CfgNode supports .dump() -> YAML; safer to walk it manually.
        import yaml  # local import; PyYAML is already a hard dep of the toolbox
        return yaml.safe_load(cfg.dump())
    except Exception:
        # Fall back to a best-effort dict cast.
        try:
            return dict(cfg)
        except Exception:
            return {"repr": repr(cfg)}


def init(config, extra_config: Optional[Dict[str, Any]] = None) -> None:
    """Initialise wandb from a frozen rPPG-Toolbox config.

    Safe to call multiple times: subsequent calls are ignored if a run is
    already active. Does nothing if ``config.WANDB.ENABLED`` is False.
    """
    global _wandb, _run, _enabled

    if _run is not None:
        # A run is already active; honour the first init.
        return

    wandb_cfg = getattr(config, "WANDB", None)
    if wandb_cfg is None or not getattr(wandb_cfg, "ENABLED", False):
        _enabled = False
        return

    try:
        import wandb  # type: ignore
    except ImportError:
        warnings.warn(
            "config.WANDB.ENABLED=True but the 'wandb' package is not installed. "
            "Install it with `pip install wandb` (or set WANDB.ENABLED: False). "
            "Continuing without wandb logging.",
            RuntimeWarning,
        )
        _enabled = False
        return

    _wandb = wandb

    init_kwargs: Dict[str, Any] = {
        "project": wandb_cfg.PROJECT or "rPPG-Toolbox",
        "name": wandb_cfg.RUN_NAME or None,
        "mode": wandb_cfg.MODE or "online",
        "config": _cfg_to_dict(config),
        "reinit": True,
    }
    if wandb_cfg.ENTITY:
        init_kwargs["entity"] = wandb_cfg.ENTITY
    if wandb_cfg.GROUP:
        init_kwargs["group"] = wandb_cfg.GROUP
    if wandb_cfg.JOB_TYPE:
        init_kwargs["job_type"] = wandb_cfg.JOB_TYPE
    if wandb_cfg.TAGS:
        init_kwargs["tags"] = list(wandb_cfg.TAGS)
    if wandb_cfg.NOTES:
        init_kwargs["notes"] = wandb_cfg.NOTES
    if extra_config:
        # Merge extra config under a sub-key so we never collide with toolbox keys.
        init_kwargs["config"] = {**init_kwargs["config"], "_runtime": extra_config}

    try:
        _run = _wandb.init(**init_kwargs)
        _enabled = True
        print(f"[wandb] Initialised run '{_run.name}' (id={_run.id}) "
              f"in project '{init_kwargs['project']}' [mode={init_kwargs['mode']}].")
        # Tell wandb which custom x-axes to use for the toolbox metrics so the
        # dashboard plots "loss vs epoch" rather than "loss vs implicit step".
        try:
            _wandb.define_metric("epoch")
            _wandb.define_metric("scheduler_step")
            _wandb.define_metric("global_step")
            _wandb.define_metric("train/epoch_loss", step_metric="epoch", summary="min")
            _wandb.define_metric("valid/epoch_loss", step_metric="epoch", summary="min")
            _wandb.define_metric("train/lr", step_metric="scheduler_step")
            _wandb.define_metric("train/batch_loss", step_metric="global_step")
            _wandb.define_metric("train/batch_lr", step_metric="global_step")
        except Exception:
            pass
    except Exception as exc:  # pragma: no cover - depends on network/auth
        warnings.warn(f"[wandb] init failed ({exc!r}); continuing without wandb.",
                      RuntimeWarning)
        _wandb = None
        _run = None
        _enabled = False


def watch(model, log: str = "gradients", log_freq: int = 100) -> None:
    """Forward to ``wandb.watch`` if wandb is enabled."""
    if not is_enabled():
        return
    try:
        _wandb.watch(model, log=log, log_freq=log_freq)
    except Exception as exc:  # pragma: no cover
        _warn_once(f"[wandb] watch failed: {exc!r}")


def log(metrics: Dict[str, Any], step: Optional[int] = None,
        commit: Optional[bool] = None) -> None:
    """Log a metrics dict. No-op when disabled."""
    if not is_enabled() or not metrics:
        return
    try:
        if step is not None:
            _wandb.log(metrics, step=step, commit=commit)
        else:
            _wandb.log(metrics, commit=commit)
    except Exception as exc:  # pragma: no cover
        _warn_once(f"[wandb] log failed: {exc!r}")


def log_image(key: str, path: str, caption: Optional[str] = None,
              step: Optional[int] = None) -> None:
    """Log a single image file to wandb under ``key``.

    PDFs are uploaded as wandb Artifacts/Files since wandb.Image only renders
    raster formats well; for non-image files we fall back to ``run.save``.
    """
    if not is_enabled() or not path or not os.path.exists(path):
        return
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext in (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"):
            payload = {key: _wandb.Image(path, caption=caption)}
            log(payload, step=step)
        else:
            # PDFs / SVGs: save the file alongside the run so it shows up under
            # the run's "Files" tab.
            try:
                _run.save(path, base_path=os.path.dirname(path) or ".",
                          policy="now")
            except Exception:
                # Older wandb versions: just upload via Artifact-like fallback.
                _run.log_artifact(path) if hasattr(_run, "log_artifact") else None
    except Exception as exc:  # pragma: no cover
        _warn_once(f"[wandb] log_image failed for {path}: {exc!r}")


def log_summary(metrics: Dict[str, Any]) -> None:
    """Update the run's summary metrics (sticky values shown in the dashboard)."""
    if not is_enabled() or not metrics:
        return
    try:
        for k, v in metrics.items():
            _run.summary[k] = v
    except Exception as exc:  # pragma: no cover
        _warn_once(f"[wandb] summary update failed: {exc!r}")


def define_metric(name: str, step_metric: str = "epoch",
                  summary: Optional[str] = None) -> None:
    """Wrapper around ``wandb.define_metric`` (silently ignored when disabled)."""
    if not is_enabled():
        return
    try:
        kwargs = {"step_metric": step_metric}
        if summary is not None:
            kwargs["summary"] = summary
        _wandb.define_metric(name, **kwargs)
    except Exception as exc:  # pragma: no cover
        _warn_once(f"[wandb] define_metric failed: {exc!r}")


def log_train_step(loss: float, lr: Optional[float], epoch: int,
                   batch_idx: int, every: int = 50,
                   extra: Optional[Dict[str, Any]] = None) -> None:
    """Log a per-batch training step.

    Maintains an internal monotonically-increasing ``global_step`` so the
    caller does not have to. ``every`` controls log frequency in batches; set
    to 0 or a negative value to disable batch-level logging entirely. Always
    a no-op when wandb is not enabled.
    """
    global _global_step
    _global_step += 1
    if not is_enabled() or every is None or every <= 0:
        return
    if (_global_step % every) != 0:
        return
    payload: Dict[str, Any] = {
        "train/batch_loss": float(loss),
        "epoch": int(epoch),
        "batch": int(batch_idx),
        "global_step": int(_global_step),
    }
    if lr is not None:
        try:
            payload["train/batch_lr"] = float(lr)
        except (TypeError, ValueError):
            pass
    if extra:
        payload.update(extra)
    log(payload)


def reset_step_counter() -> None:
    """Reset the internal global-step counter (useful between runs in tests)."""
    global _global_step
    _global_step = 0


def finish() -> None:
    """Close the active run, if any. Safe to call multiple times."""
    global _run, _enabled
    if _run is None:
        return
    try:
        _wandb.finish()
    except Exception as exc:  # pragma: no cover
        warnings.warn(f"[wandb] finish failed: {exc!r}", RuntimeWarning)
    _run = None
    _enabled = False


def _warn_once(msg: str) -> None:
    global _warned_log_failure
    if _warned_log_failure:
        return
    _warned_log_failure = True
    warnings.warn(msg, RuntimeWarning)


__all__ = [
    "init",
    "watch",
    "log",
    "log_image",
    "log_summary",
    "define_metric",
    "log_train_step",
    "reset_step_counter",
    "finish",
    "is_enabled",
]
