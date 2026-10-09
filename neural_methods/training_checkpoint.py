"""Atomic, epoch-boundary training recovery shared by every neural trainer.

Resume files are trusted local PyTorch checkpoints, not inference weights.
An unfinished epoch is deliberately replayed rather than guessing sampler state.
"""

import copy
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch
import yaml

from neural_methods import wandb_logger

PAUSED_EXIT_CODE = 75
FORMAT_VERSION = 1


class TrainingPaused(RuntimeError):
    """A checkpoint is committed and the process can release its resources."""


def atomic_save(value, path, writer):
    """Keep the previous committed file intact if writing is interrupted."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            writer(value, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def resume_path(model_dir, model_file_name):
    return Path(model_dir) / (model_file_name + '_resume.pt')


def _signature(trainer, loaders):
    """Reject changes that invalidate the saved optimizer/schedule or data split."""
    cfg = yaml.safe_load(trainer.config.dump())
    model = copy.deepcopy(cfg['MODEL'])
    for key in ('MODEL_DIR', 'RESUME'):
        model.pop(key, None)
    train = copy.deepcopy(cfg['TRAIN'])
    for key in ('PLOT_LOSSES_AND_LR', 'SAVE_RESUME'):
        train.pop(key, None)
    valid = copy.deepcopy(cfg['VALID'])
    for section in (train, valid):
        section['DATA'].pop('DO_PREPROCESS', None)
        section['DATA']['PREPROCESS'].pop('NUM_WORKERS', None)
    file_lists = {}
    for key in ('train', 'valid'):
        loader = loaders.get(key)
        inputs = getattr(getattr(loader, 'dataset', None), 'inputs', None)
        if inputs is not None:
            file_lists[key] = hashlib.sha256(json.dumps(list(inputs)).encode('utf-8')).hexdigest()
    return {
        'trainer': type(trainer).__name__, 'model': model, 'train': train, 'valid': valid,
        'optimizer': type(trainer.optimizer).__name__,
        'scheduler': type(getattr(trainer, 'scheduler', None)).__name__,
        'file_lists': file_lists,
        'use_last_epoch': cfg['TEST']['USE_LAST_EPOCH'],
        'num_gpus': cfg['NUM_OF_GPU_TRAIN'],
        'batches': {key: len(loaders[key]) if loaders.get(key) is not None else None
                    for key in ('train', 'valid')},
    }


def _rng_state(loaders):
    return {
        'python': random.getstate(), 'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        'generators': {key: loader.generator.get_state() for key, loader in loaders.items()
                       if loader is not None and getattr(loader, 'generator', None) is not None},
    }


def _restore_rng(state, loaders):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None and torch.cuda.is_available():
        if len(state['cuda']) != torch.cuda.device_count():
            raise ValueError('Resume requires the same number of visible CUDA devices.')
        torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda']])
    for key, value in state['generators'].items():
        loader = loaders.get(key)
        if loader is not None and getattr(loader, 'generator', None) is not None:
            loader.generator.set_state(value.cpu())


def _pause_if_requested(path):
    control = os.environ.get('RPPG_TRAINING_CONTROL')
    if not control:
        return
    try:
        requested = json.loads(Path(control).read_text(encoding='utf-8')).get('action') == 'pause'
    except FileNotFoundError:
        return
    if requested:
        print(f'[training] Paused at a safe epoch boundary. Resume checkpoint: {path}', flush=True)
        raise TrainingPaused(str(path))


def training_epochs(trainer, loaders, histories):
    """Yield epochs, committing only after the caller finishes training + validation.

    The histories dictionary holds the trainer's mutable lists so restoration is
    visible to the existing plotting code without changing trainer APIs.
    """
    enabled = trainer.config.TRAIN.SAVE_RESUME
    source = trainer.config.MODEL.RESUME
    if source and not enabled:
        raise ValueError('MODEL.RESUME requires TRAIN.SAVE_RESUME: true.')
    if not enabled:
        yield from range(trainer.max_epoch_num)
        return

    target = resume_path(trainer.model_dir, trainer.model_file_name)
    signature = _signature(trainer, loaders)
    start_epoch = 0
    best_weights = None
    best_saved_epoch = None
    if source:
        # Contains optimizer + Python/NumPy RNG state: only load trusted files.
        saved = torch.load(source, map_location='cpu', weights_only=False)
        if not isinstance(saved, dict) or saved.get('format_version') != FORMAT_VERSION:
            raise ValueError('Not a full training resume checkpoint; use an *_resume.pt file, '
                             'not an inference-only *_EpochN.pth file.')
        if saved['signature'] != signature:
            raise ValueError('Resume configuration/data differs from the checkpoint. Keep the same '
                             'model, epochs, batch size, optimizer settings and training/validation splits.')
        trainer.model.load_state_dict(saved['model'])
        trainer.optimizer.load_state_dict(saved['optimizer'])
        scheduler = getattr(trainer, 'scheduler', None)
        if scheduler is not None:
            scheduler.load_state_dict(saved['scheduler'])
        scaler = getattr(trainer, 'scaler', None)
        if scaler is not None:
            scaler.load_state_dict(saved['scaler'])
        for name, value in saved['selection'].items():
            setattr(trainer, name, value)
        for name, values in saved['histories'].items():
            if name in histories:
                histories[name][:] = values
        start_epoch = saved['next_epoch']
        best_weights = saved['best_model']
        best_saved_epoch = saved['best_saved_epoch']
        # Preserve the inference/test contract even when copying a resume file
        # to a different output folder and the old weight files are unavailable.
        if start_epoch > 0:
            atomic_save(saved['model'], target.parent / f'{trainer.model_file_name}_Epoch{start_epoch - 1}.pth', torch.save)
        if best_weights is not None:
            atomic_save(best_weights, target.parent / f'{trainer.model_file_name}_Epoch{best_saved_epoch}.pth', torch.save)
        wandb_logger.restore_training_step(saved['wandb_step'])
        _restore_rng(saved['rng'], loaders)
        print(f'[training] Resuming at epoch {start_epoch} / {trainer.max_epoch_num} from {source}', flush=True)

    def commit(next_epoch):
        nonlocal best_weights, best_saved_epoch
        selected = getattr(trainer, 'used_epoch', getattr(trainer, 'best_epoch', None))
        if next_epoch > 0 and not trainer.config.TEST.USE_LAST_EPOCH and selected != best_saved_epoch:
            weight_file = target.parent / f'{trainer.model_file_name}_Epoch{selected}.pth'
            best_weights = torch.load(weight_file, map_location='cpu', weights_only=True)
            best_saved_epoch = selected
        scheduler = getattr(trainer, 'scheduler', None)
        scaler = getattr(trainer, 'scaler', None)
        payload = {
            'format_version': FORMAT_VERSION, 'signature': signature, 'next_epoch': next_epoch,
            'model': trainer.model.state_dict(), 'optimizer': trainer.optimizer.state_dict(),
            'scheduler': scheduler.state_dict() if scheduler is not None else None,
            'scaler': scaler.state_dict() if scaler is not None else None,
            'selection': {key: getattr(trainer, key) for key in ('best_epoch', 'min_valid_loss', 'used_epoch')
                          if hasattr(trainer, key)},
            'best_model': best_weights, 'best_saved_epoch': best_saved_epoch,
            'histories': histories, 'rng': _rng_state(loaders),
            'wandb_step': wandb_logger.training_step(),
        }
        atomic_save(payload, target, torch.save)
        metadata = {'format_version': FORMAT_VERSION, 'next_epoch': next_epoch,
                    'epochs_total': trainer.max_epoch_num, 'training_complete': next_epoch == trainer.max_epoch_num}
        atomic_save(metadata, target.with_suffix('.json'),
                    lambda value, handle: handle.write(json.dumps(value).encode('utf-8')))
        print(f'[training] Resume checkpoint saved (next epoch {next_epoch}): {target}', flush=True)

    # Also protect a crash/stop during the first epoch.
    commit(start_epoch)
    _pause_if_requested(target)
    for epoch in range(start_epoch, trainer.max_epoch_num):
        yield epoch
        commit(epoch + 1)
        _pause_if_requested(target)