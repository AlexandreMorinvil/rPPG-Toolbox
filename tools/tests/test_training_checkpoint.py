"""Small CPU runs prove full training continuity without dataset/GPU/W&B access."""

import json
import importlib
import os
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from config import _C
from neural_methods import wandb_logger
from neural_methods.training_checkpoint import (
    TrainingPaused, atomic_save, resume_path, training_epochs,
)


class ToyTrainer:
    def __init__(self, folder, source='', epochs=4):
        self.config = _C.clone()
        self.config.TRAIN.EPOCHS = epochs
        self.config.MODEL.RESUME = source
        self.config.TEST.USE_LAST_EPOCH = False
        self.max_epoch_num = epochs
        self.model_dir = str(folder)
        self.model_file_name = 'toy'
        self.best_epoch = 0
        self.min_valid_loss = None
        self.model = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.Dropout(0.3), torch.nn.Linear(3, 1))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=0.01,
                                                             epochs=epochs, steps_per_epoch=3)
        generator = torch.Generator().manual_seed(17)
        self.loaders = {'train': DataLoader(TensorDataset(torch.arange(12).reshape(6, 2).float() / 10),
                                           batch_size=2, shuffle=True, generator=generator), 'valid': None}
        self.history = {'train_loss': [], 'valid_loss': [], 'lrs': [], 'order': []}

    def run_epoch(self, epoch):
        self.model.train()
        losses = []
        for (data,) in self.loaders['train']:
            self.history['order'].append(data.tolist())
            target = data.sum(dim=1, keepdim=True) + random.random() + np.random.random()
            self.optimizer.zero_grad()
            loss = (self.model(data) - target).square().mean()
            loss.backward()
            self.history['lrs'].append(self.scheduler.get_last_lr())
            self.optimizer.step()
            self.scheduler.step()
            wandb_logger.log_train_step(loss.item(), None, epoch, 0, every=0)
            losses.append(loss.item())
        self.history['train_loss'].append(sum(losses) / len(losses))
        valid = 0.1 + epoch
        self.history['valid_loss'].append(valid)
        if self.min_valid_loss is None or valid < self.min_valid_loss:
            self.min_valid_loss = valid
            self.best_epoch = epoch
        torch.save(self.model.state_dict(), Path(self.model_dir) / f'toy_Epoch{epoch}.pth')

    def run(self):
        for epoch in training_epochs(self, self.loaders, self.history):
            self.run_epoch(epoch)


def seed():
    random.seed(11)
    np.random.seed(11)
    torch.manual_seed(11)
    wandb_logger.restore_training_step(0)


class TrainingCheckpointTests(unittest.TestCase):
    def test_pause_resume_matches_uninterrupted_training(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            seed()
            baseline = ToyTrainer(folder / 'baseline')
            baseline.run()
            expected_rng = (random.random(), np.random.random(), torch.rand(1))
            expected_step = wandb_logger.training_step()
            seed()
            partial = ToyTrainer(folder / 'partial')
            control = folder / 'control.json'
            with patch.dict(os.environ, {'RPPG_TRAINING_CONTROL': str(control)}):
                iterator = training_epochs(partial, partial.loaders, partial.history)
                epoch = next(iterator)
                partial.run_epoch(epoch)
                control.write_text(json.dumps({'action': 'pause'}))
                with self.assertRaises(TrainingPaused):
                    next(iterator)
            path = resume_path(partial.model_dir, 'toy')
            checkpoint = torch.load(path, weights_only=False)
            self.assertEqual(checkpoint['next_epoch'], 1)
            # Simulate process restart and a new output directory, without weight files.
            seed()
            resumed = ToyTrainer(folder / 'resumed', str(path))
            resumed.run()
            self.assertEqual(baseline.history, resumed.history)
            self.assertEqual(resumed.best_epoch, 0)
            self.assertEqual(resumed.min_valid_loss, 0.1)
            self.assertEqual(wandb_logger.training_step(), expected_step)
            for key, tensor in baseline.model.state_dict().items():
                self.assertTrue(torch.equal(tensor, resumed.model.state_dict()[key]), key)
            self.assertEqual(baseline.scheduler.state_dict(), resumed.scheduler.state_dict())
            for key, state in baseline.optimizer.state_dict()['state'].items():
                for name, value in state.items():
                    self.assertTrue(torch.equal(value, resumed.optimizer.state_dict()['state'][key][name]))
            actual_rng = (random.random(), np.random.random(), torch.rand(1))
            self.assertEqual(expected_rng[:2], actual_rng[:2])
            self.assertTrue(torch.equal(expected_rng[2], actual_rng[2]))
            best = torch.load(folder / 'resumed/toy_Epoch0.pth', weights_only=True)
            original_best = torch.load(folder / 'baseline/toy_Epoch0.pth', weights_only=True)
            self.assertTrue(all(torch.equal(best[key], original_best[key]) for key in best))

    def test_crash_replays_unfinished_epoch_and_completed_run_skips_training(self):
        with tempfile.TemporaryDirectory() as directory:
            seed()
            trainer = ToyTrainer(directory)
            iterator = training_epochs(trainer, trainer.loaders, trainer.history)
            trainer.run_epoch(next(iterator))
            self.assertEqual(next(iterator), 1)  # epoch 0 is now committed
            with torch.no_grad():
                next(trainer.model.parameters()).add_(999)  # uncommitted partial epoch
            iterator.close()
            path = resume_path(directory, 'toy')
            recovered = ToyTrainer(directory, str(path))
            recovered.run()
            completed = ToyTrainer(directory, str(path))
            self.assertEqual(list(training_epochs(completed, completed.loaders, completed.history)), [])
            self.assertEqual(len(completed.history['train_loss']), 4)

    def test_rejects_weight_only_and_incompatible_schedule(self):
        with tempfile.TemporaryDirectory() as directory:
            seed()
            trainer = ToyTrainer(directory)
            trainer.run()
            path = resume_path(directory, 'toy')
            invalid = ToyTrainer(directory, str(path), epochs=5)
            with self.assertRaisesRegex(ValueError, 'configuration/data differs'):
                next(training_epochs(invalid, invalid.loaders, invalid.history))
            invalid.config.MODEL.RESUME = str(Path(directory) / 'toy_Epoch0.pth')
            with self.assertRaisesRegex(ValueError, 'Not a full training resume'):
                next(training_epochs(invalid, invalid.loaders, invalid.history))

    def test_atomic_failure_preserves_previous_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'resume.pt'
            atomic_save({'committed': 1}, path, torch.save)
            def fail(value, handle):
                handle.write(b'incomplete')
                raise OSError('disk full')
            with self.assertRaises(OSError):
                atomic_save({}, path, fail)
            self.assertEqual(torch.load(path, weights_only=True), {'committed': 1})
            self.assertEqual([file.name for file in Path(directory).iterdir()], ['resume.pt'])

    def test_real_factorize_training_loop_commits_after_validation(self):
        module = importlib.import_module('neural_methods.trainer.FactorizePhysTrainer')

        class TinyFactorizeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.mix = torch.nn.Parameter(torch.tensor([0.4, 0.6, 0.8]))
                self.dropout = torch.nn.Dropout(0.1)

            def forward(self, data):
                waveform = (data[:, :, :-1].mean(dim=(3, 4)) * self.mix[None, :, None]).sum(dim=1)
                return self.dropout(waveform), waveform

        with tempfile.TemporaryDirectory() as directory:
            seed()
            cfg = _C.clone()
            cfg.TOOLBOX_MODE = 'train_and_test'
            cfg.MODEL.NAME = 'FactorizePhys'
            cfg.MODEL.MODEL_DIR = directory
            cfg.TRAIN.MODEL_FILE_NAME = 'tiny'
            cfg.TRAIN.EPOCHS = 2
            cfg.TRAIN.PLOT_LOSSES_AND_LR = False
            cfg.TEST.USE_LAST_EPOCH = False
            cfg.NUM_OF_GPU_TRAIN = 0
            cfg.DEVICE = 'cpu'
            cfg.MODEL.FactorizePhys.TYPE = 'Standard'
            cfg.MODEL.FactorizePhys.MD_FSAM = False
            dataset = TensorDataset(torch.rand(4, 3, 16, 2, 2), torch.rand(4, 16))
            loaders = {'train': DataLoader(dataset, batch_size=2, shuffle=True,
                                          generator=torch.Generator().manual_seed(17)),
                       'valid': DataLoader(dataset, batch_size=2)}
            control = Path(directory) / 'control.json'
            with (
                patch.object(module, 'FactorizePhys', side_effect=lambda **kwargs: TinyFactorizeModel()),
                patch.object(module.torch.nn, 'DataParallel', side_effect=lambda model, **kwargs: model),
            ):
                trainer = module.FactorizePhysTrainer(cfg, loaders)
                save_model = trainer.save_model
                def save_and_request_pause(epoch):
                    save_model(epoch)
                    control.write_text(json.dumps({'action': 'pause'}))
                with patch.dict(os.environ, {'RPPG_TRAINING_CONTROL': str(control)}), \
                        patch.object(trainer, 'save_model', side_effect=save_and_request_pause):
                    with self.assertRaises(TrainingPaused):
                        trainer.train(loaders)
                source = resume_path(directory, 'tiny')
                checkpoint = torch.load(source, weights_only=False)
                self.assertEqual(checkpoint['next_epoch'], 1)
                self.assertIsNotNone(checkpoint['selection']['min_valid_loss'])
                self.assertEqual(len(checkpoint['histories']['valid_loss']), 1)
                cfg.MODEL.RESUME = str(source)
                resumed = module.FactorizePhysTrainer(cfg, loaders)
                with patch.object(resumed, 'save_model', wraps=resumed.save_model) as save:
                    resumed.train(loaders)
                    self.assertEqual([call.args[0] for call in save.call_args_list], [1])
                cfg.TRAIN.PLOT_LOSSES_AND_LR = True
                completed = module.FactorizePhysTrainer(cfg, loaders)
                with patch.object(completed, 'save_model') as save, \
                        patch.object(completed, 'plot_losses_and_lrs') as plots:
                    completed.train(loaders)
                    save.assert_not_called()
                    self.assertEqual(len(plots.call_args.args[0]), 2)
                    self.assertEqual(len(plots.call_args.args[1]), 2)


if __name__ == '__main__':
    unittest.main()