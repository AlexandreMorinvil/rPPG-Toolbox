import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, MaxNLocator
import os
import math

from neural_methods import wandb_logger
from neural_methods.training_checkpoint import training_epochs


class BaseTrainer:
    @staticmethod
    def add_trainer_args(parser):
        """Adds arguments to Paser for training process"""
        parser.add_argument('--lr', default=None, type=float)
        parser.add_argument('--model_file_name', default=None, type=float)
        return parser

    def __init__(self):
        pass

    def train(self, data_loader):
        pass

    def valid(self, data_loader):
        pass

    def test(self):
        pass

    def training_epochs(self, data_loader, **histories):
        """Shared epoch-boundary checkpoint/pause/resume lifecycle."""
        init_weights = self.config.MODEL.INIT_WEIGHTS
        if init_weights and not self.config.MODEL.RESUME:
            # Fine-tuning: start from released weights (e.g. final_model_release/PURE_*.pth).
            self.model.load_state_dict(torch.load(init_weights, map_location='cpu', weights_only=False))
            print(f'[training] Initialised model weights from {init_weights}', flush=True)
        return training_epochs(self, data_loader, histories)

    def record_validation_loss(self, loss, epoch):
        """Select the lowest finite loss and update shared early-stopping state."""
        loss = float(loss)
        is_best = math.isfinite(loss) and (self.min_valid_loss is None or loss < self.min_valid_loss)
        if is_best:
            self.min_valid_loss = loss
            self.best_epoch = epoch
            print("Update best model! Best epoch: {}".format(epoch))
        capture = getattr(self, '_valid_capture', None)
        if capture is not None:
            capture.commit(epoch, is_best)
        stopping = getattr(self, '_early_stopping', None)
        if stopping is not None:
            stopping.update(loss)
            self._early_stopping_epoch = epoch
            print(f'[training] Early stopping patience: {stopping.bad_epochs}/{stopping.patience} '
                  f'(tolerance {stopping.tolerance:g})', flush=True)

    def _wandb_log_batch(self, loss, epoch, batch_idx, lr=None, extra=None):
        """Forward a per-batch metric to wandb honouring config.WANDB.* settings.

        Safe to call from every trainer's batch loop: a no-op when wandb is
        disabled, when WANDB.LOG_BATCH_LOSS is False, or when the configured
        WANDB.LOG_FREQ skips the current step.
        """
        cfg = getattr(self, "config", None)
        wandb_cfg = getattr(cfg, "WANDB", None) if cfg is not None else None
        if wandb_cfg is None or not getattr(wandb_cfg, "LOG_BATCH_LOSS", False):
            return
        every = int(getattr(wandb_cfg, "LOG_FREQ", 50) or 0)
        if lr is None:
            opt = getattr(self, "optimizer", None)
            if opt is not None and getattr(opt, "param_groups", None):
                try:
                    lr = opt.param_groups[0].get("lr")
                except Exception:
                    lr = None
        try:
            loss_val = loss.item() if hasattr(loss, "item") else float(loss)
        except Exception:
            return
        wandb_logger.log_train_step(
            loss=loss_val,
            lr=lr,
            epoch=epoch,
            batch_idx=batch_idx,
            every=every,
            extra=extra,
        )

    def capture_valid(self, batch, prediction, label, frames=None):
        """Keep validation outputs for visualization (see config VISUALIZE); call once per batch.

        frames: samples per clip when the trainer flattens a batch to (N * D, 1) outputs.
        """
        if len(batch) < 4:
            return  # no recording/chunk identifiers
        capture = getattr(self, '_valid_capture', None)
        if capture is None:
            from evaluation.prediction_store import ValidationCapture
            capture = self._valid_capture = ValidationCapture(self.config)
        if capture.enabled:
            try:
                capture.add_batch(batch[2], batch[3], prediction, label, frames)
            except Exception as error:  # visualization extras must never stop training
                print(f'[outputs] Validation batch not captured: {error}')

    def save_test_outputs(self, predictions, labels, config):
        from evaluation import prediction_store

        output_dir = config.TEST.OUTPUT_SAVE_DIR
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # Filename ID to be used in any output files that get saved
        if config.TOOLBOX_MODE == 'train_and_test':
            filename_id = self.model_file_name
        elif config.TOOLBOX_MODE == 'only_test':
            model_file_root = config.INFERENCE.MODEL_PATH.split("/")[-1].split(".pth")[0]
            filename_id = model_file_root + "_" + config.TEST.DATA.DATASET
        else:
            raise ValueError('Metrics.py evaluation only supports train_and_test and only_test!')
        output_path = os.path.join(output_dir, filename_id + '_outputs.pickle')

        epoch = None
        if config.TOOLBOX_MODE == 'train_and_test':
            epoch = getattr(self, 'max_epoch_num', 0) - 1 if config.TEST.USE_LAST_EPOCH                 else getattr(self, 'best_epoch', None)
        meta = prediction_store.run_meta(
            config, config.TEST.DATA, 'test', name=filename_id, epoch=epoch,
            prediction_is_diff=prediction_store.prediction_is_diff(config))
        data = prediction_store.write_outputs_pickle(
            output_path, predictions, labels, config.TEST.DATA.PREPROCESS.LABEL_TYPE, config.TEST.DATA.FS,
            split='test', meta=meta,
            gt=prediction_store.vhrm_ground_truth_or_none(config.TEST.DATA, predictions))
        if config.VISUALIZE.HIGHLIGHTS:
            from evaluation.prediction_plots import render_highlights
            render_highlights(data, output_dir, worst=config.VISUALIZE.HIGHLIGHT_WORST)

    def plot_losses_and_lrs(self, train_loss, valid_loss, lrs, config):

        output_dir = os.path.join(config.LOG.PATH, config.TRAIN.DATA.EXP_DATA_NAME, 'plots')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # Filename ID to be used in plots that get saved
        if config.TOOLBOX_MODE == 'train_and_test':
            filename_id = self.model_file_name
        else:
            raise ValueError('Metrics.py evaluation only supports train_and_test and only_test!')
        
        # Create a single plot for training and validation losses
        plt.figure(figsize=(10, 6))
        epochs = range(0, len(train_loss))  # Integer values for x-axis
        plt.plot(epochs, train_loss, label='Training Loss')
        if len(valid_loss) > 0:
            plt.plot(epochs, valid_loss, label='Validation Loss')
        else:
            print("The list of validation losses is empty. The validation loss will not be plotted!")
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'{filename_id} Losses')
        plt.legend()
        plt.xticks(epochs)

        # Set y-axis ticks with more granularity
        ax = plt.gca()
        ax.yaxis.set_major_locator(MaxNLocator(integer=False, prune='both'))

        loss_plot_filename = os.path.join(output_dir, filename_id + '_losses.pdf')
        plt.savefig(loss_plot_filename, dpi=300)
        # Also save a PNG sibling so it can be embedded as a wandb image.
        loss_plot_png = os.path.join(output_dir, filename_id + '_losses.png')
        plt.savefig(loss_plot_png, dpi=150, bbox_inches='tight')
        plt.close()

        # Create a separate plot for learning rates
        plt.figure(figsize=(6, 4))
        scheduler_steps = range(0, len(lrs))
        plt.plot(scheduler_steps, lrs, label='Learning Rate')
        plt.xlabel('Scheduler Step')
        plt.ylabel('Learning Rate')
        plt.title(f'{filename_id} LR Schedule')
        plt.legend()

        # Set y-axis values in scientific notation
        ax = plt.gca()
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True, useOffset=False))
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))  # Force scientific notation

        lr_plot_filename = os.path.join(output_dir, filename_id + '_learning_rates.pdf')
        plt.savefig(lr_plot_filename, bbox_inches='tight', dpi=300)
        lr_plot_png = os.path.join(output_dir, filename_id + '_learning_rates.png')
        plt.savefig(lr_plot_png, bbox_inches='tight', dpi=150)
        plt.close()

        print('Saving plots of losses and learning rates to:', output_dir)

        # ---- wandb integration -------------------------------------------------
        # Replay the per-epoch curves so the wandb run shows train/valid loss
        # vs epoch, and per-step learning rates.
        if wandb_logger.is_enabled():
            for epoch_idx, tl in enumerate(train_loss):
                metrics = {"train/epoch_loss": float(tl), "epoch": epoch_idx}
                if epoch_idx < len(valid_loss):
                    metrics["valid/epoch_loss"] = float(valid_loss[epoch_idx])
                wandb_logger.log(metrics)
            for step_idx, lr in enumerate(lrs):
                # get_last_lr() returns one value per optimizer parameter group.
                # Match batch logging by reporting the first group's scalar LR;
                # the saved plot above still includes every parameter group.
                if isinstance(lr, (list, tuple)):
                    lr = lr[0]
                wandb_logger.log({"train/lr": float(lr), "scheduler_step": step_idx})
            wandb_logger.log_image("plots/losses", loss_plot_png)
            wandb_logger.log_image("plots/learning_rate", lr_plot_png)
