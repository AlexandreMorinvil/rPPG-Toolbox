import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, MaxNLocator
import os
import pickle

from neural_methods import wandb_logger


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

    def save_test_outputs(self, predictions, labels, config):
    
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

        data = dict()
        data['predictions'] = predictions
        data['labels'] = labels
        data['label_type'] = config.TEST.DATA.PREPROCESS.LABEL_TYPE
        data['fs'] = config.TEST.DATA.FS

        with open(output_path, 'wb') as handle: # save out frame dict pickle file
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print('Saving outputs to:', output_path)

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
                wandb_logger.log({"train/lr": float(lr), "scheduler_step": step_idx})
            wandb_logger.log_image("plots/losses", loss_plot_png)
            wandb_logger.log_image("plots/learning_rate", lr_plot_png)
