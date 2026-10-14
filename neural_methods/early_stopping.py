"""Validation-loss patience, independent of model checkpoint selection."""

import math


def validate_settings(settings, use_last_epoch):
    """Fail before training if enabled settings cannot monitor validation."""
    if not settings.ENABLED:
        return
    if use_last_epoch:
        raise ValueError('Early stopping requires TEST.USE_LAST_EPOCH: false and a validation split.')
    if isinstance(settings.PATIENCE, bool) or not isinstance(settings.PATIENCE, int) or settings.PATIENCE < 1:
        raise ValueError('TRAIN.EARLY_STOPPING.PATIENCE must be an integer of at least 1.')
    if not math.isfinite(settings.TOLERANCE) or settings.TOLERANCE < 0:
        raise ValueError('TRAIN.EARLY_STOPPING.TOLERANCE must be finite and non-negative.')


class EarlyStopping:
    """Stop after patience consecutive epochs without a significant reduction.

    Tolerance is an absolute minimum reduction, compared with the last
    significant improvement (so several small reductions can accumulate).
    Actual lowest-loss checkpoint selection remains independent of tolerance.
    Non-finite losses count as non-improving epochs.
    """

    def __init__(self, patience, tolerance):
        self.patience = patience
        self.tolerance = tolerance
        self.best_loss = None
        self.bad_epochs = 0
        self.stopped = False

    def update(self, loss):
        loss = float(loss)
        if math.isfinite(loss) and (self.best_loss is None or self.best_loss - loss > self.tolerance):
            self.best_loss = loss
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        self.stopped = self.bad_epochs >= self.patience
        return self.stopped

    def state_dict(self):
        return {'best_loss': self.best_loss, 'bad_epochs': self.bad_epochs, 'stopped': self.stopped}

    def load_state_dict(self, state):
        self.best_loss = state['best_loss']
        self.bad_epochs = state['bad_epochs']
        self.stopped = state['stopped']