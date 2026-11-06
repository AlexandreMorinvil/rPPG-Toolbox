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
    if isinstance(settings.MIN_EPOCHS, bool) or not isinstance(settings.MIN_EPOCHS, int) or settings.MIN_EPOCHS < 0:
        raise ValueError('TRAIN.EARLY_STOPPING.MIN_EPOCHS must be a non-negative integer.')
    if not isinstance(settings.PATIENCE_AFTER_MIN_EPOCHS, bool):
        raise ValueError('TRAIN.EARLY_STOPPING.PATIENCE_AFTER_MIN_EPOCHS must be true or false.')


class EarlyStopping:
    """Stop after patience consecutive epochs without a significant reduction, never before min_epochs epochs.

    Tolerance is an absolute minimum reduction, compared with the last
    significant improvement (so several small reductions can accumulate).
    Actual lowest-loss checkpoint selection remains independent of tolerance.
    Non-finite losses count as non-improving epochs. Epochs before min_epochs
    still count against the patience, so a patience already used up stops
    training as soon as min_epochs epochs are done. With
    patience_after_min_epochs, only the epochs after min_epochs count, compared
    with the best loss so far, so training lasts at least min_epochs + patience.
    """

    def __init__(self, patience, tolerance, min_epochs=0, patience_after_min_epochs=False):
        self.patience = patience
        self.tolerance = tolerance
        self.min_epochs = min_epochs
        self.patience_after_min_epochs = patience_after_min_epochs
        self.epochs = 0
        self.best_loss = None
        self.bad_epochs = 0
        self.stopped = False

    def _should_stop(self):
        return self.bad_epochs >= self.patience and self.epochs >= self.min_epochs

    def update(self, loss):
        loss = float(loss)
        self.epochs += 1
        if math.isfinite(loss) and (self.best_loss is None or self.best_loss - loss > self.tolerance):
            self.best_loss = loss
            self.bad_epochs = 0
        elif not self.patience_after_min_epochs or self.epochs > self.min_epochs:
            self.bad_epochs += 1
        self.stopped = self._should_stop()
        return self.stopped

    def state_dict(self):
        return {'best_loss': self.best_loss, 'bad_epochs': self.bad_epochs, 'stopped': self.stopped}

    def load_state_dict(self, state, epochs):
        """epochs: how many are done, which the checkpoint records as its next epoch."""
        self.epochs = epochs
        self.best_loss = state['best_loss']
        self.bad_epochs = state['bad_epochs']
        if self.patience_after_min_epochs:
            self.bad_epochs = min(self.bad_epochs, max(epochs - self.min_epochs, 0))
        self.stopped = self._should_stop()  # the settings may differ from the saved ones

    def replay(self, losses):
        """The state these settings would have reached on these validation losses."""
        self.epochs, self.best_loss, self.bad_epochs, self.stopped = 0, None, 0, False
        for loss in losses:
            self.update(loss)