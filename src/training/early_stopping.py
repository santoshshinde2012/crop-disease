"""
Early stopping callback.

Monitors a metric and signals when training should stop
to prevent overfitting.  The logic is extracted from Trainer.fit()
so it can be tested and reused independently.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Track a monitored metric and signal when to stop.

    Args:
        patience: Number of epochs without improvement before stopping.
        mode: ``'max'`` (higher is better) or ``'min'`` (lower is better).
    """

    def __init__(self, patience: int = 5, mode: str = "max"):
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got '{mode}'")

        self.patience = patience
        self.mode = mode
        self.counter = 0
        self.best_score: float | None = None
        self.should_stop = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self, metric: float) -> bool:
        """Update the tracker with the latest metric value.

        Args:
            metric: The current epoch's monitored metric.

        Returns:
            ``True`` if training should stop, ``False`` otherwise.
        """
        if self.best_score is None or self._is_improvement(metric):
            self.best_score = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(
                    "Early stopping triggered (patience=%d)", self.patience
                )

        return self.should_stop

    @property
    def improved(self) -> bool:
        """Return ``True`` if the last ``step()`` recorded an improvement."""
        return self.counter == 0

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _is_improvement(self, metric: float) -> bool:
        if self.mode == "max":
            return metric > self.best_score
        return metric < self.best_score
