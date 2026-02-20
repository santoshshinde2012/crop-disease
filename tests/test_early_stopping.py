"""Tests for EarlyStopping callback."""

import pytest

from src.training.early_stopping import EarlyStopping


class TestEarlyStopping:
    """Test suite for the EarlyStopping class."""

    def test_no_stop_while_improving(self):
        """Should not stop if the metric keeps improving."""
        stopper = EarlyStopping(patience=3, mode="max")
        for val in [0.1, 0.2, 0.3, 0.4, 0.5]:
            assert stopper.step(val) is False

    def test_stops_after_patience(self):
        """Should stop after *patience* epochs without improvement."""
        stopper = EarlyStopping(patience=3, mode="max")
        stopper.step(0.5)  # New best
        stopper.step(0.4)  # No improvement (1)
        stopper.step(0.3)  # No improvement (2)
        assert stopper.step(0.2) is True  # No improvement (3) → stop

    def test_improved_flag(self):
        """improved should be True only on improvement epochs."""
        stopper = EarlyStopping(patience=5, mode="max")
        stopper.step(0.5)
        assert stopper.improved is True

        stopper.step(0.4)
        assert stopper.improved is False

        stopper.step(0.6)
        assert stopper.improved is True

    def test_min_mode(self):
        """In min mode, lower values are improvements."""
        stopper = EarlyStopping(patience=2, mode="min")
        stopper.step(1.0)
        assert stopper.improved is True

        stopper.step(0.5)
        assert stopper.improved is True

        stopper.step(0.6)
        assert stopper.improved is False

    def test_invalid_mode_raises(self):
        """Invalid mode should raise ValueError."""
        with pytest.raises(ValueError, match="mode must be"):
            EarlyStopping(patience=3, mode="invalid")

    def test_counter_resets_on_improvement(self):
        """Counter should reset when a new best is found."""
        stopper = EarlyStopping(patience=3, mode="max")
        stopper.step(0.5)
        stopper.step(0.4)  # counter=1
        stopper.step(0.3)  # counter=2
        stopper.step(0.6)  # improvement → counter=0
        stopper.step(0.5)  # counter=1
        assert stopper.should_stop is False
