"""Tests for scheduler factory."""

import pytest
import torch

from src.training.scheduler import create_scheduler


class TestScheduler:
    """Test suite for create_scheduler."""

    @pytest.fixture
    def optimizer(self):
        """Dummy optimizer with a single parameter."""
        param = torch.nn.Parameter(torch.zeros(1))
        return torch.optim.AdamW([param], lr=1e-3)

    @pytest.mark.parametrize("name", ["cosine", "step", "plateau"])
    def test_supported_schedulers(self, optimizer, name):
        """All supported names should return a scheduler instance."""
        scheduler = create_scheduler(optimizer, name, num_epochs=10)
        assert scheduler is not None

    def test_unknown_scheduler_raises(self, optimizer):
        """Unknown scheduler name should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown scheduler"):
            create_scheduler(optimizer, "unknown_scheduler", num_epochs=10)

    def test_cosine_type(self, optimizer):
        """Cosine scheduler should be CosineAnnealingLR."""
        scheduler = create_scheduler(optimizer, "cosine", num_epochs=10)
        assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)

    def test_step_type(self, optimizer):
        """Step scheduler should be StepLR."""
        scheduler = create_scheduler(optimizer, "step", num_epochs=10)
        assert isinstance(scheduler, torch.optim.lr_scheduler.StepLR)

    def test_plateau_type(self, optimizer):
        """Plateau scheduler should be ReduceLROnPlateau."""
        scheduler = create_scheduler(optimizer, "plateau", num_epochs=10)
        assert isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
