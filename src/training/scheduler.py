"""
Learning rate scheduler factory.

Creates scheduler instances from a string identifier.
Decoupled from the Trainer so schedulers can be tested and reused
independently.
"""

import torch


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str,
    num_epochs: int,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Create a learning rate scheduler.

    Args:
        optimizer: The optimizer to schedule.
        scheduler_type: One of ``'cosine'``, ``'step'``, ``'plateau'``.
        num_epochs: Total number of training epochs (used by cosine).

    Returns:
        A PyTorch learning rate scheduler instance.

    Raises:
        ValueError: If *scheduler_type* is not recognised.
    """
    if scheduler_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs, eta_min=1e-7
        )
    elif scheduler_type == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=5, gamma=0.5
        )
    elif scheduler_type == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3
        )
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_type}")
