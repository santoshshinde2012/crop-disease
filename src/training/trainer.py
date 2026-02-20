"""
Training engine with three-stage progressive fine-tuning.

Handles training loop, validation, checkpointing, mixed precision,
and gradient clipping.  Scheduler creation and early-stopping logic
are delegated to dedicated modules for single-responsibility.
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .early_stopping import EarlyStopping
from .scheduler import create_scheduler

logger = logging.getLogger(__name__)

try:
    import torchmetrics
    HAS_TORCHMETRICS = True
except ImportError:
    HAS_TORCHMETRICS = False


class Trainer:
    """Training engine for plant disease classification.

    Args:
        model: The neural network model.
        num_classes: Number of output classes.
        learning_rate: Initial learning rate.
        weight_decay: AdamW decoupled weight decay.
        label_smoothing: Label smoothing factor for CrossEntropyLoss.
        device: Torch device (cuda/cpu/mps).
        checkpoint_dir: Directory to save model checkpoints.
        model_name: Architecture name for checkpoint filenames.
        max_grad_norm: Gradient clipping ceiling.
        use_amp: If True, use mixed precision training.
    """

    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        label_smoothing: float = 0.1,
        device: torch.device = None,
        checkpoint_dir: Path = Path("models"),
        model_name: str = "model",
        max_grad_norm: float = 1.0,
        use_amp: bool = True,
        param_groups: Optional[list] = None,
    ):
        self.model = model
        self.num_classes = num_classes
        self.device = device or torch.device("cpu")
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.max_grad_norm = max_grad_norm

        # Determine AMP device type
        if self.device.type == "cuda":
            self.amp_device = "cuda"
            self.use_amp = use_amp
        elif self.device.type == "mps":
            self.amp_device = "cpu"  # AMP not fully supported on MPS
            self.use_amp = False
        else:
            self.amp_device = "cpu"
            self.use_amp = False

        # Loss function
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        # Optimizer
        if param_groups is not None:
            self.optimizer = torch.optim.AdamW(
                param_groups, weight_decay=weight_decay
            )
        else:
            trainable_params = filter(
                lambda p: p.requires_grad, model.parameters()
            )
            self.optimizer = torch.optim.AdamW(
                trainable_params, lr=learning_rate, weight_decay=weight_decay
            )

        # Metrics
        if HAS_TORCHMETRICS:
            self.train_acc = torchmetrics.Accuracy(
                task="multiclass", num_classes=num_classes
            ).to(self.device)
            self.val_acc = torchmetrics.Accuracy(
                task="multiclass", num_classes=num_classes
            ).to(self.device)
            self.val_f1 = torchmetrics.F1Score(
                task="multiclass", num_classes=num_classes, average="macro"
            ).to(self.device)
        else:
            self.train_acc = None
            self.val_acc = None
            self.val_f1 = None

        # Mixed precision scaler
        self.scaler = torch.amp.GradScaler(enabled=self.use_amp)

        # Best metric tracking
        self.best_val_f1 = 0.0

    def _train_one_epoch(self, loader: DataLoader) -> Tuple[float, float]:
        """Run one training epoch.

        Returns:
            Tuple of (average_loss, average_accuracy).
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        if self.train_acc is not None:
            self.train_acc.reset()

        pbar = tqdm(loader, desc="Training", leave=False)
        for images, labels in pbar:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=self.amp_device, enabled=self.use_amp):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.max_grad_norm
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if self.train_acc is not None:
                self.train_acc.update(preds, labels)

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = running_loss / total
        if self.train_acc is not None:
            avg_acc = self.train_acc.compute().item()
        else:
            avg_acc = correct / total

        return avg_loss, avg_acc

    @torch.no_grad()
    def _validate(self, loader: DataLoader) -> Tuple[float, float, float]:
        """Run validation.

        Returns:
            Tuple of (average_loss, average_accuracy, average_f1).
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        if self.val_acc is not None:
            self.val_acc.reset()
        if self.val_f1 is not None:
            self.val_f1.reset()

        for images, labels in tqdm(loader, desc="Validating", leave=False):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with torch.amp.autocast(device_type=self.amp_device, enabled=self.use_amp):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if self.val_acc is not None:
                self.val_acc.update(preds, labels)
            if self.val_f1 is not None:
                self.val_f1.update(preds, labels)

        avg_loss = running_loss / total
        if self.val_acc is not None:
            avg_acc = self.val_acc.compute().item()
        else:
            avg_acc = correct / total

        if self.val_f1 is not None:
            avg_f1 = self.val_f1.compute().item()
        else:
            avg_f1 = avg_acc  # Fallback: use accuracy if torchmetrics unavailable

        return avg_loss, avg_acc, avg_f1

    def _save_checkpoint(self, epoch: int, val_f1: float) -> None:
        """Save model checkpoint with full reconstruction info."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_f1": val_f1,
            "model_name": self.model_name,
            "num_classes": self.num_classes,
        }
        path = self.checkpoint_dir / f"{self.model_name}_best.pth"
        torch.save(checkpoint, path)
        logger.info("Checkpoint saved to %s", path)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        scheduler_type: str = "cosine",
        patience: int = 5,
    ) -> Dict[str, List]:
        """Train the model with early stopping.

        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.
            num_epochs: Maximum number of epochs.
            scheduler_type: LR scheduler type.
            patience: Early stopping patience (epochs without improvement).

        Returns:
            Training history dict with keys:
            train_loss, train_acc, val_loss, val_acc, val_f1, lr, epoch_time
        """
        scheduler = create_scheduler(self.optimizer, scheduler_type, num_epochs)
        stopper = EarlyStopping(patience=patience, mode="max")

        history: Dict[str, List] = {
            "train_loss": [], "train_acc": [],
            "val_loss": [], "val_acc": [], "val_f1": [],
            "lr": [], "epoch_time": [],
        }

        for epoch in range(1, num_epochs + 1):
            start_time = time.time()

            # Training
            train_loss, train_acc = self._train_one_epoch(train_loader)

            # Validation
            val_loss, val_acc, val_f1 = self._validate(val_loader)

            # Scheduler step
            current_lr = self.optimizer.param_groups[0]["lr"]
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

            epoch_time = time.time() - start_time

            # Log history
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["val_f1"].append(val_f1)
            history["lr"].append(current_lr)
            history["epoch_time"].append(epoch_time)

            # Print epoch summary
            logger.info(
                "Epoch %d/%d | Train Loss: %.4f | Train Acc: %.4f | "
                "Val Loss: %.4f | Val Acc: %.4f | Val F1: %.4f | "
                "LR: %.2e | Time: %.1fs",
                epoch, num_epochs, train_loss, train_acc,
                val_loss, val_acc, val_f1, current_lr, epoch_time,
            )

            # Checkpointing and early stopping
            should_stop = stopper.step(val_f1)

            if stopper.improved:
                self.best_val_f1 = val_f1
                self._save_checkpoint(epoch, val_f1)

            if should_stop:
                logger.info(
                    "Early stopping at epoch %d (patience=%d)",
                    epoch, patience,
                )
                break

        return history
