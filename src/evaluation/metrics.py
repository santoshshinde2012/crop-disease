"""
Prediction generation and classification metrics.

Functions for model inference, classification report, and summary metrics.
"""

import logging
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score, accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


@torch.no_grad()
def compute_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate predictions on a dataset.

    Uses mixed precision (AMP) on CUDA for faster inference.

    Args:
        model: Trained model in eval mode.
        loader: DataLoader for the dataset.
        device: Torch device.

    Returns:
        Tuple of (all_preds, all_labels, all_probs) as numpy arrays.
        Shapes: (N,), (N,), (N, C)
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    use_amp = device.type == "cuda"
    amp_device = "cuda" if use_amp else "cpu"

    for images, labels in tqdm(loader, desc="Generating predictions", leave=False):
        images = images.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=amp_device, enabled=use_amp):
            outputs = model(images)

        probs = torch.softmax(outputs.float(), dim=1)
        preds = probs.argmax(dim=1)

        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.numpy())
        all_probs.append(probs.cpu().numpy())

    return (
        np.concatenate(all_preds),
        np.concatenate(all_labels),
        np.concatenate(all_probs),
    )


def generate_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
) -> str:
    """Generate a formatted classification report.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        class_names: List of class name strings.

    Returns:
        Formatted classification report string.
    """
    return classification_report(
        y_true, y_pred,
        target_names=class_names,
        digits=4,
    )


def compute_summary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Compute summary classification metrics.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.

    Returns:
        Dict with keys: accuracy, f1_macro, f1_weighted.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
    }
