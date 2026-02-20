"""
Confusion matrix visualization.

Generates normalized confusion matrices with shortened class names
and detailed heatmap visualization.
"""

from pathlib import Path
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

from ..utils.text_helpers import shorten_class_name


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    normalize: bool = True,
    save_path: Optional[Union[str, Path]] = None,
    figsize: tuple = (12, 10),
) -> plt.Figure:
    """Plot a confusion matrix heatmap.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        class_names: List of class name strings.
        normalize: If True, normalize by row (shows recall per class).
        save_path: Optional path to save the figure.
        figsize: Figure size.

    Returns:
        matplotlib Figure object.
    """
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        fmt = ".2f"
        title = "Normalized Confusion Matrix (Recall per Class)"
    else:
        fmt = "d"
        title = "Confusion Matrix"

    short_names = [shorten_class_name(n) for n in class_names]

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        square=True,
        xticklabels=short_names,
        yticklabels=short_names,
        ax=ax,
        vmin=0,
        vmax=1.0 if normalize else None,
        linewidths=0.5,
    )

    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(title, fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)

    return fig
