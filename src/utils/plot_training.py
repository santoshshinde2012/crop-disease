"""
Training analysis plots — training curves and model comparison charts.

Visualises metrics collected *during* or *after* training runs.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt


def plot_training_curves(
    histories: Dict[str, dict],
    stage_boundaries: List[int] = None,
    figsize: Tuple[int, int] = (16, 12),
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plot training curves in a 2x2 subplot grid.

    Top-left:  Train Loss vs Val Loss
    Top-right: Train Accuracy vs Val Accuracy
    Bottom-left:  Validation F1 Score
    Bottom-right: Learning Rate schedule

    Args:
        histories: Dict mapping model name to training history dict.
        stage_boundaries: List of epoch numbers where stages transition.
        figsize: Figure size.
        save_path: Optional path to save the figure.

    Returns:
        matplotlib Figure object.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    for model_name, history in histories.items():
        epochs = range(1, len(history["train_loss"]) + 1)

        axes[0, 0].plot(epochs, history["train_loss"], label=f"{model_name} (train)")
        axes[0, 0].plot(epochs, history["val_loss"], "--", label=f"{model_name} (val)")

        axes[0, 1].plot(epochs, history["train_acc"], label=f"{model_name} (train)")
        axes[0, 1].plot(epochs, history["val_acc"], "--", label=f"{model_name} (val)")

        axes[1, 0].plot(epochs, history["val_f1"], label=model_name)

        axes[1, 1].plot(epochs, history["lr"], label=model_name)

    titles = ["Train/Val Loss", "Train/Val Accuracy", "Validation F1 (Macro)", "Learning Rate"]
    ylabels = ["Loss", "Accuracy", "F1 Score", "Learning Rate"]

    for ax, title, ylabel in zip(axes.flat, titles, ylabels):
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        if stage_boundaries:
            stage_labels = ["Stage 1", "Stage 2", "Stage 3"]
            for i, boundary in enumerate(stage_boundaries):
                ax.axvline(boundary, color="gray", linestyle="--", alpha=0.5)
                if ax == axes[0, 0] and i < len(stage_labels) - 1:
                    ax.text(
                        boundary + 0.5, ax.get_ylim()[1] * 0.95,
                        stage_labels[i + 1], fontsize=8, alpha=0.7,
                    )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)

    return fig


def plot_model_comparison(
    comparison_df: "pd.DataFrame",
    figsize: Tuple[int, int] = (14, 6),
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plot model comparison scatter plots.

    Left:  Accuracy vs CPU Latency
    Right: Accuracy vs Model Size

    Args:
        comparison_df: DataFrame with columns:
            model, accuracy, f1_macro, model_size_mb, cpu_latency_mean_ms
        figsize: Figure size.
        save_path: Optional path to save the figure.

    Returns:
        matplotlib Figure object.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]

    # Accuracy vs CPU Latency
    for i, (_, row) in enumerate(comparison_df.iterrows()):
        ax1.scatter(
            row["cpu_latency_mean_ms"], row["accuracy"],
            s=200, c=colors[i % len(colors)], zorder=5, edgecolors="black",
        )
        ax1.annotate(
            row["model"], (row["cpu_latency_mean_ms"], row["accuracy"]),
            textcoords="offset points", xytext=(10, 5), fontsize=10,
        )

    ax1.set_xlabel("CPU Latency (ms)", fontsize=12)
    ax1.set_ylabel("Accuracy", fontsize=12)
    ax1.set_title("Accuracy vs CPU Latency", fontsize=14)
    ax1.grid(True, alpha=0.3)

    # Accuracy vs Model Size
    for i, (_, row) in enumerate(comparison_df.iterrows()):
        ax2.scatter(
            row["model_size_mb"], row["accuracy"],
            s=200, c=colors[i % len(colors)], zorder=5, edgecolors="black",
        )
        ax2.annotate(
            row["model"], (row["model_size_mb"], row["accuracy"]),
            textcoords="offset points", xytext=(10, 5), fontsize=10,
        )

    ax2.set_xlabel("Model Size (MB)", fontsize=12)
    ax2.set_ylabel("Accuracy", fontsize=12)
    ax2.set_title("Accuracy vs Model Size", fontsize=14)
    ax2.grid(True, alpha=0.3)

    # Highlight sweet spot region
    for ax in (ax1, ax2):
        ymin, ymax = ax.get_ylim()
        xmin, xmax = ax.get_xlim()
        sweet_y = ymin + (ymax - ymin) * 0.7
        ax.axhspan(sweet_y, ymax, xmin=0, xmax=0.4, alpha=0.1, color="green", label="Sweet Spot")
        ax.legend(fontsize=9)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)

    return fig
