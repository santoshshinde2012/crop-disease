"""
Data exploration plots — sample images, class distribution, augmentation.

Visualises dataset characteristics before any training begins.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from PIL import Image

from .text_helpers import get_crop_name, shorten_class_name


def plot_sample_images(
    dataset: "Dataset",
    num_classes: int = 5,
    images_per_class: int = 5,
    figsize: Tuple[int, int] = (20, 20),
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plot a grid of sample images from the dataset.

    Layout: *num_classes* rows x *images_per_class* columns.

    Args:
        dataset: PlantDiseaseDataset instance (without transforms).
        num_classes: Number of classes to display.
        images_per_class: Number of images per class.
        figsize: Figure size.
        save_path: Optional path to save the figure.

    Returns:
        matplotlib Figure object.
    """
    class_names = list(dataset.class_to_idx.keys())
    indices = np.linspace(0, len(class_names) - 1, num_classes, dtype=int)
    selected_classes = [class_names[i] for i in indices]

    fig, axes = plt.subplots(num_classes, images_per_class, figsize=figsize)
    fig.suptitle("Sample Images from PlantVillage Dataset", fontsize=18, y=1.02)

    for row, cls_name in enumerate(selected_classes):
        cls_idx = dataset.class_to_idx[cls_name]
        cls_samples = [
            i for i, (_, label) in enumerate(dataset.samples)
            if label == cls_idx
        ][:images_per_class]

        for col, sample_idx in enumerate(cls_samples):
            ax = axes[row, col] if num_classes > 1 else axes[col]
            img, _ = dataset[sample_idx]

            if hasattr(img, "numpy"):
                img = img.permute(1, 2, 0).numpy()

            ax.imshow(img)
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(
                    shorten_class_name(cls_name),
                    fontsize=11, rotation=0, labelpad=100, va="center",
                )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)

    return fig


def plot_class_distribution(
    class_counts: Dict[str, int],
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plot horizontal bar chart of class distribution.

    Bars are colour-coded by crop and sorted by count descending.

    Args:
        class_counts: Dict mapping class name to image count.
        figsize: Figure size.
        save_path: Optional path to save the figure.

    Returns:
        matplotlib Figure object.
    """
    sorted_items = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    names = [shorten_class_name(k) for k, _ in sorted_items]
    full_names = [k for k, _ in sorted_items]
    counts = [v for _, v in sorted_items]

    crop_colors = {
        "Tomato": "#FF6B6B",
        "Potato": "#4ECDC4",
        "Pepper": "#45B7D1",
        "Corn": "#96CEB4",
        "Apple": "#FFEAA7",
        "Other": "#DFE6E9",
    }
    colors = [crop_colors.get(get_crop_name(fn), "#DFE6E9") for fn in full_names]

    fig, ax = plt.subplots(figsize=figsize)

    bars = ax.barh(range(len(names)), counts, color=colors, edgecolor="white")

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    ax.invert_yaxis()

    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_width() + 20, bar.get_y() + bar.get_height() / 2,
            f"{count:,}", va="center", fontsize=9,
        )

    mean_count = np.mean(counts)
    ax.axvline(mean_count, color="gray", linestyle="--", alpha=0.7, label=f"Mean: {mean_count:.0f}")

    ax.set_xlabel("Number of Images", fontsize=12)
    ax.set_title("Class Distribution in Selected Dataset", fontsize=14)
    ax.legend(fontsize=10)

    crops_present = sorted(set(get_crop_name(fn) for fn in full_names))
    legend_patches = [
        Patch(color=crop_colors.get(c, "#DFE6E9"), label=c) for c in crops_present
    ]
    ax.legend(handles=legend_patches, title="Crop", loc="lower right", fontsize=9)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)

    return fig


def plot_augmentation_examples(
    image_path: str,
    transform: "Callable",
    num_augmented: int = 5,
    figsize: Tuple[int, int] = (18, 4),
) -> plt.Figure:
    """Show original image alongside augmented versions.

    Args:
        image_path: Path to a sample image.
        transform: Training augmentation transform.
        num_augmented: Number of augmented versions to show.
        figsize: Figure size.

    Returns:
        matplotlib Figure object.
    """
    img = Image.open(image_path).convert("RGB")

    fig, axes = plt.subplots(1, 1 + num_augmented, figsize=figsize)

    # Original
    axes[0].imshow(img)
    axes[0].set_title("Original", fontsize=11)
    axes[0].axis("off")

    # Augmented versions
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    for i in range(num_augmented):
        aug_img = transform(img)
        img_np = aug_img.permute(1, 2, 0).numpy()
        img_np = std * img_np + mean
        img_np = np.clip(img_np, 0, 1)

        axes[i + 1].imshow(img_np)
        axes[i + 1].set_title(f"Augmented {i + 1}", fontsize=11)
        axes[i + 1].axis("off")

    plt.suptitle("Data Augmentation Examples", fontsize=14)
    plt.tight_layout()
    return fig
