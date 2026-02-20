"""
Prediction visualization — correct and incorrect examples.

Shows individual predictions with confidence scores and color-coded results.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from PIL import Image

from ..utils.text_helpers import shorten_class_name


@torch.no_grad()
def get_prediction_examples(
    model: nn.Module,
    dataset: "Dataset",
    idx_to_class: Dict[int, str],
    device: torch.device,
    num_correct: int = 5,
    num_incorrect: int = 5,
) -> Tuple[List[dict], List[dict]]:
    """Collect correct and incorrect prediction examples.

    Iterates through the dataset and collects examples until both lists are full.

    Args:
        model: Trained model in eval mode.
        dataset: SplitDataset or PlantDiseaseDataset with transform.
        idx_to_class: Mapping from label index to class name.
        device: Torch device.
        num_correct: Number of correct examples to collect.
        num_incorrect: Number of incorrect examples to collect.

    Returns:
        Tuple of (correct_examples, incorrect_examples).
        Each example is a dict: {image_path, true_label, pred_label, confidence}
    """
    model.eval()
    correct_examples = []
    incorrect_examples = []

    for idx in range(len(dataset)):
        if len(correct_examples) >= num_correct and len(incorrect_examples) >= num_incorrect:
            break

        image_tensor, label = dataset[idx]
        img_input = image_tensor.unsqueeze(0).to(device)

        output = model(img_input)
        probs = torch.softmax(output, dim=1).squeeze()
        pred_idx = probs.argmax().item()
        confidence = probs[pred_idx].item()

        true_label = idx_to_class[label]
        pred_label = idx_to_class[pred_idx]

        # Get image path from the underlying dataset
        if hasattr(dataset, "samples"):
            image_path = str(dataset.samples[idx][0])
        else:
            image_path = f"sample_{idx}"

        example = {
            "image_path": image_path,
            "true_label": true_label,
            "pred_label": pred_label,
            "confidence": confidence,
        }

        if pred_idx == label and len(correct_examples) < num_correct:
            correct_examples.append(example)
        elif pred_idx != label and len(incorrect_examples) < num_incorrect:
            incorrect_examples.append(example)

    return correct_examples, incorrect_examples


def plot_prediction_grid(
    examples: List[dict],
    title: str,
    save_path: Optional[Union[str, Path]] = None,
    figsize: Optional[Tuple[int, int]] = None,
) -> plt.Figure:
    """Plot a grid of prediction examples.

    Each subplot shows the raw image with true/predicted label and confidence.

    Args:
        examples: List of prediction example dicts.
        title: Overall figure title.
        save_path: Optional path to save the figure.
        figsize: Optional figure size. Auto-calculated if None.

    Returns:
        matplotlib Figure object.
    """
    n = len(examples)
    if n == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No examples found", ha="center", va="center")
        return fig

    if figsize is None:
        figsize = (4 * n, 5)

    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]

    for ax, example in zip(axes, examples):
        # Load and display raw image
        img = Image.open(example["image_path"]).convert("RGB")
        ax.imshow(img)

        true_short = shorten_class_name(example["true_label"])
        pred_short = shorten_class_name(example["pred_label"])
        conf = example["confidence"]

        is_correct = example["true_label"] == example["pred_label"]
        color = "green" if is_correct else "red"

        ax.set_title(
            f"True: {true_short}\nPred: {pred_short}\nConf: {conf:.1%}",
            fontsize=9,
            color=color,
            fontweight="bold",
        )
        ax.axis("off")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)

    return fig
