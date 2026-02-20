"""
Layer freezing and unfreezing for progressive fine-tuning.

Stage 1 — freeze_backbone:   only classifier head trains.
Stage 2 — partial_unfreeze:  top backbone layers + head train.
Stage 3 — full_unfreeze:     all parameters train.
"""

import torch.nn as nn

# Maps model name → the attribute name that holds the classifier head.
# Shared with factory.py and params.py via import.
_CLASSIFIER_ATTR = {
    "resnet50": "fc",
    "efficientnet_b0": "classifier",
    "mobilenetv3": "classifier",
}


def freeze_backbone(model: nn.Module, model_name: str) -> None:
    """Freeze all backbone parameters, leaving only classifier trainable.

    Stage 1: Only the classifier head is trained.

    Args:
        model: The model to freeze.
        model_name: Architecture name for determining freeze logic.
    """
    classifier_key = _CLASSIFIER_ATTR[model_name.lower()]
    for name, param in model.named_parameters():
        param.requires_grad = classifier_key in name


def partial_unfreeze(model: nn.Module, model_name: str) -> None:
    """Partially unfreeze backbone — top layers become trainable.

    Stage 2: High-level feature layers + classifier head are trained.

    Args:
        model: The model to partially unfreeze.
        model_name: Architecture name for determining unfreeze logic.
    """
    model_name = model_name.lower()

    if model_name == "resnet50":
        # Unfreeze layer3, layer4, and fc
        for name, param in model.named_parameters():
            if any(key in name for key in ["layer3", "layer4", "fc"]):
                param.requires_grad = True
            else:
                param.requires_grad = False

    elif model_name in ("efficientnet_b0", "mobilenetv3"):
        # Unfreeze last 30% of features + classifier
        features = list(model.features.children())
        unfreeze_from = int(len(features) * 0.7)

        # First freeze everything
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze last 30% of features
        for child in features[unfreeze_from:]:
            for param in child.parameters():
                param.requires_grad = True

        # Unfreeze classifier
        for param in model.classifier.parameters():
            param.requires_grad = True


def full_unfreeze(model: nn.Module) -> None:
    """Unfreeze all model parameters.

    Stage 3: End-to-end fine-tuning.

    Args:
        model: The model to fully unfreeze.
    """
    for param in model.parameters():
        param.requires_grad = True
