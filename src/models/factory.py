"""
Model factory for creating classification architectures.

Supports ResNet-50, EfficientNet-B0, and MobileNetV3-Small with
custom classifier heads.  Layer-freezing utilities live in
``freeze.py``; this module focuses on model construction and
parameter utilities.
"""

from typing import Dict, List, Optional, Union

import torch.nn as nn
from torchvision import models

from .freeze import _CLASSIFIER_ATTR


def _build_classifier_head(
    in_features: int,
    num_classes: int,
    dropout: float = 0.3,
    activation: Optional[nn.Module] = None,
) -> nn.Sequential:
    """Build shared classifier head structure.

    Architecture: Dropout(0.3) → Linear(in, 512) → Activation → Dropout(0.15) → Linear(512, num_classes)

    Args:
        in_features: Number of input features from backbone.
        num_classes: Number of output classes.
        dropout: First dropout rate.
        activation: Activation function module.

    Returns:
        nn.Sequential classifier head.
    """
    if activation is None:
        activation = nn.ReLU(inplace=True)

    return nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, 512),
        activation,
        nn.Dropout(p=0.15),
        nn.Linear(512, num_classes),
    )


def get_model(
    name: str = "resnet50",
    num_classes: int = 12,
    pretrained: bool = True,
    dropout: float = 0.3,
) -> nn.Module:
    """Create a model with pretrained backbone and custom classifier head.

    Args:
        name: Architecture name. Options: resnet50, efficientnet_b0, mobilenetv3
        num_classes: Number of output classes.
        pretrained: If True, load ImageNet pretrained weights.
        dropout: Dropout rate for the classifier head.

    Returns:
        nn.Module with custom classifier head.

    Raises:
        ValueError: If model name is not supported.
    """
    name = name.lower()

    if name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features  # 2048
        model.fc = _build_classifier_head(
            in_features, num_classes, dropout, nn.ReLU(inplace=True)
        )

    elif name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features  # 1280
        model.classifier = _build_classifier_head(
            in_features, num_classes, dropout, nn.SiLU(inplace=True)
        )

    elif name == "mobilenetv3":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        # MobileNetV3 classifier is already multi-layer; access in_features
        in_features = model.classifier[0].in_features  # 576
        model.classifier = _build_classifier_head(
            in_features, num_classes, dropout, nn.Hardswish()
        )

    else:
        supported = ", ".join(_CLASSIFIER_ATTR)
        raise ValueError(
            f"Unsupported model: {name}. Options: {supported}"
        )

    return model


def count_parameters(model: nn.Module) -> Dict[str, Union[int, float]]:
    """Count model parameters by category.

    Returns:
        Dict with keys: total, trainable, frozen (int) and total_mb (float).
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    total_mb = total * 4 / (1024 ** 2)  # float32 = 4 bytes

    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
        "total_mb": round(total_mb, 2),
    }


def get_differential_lr_params(
    model: nn.Module, model_name: str, backbone_lr: float, head_lr: float,
) -> List[Dict[str, object]]:
    """Get parameter groups with differential learning rates.

    Used in Stage 3 for full fine-tuning with lower backbone LR.

    Args:
        model: The model.
        model_name: Architecture name.
        backbone_lr: Learning rate for backbone parameters.
        head_lr: Learning rate for classifier head parameters.

    Returns:
        List of parameter group dicts for optimizer.
    """
    model_name = model_name.lower()
    classifier_key = _CLASSIFIER_ATTR[model_name]

    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            if classifier_key in name:
                head_params.append(param)
            else:
                backbone_params.append(param)

    return [
        {"params": backbone_params, "lr": backbone_lr},
        {"params": head_params, "lr": head_lr},
    ]
