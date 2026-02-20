"""
Central configuration for Crop Disease Classification.

Every hyperparameter lives here — zero magic numbers in other modules.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""

    raw_data_dir: Path = Path("data/raw/PlantVillage")
    image_size: int = 224

    selected_classes: List[str] = field(default_factory=lambda: [
        # Tomato — 7 classes (multiple diseases per crop tests intra-crop discrimination)
        "Tomato_Bacterial_spot",
        "Tomato_Early_blight",
        "Tomato_Late_blight",
        "Tomato_Leaf_Mold",
        "Tomato__Target_Spot",
        "Tomato_Septoria_leaf_spot",
        "Tomato_healthy",
        # Potato — 3 classes (shared disease names with Tomato tests cross-crop confusion)
        "Potato___Early_blight",
        "Potato___Late_blight",
        "Potato___healthy",
        # Pepper — 2 classes (extends to another crop type)
        "Pepper__bell___Bacterial_spot",
        "Pepper__bell___healthy",
    ])

    split_ratios: Dict[str, float] = field(default_factory=lambda: {
        "train": 0.70,
        "val": 0.15,
        "test": 0.15,
    })

    random_seed: int = 42


@dataclass
class TrainConfig:
    """Configuration for model training."""

    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True

    # Stage 1: Frozen backbone, head only
    stage1_epochs: int = 5
    stage1_lr: float = 1e-3

    # Stage 2: Partial unfreeze
    stage2_epochs: int = 10
    stage2_lr: float = 1e-4

    # Stage 3: Full fine-tune
    stage3_epochs: int = 10
    stage3_lr: float = 1e-5

    weight_decay: float = 1e-4
    label_smoothing: float = 0.1
    early_stopping_patience: int = 5

    scheduler: str = "cosine"  # Options: cosine, step, plateau
    optimizer: str = "adamw"   # Options: adamw, sgd

    checkpoint_dir: Path = Path("models")
    monitor_metric: str = "val_f1"  # Macro F1 — not accuracy

    max_grad_norm: float = 1.0
    use_amp: bool = True  # Mixed precision on CUDA


@dataclass
class ModelConfig:
    """Configuration for model architecture."""

    name: str = "resnet50"  # Options: resnet50, mobilenetv3, efficientnet_b0
    pretrained: bool = True  # ImageNet weights
    dropout: float = 0.3    # First dropout layer in classifier head
    num_classes: int = 12   # Must match len(selected_classes)


@dataclass
class Config:
    """Master configuration combining all sub-configs."""

    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    def __post_init__(self):
        """Validate configuration consistency."""
        if self.model.num_classes != len(self.data.selected_classes):
            raise ValueError(
                f"num_classes ({self.model.num_classes}) must match "
                f"len(selected_classes) ({len(self.data.selected_classes)})"
            )
        ratios = self.data.split_ratios
        if abs(sum(ratios.values()) - 1.0) >= 1e-6:
            raise ValueError(
                f"Split ratios must sum to 1.0, got {sum(ratios.values())}"
            )
