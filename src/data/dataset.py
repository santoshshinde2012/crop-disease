"""
Custom PyTorch Dataset for PlantVillage plant disease images.

Handles folder-per-class layout where each subdirectory name is the label.
"""

import logging
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def safe_load_image(
    img_path: Union[str, Path], fallback_size: tuple = (224, 224)
) -> Image.Image:
    """Load an image safely, returning a blank fallback on corruption.

    Centralizes corrupt-image handling used by both PlantDiseaseDataset
    and SplitDataset.

    Args:
        img_path: Path to the image file.
        fallback_size: (width, height) for blank fallback image.

    Returns:
        PIL.Image in RGB mode.
    """
    try:
        return Image.open(img_path).convert("RGB")
    except (OSError, IOError, SyntaxError) as e:
        logger.warning("Corrupt image %s: %s -- returning blank", img_path, e)
        return Image.new("RGB", fallback_size, (0, 0, 0))


class PlantDiseaseDataset(Dataset):
    """Dataset for plant disease classification from folder-structured images.

    Args:
        root_dir: Root directory containing class subdirectories.
        selected_classes: Optional list of class names to include.
            If None, all subdirectories are used.
        transform: Optional torchvision transform to apply to images.
    """

    def __init__(
        self,
        root_dir: Union[str, Path],
        selected_classes: Optional[List[str]] = None,
        transform: Optional[Callable] = None,
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform

        if not self.root_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.root_dir}")

        # Scan root directory for subdirectories (skip hidden folders)
        all_classes = sorted([
            d.name for d in self.root_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ])

        # Filter to selected classes if provided
        if selected_classes is not None:
            missing = set(selected_classes) - set(all_classes)
            if missing:
                raise ValueError(
                    f"Classes not found in {self.root_dir}: {missing}. "
                    f"Available classes: {all_classes}"
                )
            # Sort alphabetically for deterministic class-to-index mapping
            classes = sorted(selected_classes)
        else:
            classes = all_classes

        # Build class-to-index mappings
        self.class_to_idx: Dict[str, int] = {
            cls_name: idx for idx, cls_name in enumerate(classes)
        }
        self.idx_to_class: Dict[int, str] = {
            idx: cls_name for cls_name, idx in self.class_to_idx.items()
        }

        # Build samples list: (path, label_idx)
        self.samples: List[Tuple[Path, int]] = []
        for cls_name in classes:
            cls_dir = self.root_dir / cls_name
            label_idx = self.class_to_idx[cls_name]
            for img_path in sorted(cls_dir.iterdir()):
                if img_path.suffix.lower() in VALID_EXTENSIONS:
                    self.samples.append((img_path, label_idx))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No valid images found in {self.root_dir} for classes {classes}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple:
        """Load and return image and label at the given index.

        Returns:
            Tuple of (image_tensor, label_int) if transform is provided,
            otherwise (PIL.Image, label_int).
        """
        img_path, label = self.samples[idx]
        image = safe_load_image(img_path)

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def get_class_counts(self) -> Dict[str, int]:
        """Return per-class image counts.

        Returns:
            Dict mapping class name to number of images.
        """
        label_counts = Counter(label for _, label in self.samples)
        return {
            self.idx_to_class[idx]: count
            for idx, count in sorted(label_counts.items())
        }

    @property
    def num_classes(self) -> int:
        """Return the number of classes."""
        return len(self.class_to_idx)
