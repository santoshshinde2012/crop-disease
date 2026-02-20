"""DataLoader factory for train/val/test loaders."""

import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from torch.utils.data import DataLoader, Dataset

from .dataset import safe_load_image

logger = logging.getLogger(__name__)


class SplitDataset(Dataset):
    """Dataset wrapper for pre-split samples with transforms.

    Takes a list of (path, label) tuples and applies transforms on load.

    Args:
        samples: List of (image_path, label_index) tuples.
        transform: Optional torchvision transform to apply.
    """

    def __init__(
        self,
        samples: List[Tuple[Path, int]],
        transform: Optional[Callable] = None,
    ):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple:
        img_path, label = self.samples[idx]
        image = safe_load_image(img_path)

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def create_dataloaders(
    splits: dict,
    train_transform: Callable,
    val_transform: Callable,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Dict[str, DataLoader]:
    """Create DataLoaders for train/val/test splits.

    Args:
        splits: Dict with 'train', 'val', 'test' sample lists.
        train_transform: Augmentation pipeline for training data.
        val_transform: Preprocessing pipeline for val/test data.
        batch_size: Batch size for all loaders.
        num_workers: Number of parallel data loading workers.
        pin_memory: If True, enables faster CPU→GPU transfers.

    Returns:
        Dict with 'train', 'val', 'test' DataLoader instances.
    """
    train_dataset = SplitDataset(splits["train"], transform=train_transform)
    val_dataset = SplitDataset(splits["val"], transform=val_transform)
    test_dataset = SplitDataset(splits["test"], transform=val_transform)

    use_persistent = num_workers > 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=use_persistent,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=use_persistent,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=use_persistent,
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }
