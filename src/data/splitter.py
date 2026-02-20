"""
Stratified dataset splitting for train/val/test partitions.

Uses two-stage sklearn split to produce exact 70/15/15 proportions
with stratification preserved in each partition.
"""

from pathlib import Path
from typing import Dict, List, Tuple

from sklearn.model_selection import train_test_split


def create_stratified_split(
    samples: List[Tuple[Path, int]],
    split_ratios: Dict[str, float],
    seed: int = 42,
) -> Dict[str, List[Tuple[Path, int]]]:
    """Split samples into stratified train/val/test partitions.

    Two-stage split: first separate test, then split remainder into train/val.
    This ensures exact proportions with stratification preserved.

    Args:
        samples: List of (image_path, label_index) tuples.
        split_ratios: Dict with 'train', 'val', 'test' keys summing to 1.0.
        seed: Random seed for reproducibility.

    Returns:
        Dict with 'train', 'val', 'test' keys mapping to sample lists.
    """
    if abs(sum(split_ratios.values()) - 1.0) >= 1e-6:
        raise ValueError(
            f"Split ratios must sum to 1.0, got {sum(split_ratios.values())}"
        )

    paths = [s[0] for s in samples]
    labels = [s[1] for s in samples]

    test_ratio = split_ratios["test"]
    # Adjusted val ratio relative to remaining data after test split
    val_ratio = split_ratios["val"] / (1 - test_ratio)

    # First split: separate test set
    paths_trainval, paths_test, labels_trainval, labels_test = train_test_split(
        paths, labels,
        test_size=test_ratio,
        stratify=labels,
        random_state=seed,
    )

    # Second split: separate val from train
    paths_train, paths_val, labels_train, labels_val = train_test_split(
        paths_trainval, labels_trainval,
        test_size=val_ratio,
        stratify=labels_trainval,
        random_state=seed,
    )

    return {
        "train": list(zip(paths_train, labels_train)),
        "val": list(zip(paths_val, labels_val)),
        "test": list(zip(paths_test, labels_test)),
    }
