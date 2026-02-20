"""Tests for data splitting."""

import pytest
from pathlib import Path

from src.data.splitter import create_stratified_split


class TestSplitter:
    """Test suite for stratified data splitting."""

    @pytest.fixture
    def mock_samples(self):
        """Create mock samples with 3 classes, 30 samples each."""
        samples = []
        for label in range(3):
            for i in range(30):
                samples.append((Path(f"class{label}/img_{i}.jpg"), label))
        return samples

    def test_split_proportions(self, mock_samples):
        """Split should produce approximately correct proportions."""
        ratios = {"train": 0.70, "val": 0.15, "test": 0.15}
        splits = create_stratified_split(mock_samples, ratios, seed=42)

        total = len(mock_samples)
        assert len(splits["train"]) + len(splits["val"]) + len(splits["test"]) == total

        train_ratio = len(splits["train"]) / total
        assert abs(train_ratio - 0.70) < 0.05, f"Train ratio {train_ratio} too far from 0.70"

    def test_no_overlap(self, mock_samples):
        """Train, val, test should have no overlapping paths."""
        ratios = {"train": 0.70, "val": 0.15, "test": 0.15}
        splits = create_stratified_split(mock_samples, ratios, seed=42)

        train_paths = {s[0] for s in splits["train"]}
        val_paths = {s[0] for s in splits["val"]}
        test_paths = {s[0] for s in splits["test"]}

        assert len(train_paths & val_paths) == 0, "Train and val overlap"
        assert len(train_paths & test_paths) == 0, "Train and test overlap"
        assert len(val_paths & test_paths) == 0, "Val and test overlap"

    def test_stratification_preserved(self, mock_samples):
        """Each split should contain all classes."""
        ratios = {"train": 0.70, "val": 0.15, "test": 0.15}
        splits = create_stratified_split(mock_samples, ratios, seed=42)

        for split_name in ["train", "val", "test"]:
            labels = {s[1] for s in splits[split_name]}
            assert labels == {0, 1, 2}, f"{split_name} missing classes: {labels}"

    def test_invalid_ratios_raises(self):
        """Ratios not summing to 1.0 should raise ValueError."""
        samples = [(Path("img.jpg"), 0)] * 20
        bad_ratios = {"train": 0.5, "val": 0.3, "test": 0.3}
        with pytest.raises(ValueError, match="sum to 1.0"):
            create_stratified_split(samples, bad_ratios)

    def test_reproducibility(self, mock_samples):
        """Same seed should produce identical splits."""
        ratios = {"train": 0.70, "val": 0.15, "test": 0.15}
        split1 = create_stratified_split(mock_samples, ratios, seed=42)
        split2 = create_stratified_split(mock_samples, ratios, seed=42)

        for key in ["train", "val", "test"]:
            paths1 = [s[0] for s in split1[key]]
            paths2 = [s[0] for s in split2[key]]
            assert paths1 == paths2, f"{key} split not reproducible"
