"""Tests for dataset loading and corrupt image handling."""

import pytest
from pathlib import Path

from PIL import Image

from src.data.dataset import PlantDiseaseDataset


class TestPlantDiseaseDataset:
    """Test suite for PlantDiseaseDataset."""

    @pytest.fixture
    def temp_dataset(self, tmp_path):
        """Create a temporary dataset with 2 classes and 5 images each."""
        classes = ["Tomato_healthy", "Tomato_Early_blight"]
        for cls in classes:
            cls_dir = tmp_path / cls
            cls_dir.mkdir()
            for i in range(5):
                img = Image.new("RGB", (100, 100), color=(i * 50, 100, 150))
                img.save(cls_dir / f"img_{i}.jpg")
        return tmp_path, classes

    def test_dataset_length(self, temp_dataset):
        """Dataset should report correct number of samples."""
        root, classes = temp_dataset
        ds = PlantDiseaseDataset(root, selected_classes=classes)
        assert len(ds) == 10

    def test_num_classes(self, temp_dataset):
        """num_classes should match number of selected classes."""
        root, classes = temp_dataset
        ds = PlantDiseaseDataset(root, selected_classes=classes)
        assert ds.num_classes == 2

    def test_class_mapping(self, temp_dataset):
        """class_to_idx should map sorted class names to indices."""
        root, classes = temp_dataset
        ds = PlantDiseaseDataset(root, selected_classes=classes)
        assert "Tomato_Early_blight" in ds.class_to_idx
        assert "Tomato_healthy" in ds.class_to_idx

    def test_getitem_returns_tuple(self, temp_dataset):
        """__getitem__ should return (image, label) tuple."""
        root, classes = temp_dataset
        from src.data.transforms import get_val_transforms
        ds = PlantDiseaseDataset(root, selected_classes=classes, transform=get_val_transforms(224))
        img, label = ds[0]
        assert img.shape == (3, 224, 224)
        assert isinstance(label, int)

    def test_corrupt_image_handled(self, temp_dataset):
        """Corrupt images should not crash; should return a blank image."""
        root, classes = temp_dataset
        # Create a corrupt file
        corrupt_path = root / classes[0] / "corrupt.jpg"
        corrupt_path.write_bytes(b"not a real image file content")

        from src.data.transforms import get_val_transforms
        ds = PlantDiseaseDataset(root, selected_classes=classes, transform=get_val_transforms(224))

        # Find the corrupt image index
        corrupt_idx = None
        for i, (path, _) in enumerate(ds.samples):
            if "corrupt" in str(path):
                corrupt_idx = i
                break

        assert corrupt_idx is not None, "Corrupt image should be in dataset"

        # Should not raise
        img, label = ds[corrupt_idx]
        assert img.shape == (3, 224, 224)

    def test_get_class_counts(self, temp_dataset):
        """get_class_counts should return counts for each class."""
        root, classes = temp_dataset
        ds = PlantDiseaseDataset(root, selected_classes=classes)
        counts = ds.get_class_counts()
        assert len(counts) == 2
        for cls in classes:
            assert counts[cls] == 5
