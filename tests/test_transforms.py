"""Tests for data transforms."""

import pytest
import torch
from PIL import Image

from src.data.transforms import get_train_transforms, get_val_transforms


class TestTransforms:
    """Test suite for augmentation pipelines."""

    def test_train_transform_output_shape(self):
        """Train transform should produce (3, 224, 224) tensor."""
        transform = get_train_transforms(224)
        img = Image.new("RGB", (400, 400), color=(128, 128, 128))
        tensor = transform(img)
        assert tensor.shape == (3, 224, 224)
        assert tensor.dtype == torch.float32

    def test_val_transform_output_shape(self):
        """Val transform should produce (3, 224, 224) tensor."""
        transform = get_val_transforms(224)
        img = Image.new("RGB", (400, 400), color=(128, 128, 128))
        tensor = transform(img)
        assert tensor.shape == (3, 224, 224)

    def test_val_transform_deterministic(self):
        """Val transforms should be deterministic (no randomness)."""
        transform = get_val_transforms(224)
        img = Image.new("RGB", (400, 400), color=(100, 150, 200))
        t1 = transform(img)
        t2 = transform(img)
        assert torch.allclose(t1, t2), "Val transforms should be deterministic"

    def test_custom_image_size(self):
        """Transforms should respect custom image sizes."""
        for size in [128, 256, 384]:
            t = get_val_transforms(size)
            img = Image.new("RGB", (500, 500))
            tensor = t(img)
            assert tensor.shape == (3, size, size)

    def test_small_image_upscaled(self):
        """Small images should be upscaled correctly."""
        transform = get_val_transforms(224)
        img = Image.new("RGB", (32, 32))
        tensor = transform(img)
        assert tensor.shape == (3, 224, 224)
