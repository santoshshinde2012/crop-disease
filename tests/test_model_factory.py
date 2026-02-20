"""Tests for model factory."""

import pytest
import torch

from src.models.factory import (
    get_model,
    count_parameters,
)
from src.models.freeze import (
    freeze_backbone,
    partial_unfreeze,
    full_unfreeze,
)


class TestModelFactory:
    """Test suite for model creation and layer management."""

    @pytest.mark.parametrize("name", ["resnet50", "efficientnet_b0", "mobilenetv3"])
    def test_model_output_shape(self, name):
        """Each architecture should produce correct output dimensions."""
        num_classes = 12
        model = get_model(name=name, num_classes=num_classes, pretrained=False)
        model.eval()

        dummy = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            output = model(dummy)

        assert output.shape == (2, num_classes), f"{name}: expected (2, {num_classes}), got {output.shape}"

    @pytest.mark.parametrize("name", ["resnet50", "efficientnet_b0", "mobilenetv3"])
    def test_freeze_backbone(self, name):
        """freeze_backbone should freeze all but classifier parameters."""
        model = get_model(name=name, num_classes=12, pretrained=False)
        freeze_backbone(model, name)

        info = count_parameters(model)
        assert info["trainable"] < info["total"], f"{name}: freeze should reduce trainable params"

    @pytest.mark.parametrize("name", ["resnet50", "efficientnet_b0", "mobilenetv3"])
    def test_full_unfreeze(self, name):
        """full_unfreeze should make all parameters trainable."""
        model = get_model(name=name, num_classes=12, pretrained=False)
        freeze_backbone(model, name)
        full_unfreeze(model)

        info = count_parameters(model)
        assert info["total"] == info["trainable"], f"{name}: all params should be trainable after full_unfreeze"

    def test_invalid_model_name_raises(self):
        """Unknown model name should raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported model"):
            get_model(name="invalid_model", num_classes=12)

    def test_custom_dropout(self):
        """Models should accept custom dropout values."""
        model = get_model(name="efficientnet_b0", num_classes=5, pretrained=False, dropout=0.5)
        # Should not raise — just verify it creates successfully
        assert model is not None

    @pytest.mark.parametrize("name", ["resnet50", "efficientnet_b0", "mobilenetv3"])
    def test_partial_unfreeze(self, name):
        """partial_unfreeze should increase trainable params vs frozen state."""
        model = get_model(name=name, num_classes=12, pretrained=False)
        freeze_backbone(model, name)
        frozen_trainable = count_parameters(model)["trainable"]

        partial_unfreeze(model, name)
        partial_trainable = count_parameters(model)["trainable"]

        assert partial_trainable > frozen_trainable, (
            f"{name}: partial unfreeze should have more trainable params than frozen"
        )
