"""Tests for ONNX export utility."""

import pytest
import torch

from src.models.factory import get_model

# Skip entire module if onnxscript (required by torch.onnx.export in PyTorch 2.6+) is missing
onnxscript = pytest.importorskip("onnxscript", reason="onnxscript required for ONNX export")


class TestONNXExport:
    """Test suite for ONNX export functionality."""

    def test_export_creates_file(self, tmp_path):
        """export_to_onnx should create a valid .onnx file."""
        from src.utils.export import export_to_onnx

        model = get_model(name="mobilenetv3", num_classes=12, pretrained=False)
        output_path = tmp_path / "test_model.onnx"

        result = export_to_onnx(model, output_path=str(output_path))
        assert result.exists(), "ONNX file should be created"
        assert result.stat().st_size > 0, "ONNX file should not be empty"

    def test_export_model_still_works(self, tmp_path):
        """Model should still work after export."""
        from src.utils.export import export_to_onnx

        model = get_model(name="mobilenetv3", num_classes=12, pretrained=False)
        export_to_onnx(model, output_path=str(tmp_path / "test.onnx"))

        # Model should still be usable
        model.eval()
        dummy = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(dummy)
        assert output.shape == (1, 12)
