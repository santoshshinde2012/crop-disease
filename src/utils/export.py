"""
ONNX export utility for deployment.

Converts a trained PyTorch model to ONNX format for
cross-platform inference (TensorFlow Lite, ONNX Runtime, etc.).
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def export_to_onnx(
    model: nn.Module,
    output_path: str = "models/model.onnx",
    input_size: Tuple[int, ...] = (1, 3, 224, 224),
    opset_version: int = 17,
    dynamic_batch: bool = True,
    simplify: bool = True,
) -> Path:
    """Export a PyTorch model to ONNX format.

    Args:
        model: Trained PyTorch model (will be set to eval mode).
        output_path: Destination path for the .onnx file.
        input_size: Input tensor shape (batch, channels, height, width).
        opset_version: ONNX opset version (17 is widely supported).
        dynamic_batch: If True, allows variable batch size at inference.
        simplify: If True, run onnx-simplifier to optimize the graph.

    Returns:
        Path to the exported ONNX file.

    Raises:
        ImportError: If onnx package is not installed.
        RuntimeError: If export or validation fails.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    model.cpu()

    dummy_input = torch.randn(*input_size)

    # Dynamic axes for variable batch size
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        }

    logger.info("Exporting model to ONNX: %s", output_path)

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
    )

    # Validate the exported model
    try:
        import onnx

        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        logger.info("ONNX model validation passed")
    except ImportError:
        logger.warning(
            "onnx package not installed — skipping validation. "
            "Install with: pip install onnx"
        )
    except Exception as e:
        raise RuntimeError(f"ONNX validation failed: {e}") from e

    # Optional simplification
    if simplify:
        try:
            import onnxsim

            onnx_model = onnx.load(str(output_path))
            simplified, check = onnxsim.simplify(onnx_model)
            if check:
                onnx.save(simplified, str(output_path))
                logger.info("ONNX model simplified successfully")
            else:
                logger.warning("ONNX simplification check failed — keeping original")
        except ImportError:
            logger.info(
                "onnxsim not installed — skipping simplification. "
                "Install with: pip install onnxsim"
            )

    file_size_mb = output_path.stat().st_size / (1024 ** 2)
    logger.info("ONNX export complete: %.2f MB → %s", file_size_mb, output_path)

    return output_path


def load_checkpoint_and_export(
    checkpoint_path: str,
    model_name: str = "efficientnet_b0",
    num_classes: int = 12,
    output_path: str = "models/model.onnx",
    dropout: float = 0.3,
) -> Path:
    """Load a checkpoint and export the model to ONNX.

    Convenience function that handles model creation, weight loading,
    and ONNX export in one call.

    Args:
        checkpoint_path: Path to the .pth checkpoint file.
        model_name: Architecture name (resnet50, efficientnet_b0, mobilenetv3).
        num_classes: Number of output classes.
        output_path: Destination path for the .onnx file.
        dropout: Dropout rate used during training.

    Returns:
        Path to the exported ONNX file.
    """
    from src.models.factory import get_model

    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )

    # Try to read model config from checkpoint (enhanced checkpoints)
    model_name = checkpoint.get("model_name", model_name)
    num_classes = checkpoint.get("num_classes", num_classes)

    model = get_model(
        name=model_name,
        num_classes=num_classes,
        pretrained=False,
        dropout=dropout,
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    logger.info(
        "Loaded checkpoint: %s (epoch=%d, val_f1=%.4f)",
        checkpoint_path,
        checkpoint.get("epoch", -1),
        checkpoint.get("val_f1", -1),
    )

    return export_to_onnx(model, output_path=output_path)
