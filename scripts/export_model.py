"""
Model export pipeline — PyTorch → ONNX (→ TFLite).

Usage:
    python scripts/export_model.py
    python scripts/export_model.py --model efficientnet_b0 --checkpoint models/efficientnet_b0_best.pth
    python scripts/export_model.py --tflite
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.export import load_checkpoint_and_export  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---- Default model priorities (same as app.config) ----
MODEL_PREFERENCES = [
    ("efficientnet_b0", "efficientnet_b0_best.pth"),
    ("resnet50", "resnet50_best.pth"),
    ("mobilenetv3", "mobilenetv3_best.pth"),
]


def find_best_checkpoint(models_dir: Path) -> tuple[str, Path]:
    """Find the best available checkpoint in priority order.

    Returns:
        Tuple of (model_name, checkpoint_path).

    Raises:
        FileNotFoundError: If no checkpoint is found.
    """
    for model_name, filename in MODEL_PREFERENCES:
        path = models_dir / filename
        if path.exists():
            logger.info("Found checkpoint: %s", path)
            return model_name, path

    raise FileNotFoundError(
        f"No checkpoint found in {models_dir}. "
        f"Expected one of: {[f for _, f in MODEL_PREFERENCES]}"
    )


def export_pytorch_to_onnx(
    checkpoint_path: Path,
    model_name: str,
    num_classes: int,
    output_path: Path,
    dropout: float,
) -> Path:
    """Step 1: Convert PyTorch checkpoint to ONNX.

    Args:
        checkpoint_path: Path to .pth file.
        model_name: Architecture name.
        num_classes: Number of output classes.
        output_path: Destination .onnx file.
        dropout: Dropout rate used during training.

    Returns:
        Path to the exported ONNX model.
    """
    logger.info("=" * 60)
    logger.info("Step 1: PyTorch → ONNX")
    logger.info("=" * 60)

    onnx_path = load_checkpoint_and_export(
        checkpoint_path=str(checkpoint_path),
        model_name=model_name,
        num_classes=num_classes,
        output_path=str(output_path),
        dropout=dropout,
    )

    logger.info("ONNX model saved: %s (%.2f MB)", onnx_path, onnx_path.stat().st_size / 1e6)
    return onnx_path


def export_onnx_to_tflite(onnx_path: Path, output_dir: Path) -> Path | None:
    """Step 2: Convert ONNX to TFLite with INT8 quantization.

    Requires ``onnx2tf`` and ``tensorflow``. Returns None if not installed.

    Args:
        onnx_path: Path to the .onnx model.
        output_dir: Directory for TFLite output.

    Returns:
        Path to TFLite model, or None if conversion is skipped.
    """
    logger.info("=" * 60)
    logger.info("Step 2: ONNX → TFLite (INT8 quantization)")
    logger.info("=" * 60)

    try:
        import onnx2tf  # noqa: F401
    except ImportError:
        logger.warning(
            "onnx2tf not installed — skipping TFLite conversion. "
            "Install with: pip install onnx2tf tensorflow"
        )
        return None

    output_dir.mkdir(parents=True, exist_ok=True)

    onnx2tf.convert(
        input_onnx_file_path=str(onnx_path),
        output_folder_path=str(output_dir),
        output_integer_quantized_tflite=True,
    )

    # Find the generated TFLite file
    tflite_files = list(output_dir.glob("*.tflite"))
    if tflite_files:
        tflite_path = tflite_files[0]
        logger.info(
            "TFLite model saved: %s (%.2f MB)",
            tflite_path,
            tflite_path.stat().st_size / 1e6,
        )
        return tflite_path

    logger.warning("TFLite conversion completed but no .tflite file found in %s", output_dir)
    return None


def main() -> None:
    """CLI entry point for the export pipeline."""
    parser = argparse.ArgumentParser(
        description="Export trained PyTorch model to ONNX (and optionally TFLite)"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Architecture name (auto-detected from checkpoint if omitted)",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to .pth checkpoint (auto-discovers best if omitted)",
    )
    parser.add_argument(
        "--num-classes", type=int, default=12,
        help="Number of output classes (default: 12)",
    )
    parser.add_argument(
        "--dropout", type=float, default=0.3,
        help="Dropout rate used during training (default: 0.3)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="models",
        help="Output directory (default: models/)",
    )
    parser.add_argument(
        "--tflite", action="store_true",
        help="Also convert to TFLite (requires onnx2tf + tensorflow)",
    )
    args = parser.parse_args()

    models_dir = PROJECT_ROOT / args.output_dir

    # ---- Discover checkpoint ----
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        model_name = args.model or "efficientnet_b0"
    else:
        model_name, checkpoint_path = find_best_checkpoint(models_dir)
        if args.model:
            model_name = args.model

    # ---- Step 1: PyTorch → ONNX ----
    onnx_filename = f"{model_name}.onnx"
    onnx_path = export_pytorch_to_onnx(
        checkpoint_path=checkpoint_path,
        model_name=model_name,
        num_classes=args.num_classes,
        output_path=models_dir / onnx_filename,
        dropout=args.dropout,
    )

    # ---- Step 2: ONNX → TFLite (optional) ----
    if args.tflite:
        tflite_dir = models_dir / "tflite"
        export_onnx_to_tflite(onnx_path, tflite_dir)
    else:
        logger.info(
            "Skipping TFLite conversion. Use --tflite flag to enable."
        )

    # ---- Summary ----
    logger.info("=" * 60)
    logger.info("Export pipeline complete!")
    logger.info("  ONNX:   %s", onnx_path)
    if args.tflite:
        logger.info("  TFLite: %s/", models_dir / "tflite")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
