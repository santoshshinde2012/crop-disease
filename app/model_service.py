"""Model loading and inference - checkpoint discovery, instantiation, prediction."""

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from PIL import Image

from .config import (
    CLASS_MAPPING_PATH,
    DEFAULT_DROPOUT,
    DEFAULT_MODEL_NAME,
    IMAGE_SIZE,
    MODEL_PREFERENCES,
    MODELS_DIR,
    PROJECT_ROOT,
    TOP_K,
)

# Add project root so `src` package is importable via `streamlit run`
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.transforms import get_val_transforms  # noqa: E402
from src.models.factory import get_model  # noqa: E402

logger = logging.getLogger(__name__)


# ---- Data transfer object for prediction results ----
@dataclass(frozen=True)
class Prediction:
    """Single class prediction with confidence score."""

    class_name: str
    probability: float


def _load_class_mapping() -> Dict[str, str]:
    """Load idx-to-class-name mapping from JSON or fall back to Config.

    Returns:
        Dict mapping string index to class name (e.g. ``{"0": "Pepper..."}``).
    """
    if CLASS_MAPPING_PATH.exists():
        with open(CLASS_MAPPING_PATH) as fh:
            return json.load(fh)

    logger.warning("class_mapping.json not found -- generating from Config defaults")
    from src.config import Config

    cfg = Config()
    return {str(i): name for i, name in enumerate(sorted(cfg.data.selected_classes))}


def load_model() -> Tuple[torch.nn.Module, Dict[str, str], str]:
    """Discover and load the best available checkpoint.

    Tries model architectures in priority order defined in
    ``app.config.MODEL_PREFERENCES``.

    Returns:
        Tuple of (model, class_mapping, model_display_name).
        The model is already in ``eval`` mode on CPU.
    """
    class_mapping = _load_class_mapping()
    num_classes = len(class_mapping)

    for model_name, filename in MODEL_PREFERENCES:
        checkpoint_path = MODELS_DIR / filename
        if not checkpoint_path.exists():
            continue

        model = get_model(
            name=model_name,
            num_classes=num_classes,
            pretrained=False,
            dropout=DEFAULT_DROPOUT,
        )
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        logger.info(
            "Loaded %s (epoch=%d, val_f1=%.4f)",
            model_name,
            checkpoint.get("epoch", -1),
            checkpoint.get("val_f1", -1),
        )
        return model, class_mapping, model_name

    # No checkpoint found -- return an untrained model for demo
    logger.warning("No trained checkpoint found; loading untrained %s", DEFAULT_MODEL_NAME)
    model = get_model(
        name=DEFAULT_MODEL_NAME,
        num_classes=num_classes,
        pretrained=True,
        dropout=DEFAULT_DROPOUT,
    )
    model.eval()
    return model, class_mapping, f"{DEFAULT_MODEL_NAME} (untrained)"


def get_transform():
    """Return the deterministic validation transform pipeline."""
    return get_val_transforms(IMAGE_SIZE)


@torch.no_grad()
def predict(
    image: Image.Image,
    model: torch.nn.Module,
    class_mapping: Dict[str, str],
    transform,
    top_k: int = TOP_K,
) -> List[Prediction]:
    """Run inference on a single PIL image.

    Args:
        image: RGB PIL Image.
        model: Model in eval mode.
        class_mapping: Index-to-class-name mapping.
        transform: Preprocessing transform.
        top_k: Number of top predictions to return.

    Returns:
        List of ``Prediction`` objects sorted by descending probability.
    """
    img_tensor = transform(image).unsqueeze(0)
    output = model(img_tensor)
    probs = torch.softmax(output, dim=1).squeeze()
    top_probs, top_indices = probs.topk(top_k)

    return [
        Prediction(
            class_name=class_mapping[str(idx.item())],
            probability=prob.item(),
        )
        for prob, idx in zip(top_probs, top_indices)
    ]
