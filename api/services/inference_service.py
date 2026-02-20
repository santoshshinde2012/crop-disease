"""PyTorch inference service - model loading and prediction."""

import logging
from typing import Dict, List

from PIL import Image

from app.config import TOP_K
from app.model_service import (
    Prediction,
    get_transform,
    load_model,
    predict,
)

logger = logging.getLogger(__name__)


class PyTorchInferenceService:
    """Concrete inference service backed by PyTorch."""

    def __init__(self) -> None:
        """Load the best available checkpoint at construction time."""
        model, class_mapping, model_display_name = load_model()
        self._model = model
        self._class_mapping = class_mapping
        self._model_name = model_display_name
        self._transform = get_transform()
        logger.info("PyTorchInferenceService ready: %s", self._model_name)

    # ---- InferenceService protocol ----

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def num_classes(self) -> int:
        return len(self._class_mapping)

    def predict(self, image: Image.Image, top_k: int = TOP_K) -> List[Dict]:
        """Run PyTorch inference and return raw predictions.

        Returns:
            List of dicts: ``[{"class_name": str, "probability": float}, ...]``
        """
        predictions: List[Prediction] = predict(
            image, self._model, self._class_mapping, self._transform, top_k=top_k,
        )
        return [
            {"class_name": p.class_name, "probability": round(p.probability, 4)}
            for p in predictions
        ]
