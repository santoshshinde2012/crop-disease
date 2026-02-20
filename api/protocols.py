"""
Abstract interfaces (Protocols) for the API layer.

Dependency Inversion Principle: high-level modules (routes) depend on
these abstractions, not on concrete implementations.

Interface Segregation Principle: each protocol is small and focused —
clients only depend on the methods they actually use.
"""

from typing import Dict, List, Protocol, runtime_checkable

from PIL import Image


@runtime_checkable
class InferenceService(Protocol):
    """Contract for any model inference backend (PyTorch, ONNX, etc.).

    Liskov Substitution Principle: any class satisfying this protocol
    can replace another without breaking callers.
    """

    @property
    def model_name(self) -> str:
        """Human-readable model display name."""
        ...

    @property
    def num_classes(self) -> int:
        """Number of output classes."""
        ...

    def predict(self, image: Image.Image, top_k: int = 3) -> List[Dict]:
        """Run inference on a PIL image.

        Args:
            image: RGB PIL Image.
            top_k: Number of top predictions to return.

        Returns:
            List of dicts with keys ``class_name`` and ``probability``.
        """
        ...


@runtime_checkable
class DiseaseInfoService(Protocol):
    """Contract for disease metadata lookup.

    Interface Segregation: prediction routes only need ``enrich()``,
    they don't care how the data is stored.
    """

    def enrich(self, class_name: str) -> Dict[str, str]:
        """Look up disease info for a predicted class name.

        Args:
            class_name: Raw model class name (e.g. ``Tomato_Early_blight``).

        Returns:
            Dict with keys: crop, disease, severity, action, product.
        """
        ...
