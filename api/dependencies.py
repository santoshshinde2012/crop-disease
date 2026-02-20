"""
FastAPI dependency injection — wiring layer.

Dependency Inversion Principle: routes receive service instances via
``Depends()``, never importing concrete classes directly.  Swap
implementations by changing only this module.

Single Responsibility: this module's only job is constructing and
providing service singletons.
"""

import logging
from functools import lru_cache

from api.protocols import DiseaseInfoService, InferenceService
from api.services.disease_service import DiseaseInfoLookupService
from api.services.inference_service import PyTorchInferenceService

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_inference_service() -> InferenceService:
    """Singleton factory for the inference service.

    ``lru_cache`` ensures the model is loaded exactly once and
    reused across all requests.

    To switch to ONNX Runtime, replace the return type and
    constructor here — no route changes required (OCP).
    """
    logger.info("Initializing inference service…")
    return PyTorchInferenceService()


@lru_cache(maxsize=1)
def get_disease_service() -> DiseaseInfoService:
    """Singleton factory for the disease info service."""
    return DiseaseInfoLookupService()
