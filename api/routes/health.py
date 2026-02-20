"""
Health and version endpoints.

Single Responsibility: only health/readiness probes and model metadata.
Interface Segregation: separated from prediction routes so monitoring
tools don't depend on the prediction interface.
"""

from fastapi import APIRouter, Depends

from api import __version__
from api.dependencies import get_inference_service
from api.protocols import InferenceService
from api.schemas import HealthResponse, ModelVersionResponse

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Returns service health status. Used by load balancers and mobile app connectivity probes.",
)
def health_check(
    inference: InferenceService = Depends(get_inference_service),
) -> HealthResponse:
    return HealthResponse(status="healthy", model=inference.model_name)


@router.get(
    "/model/version",
    response_model=ModelVersionResponse,
    summary="Model version",
    description="Returns current model details — used by mobile app for update checks.",
)
def model_version(
    inference: InferenceService = Depends(get_inference_service),
) -> ModelVersionResponse:
    return ModelVersionResponse(
        model=inference.model_name,
        num_classes=inference.num_classes,
        version=__version__,
    )
