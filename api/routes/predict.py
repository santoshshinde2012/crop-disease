"""Prediction endpoint - delegates to inference and disease services."""

from __future__ import annotations

import io
import logging

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from PIL import Image

from app.config import CONFIDENCE_THRESHOLD
from api.dependencies import get_disease_service, get_inference_service
from api.protocols import DiseaseInfoService, InferenceService
from api.schemas import PredictionDetail, PredictionResponse

logger = logging.getLogger(__name__)

# Allowed MIME types for uploaded images.
_ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png"}

router = APIRouter(tags=["Prediction"])


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict crop disease",
    description=(
        "Upload a leaf image and receive the top-3 disease predictions "
        "with severity, treatment, and product recommendations. "
        "The ``confident`` flag is True when the top prediction exceeds 70%."
    ),
)
async def predict_endpoint(
    file: UploadFile = File(
        ..., description="Leaf image in JPEG or PNG format"
    ),
    inference: InferenceService = Depends(get_inference_service),
    disease_svc: DiseaseInfoService = Depends(get_disease_service),
) -> PredictionResponse:
    # ---- Validate upload ----
    if file.content_type not in _ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Only JPEG and PNG images are supported (got {file.content_type})",
        )

    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as exc:
        logger.warning("Image decode failed: %s", exc)
        raise HTTPException(
            status_code=400, detail="Could not decode the uploaded image"
        )

    # ---- Run inference (SRP — delegated to service) ----
    raw_predictions = inference.predict(image)

    # ---- Enrich with disease metadata (SRP — delegated to service) ----
    results: list[PredictionDetail] = []
    for pred in raw_predictions:
        info = disease_svc.enrich(pred["class_name"])
        results.append(
            PredictionDetail(
                class_name=pred["class_name"],
                probability=pred["probability"],
                crop=info["crop"],
                disease=info["disease"],
                severity=info["severity"],
                action=info["action"],
                product=info["product"],
            )
        )

    confident = (
        results[0].probability >= CONFIDENCE_THRESHOLD if results else False
    )

    return PredictionResponse(
        model=inference.model_name,
        confident=confident,
        predictions=results,
    )
