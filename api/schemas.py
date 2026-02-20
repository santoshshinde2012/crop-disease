"""
Pydantic request/response schemas for the API.

Single Responsibility: only data validation and serialization lives here.
No business logic, no model loading, no I/O.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Response schema for the health check endpoint."""

    status: str = Field(..., examples=["healthy"])
    model: str = Field(..., examples=["efficientnet_b0"])


class ModelVersionResponse(BaseModel):
    """Response schema for model version endpoint."""

    model: str = Field(..., examples=["efficientnet_b0"])
    num_classes: int = Field(..., examples=[12])
    version: str = Field(..., examples=["1.0.0"])


class PredictionDetail(BaseModel):
    """Single prediction with disease metadata."""

    class_name: str = Field(..., examples=["Tomato_Early_blight"])
    probability: float = Field(..., ge=0.0, le=1.0, examples=[0.9234])
    crop: str = Field(..., examples=["Tomato"])
    disease: str = Field(..., examples=["Early Blight"])
    severity: str = Field(..., examples=["Moderate"])
    action: str = Field(
        ...,
        examples=["Apply fungicide at first sign. Remove lower infected leaves."],
    )
    product: str = Field(
        ..., examples=["Score (Difenoconazole) or Mancozeb"]
    )


class PredictionResponse(BaseModel):
    """Top-level prediction response with confidence flag."""

    model: str = Field(..., examples=["efficientnet_b0"])
    confident: bool = Field(
        ...,
        description="True when top prediction exceeds the confidence threshold (70%)",
    )
    predictions: list[PredictionDetail]
