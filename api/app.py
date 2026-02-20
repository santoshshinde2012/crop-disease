"""
FastAPI application factory.

Single Responsibility: assembles the application from routes and
middleware — no business logic lives here.

Run with:
    uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload

Production (multi-worker):
    uvicorn api.app:app --host 0.0.0.0 --port 8000 --workers 4
"""

import logging

from fastapi import FastAPI

from api import __version__
from api.middleware import register_middleware
from api.routes import health, predict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)


def create_app() -> FastAPI:
    """Application factory — builds and configures the FastAPI instance.

    Open/Closed: register new route modules here without modifying
    existing ones.
    """
    application = FastAPI(
        title="Crop Disease Detection API",
        description=(
            "Upload a leaf image → get disease prediction with treatment "
            "recommendations and product suggestions."
        ),
        version=__version__,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ---- Middleware (OCP — add without modifying existing) ----
    register_middleware(application)

    # ---- Routes (ISP — each router is a focused interface) ----
    application.include_router(health.router)
    application.include_router(predict.router)

    return application


# Module-level instance for ``uvicorn api.app:app``
app = create_app()
