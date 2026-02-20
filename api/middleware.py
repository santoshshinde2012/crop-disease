"""
Cross-cutting middleware configuration.

Open/Closed Principle: add new middleware by appending to
``register_middleware()`` — existing middleware is never modified.

Single Responsibility: this module only manages middleware;
no route logic, no business rules.
"""

import logging
import os
import time

from typing import Callable

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import Response

logger = logging.getLogger(__name__)


def register_middleware(app: FastAPI) -> None:
    """Attach all middleware to the FastAPI application.

    Args:
        app: The FastAPI application instance.
    """
    # ---- CORS ----
    allowed_origins = os.getenv("CORS_ORIGINS", "*").split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_methods=["POST", "GET"],
        allow_headers=["*"],
    )

    # ---- Request logging / timing ----
    @app.middleware("http")
    async def log_requests(
        request: Request, call_next: Callable
    ) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "%s %s → %d (%.1f ms)",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        return response
