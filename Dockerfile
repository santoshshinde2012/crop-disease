# ---- Build stage ----
FROM python:3.11-slim AS builder

WORKDIR /build

# Install only API dependencies (not notebook/test/streamlit)
COPY requirements-api.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements-api.txt


# ---- Runtime stage ----
FROM python:3.11-slim

LABEL maintainer="Santosh Shinde"
LABEL description="Crop Disease Detection REST API"

WORKDIR /app

# Copy only the installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code (order matters for layer caching)
COPY src/ src/
COPY app/ app/
COPY api/ api/

# Models directory — copy everything in models/ (.gitkeep, .json, .pth if present)
# Use Git LFS or a download step to populate models/ before building
COPY models/ models/

# Non-root user for security
RUN adduser --disabled-password --no-create-home appuser
USER appuser

# Expose default API port (overridden by $PORT on cloud platforms)
EXPOSE 8000

# Health check for container orchestrators
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT:-8000}/health')" || exit 1

# Production: respect $PORT env var (Render, HF Spaces, etc.)
CMD ["sh", "-c", "uvicorn api.app:app --host 0.0.0.0 --port ${PORT:-8000} --workers 2"]
