"""
Crop Disease Detection — REST API package.

SOLID Architecture:
    - schemas.py        → Pydantic request/response models (SRP)
    - protocols.py      → Abstract interfaces (DIP, ISP)
    - services/         → Business logic implementations (SRP, OCP, LSP)
    - routes/           → Endpoint handlers (SRP, ISP)
    - dependencies.py   → FastAPI dependency injection (DIP)
    - middleware.py      → Cross-cutting concerns (OCP)
    - app.py            → Application factory (SRP)
"""

__version__ = "1.0.0"
