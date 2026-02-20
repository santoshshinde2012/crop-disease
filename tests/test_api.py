"""
Tests for the Crop Disease Detection REST API.

Uses FastAPI's TestClient (backed by httpx) so no real server is needed.
Tests are organized by endpoint, following the same SOLID structure as the API.
"""

import io
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api.app import create_app
from api.schemas import PredictionResponse


# ---- Fixtures ----


@pytest.fixture(scope="module")
def mock_inference_service():
    """Create a mock inference service that satisfies the InferenceService protocol."""
    service = MagicMock()
    service.model_name = "test_model"
    service.num_classes = 12
    service.predict.return_value = [
        {"class_name": "Tomato_Early_blight", "probability": 0.9234},
        {"class_name": "Tomato_Late_blight", "probability": 0.0453},
        {"class_name": "Tomato_Septoria_leaf_spot", "probability": 0.0189},
    ]
    return service


@pytest.fixture(scope="module")
def mock_disease_service():
    """Create a mock disease service that satisfies the DiseaseInfoService protocol."""
    service = MagicMock()

    def _enrich(class_name: str):
        db = {
            "Tomato_Early_blight": {
                "crop": "Tomato",
                "disease": "Early Blight",
                "severity": "Moderate",
                "action": "Apply fungicide at first sign.",
                "product": "Score (Difenoconazole)",
            },
            "Tomato_Late_blight": {
                "crop": "Tomato",
                "disease": "Late Blight",
                "severity": "High -- Urgent",
                "action": "Apply systemic fungicide immediately.",
                "product": "Ridomil Gold MZ",
            },
            "Tomato_Septoria_leaf_spot": {
                "crop": "Tomato",
                "disease": "Septoria Leaf Spot",
                "severity": "Moderate",
                "action": "Remove infected leaves.",
                "product": "Bravo + Score tank mix",
            },
        }
        return db.get(class_name, {
            "crop": "Unknown",
            "disease": "Unknown",
            "severity": "Unknown",
            "action": "Consult agronomist.",
            "product": "N/A",
        })

    service.enrich.side_effect = _enrich
    return service


@pytest.fixture(scope="module")
def client(mock_inference_service, mock_disease_service):
    """Create a TestClient with mocked services (no real model loading)."""
    from api.dependencies import get_disease_service, get_inference_service

    app = create_app()

    # Override dependencies with mocks (DIP in action)
    app.dependency_overrides[get_inference_service] = lambda: mock_inference_service
    app.dependency_overrides[get_disease_service] = lambda: mock_disease_service

    with TestClient(app) as test_client:
        yield test_client

    # Cleanup
    app.dependency_overrides.clear()


def _create_test_image(fmt: str = "JPEG") -> io.BytesIO:
    """Helper: create a minimal in-memory test image."""
    img = Image.new("RGB", (224, 224), color=(34, 139, 34))
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    buf.seek(0)
    return buf


# ---- Health endpoint tests ----


class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_response_structure(self, client):
        data = client.get("/health").json()
        assert data["status"] == "healthy"
        assert data["model"] == "test_model"

    def test_health_content_type(self, client):
        resp = client.get("/health")
        assert "application/json" in resp.headers["content-type"]


# ---- Model version endpoint tests ----


class TestModelVersionEndpoint:
    """Tests for GET /model/version."""

    def test_version_returns_200(self, client):
        resp = client.get("/model/version")
        assert resp.status_code == 200

    def test_version_response_structure(self, client):
        data = client.get("/model/version").json()
        assert data["model"] == "test_model"
        assert data["num_classes"] == 12
        assert "version" in data

    def test_version_string_format(self, client):
        data = client.get("/model/version").json()
        parts = data["version"].split(".")
        assert len(parts) == 3, "Version should be semver (x.y.z)"


# ---- Prediction endpoint tests ----


class TestPredictEndpoint:
    """Tests for POST /predict."""

    def test_predict_returns_200(self, client):
        img = _create_test_image()
        resp = client.post(
            "/predict",
            files={"file": ("leaf.jpg", img, "image/jpeg")},
        )
        assert resp.status_code == 200

    def test_predict_response_model(self, client):
        """Response should match the PredictionResponse schema."""
        img = _create_test_image()
        resp = client.post(
            "/predict",
            files={"file": ("leaf.jpg", img, "image/jpeg")},
        )
        data = resp.json()

        # Validate via Pydantic
        parsed = PredictionResponse(**data)
        assert parsed.model == "test_model"
        assert len(parsed.predictions) == 3

    def test_predict_confident_flag(self, client):
        """Top prediction >= 0.70 → confident=True."""
        img = _create_test_image()
        data = client.post(
            "/predict",
            files={"file": ("leaf.jpg", img, "image/jpeg")},
        ).json()
        assert data["confident"] is True  # 0.9234 >= 0.70

    def test_predict_enriched_with_disease_info(self, client):
        """Each prediction should have disease metadata."""
        img = _create_test_image()
        data = client.post(
            "/predict",
            files={"file": ("leaf.jpg", img, "image/jpeg")},
        ).json()

        top = data["predictions"][0]
        assert top["crop"] == "Tomato"
        assert top["disease"] == "Early Blight"
        assert top["severity"] == "Moderate"
        assert "fungicide" in top["action"].lower()
        assert "Score" in top["product"]

    def test_predict_probabilities_ordered(self, client):
        """Predictions should be ordered by descending probability."""
        img = _create_test_image()
        data = client.post(
            "/predict",
            files={"file": ("leaf.jpg", img, "image/jpeg")},
        ).json()

        probs = [p["probability"] for p in data["predictions"]]
        assert probs == sorted(probs, reverse=True)

    def test_predict_accepts_png(self, client):
        """PNG images should also be accepted."""
        img = _create_test_image(fmt="PNG")
        resp = client.post(
            "/predict",
            files={"file": ("leaf.png", img, "image/png")},
        )
        assert resp.status_code == 200

    def test_predict_rejects_non_image(self, client):
        """Non-image files should return 400."""
        buf = io.BytesIO(b"not an image")
        resp = client.post(
            "/predict",
            files={"file": ("data.txt", buf, "text/plain")},
        )
        assert resp.status_code == 400

    def test_predict_rejects_corrupt_image(self, client):
        """Corrupt image data with image MIME type should return 400."""
        buf = io.BytesIO(b"\x89PNG\r\n\x1a\ncorrupt")
        resp = client.post(
            "/predict",
            files={"file": ("bad.png", buf, "image/png")},
        )
        assert resp.status_code == 400

    def test_predict_low_confidence_flag(self, client, mock_inference_service):
        """Top prediction < 0.70 → confident=False."""
        # Temporarily override predict to return low-confidence results
        original = mock_inference_service.predict.return_value
        mock_inference_service.predict.return_value = [
            {"class_name": "Tomato_Early_blight", "probability": 0.45},
            {"class_name": "Tomato_Late_blight", "probability": 0.30},
            {"class_name": "Tomato_healthy", "probability": 0.15},
        ]
        try:
            img = _create_test_image()
            data = client.post(
                "/predict",
                files={"file": ("leaf.jpg", img, "image/jpeg")},
            ).json()
            assert data["confident"] is False  # 0.45 < 0.70
        finally:
            mock_inference_service.predict.return_value = original


# ---- Protocol compliance tests ----


class TestProtocolCompliance:
    """Verify that mock services satisfy the protocol contracts."""

    def test_inference_service_protocol(self, mock_inference_service):
        from api.protocols import InferenceService

        assert isinstance(mock_inference_service, InferenceService)

    def test_disease_service_protocol(self, mock_disease_service):
        from api.protocols import DiseaseInfoService

        assert isinstance(mock_disease_service, DiseaseInfoService)


# ---- Schema validation tests ----


class TestSchemas:
    """Tests for Pydantic schema validation."""

    def test_prediction_detail_bounds(self):
        from api.schemas import PredictionDetail

        # Valid
        p = PredictionDetail(
            class_name="Tomato_healthy",
            probability=0.95,
            crop="Tomato",
            disease="Healthy",
            severity="None",
            action="No treatment needed.",
            product="N/A",
        )
        assert p.probability == 0.95

    def test_prediction_detail_probability_range(self):
        from api.schemas import PredictionDetail

        with pytest.raises(Exception):
            PredictionDetail(
                class_name="test",
                probability=1.5,  # exceeds 1.0
                crop="X",
                disease="X",
                severity="X",
                action="X",
                product="X",
            )
