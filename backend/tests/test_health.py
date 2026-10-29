from fastapi.testclient import TestClient
from unittest.mock import patch

from backend.api.deps import get_model_service
from backend.main import app
from backend.tests.conftest import DummyAvailableModelService, DummyUnavailableModelService


def test_root_endpoint(client: TestClient):
    """Verify root endpoint returns basic service metadata and documentation links."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "documentation" in data
    assert data["documentation"] == "/docs"


def test_health_endpoint_degraded_when_model_unavailable(client: TestClient):
    """When DB is mocked/connected but model is untrained, health is degraded."""
    with patch("backend.db.mongodb.db_manager.ping", return_value=True):
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["api"] == "available"
        assert data["database"]["status"] == "connected"
        assert data["model"]["status"] == "unavailable"
        assert data["model"]["loaded"] is False
        assert data["status"] == "degraded"


def test_health_endpoint_healthy_when_all_available(client: TestClient, available_model_svc):
    """When both DB and model are fully operational, health returns 'healthy'."""
    app.dependency_overrides[get_model_service] = lambda: available_model_svc
    with patch("backend.db.mongodb.db_manager.ping", return_value=True):
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["database"]["status"] == "connected"
        assert data["model"]["status"] == "available"
        assert data["model"]["loaded"] is True


def test_health_endpoint_degraded_when_database_down(client: TestClient, available_model_svc):
    """When DB is unreachable, health returns disconnected database status."""
    app.dependency_overrides[get_model_service] = lambda: available_model_svc
    with patch("backend.db.mongodb.db_manager.ping", return_value=False):
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["database"]["status"] == "disconnected"
        assert data["status"] == "degraded"


def test_models_list_endpoint(client: TestClient):
    """Verify /api/v1/models returns planned models and truthful availability."""
    response = client.get("/api/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert "active_model" in data
    assert "models" in data
    assert data["total_models"] == 3

    model_names = [m["name"] for m in data["models"]]
    assert "baseline" in model_names
    assert "deberta_text" in model_names
    assert "deberta_fusion" in model_names

    for m in data["models"]:
        assert "display_name" in m
        assert "architecture" in m
        assert "input_type" in m
        assert "is_loaded" in m
        assert "is_available" in m
        assert "status_message" in m


def test_models_metrics_endpoint(client: TestClient):
    """Verify /api/v1/models/metrics returns real baseline metrics and truthful DeBERTa status."""
    response = client.get("/api/v1/models/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "active_model" in data
    assert "baseline" in data
    assert "deberta_text" in data
    assert "deberta_fusion" in data

    # Baseline has evaluated metrics on disk
    baseline = data["baseline"]
    assert baseline["is_evaluated"] is True
    assert baseline["status"] == "ready"
    assert baseline["metrics"]["roc_auc"] == 0.6691
    assert baseline["metrics"]["pr_auc"] == 0.6118

    # DeBERTa is in training (unless colab weights were placed)
    deberta = data["deberta_text"]
    assert "training_environment" in deberta
    assert deberta["status"] in ["training", "ready"]


def test_direct_health_endpoint(client: TestClient):
    """Verify root GET /health reports model loaded, device, precision, and modality without secrets."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert "model" in data
    assert "device" in data
    assert data["precision"] == "fp32"
    assert data["modality"] == "text-only"
    # Never expose local paths, credentials or secrets
    raw_text = response.text
    assert "password" not in raw_text.lower()
    assert "secret" not in raw_text.lower()
    assert "d:\\" not in raw_text.lower()
    assert "c:\\" not in raw_text.lower()


def test_direct_predict_endpoint_valid_input(client: TestClient):
    """Verify POST /predict returns valid prediction, probabilities summing to 1, and label."""
    payload = {"review_text": "The product works exactly as described."}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] in ["genuine", "fake"]
    assert data["label"] in [0, 1]
    assert 0.0 <= data["fake_probability"] <= 1.0
    assert 0.0 <= data["genuine_probability"] <= 1.0
    assert abs(data["fake_probability"] + data["genuine_probability"] - 1.0) < 0.01
    assert data["model"] == "deberta-v3-base"


def test_direct_predict_endpoint_empty_and_short_input(client: TestClient):
    """Verify POST /predict safely rejects empty and short inputs."""
    # Empty string
    response = client.post("/predict", json={"review_text": ""})
    assert response.status_code in [400, 422]

    # Short string (< 3 chars)
    response = client.post("/predict", json={"review_text": "ab"})
    assert response.status_code in [400, 422]

    # Missing field
    response = client.post("/predict", json={})
    assert response.status_code in [400, 422]

