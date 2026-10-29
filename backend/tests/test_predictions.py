import io
from fastapi.testclient import TestClient

from backend.api.deps import get_model_service, get_prediction_repository
from backend.main import app
from backend.tests.conftest import DummyAvailableModelService


def test_predict_model_unavailable(client: TestClient):
    """When configured model is missing/untrained, POST /api/v1/predict returns HTTP 503."""
    payload = {
        "review_text": "This product was completely broken and customer service was unhelpful.",
        "rating": 1.0,
        "verified_purchase": True,
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 503
    data = response.json()
    assert "detail" in data
    assert "unavailable" in data["detail"].lower() or "not trained" in data["detail"].lower()


def test_predict_success(client: TestClient, available_model_svc):
    """When a trained model and DB are available, return prediction and save in MongoDB."""
    app.dependency_overrides[get_model_service] = lambda: available_model_svc

    payload = {
        "review_text": "Get free gift cards by clicking this link! Amazing promo scam!",
        "rating": 5.0,
        "verified_purchase": False,
        "product_category": "Gift Cards",
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert "prediction_id" in data
    assert len(data["prediction_id"]) == 24  # Valid 24-char hex ObjectId
    assert data["predicted_label"] == "fake"
    assert data["fake_probability"] >= 0.50
    assert data["confidence"] >= 0.50
    assert data["model_name"] == "test_model"
    assert data["metadata"]["product_category"] == "Gift Cards"

    # Confirm it can be retrieved from history
    history_resp = client.get(f"/api/v1/history/{data['prediction_id']}")
    assert history_resp.status_code == 200
    hist_data = history_resp.json()
    assert hist_data["id"] == data["prediction_id"]
    assert hist_data["predicted_label"] == "fake"


def test_predict_database_unavailable(client: TestClient, available_model_svc, disconnected_repo):
    """When DB is unavailable, /api/v1/predict must return 503 and never pretend writes succeeded."""
    app.dependency_overrides[get_model_service] = lambda: available_model_svc
    app.dependency_overrides[get_prediction_repository] = lambda: disconnected_repo

    payload = {
        "review_text": "Great shoes, comfortable fit and true to size.",
        "rating": 4.0,
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 503
    data = response.json()
    assert "database service unavailable" in data["detail"].lower()


def test_batch_predict_model_unavailable(client: TestClient):
    """When model is unavailable, batch-predict returns 503 without fabricating outputs."""
    csv_content = b"review_text,rating\nGreat product!,5.0\nTerrible scam!,1.0\n"
    files = {"file": ("reviews.csv", io.BytesIO(csv_content), "text/csv")}

    response = client.post("/api/v1/batch-predict", files=files)
    assert response.status_code == 503
    data = response.json()
    assert "unavailable" in data["detail"].lower()


def test_batch_predict_invalid_extension(client: TestClient, available_model_svc):
    """Uploading non-CSV files must return HTTP 400."""
    app.dependency_overrides[get_model_service] = lambda: available_model_svc
    files = {"file": ("reviews.txt", io.BytesIO(b"Just plain text"), "text/plain")}

    response = client.post("/api/v1/batch-predict", files=files)
    assert response.status_code == 400
    assert "must have a .csv extension" in response.json()["detail"].lower()


def test_batch_predict_missing_column(client: TestClient, available_model_svc):
    """CSV missing 'review_text' must return HTTP 400."""
    app.dependency_overrides[get_model_service] = lambda: available_model_svc
    csv_content = b"product_id,rating\nB001,5.0\nB002,1.0\n"
    files = {"file": ("reviews.csv", io.BytesIO(csv_content), "text/csv")}

    response = client.post("/api/v1/batch-predict", files=files)
    assert response.status_code == 400
    assert "review_text" in response.json()["detail"]


def test_batch_predict_too_many_rows(client: TestClient, available_model_svc):
    """CSV exceeding 500 rows must return HTTP 400."""
    app.dependency_overrides[get_model_service] = lambda: available_model_svc
    header = "review_text,rating\n"
    rows = "\n".join([f"Review number {i} is here,5.0" for i in range(505)])
    csv_content = (header + rows).encode("utf-8")
    files = {"file": ("large_batch.csv", io.BytesIO(csv_content), "text/csv")}

    response = client.post("/api/v1/batch-predict", files=files)
    assert response.status_code == 400
    assert "exceeds maximum limit" in response.json()["detail"].lower()


def test_batch_predict_success(client: TestClient, available_model_svc):
    """Batch predict processes valid rows, skips malformed rows, and persists to DB."""
    app.dependency_overrides[get_model_service] = lambda: available_model_svc
    csv_content = (
        "review_text,rating,verified_purchase\n"
        "Excellent quality, highly recommended!,5.0,True\n"
        "Free gift card bonus scam click now!,1.0,False\n"
        ",3.0,True\n"  # Empty review text
        "A\n"          # Too short (< 3 chars)
    ).encode("utf-8")
    files = {"file": ("test_reviews.csv", io.BytesIO(csv_content), "text/csv")}

    response = client.post("/api/v1/batch-predict", files=files)
    assert response.status_code == 200
    data = response.json()

    assert data["total_submitted"] == 4
    assert data["total_successful"] == 2
    assert data["total_failed"] == 2

    # Check successful rows
    assert data["predictions"][0]["status"] == "success"
    assert data["predictions"][0]["predicted_label"] == "genuine"
    assert data["predictions"][1]["status"] == "success"
    assert data["predictions"][1]["predicted_label"] == "fake"

    # Check failed rows
    assert data["predictions"][2]["status"] == "error"
    assert data["predictions"][3]["status"] == "error"
