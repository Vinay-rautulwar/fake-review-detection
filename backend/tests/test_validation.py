from datetime import datetime, timezone
from bson import ObjectId
from fastapi.testclient import TestClient

from backend.api.deps import get_model_service, get_prediction_repository
from backend.main import app
from backend.tests.conftest import DummyAvailableModelService


def test_empty_review_text(client: TestClient):
    """Empty review text must fail Pydantic validation with HTTP 422."""
    response = client.post("/api/v1/predict", json={"review_text": ""})
    assert response.status_code == 422
    data = response.json()
    assert "errors" in data or "detail" in data


def test_whitespace_only_review_text(client: TestClient):
    """Whitespace-only review text must fail validation with HTTP 422."""
    response = client.post("/api/v1/predict", json={"review_text": "     "})
    assert response.status_code == 422


def test_short_review_text(client: TestClient):
    """Review text with less than 3 characters must fail validation with HTTP 422."""
    response = client.post("/api/v1/predict", json={"review_text": "ok"})
    assert response.status_code == 422


def test_rating_out_of_bounds_high(client: TestClient):
    """Star rating above 5.0 must return HTTP 422."""
    response = client.post(
        "/api/v1/predict",
        json={"review_text": "Decent product overall.", "rating": 5.5},
    )
    assert response.status_code == 422


def test_rating_out_of_bounds_low(client: TestClient):
    """Star rating below 1.0 must return HTTP 422."""
    response = client.post(
        "/api/v1/predict",
        json={"review_text": "Worst purchase ever.", "rating": 0.5},
    )
    assert response.status_code == 422


def test_negative_price(client: TestClient):
    """Negative product price must return HTTP 422."""
    response = client.post(
        "/api/v1/predict",
        json={"review_text": "Good value for money.", "product_price": -15.0},
    )
    assert response.status_code == 422


def test_history_invalid_object_id(client: TestClient):
    """Invalid ObjectId format must return HTTP 400 Bad Request."""
    response = client.get("/api/v1/history/123-invalid-id")
    assert response.status_code == 400
    assert "invalid prediction id" in response.json()["detail"].lower()


def test_history_not_found(client: TestClient):
    """Valid ObjectId that does not exist in the database must return HTTP 404."""
    non_existent_id = str(ObjectId())
    response = client.get(f"/api/v1/history/{non_existent_id}")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_history_invalid_sort_order(client: TestClient):
    """Sort order other than 'asc' or 'desc' must return HTTP 400."""
    response = client.get("/api/v1/history?sort_order=alphabetical")
    assert response.status_code == 400
    assert "sort_order" in response.json()["detail"]


def test_history_invalid_class_filter(client: TestClient):
    """Prediction class filter other than 'fake' or 'genuine' must return HTTP 400."""
    response = client.get("/api/v1/history?prediction_class=suspicious")
    assert response.status_code == 400
    assert "prediction_class" in response.json()["detail"]


def test_history_pagination_and_filters(client: TestClient, mock_repo):
    """Verify history pagination, class filtering, and sorting."""
    # Seed mock database with known records
    for i in range(15):
        label = "fake" if i % 2 == 0 else "genuine"
        mock_repo.save_prediction(
            {
                "review_text": f"Sample review number {i}",
                "predicted_label": label,
                "fake_probability": 0.8 if label == "fake" else 0.2,
                "confidence": 0.8,
                "model_name": "test_model",
                "timestamp": datetime.now(timezone.utc),
            }
        )

    # 1. Fetch page 1 (5 items)
    resp_p1 = client.get("/api/v1/history?page=1&page_size=5")
    assert resp_p1.status_code == 200
    data_p1 = resp_p1.json()
    assert data_p1["total"] == 15
    assert len(data_p1["items"]) == 5
    assert data_p1["total_pages"] == 3

    # 2. Filter by fake class
    resp_fake = client.get("/api/v1/history?prediction_class=fake&page_size=20")
    assert resp_fake.status_code == 200
    data_fake = resp_fake.json()
    assert data_fake["total"] == 8
    for item in data_fake["items"]:
        assert item["predicted_label"] == "fake"


def test_analytics_summary(client: TestClient, mock_repo):
    """Verify analytics summary accurately aggregates counts and includes mandatory disclaimer."""
    # Seed 3 genuine and 2 fake reviews
    for _ in range(3):
        mock_repo.save_prediction(
            {
                "review_text": "Genuine sample text",
                "predicted_label": "genuine",
                "fake_probability": 0.1,
                "confidence": 0.9,
                "model_name": "test_model",
                "timestamp": datetime.now(timezone.utc),
            }
        )
    for _ in range(2):
        mock_repo.save_prediction(
            {
                "review_text": "Fake sample promo",
                "predicted_label": "fake",
                "fake_probability": 0.95,
                "confidence": 0.95,
                "model_name": "test_model",
                "timestamp": datetime.now(timezone.utc),
            }
        )

    response = client.get("/api/v1/analytics/summary")
    assert response.status_code == 200
    data = response.json()

    assert data["total_predictions"] == 5
    assert data["fake_count"] == 2
    assert data["genuine_count"] == 3
    assert data["fake_percentage"] == 40.0
    assert "disclaimer" in data
    # Mandatory distinction between user submissions and model evaluation benchmarks
    assert "must not be confused with" in data["disclaimer"].lower() or "operational" in data["disclaimer"].lower()
