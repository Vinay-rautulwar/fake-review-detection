from typing import Generator
import mongomock
import pytest
from fastapi.testclient import TestClient

from backend.api.deps import get_model_service, get_prediction_repository
from backend.db.mongodb import DatabaseUnavailableException, db_manager
from backend.main import app
from backend.repositories.prediction_repository import PredictionRepository
from backend.services.model_service import ModelService, ModelUnavailableException, PredictionResult


class DummyAvailableModelService(ModelService):
    """Test model service stub that simulates a successfully loaded model."""

    def __init__(self):
        super().__init__()
        self.configured_model = "test_model"
        self._is_loaded = True
        self._is_available = True
        self._status_message = "Test model loaded successfully."

    def predict(self, review_text: str, metadata=None) -> PredictionResult:
        # Realistic mock prediction for tests
        is_fake = "free" in review_text.lower() or "scam" in review_text.lower()
        fake_prob = 0.85 if is_fake else 0.15
        return PredictionResult(
            predicted_label="fake" if is_fake else "genuine",
            fake_probability=fake_prob,
            confidence=0.85,
            model_name="test_model",
        )

    def predict_batch(self, reviews):
        return [self.predict(r) for r in reviews]


class DummyUnavailableModelService(ModelService):
    """Test model service stub that simulates an untrained or missing model."""

    def __init__(self):
        super().__init__()
        self.configured_model = "deberta_fusion"
        self._is_loaded = False
        self._is_available = False
        self._status_message = "DeBERTa Multimodal Fusion model is not trained yet. Artifacts missing."

    def predict(self, review_text: str, metadata=None):
        raise ModelUnavailableException(
            f"Model '{self.configured_model}' is currently unavailable: {self._status_message}"
        )

    def predict_batch(self, reviews):
        raise ModelUnavailableException(
            f"Batch inference unavailable: Model '{self.configured_model}' is not loaded."
        )


@pytest.fixture
def mock_db():
    """Provide a fresh in-memory mongomock database instance."""
    client = mongomock.MongoClient()
    db = client["fake_review_detection_test"]
    return db


@pytest.fixture
def mock_repo(mock_db):
    """Provide a PredictionRepository wired to the in-memory mongomock database."""
    return PredictionRepository(db=mock_db)


@pytest.fixture
def disconnected_repo():
    """Provide a PredictionRepository that simulates an unavailable MongoDB connection."""
    class DisconnectedRepository(PredictionRepository):
        def _get_collection(self):
            raise DatabaseUnavailableException("Simulated connection failure to MongoDB server.")

        def get_analytics_summary(self):
            raise DatabaseUnavailableException("Simulated connection failure to MongoDB server.")

    return DisconnectedRepository(db=None)


@pytest.fixture
def available_model_svc():
    return DummyAvailableModelService()


@pytest.fixture
def unavailable_model_svc():
    return DummyUnavailableModelService()


@pytest.fixture
def client(mock_repo, unavailable_model_svc) -> Generator[TestClient, None, None]:
    """Default TestClient with in-memory DB and default unavailable model (mirroring production startup)."""
    app.dependency_overrides[get_prediction_repository] = lambda: mock_repo
    app.dependency_overrides[get_model_service] = lambda: unavailable_model_svc

    with TestClient(app, raise_server_exceptions=False) as c:
        yield c

    app.dependency_overrides.clear()
