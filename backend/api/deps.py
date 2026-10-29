from fastapi import Depends
from pymongo.database import Database

from backend.db.mongodb import db_manager
from backend.repositories.prediction_repository import PredictionRepository
from backend.services.model_service import ModelService, get_model_service
from backend.services.prediction_service import PredictionService


def get_prediction_repository() -> PredictionRepository:
    """Dependency that returns a repository connected to the current database."""
    # Note: If db_manager.db is None, PredictionRepository methods will raise
    # DatabaseUnavailableException when called, properly handled by FastAPI.
    return PredictionRepository(db=db_manager.db)


def get_prediction_service(
    model_svc: ModelService = Depends(get_model_service),
    repo: PredictionRepository = Depends(get_prediction_repository),
) -> PredictionService:
    """Dependency providing PredictionService with injected model service and repository."""
    return PredictionService(model_service=model_svc, repository=repo)
