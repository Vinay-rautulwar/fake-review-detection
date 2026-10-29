from fastapi import APIRouter

from backend.api.routes import analytics, health, history, models, predictions

api_router = APIRouter()

api_router.include_router(health.router)
api_router.include_router(models.router)
api_router.include_router(predictions.router)
api_router.include_router(history.router)
api_router.include_router(analytics.router)
