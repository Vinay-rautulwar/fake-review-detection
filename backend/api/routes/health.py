from datetime import datetime, timezone
from fastapi import APIRouter, Depends

from backend.core.config import settings
from backend.db.mongodb import MongoDBManager, db_manager
from backend.schemas.common import DatabaseStatus, HealthStatus, ModelHealthStatus
from backend.services.model_service import ModelService, get_model_service

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthStatus,
    summary="System Health & Component Availability Check",
    description="Returns availability statuses for API service, MongoDB database connection, and configured ML model.",
)
async def check_health(
    model_svc: ModelService = Depends(get_model_service),
) -> HealthStatus:
    # 1. Database check
    is_db_connected = db_manager.ping()
    db_status = DatabaseStatus(
        status="connected" if is_db_connected else "disconnected",
        database_name=settings.MONGODB_DATABASE,
        details=None if is_db_connected else db_manager.last_error,
    )

    # 2. Model check
    is_model_ready = model_svc.is_available
    model_status = ModelHealthStatus(
        status="available" if is_model_ready else "unavailable",
        configured_model=model_svc.configured_model,
        loaded=model_svc.is_loaded,
        details=None if is_model_ready else model_svc.status_message,
    )

    # 3. Overall composite health
    if is_db_connected and is_model_ready:
        overall_status = "healthy"
    elif is_db_connected or is_model_ready:
        overall_status = "degraded"
    else:
        overall_status = "unhealthy"

    return HealthStatus(
        status=overall_status,
        api="available",
        database=db_status,
        model=model_status,
        timestamp=datetime.now(timezone.utc),
    )
