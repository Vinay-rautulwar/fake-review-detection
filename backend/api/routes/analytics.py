from fastapi import APIRouter, Depends, HTTPException, status

from backend.api.deps import get_prediction_service
from backend.db.mongodb import DatabaseUnavailableException
from backend.schemas.prediction import AnalyticsSummaryResponse
from backend.services.model_service import ModelService, get_model_service
from backend.services.prediction_service import PredictionService

router = APIRouter(tags=["Analytics"])


@router.get(
    "/analytics/summary",
    response_model=AnalyticsSummaryResponse,
    summary="Application Prediction Analytics & Class Distribution",
    description="Returns aggregate usage metrics, classification distribution, and recent activity from the MongoDB predictions collection.",
)
async def get_analytics_summary(
    service: PredictionService = Depends(get_prediction_service),
) -> AnalyticsSummaryResponse:
    try:
        return await service.get_analytics_summary()
    except DatabaseUnavailableException as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database service unavailable: {str(e)}",
        )


@router.get(
    "/analytics/visualizations",
    summary="Get Visual Analytics & Research Figures",
    description="Returns dataset split distribution, behavioral correlations, and research paper visualization assets.",
)
async def get_visualizations_data(
    model_svc: ModelService = Depends(get_model_service),
):
    return model_svc.get_visualizations_data()
