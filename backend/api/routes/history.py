from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status

from backend.api.deps import get_prediction_service
from backend.db.mongodb import DatabaseUnavailableException
from backend.schemas.prediction import HistoryItem, HistoryResponse
from backend.services.prediction_service import PredictionService

router = APIRouter(tags=["History"])


@router.get(
    "/history",
    response_model=HistoryResponse,
    summary="Query Prediction History with Pagination & Filters",
    description="Retrieve paginated list of stored predictions with optional filtering by class and model.",
)
async def get_history(
    page: int = Query(1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page (max 100)"),
    prediction_class: Optional[str] = Query(
        None,
        description="Filter by classification label ('fake' or 'genuine')",
    ),
    model_name: Optional[str] = Query(None, description="Filter by model identifier"),
    sort_order: str = Query("desc", description="Sort order by timestamp ('desc' or 'asc')"),
    service: PredictionService = Depends(get_prediction_service),
) -> HistoryResponse:
    if sort_order.lower() not in ["asc", "desc"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="sort_order must be either 'asc' or 'desc'.",
        )

    if prediction_class and prediction_class.lower() not in ["fake", "genuine"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="prediction_class must be either 'fake' or 'genuine'.",
        )

    try:
        return await service.get_history(
            page=page,
            page_size=page_size,
            prediction_class=prediction_class,
            model_name=model_name,
            sort_order=sort_order,
        )
    except DatabaseUnavailableException as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database service unavailable: {str(e)}",
        )


@router.get(
    "/history/{prediction_id}",
    response_model=HistoryItem,
    summary="Get Specific Stored Prediction by ID",
    description="Retrieve details of a single historical prediction using its MongoDB ObjectId.",
)
async def get_prediction_detail(
    prediction_id: str,
    service: PredictionService = Depends(get_prediction_service),
) -> HistoryItem:
    try:
        return await service.get_prediction_by_id(prediction_id)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except KeyError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e).strip("'"),
        )
    except DatabaseUnavailableException as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database service unavailable: {str(e)}",
        )
