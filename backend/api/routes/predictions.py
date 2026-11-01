from typing import Optional
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from backend.api.deps import get_prediction_service
from backend.db.mongodb import DatabaseUnavailableException
from backend.schemas.prediction import BatchPredictionResponse, PredictionRequest, PredictionResponse
from backend.services.model_service import ModelUnavailableException
from backend.services.prediction_service import PredictionService

router = APIRouter(tags=["Predictions"])


@router.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Classify a Single Product Review",
    description="Submits a review text and optional metadata for real-time fake/genuine classification and logs to database.",
)
async def predict_single_review(
    request: PredictionRequest,
    service: PredictionService = Depends(get_prediction_service),
) -> PredictionResponse:
    try:
        return await service.predict_single(request)
    except ModelUnavailableException as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
        )
    except DatabaseUnavailableException as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database service unavailable: {str(e)}",
        )


@router.post(
    "/batch-predict",
    response_model=BatchPredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Batch Classify Reviews via CSV File Upload",
    description="Upload a CSV file containing a 'review_text' column (max 500 rows) for batch inference.",
)
async def predict_batch_reviews(
    file: UploadFile = File(..., description="CSV file containing review_text column"),
    model: Optional[str] = Form(None, description="Target model name"),
    threshold: Optional[float] = Form(None, description="Classification threshold"),
    service: PredictionService = Depends(get_prediction_service),
) -> BatchPredictionResponse:
    try:
        content = await file.read()
        return await service.predict_batch(
            file_content=content,
            filename=file.filename or "upload.csv",
            model_name=model,
            threshold=threshold,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except ModelUnavailableException as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
        )
    except DatabaseUnavailableException as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database service unavailable: {str(e)}",
        )
