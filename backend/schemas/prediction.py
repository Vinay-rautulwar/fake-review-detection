from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class PredictionRequest(BaseModel):
    """Payload for single review prediction."""

    review_text: str = Field(
        ...,
        min_length=3,
        max_length=10000,
        description="The customer review text to classify",
        examples=["This product exceeded my expectations! The build quality is fantastic."],
    )
    rating: Optional[float] = Field(
        None,
        ge=1.0,
        le=5.0,
        description="Optional star rating given by the reviewer (1.0 to 5.0)",
        examples=[5.0],
    )
    verified_purchase: Optional[bool] = Field(
        None,
        description="Optional flag indicating if the reviewer had a verified purchase",
        examples=[True],
    )
    product_category: Optional[str] = Field(
        None,
        max_length=100,
        description="Optional product category",
        examples=["Electronics"],
    )
    product_price: Optional[float] = Field(
        None,
        ge=0.0,
        description="Optional product price",
        examples=[49.99],
    )
    reviewer_review_count: Optional[int] = Field(
        None,
        ge=0,
        description="Optional lifetime reviews written by this reviewer",
        examples=[12],
    )
    days_since_previous_review: Optional[float] = Field(
        None,
        ge=0.0,
        description="Optional days elapsed since reviewer's previous review",
        examples=[14.0],
    )
    model: Optional[str] = Field(
        None,
        description="Optional model identifier to use for classification (e.g. deberta_text, deberta_fusion, baseline, bilstm, svm, fakeguard)",
        examples=["deberta_text"],
    )
    threshold: Optional[float] = Field(
        None,
        ge=0.01,
        le=0.99,
        description="Optional decision threshold for classification",
        examples=[0.50],
    )

    @field_validator("review_text")
    @classmethod
    def validate_review_text(cls, v: str) -> str:
        clean = v.strip()
        if len(clean) < 3:
            raise ValueError("review_text must contain at least 3 non-whitespace characters")
        return clean


class DirectPredictionResponse(BaseModel):
    """Direct lightweight response for /predict inference."""
    prediction: str = Field(..., description="'genuine' or 'fake'")
    label: int = Field(..., description="0 for genuine, 1 for fake")
    fake_probability: float = Field(..., description="Estimated probability of review being fraudulent (0.0 to 1.0)")
    genuine_probability: float = Field(..., description="Estimated probability of review being genuine (0.0 to 1.0)")
    model: str = Field(default="deberta-v3-base", description="Model architecture identifier")
    attention_weights: Optional[List[List[float]]] = Field(None, description="Multi-view cross-attention interaction weights")


class PredictionResponse(BaseModel):
    """Result of a single review classification."""

    prediction_id: str = Field(..., description="Unique MongoDB identifier for this prediction")
    review_text: str = Field(..., description="The analyzed review text")
    predicted_label: str = Field(..., description="'fake' or 'genuine'")
    prediction: Optional[str] = Field(None, description="Direct classification: 'fake' or 'genuine'")
    label: Optional[int] = Field(None, description="Binary target label (0 = genuine, 1 = fake)")
    fake_probability: float = Field(..., description="Probability of review being fraudulent (0.0 to 1.0)")
    genuine_probability: Optional[float] = Field(None, description="Probability of review being genuine (0.0 to 1.0)")
    confidence: float = Field(..., description="Classification confidence score (0.5 to 1.0)")
    model_name: str = Field(..., description="Model identifier used for inference")
    model: Optional[str] = Field(None, description="Model identifier alias")
    timestamp: datetime = Field(..., description="UTC timestamp of the prediction")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional review metadata provided")
    attention_weights: Optional[List[List[float]]] = Field(None, description="Multi-view cross-attention interaction weights (Text, Behavior, Graph)")



class BatchPredictionItem(BaseModel):
    """Prediction result for a single row within a batch CSV upload."""

    row_index: int = Field(..., description="Zero-based row index in uploaded file")
    review_text: str = Field(..., description="The review text processed")
    predicted_label: Optional[str] = Field(None, description="'fake' or 'genuine' if successful")
    fake_probability: Optional[float] = Field(None, description="Fake probability if successful")
    confidence: Optional[float] = Field(None, description="Confidence score if successful")
    prediction_id: Optional[str] = Field(None, description="Stored MongoDB prediction ID")
    status: str = Field("success", description="'success' or 'error'")
    error: Optional[str] = Field(None, description="Error explanation if row failed")


class BatchPredictionResponse(BaseModel):
    """Aggregate response for batch CSV upload predictions."""

    total_submitted: int = Field(..., description="Total rows submitted in CSV")
    total_successful: int = Field(..., description="Count of successfully classified rows")
    total_failed: int = Field(..., description="Count of invalid/failed rows")
    model_name: str = Field(..., description="Model identifier used for inference")
    predictions: List[BatchPredictionItem] = Field(..., description="Row-by-row prediction results")


class HistoryItem(BaseModel):
    """Stored prediction document retrieved from history."""

    id: str = Field(..., description="MongoDB string ID")
    review_text: str = Field(..., description="Analyzed review text")
    predicted_label: str = Field(..., description="'fake' or 'genuine'")
    fake_probability: float = Field(..., description="Fake probability score")
    confidence: float = Field(..., description="Confidence score")
    model_name: str = Field(..., description="Model used")
    timestamp: datetime = Field(..., description="UTC timestamp")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional metadata provided")


class HistoryResponse(BaseModel):
    """Paginated prediction history response."""

    total: int = Field(..., description="Total records matching filters")
    page: int = Field(..., description="Current page number (1-indexed)")
    page_size: int = Field(..., description="Items per page")
    total_pages: int = Field(..., description="Total pages available")
    items: List[HistoryItem] = Field(..., description="List of historical predictions")


class AnalyticsSummaryResponse(BaseModel):
    """Aggregated prediction statistics for user submissions."""

    total_predictions: int = Field(..., description="Total reviews analyzed through the application")
    fake_count: int = Field(..., description="Count of reviews classified as fake")
    genuine_count: int = Field(..., description="Count of reviews classified as genuine")
    fake_percentage: float = Field(..., description="Percentage of submitted reviews classified as fake")
    model_breakdown: Dict[str, int] = Field(..., description="Prediction counts by model")
    recent_activity: List[Dict[str, Any]] = Field(..., description="Daily counts of recent analyses")
    disclaimer: str = Field(
        ...,
        description="Explicit note clarifying that these are application usage stats, not model benchmark metrics",
    )
