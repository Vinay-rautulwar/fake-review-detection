from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class DatabaseStatus(BaseModel):
    """Database connectivity and health status."""
    status: str = Field(..., description="Connection status: 'connected' or 'disconnected'")
    database_name: str = Field(..., description="Configured database name")
    details: Optional[str] = Field(None, description="Additional status details or error message")


class ModelHealthStatus(BaseModel):
    """Model loaded and availability status."""
    status: str = Field(..., description="Model availability: 'available' or 'unavailable'")
    configured_model: str = Field(..., description="Currently configured model identifier")
    loaded: bool = Field(..., description="Whether model weights/artifacts are loaded in memory")
    details: Optional[str] = Field(None, description="Reason if model is unavailable")


class HealthStatus(BaseModel):
    """Overall system health status."""
    status: str = Field(..., description="'healthy', 'degraded', or 'unhealthy'")
    api: str = Field("available", description="API service status")
    database: DatabaseStatus = Field(..., description="Database status")
    model: ModelHealthStatus = Field(..., description="Model status")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ModelInfo(BaseModel):
    """Metadata describing a supported model architecture."""
    name: str = Field(..., description="Model identifier used in API / configs")
    display_name: str = Field(..., description="Human-readable title")
    architecture: str = Field(..., description="Model architecture summary")
    input_type: str = Field(..., description="Input scope: 'text-only' or 'multimodal'")
    artifact_path: str = Field(..., description="Relative or absolute path to model artifacts")
    is_loaded: bool = Field(..., description="Whether model is currently loaded in memory")
    is_available: bool = Field(..., description="Whether model can serve predictions immediately")
    status_message: str = Field(..., description="Status summary or missing artifact message")


class ModelsListResponse(BaseModel):
    """List of all configured and planned models."""
    active_model: str = Field(..., description="Currently active model ID from environment")
    total_models: int = Field(..., description="Total supported architectures")
    models: List[ModelInfo] = Field(..., description="Detailed model descriptors")


class ErrorResponse(BaseModel):
    """Standardized error response payload."""
    detail: str = Field(..., description="Readable error description")
    status_code: int = Field(..., description="HTTP status code")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ConfusionMatrix(BaseModel):
    """2x2 confusion matrix values."""
    tn: int = Field(..., description="True Negatives (Genuine as Genuine)")
    fp: int = Field(..., description="False Positives (Genuine as Fake)")
    fn: int = Field(..., description="False Negatives (Fake as Genuine)")
    tp: int = Field(..., description="True Positives (Fake as Fake)")


class ThresholdMetrics(BaseModel):
    """Metrics calculated at a specific classification threshold."""
    threshold: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: Optional[ConfusionMatrix] = None
    classification_report: Optional[Dict[str, Any]] = None


class ModelMetricsDetail(BaseModel):
    """Detailed validation metrics from offline evaluation."""
    model_type: Optional[str] = None
    feature_scope: Optional[str] = None
    evaluation_split: Optional[str] = None
    roc_auc: Optional[float] = None
    pr_auc: Optional[float] = None
    threshold_0_5: Optional[ThresholdMetrics] = None
    optimal_threshold: Optional[ThresholdMetrics] = None


class ModelEvaluationSummary(BaseModel):
    """Summary of model validation status and metrics."""
    name: str
    display_name: str
    status: str
    is_evaluated: bool
    metrics: Optional[Dict[str, Any]] = None
    training_environment: Optional[str] = None
    status_message: Optional[str] = None


class SystemMetricsResponse(BaseModel):
    """Comparison and validation metrics across all models."""
    active_model: str
    selected_model: str
    baseline: ModelEvaluationSummary
    deberta_text: ModelEvaluationSummary
    deberta_fusion: ModelEvaluationSummary

