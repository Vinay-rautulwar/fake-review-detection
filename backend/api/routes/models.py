from typing import Optional
from fastapi import APIRouter, Depends

from backend.schemas.common import ModelInfo, ModelsListResponse, SystemMetricsResponse
from backend.services.model_service import ModelService, get_model_service

router = APIRouter(tags=["Models"])


@router.get(
    "/models",
    response_model=ModelsListResponse,
    summary="List Supported Model Architectures & Active Status",
    description="Returns metadata, architecture scopes, and availability statuses for all supported models.",
)
async def list_models(
    model_svc: ModelService = Depends(get_model_service),
) -> ModelsListResponse:
    models_raw = model_svc.list_models()
    model_infos = [ModelInfo(**m) for m in models_raw]

    return ModelsListResponse(
        active_model=model_svc.configured_model,
        total_models=len(model_infos),
        models=model_infos,
    )


@router.get(
    "/models/benchmark",
    summary="Get Unified Model Benchmark Results",
    description="Returns cross-model evaluation metrics (Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC) across all tested models.",
)
async def get_model_benchmark(
    model_svc: ModelService = Depends(get_model_service),
):
    return model_svc.get_benchmark_data()


@router.get(
    "/models/metrics",
    response_model=SystemMetricsResponse,
    summary="Get Model Validation Benchmarks & Training Evaluation Metrics",
    description="Returns offline validation metrics (ROC-AUC, PR-AUC, confusion matrix, threshold comparisons) for models.",
)
async def get_all_model_metrics(
    model_svc: ModelService = Depends(get_model_service),
) -> SystemMetricsResponse:
    metrics = model_svc.get_model_metrics()
    return SystemMetricsResponse(**metrics)


@router.get(
    "/models/{model_name}/metrics",
    response_model=SystemMetricsResponse,
    summary="Get Validation Metrics For A Specific Model",
    description="Returns validation metrics with the specified model selected.",
)
async def get_single_model_metrics(
    model_name: str,
    model_svc: ModelService = Depends(get_model_service),
) -> SystemMetricsResponse:
    metrics = model_svc.get_model_metrics(model_name=model_name)
    return SystemMetricsResponse(**metrics)

