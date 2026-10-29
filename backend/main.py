import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.api.router import api_router
from backend.core.config import settings
from backend.db.mongodb import db_manager
from backend.services.model_service import model_service

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("backend")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan event handler managing application startup and graceful shutdown."""
    logger.info("=" * 65)
    logger.info(f"Starting {settings.PROJECT_NAME} v{settings.PROJECT_VERSION}")
    logger.info("=" * 65)

    # 1. Connect to MongoDB
    db_connected = db_manager.connect()
    if not db_connected:
        logger.warning("Application started in degraded state: MongoDB is unreachable.")

    # 2. Check and load configured model artifacts
    model_service.load_configured_model()

    logger.info(f"API endpoints ready under prefix: {settings.API_V1_STR}")
    logger.info("=" * 65)

    yield

    # Graceful shutdown
    logger.info("Shutting down backend service...")
    db_manager.close()
    logger.info("Cleanup complete. Service stopped.")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.PROJECT_VERSION,
    description=(
        "REST API service for the Fake Review Detection ML system. "
        "Provides real-time text and metadata classification, batch processing, "
        "and MongoDB historical persistence."
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# CORS Configuration for Frontend Origins
origins = [
    "http://127.0.0.1:5000",
    "http://localhost:5000",
    "http://127.0.0.1:8000",
    "http://localhost:8000",
    "http://127.0.0.1:3000",
    "http://localhost:3000",
]
if settings.FLASK_ORIGIN and settings.FLASK_ORIGIN.rstrip("/") not in origins:
    origins.append(settings.FLASK_ORIGIN.rstrip("/"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Include API v1 router
app.include_router(api_router, prefix=settings.API_V1_STR)

# Mount frontend static directory under /app
from fastapi.staticfiles import StaticFiles
frontend_dir = settings.BASE_DIR / "frontend"
if frontend_dir.exists():
    app.mount("/app", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")


@app.get("/", tags=["Root"])
async def root():
    """Root entrypoint providing basic service metadata and documentation links."""
    return {
        "service": settings.PROJECT_NAME,
        "version": settings.PROJECT_VERSION,
        "status": "online",
        "documentation": "/docs",
        "health": "/health",
        "predict": "/predict",
        "health_check": f"{settings.API_V1_STR}/health",
        "models_status": f"{settings.API_V1_STR}/models",
        "models_metrics": f"{settings.API_V1_STR}/models/metrics",
        "frontend_dashboard": "/app",
    }


@app.get("/health", tags=["Health"], summary="Model and System Health Status")
async def health_check():
    """Reports model loading status, runtime device, and precision without exposing paths or secrets."""
    is_loaded = model_service.is_loaded
    return {
        "status": "healthy" if is_loaded else "degraded",
        "model_loaded": is_loaded,
        "model": "deberta-v3-base" if "deberta" in model_service.configured_model else model_service.configured_model,
        "device": model_service.device,
        "precision": "fp32",
        "modality": "text-only",
    }


from backend.schemas.prediction import DirectPredictionResponse, PredictionRequest
from backend.services.model_service import ModelUnavailableException


@app.post("/predict", response_model=DirectPredictionResponse, tags=["Inference"], summary="DeBERTa Text Classification Inference")
async def direct_predict(request: PredictionRequest):
    """Clean inference API returning prediction label, probabilities, and model name."""
    if not model_service.is_available:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model unavailable: {model_service.status_message}",
        )
    try:
        res = model_service.predict(request.review_text)
        return DirectPredictionResponse(
            prediction=res.predicted_label,
            label=res.label,
            fake_probability=res.fake_probability,
            genuine_probability=res.genuine_probability,
            model=res.model_name,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.exception("Inference execution failed")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Inference processing failed.")




# Standardized Exception Handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle schema validation errors cleanly."""
    errors = []
    for err in exc.errors():
        loc = " -> ".join(str(l) for l in err.get("loc", []))
        msg = err.get("msg", "Invalid value")
        errors.append(f"{loc}: {msg}")

    status_422 = getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT", status.HTTP_422_UNPROCESSABLE_ENTITY)
    return JSONResponse(
        status_code=status_422,
        content={
            "detail": "Request validation failed.",
            "errors": errors,
            "status_code": 422,
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Ensure consistent JSON structure for HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.detail,
            "status_code": exc.status_code,
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Catch-all error handler preventing sensitive credentials or tracebacks from leaking."""
    logger.exception(f"Unhandled error on {request.method} {request.url.path}: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "An internal server error occurred. Please contact system administrator.",
            "status_code": status.HTTP_500_INTERNAL_SERVER_ERROR,
        },
    )
