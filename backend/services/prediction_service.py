import csv
import io
import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from bson import ObjectId
from starlette.concurrency import run_in_threadpool

from backend.core.config import settings
from backend.db.mongodb import DatabaseUnavailableException
from backend.repositories.prediction_repository import PredictionRepository
from backend.schemas.prediction import (
    AnalyticsSummaryResponse,
    BatchPredictionItem,
    BatchPredictionResponse,
    HistoryItem,
    HistoryResponse,
    PredictionRequest,
    PredictionResponse,
)
from backend.services.model_service import ModelService, ModelUnavailableException, PredictionResult


class PredictionService:
    """Orchestrates prediction requests, CSV batch parsing, and database history tracking."""

    def __init__(self, model_service: ModelService, repository: PredictionRepository):
        self.model_service = model_service
        self.repository = repository

    async def predict_single(self, request: PredictionRequest) -> PredictionResponse:
        """Run single prediction and persist to MongoDB."""
        # 1. Run inference (raises ModelUnavailableException if model not ready)
        metadata = {
            k: v
            for k, v in request.model_dump().items()
            if k != "review_text" and v is not None
        }
        result: PredictionResult = self.model_service.predict(request.review_text, metadata)

        # 2. Prepare document
        now = datetime.now(timezone.utc)
        doc = {
            "review_text": request.review_text,
            "predicted_label": result.predicted_label,
            "fake_probability": result.fake_probability,
            "confidence": result.confidence,
            "model_name": result.model_name,
            "timestamp": now,
            "metadata": metadata if metadata else None,
        }

        # 3. Persist document to database without blocking event loop
        prediction_id = await run_in_threadpool(self.repository.save_prediction, doc)

        return PredictionResponse(
            prediction_id=prediction_id,
            review_text=request.review_text,
            predicted_label=result.predicted_label,
            prediction=result.predicted_label,
            label=1 if result.predicted_label == "fake" else 0,
            fake_probability=result.fake_probability,
            genuine_probability=round(1.0 - result.fake_probability, 4),
            confidence=result.confidence,
            model_name=result.model_name,
            model=result.model_name,
            timestamp=now,
            metadata=metadata if metadata else None,
        )


    async def predict_batch(self, file_content: bytes, filename: str) -> BatchPredictionResponse:
        """Process CSV batch upload, run batch inference, and persist valid predictions."""
        # 1. Validate file extension
        if not filename.lower().endswith(".csv"):
            raise ValueError("Invalid file format. Uploaded file must have a .csv extension.")

        # 2. Validate file size
        if len(file_content) > settings.MAX_FILE_SIZE_BYTES:
            max_mb = settings.MAX_FILE_SIZE_BYTES / (1024 * 1024)
            raise ValueError(f"File size exceeds maximum limit of {max_mb:.1f} MB.")

        # 3. Parse CSV rows
        try:
            content_str = file_content.decode("utf-8-sig")
        except UnicodeDecodeError:
            try:
                content_str = file_content.decode("latin-1")
            except Exception:
                raise ValueError("Could not decode CSV file. Please use UTF-8 encoding.")

        reader = csv.DictReader(io.StringIO(content_str))
        if not reader.fieldnames:
            raise ValueError("Uploaded CSV file is empty or missing headers.")

        # Check required review_text column (case-insensitive check)
        header_map = {col.strip().lower(): col for col in reader.fieldnames if col}
        if "review_text" not in header_map:
            raise ValueError("CSV missing required column: 'review_text'.")

        text_col = header_map["review_text"]
        rows = list(reader)
        total_rows = len(rows)

        if total_rows == 0:
            raise ValueError("CSV file contains headers but no review rows.")

        if total_rows > settings.MAX_BATCH_ROWS:
            raise ValueError(
                f"Batch exceeds maximum limit of {settings.MAX_BATCH_ROWS} rows. "
                f"File contains {total_rows} rows."
            )

        # 4. Check model availability before doing partial inference
        if not self.model_service.is_available:
            raise ModelUnavailableException(
                f"Batch inference unavailable: Model '{self.model_service.configured_model}' "
                f"is not loaded: {self.model_service.status_message}"
            )

        # 5. Extract rows and run batch inference
        items: List[BatchPredictionItem] = []
        valid_indices: List[int] = []
        valid_texts: List[str] = []

        for idx, row in enumerate(rows):
            raw_text = (row.get(text_col) or "").strip()
            if not raw_text or len(raw_text) < 3:
                items.append(
                    BatchPredictionItem(
                        row_index=idx,
                        review_text=raw_text,
                        status="error",
                        error="review_text is empty or shorter than 3 characters.",
                    )
                )
            else:
                valid_indices.append(idx)
                valid_texts.append(raw_text)
                # Placeholder to be filled after batch predict
                items.append(
                    BatchPredictionItem(
                        row_index=idx,
                        review_text=raw_text,
                    )
                )

        now = datetime.now(timezone.utc)
        docs_to_insert = []

        if valid_texts:
            predictions = self.model_service.predict_batch(valid_texts)
            for i, pred in enumerate(predictions):
                original_idx = valid_indices[i]
                row_data = rows[original_idx]

                # Extract optional metadata
                metadata = {}
                for k, v in row_data.items():
                    if k and k != text_col and v is not None:
                        if isinstance(v, str):
                            v_clean = v.strip()
                            if v_clean:
                                metadata[k] = v_clean
                        elif isinstance(v, (int, float, bool)):
                            metadata[k] = v

                doc = {
                    "review_text": valid_texts[i],
                    "predicted_label": pred.predicted_label,
                    "fake_probability": pred.fake_probability,
                    "confidence": pred.confidence,
                    "model_name": pred.model_name,
                    "timestamp": now,
                    "metadata": metadata if metadata else None,
                }
                docs_to_insert.append((original_idx, doc, pred))

        # 6. Bulk save to MongoDB
        if docs_to_insert:
            inserted_ids = await run_in_threadpool(
                self.repository.save_predictions_batch,
                [d[1] for d in docs_to_insert],
            )
            for (orig_idx, _, pred), ins_id in zip(docs_to_insert, inserted_ids):
                items[orig_idx].predicted_label = pred.predicted_label
                items[orig_idx].fake_probability = pred.fake_probability
                items[orig_idx].confidence = pred.confidence
                items[orig_idx].prediction_id = ins_id
                items[orig_idx].status = "success"

        successful_count = len(docs_to_insert)
        failed_count = total_rows - successful_count

        return BatchPredictionResponse(
            total_submitted=total_rows,
            total_successful=successful_count,
            total_failed=failed_count,
            model_name=self.model_service.configured_model,
            predictions=items,
        )

    async def get_history(
        self,
        page: int = 1,
        page_size: int = 20,
        prediction_class: Optional[str] = None,
        model_name: Optional[str] = None,
        sort_order: str = "desc",
    ) -> HistoryResponse:
        """Fetch paginated historical predictions."""
        items_raw, total = await run_in_threadpool(
            self.repository.get_predictions,
            page=page,
            page_size=page_size,
            prediction_class=prediction_class,
            model_name=model_name,
            sort_order=sort_order,
        )

        total_pages = math.ceil(total / page_size) if total > 0 else 1
        items = [HistoryItem(**doc) for doc in items_raw]

        return HistoryResponse(
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
            items=items,
        )

    async def get_prediction_by_id(self, prediction_id: str) -> HistoryItem:
        """Retrieve single prediction record by ID. Validate ObjectId format."""
        if not ObjectId.is_valid(prediction_id):
            raise ValueError(f"Invalid prediction ID format: '{prediction_id}'. Must be a 24-character hex string.")

        doc = await run_in_threadpool(self.repository.get_prediction_by_id, prediction_id)
        if not doc:
            raise KeyError(f"Prediction with ID '{prediction_id}' was not found.")

        return HistoryItem(**doc)

    async def get_analytics_summary(self) -> AnalyticsSummaryResponse:
        """Retrieve aggregated prediction statistics."""
        data = await run_in_threadpool(self.repository.get_analytics_summary)
        return AnalyticsSummaryResponse(**data)
