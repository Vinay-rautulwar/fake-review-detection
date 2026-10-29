from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from bson import ObjectId
from bson.errors import InvalidId
from pymongo.database import Database

from backend.db.mongodb import DatabaseUnavailableException


class PredictionRepository:
    """Repository handling all MongoDB CRUD operations for the predictions collection."""

    def __init__(self, db: Optional[Database] = None):
        self._db = db

    def _get_collection(self):
        if self._db is None:
            raise DatabaseUnavailableException("Database handle is not initialized.")
        return self._db["predictions"]

    def set_db(self, db: Database):
        """Set or update the active database handle."""
        self._db = db

    def save_prediction(self, document: Dict[str, Any]) -> str:
        """Insert a single prediction document and return its string ObjectId."""
        collection = self._get_collection()
        # Ensure UTC timestamp
        if "timestamp" not in document or not document["timestamp"]:
            document["timestamp"] = datetime.now(timezone.utc)

        result = collection.insert_one(document)
        return str(result.inserted_id)

    def save_predictions_batch(self, documents: List[Dict[str, Any]]) -> List[str]:
        """Insert multiple prediction documents in a single bulk operation."""
        if not documents:
            return []
        collection = self._get_collection()
        now = datetime.now(timezone.utc)
        for doc in documents:
            if "timestamp" not in doc or not doc["timestamp"]:
                doc["timestamp"] = now

        result = collection.insert_many(documents)
        return [str(_id) for _id in result.inserted_ids]

    def get_prediction_by_id(self, prediction_id: str) -> Optional[Dict[str, Any]]:
        """Find a prediction document by its string ObjectId."""
        collection = self._get_collection()
        try:
            oid = ObjectId(prediction_id)
        except (InvalidId, TypeError):
            return None

        doc = collection.find_one({"_id": oid})
        if not doc:
            return None
        return self._format_doc(doc)

    def get_predictions(
        self,
        page: int = 1,
        page_size: int = 20,
        prediction_class: Optional[str] = None,
        model_name: Optional[str] = None,
        sort_order: str = "desc",
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Retrieve paginated and filtered historical predictions."""
        collection = self._get_collection()

        query: Dict[str, Any] = {}
        if prediction_class:
            clean_class = prediction_class.strip().lower()
            if clean_class in ["fake", "genuine"]:
                query["predicted_label"] = clean_class

        if model_name:
            query["model_name"] = model_name.strip()

        total_count = collection.count_documents(query)

        sort_dir = -1 if sort_order.lower() == "desc" else 1
        skip_count = max(0, (page - 1) * page_size)

        cursor = (
            collection.find(query)
            .sort("timestamp", sort_dir)
            .skip(skip_count)
            .limit(page_size)
        )

        items = [self._format_doc(doc) for doc in cursor]
        return items, total_count

    def get_analytics_summary(self) -> Dict[str, Any]:
        """Aggregate prediction counts, class balance, and usage distributions."""
        collection = self._get_collection()

        total = collection.count_documents({})
        fake_count = collection.count_documents({"predicted_label": "fake"})
        genuine_count = collection.count_documents({"predicted_label": "genuine"})
        fake_pct = round((fake_count / total) * 100, 2) if total > 0 else 0.0

        # Model usage breakdown
        model_pipeline = [
            {"$group": {"_id": "$model_name", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
        ]
        model_breakdown = {}
        try:
            for item in collection.aggregate(model_pipeline):
                name = item["_id"] or "unknown"
                model_breakdown[name] = item["count"]
        except Exception:
            # Fallback for mock environments lacking aggregate support
            pass

        # Recent activity timeline (past 7 days)
        activity_pipeline = [
            {
                "$project": {
                    "date": {
                        "$dateToString": {"format": "%Y-%m-%d", "date": "$timestamp"}
                    },
                    "predicted_label": 1,
                }
            },
            {
                "$group": {
                    "_id": "$date",
                    "total": {"$sum": 1},
                    "fake": {
                        "$sum": {
                            "$cond": [{"$eq": ["$predicted_label", "fake"]}, 1, 0]
                        }
                    },
                    "genuine": {
                        "$sum": {
                            "$cond": [{"$eq": ["$predicted_label", "genuine"]}, 1, 0]
                        }
                    },
                }
            },
            {"$sort": {"_id": -1}},
            {"$limit": 7},
        ]
        recent_activity = []
        try:
            for item in collection.aggregate(activity_pipeline):
                recent_activity.append(
                    {
                        "date": item["_id"],
                        "total": item["total"],
                        "fake": item["fake"],
                        "genuine": item["genuine"],
                    }
                )
        except Exception:
            pass

        return {
            "total_predictions": total,
            "fake_count": fake_count,
            "genuine_count": genuine_count,
            "fake_percentage": fake_pct,
            "model_breakdown": model_breakdown,
            "recent_activity": recent_activity,
            "disclaimer": (
                "These metrics reflect actual prediction requests recorded in MongoDB during application use. "
                "They are operational usage statistics and must NOT be confused with ground-truth model "
                "evaluation benchmarks (e.g. F1, ROC-AUC) on validated test splits."
            ),
        }

    @staticmethod
    def _format_doc(doc: Dict[str, Any]) -> Dict[str, Any]:
        """Convert MongoDB document to API schema format."""
        formatted = dict(doc)
        formatted["id"] = str(formatted.pop("_id"))
        return formatted
