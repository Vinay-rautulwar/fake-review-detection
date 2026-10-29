import logging
from typing import Optional
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.errors import ConnectionFailure, PyMongoError, ServerSelectionTimeoutError

from backend.core.config import settings

logger = logging.getLogger(__name__)


class DatabaseUnavailableException(Exception):
    """Raised when an operation is attempted but MongoDB is disconnected or unavailable."""
    pass


class MongoDBManager:
    """Manages the MongoDB client connection, database handle, and index creation."""

    def __init__(self):
        self.client: Optional[MongoClient] = None
        self.db: Optional[Database] = None
        self._is_connected: bool = False
        self._last_error: Optional[str] = None

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

    def connect(self) -> bool:
        """Establish connection to MongoDB and create collections/indexes."""
        try:
            logger.info(f"Connecting to MongoDB at {settings.MONGODB_URI} (Database: {settings.MONGODB_DATABASE})...")
            self.client = MongoClient(
                settings.MONGODB_URI,
                serverSelectionTimeoutMS=settings.MONGODB_TIMEOUT_MS,
                connectTimeoutMS=settings.MONGODB_TIMEOUT_MS,
            )
            # Ping database to verify active connection
            self.client.admin.command("ping")
            self.db = self.client[settings.MONGODB_DATABASE]
            self._is_connected = True
            self._last_error = None
            logger.info("Successfully connected to MongoDB server.")

            # Create required indexes
            self.init_indexes()
            return True

        except (ServerSelectionTimeoutError, ConnectionFailure) as e:
            self._is_connected = False
            self.db = None
            self._last_error = "Could not connect to MongoDB server. Ensure MongoDB service (mongod) is running."
            logger.warning(f"MongoDB connection failed: {self._last_error}")
            return False
        except Exception as e:
            self._is_connected = False
            self.db = None
            self._last_error = f"Database initialization error: {type(e).__name__}"
            logger.warning(f"MongoDB unexpected error: {self._last_error}")
            return False

    def close(self):
        """Close MongoDB connection gracefully."""
        if self.client:
            logger.info("Closing MongoDB connection...")
            self.client.close()
            self.client = None
            self.db = None
            self._is_connected = False
            self._last_error = None

    def init_indexes(self):
        """Create indexes on collections for fast querying and sorting."""
        if self.db is None:
            return

        try:
            predictions_col = self.db["predictions"]
            # Fast timestamp sorting for history and analytics
            predictions_col.create_index([("timestamp", -1)], background=True)
            # Filtering by predicted class (fake/genuine)
            predictions_col.create_index([("predicted_label", 1)], background=True)
            # Filtering by model name
            predictions_col.create_index([("model_name", 1)], background=True)
            # Compound index for filtered timeline queries
            predictions_col.create_index(
                [("model_name", 1), ("predicted_label", 1), ("timestamp", -1)],
                background=True,
            )

            # Metadata runs collection index
            model_runs_col = self.db["model_runs"]
            model_runs_col.create_index([("created_at", -1)], background=True)

            logger.info("MongoDB collections and indexes initialized successfully.")
        except PyMongoError as e:
            logger.error(f"Failed to create MongoDB indexes: {e}")

    def ping(self) -> bool:
        """Check live connection status."""
        if not self.client:
            return False
        try:
            self.client.admin.command("ping")
            self._is_connected = True
            self._last_error = None
            return True
        except Exception as e:
            self._is_connected = False
            self._last_error = "Database ping failed. MongoDB server unreachable."
            return False

    def get_database(self) -> Database:
        """Retrieve the database handle or raise DatabaseUnavailableException."""
        if not self._is_connected or self.db is None:
            if not self.ping():
                raise DatabaseUnavailableException(
                    self._last_error or "Database connection is not available. Please ensure MongoDB is running."
                )
        return self.db


# Global database manager instance
db_manager = MongoDBManager()


def get_db() -> Database:
    """FastAPI dependency to retrieve the active MongoDB database."""
    return db_manager.get_database()
