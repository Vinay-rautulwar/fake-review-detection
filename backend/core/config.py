import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    # Project root directory
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent

    # Project Information
    PROJECT_NAME: str = "Fake Review Detection API"
    PROJECT_VERSION: str = "2.0.0"
    API_V1_STR: str = "/api/v1"

    # MongoDB Configuration
    MONGODB_URI: str = "mongodb://localhost:27017"
    MONGODB_DATABASE: str = "fake_review_detection"
    MONGODB_TIMEOUT_MS: int = 3000

    # CORS Configuration
    FLASK_ORIGIN: str = "http://127.0.0.1:5000"

    # Model Configuration
    # Options: "deberta_text", "deberta_fusion", "baseline" / "tfidf_lightgbm"
    MODEL_TYPE: str = "deberta_text"
    MODEL_DIR: str = "backend/models/deberta_fp32_final"
    MODEL_ARTIFACTS_PATH: str = ""

    # ML Inference Service Gateway Settings
    ML_SERVICE_URL: str = "http://127.0.0.1:8000"
    ML_SERVICE_HOST: str = "127.0.0.1"
    ML_SERVICE_PORT: int = 8000

    # Operational Constraints
    MAX_BATCH_ROWS: int = 500
    MAX_FILE_SIZE_BYTES: int = 5 * 1024 * 1024  # 5 MB

    model_config = SettingsConfigDict(
        env_file=(".env", "backend/.env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def get_artifacts_dir(self, model_type: str = None) -> Path:
        """Resolve the directory containing model artifacts using relative paths."""
        m_type = model_type or self.MODEL_TYPE

        # If baseline requested, return baseline artifacts directory
        if m_type in ["baseline", "tfidf_lightgbm"]:
            return self.BASE_DIR / "training" / "artifacts" / "baseline"

        # If DeBERTa or active model, check MODEL_DIR first
        if self.MODEL_DIR and (model_type is None or m_type in ["deberta_text", "deberta", "deberta-v3-base", "deberta_v3", "deberta_fp32_final"]):
            custom_path = Path(self.MODEL_DIR)
            resolved = custom_path if custom_path.is_absolute() else self.BASE_DIR / custom_path
            if resolved.exists():
                return resolved

        if self.MODEL_ARTIFACTS_PATH:
            custom_path = Path(self.MODEL_ARTIFACTS_PATH)
            resolved = custom_path if custom_path.is_absolute() else self.BASE_DIR / custom_path
            if resolved.exists():
                return resolved

        # Standard model locations
        if m_type in ["baseline", "tfidf_lightgbm"]:
            return self.BASE_DIR / "training" / "artifacts" / "baseline"
        elif m_type in ["deberta_text", "deberta", "deberta-v3-base", "deberta_v3", "deberta_fp32_final"]:
            # Prefer extracted local model directory
            local_extracted = self.BASE_DIR / "backend" / "models" / "deberta_fp32_final"
            if (local_extracted / "config.json").exists():
                return local_extracted
            return self.BASE_DIR / "training" / "artifacts" / "deberta_text"
        elif m_type == "deberta_fusion":
            return self.BASE_DIR / "training" / "artifacts" / "deberta_fusion"
        else:
            return self.BASE_DIR / "training" / "artifacts" / m_type



settings = Settings()
