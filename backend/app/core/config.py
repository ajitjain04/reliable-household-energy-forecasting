"""
Application configuration and path management.
Anchors all project directories to the absolute project root.
"""

import os
from pathlib import Path
from typing import List
from pydantic import Field
from pydantic_settings import BaseSettings


# Determine absolute project root (4 levels up from this file)
DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """Application settings with environment variable override support."""

    # Project Information
    APP_NAME: str = "Reliable Multimodal Household Energy Forecasting API"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: str = Field(default="development")
    DEBUG: bool = Field(default=True)
    API_PREFIX: str = "/api/v1"

    # Server Bindings
    HOST: str = Field(default="127.0.0.1")
    PORT: int = Field(default=8000)

    # Security & Authentication Configuration
    JWT_SECRET_KEY: str = Field(default="reliable_energy_forecasting_secret_key_change_in_prod")
    JWT_ALGORITHM: str = Field(default="HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=60 * 24)  # 24 hours
    INITIAL_DEV_USERNAME: str = Field(default="")
    INITIAL_DEV_PASSWORD: str = Field(default="")

    # CORS Allowed Origins
    CORS_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
    ]

    # Absolute Directories Anchored to Project Root
    PROJECT_ROOT: Path = Path(os.getenv("PROJECT_ROOT", str(DEFAULT_PROJECT_ROOT)))

    @property
    def OUTPUTS_DIR(self) -> Path:
        return self.PROJECT_ROOT / "outputs"

    @property
    def TABLES_DIR(self) -> Path:
        return self.OUTPUTS_DIR / "tables"

    @property
    def PREDICTIONS_DIR(self) -> Path:
        return self.OUTPUTS_DIR / "predictions"

    @property
    def MODELS_DIR(self) -> Path:
        return self.OUTPUTS_DIR / "models"

    @property
    def SCALERS_DIR(self) -> Path:
        return self.MODELS_DIR / "scalers"

    @property
    def UNCERTAINTY_DIR(self) -> Path:
        return self.OUTPUTS_DIR / "uncertainty"

    @property
    def UNCERTAINTY_ARTIFACTS_DIR(self) -> Path:
        return self.UNCERTAINTY_DIR / "artifacts"

    @property
    def XAI_DIR(self) -> Path:
        return self.OUTPUTS_DIR / "xai"

    @property
    def XAI_TABLES_DIR(self) -> Path:
        return self.XAI_DIR / "tables"

    @property
    def REPORTS_DIR(self) -> Path:
        return self.OUTPUTS_DIR / "reports"

    @property
    def DATA_DIR(self) -> Path:
        return self.PROJECT_ROOT / "data"

    @property
    def PROCESSED_DATA_DIR(self) -> Path:
        return self.DATA_DIR / "processed"

    @property
    def DATABASE_PATH(self) -> Path:
        return Path(os.getenv("DATABASE_PATH", str(self.DATA_DIR / "forecast_records.db")))

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


settings = Settings()
