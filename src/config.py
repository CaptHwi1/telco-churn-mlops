"""
Project configuration using Pydantic Settings.
Loads configuration from environment variables or .env file.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
from typing import Optional

class Settings(BaseSettings):
    # Project settings
    project_name: str = "telco_churn"
    environment: str = "dev"
    debug: bool = True
    
    # Data paths (relative to project root)
    data_raw_path: str = "data/raw"
    data_processed_path: str = "data/processed"
    models_path: str = "models"
    artifacts_path: str = "artifacts"
    
    # MLflow settings
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "telco_churn_baseline"
    model_registry_name: str = "telco_churn_model"
    
    # Model settings
    random_state: int = 42
    test_size: float = 0.2
    cv_folds: int = 5
    
    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 1
    
    # Huawei Cloud settings (optional)
    huawei_region: Optional[str] = None
    huawei_project_id: Optional[str] = None
    swr_registry: Optional[str] = None
    
    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = None
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

# Global settings instance
settings = Settings()