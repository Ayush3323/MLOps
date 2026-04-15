from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Review Intelligence API"
    app_env: str = "dev"
    app_host: str = "0.0.0.0"
    app_port: int = 8000

    model_path: str = "./checkpoints/best-model"
    chroma_path: str = "./chroma_db"
    mlflow_tracking_uri: str = "./experiments/mlruns"
    processed_data_dir: str = "./data/processed"

    device: str = "cpu"
    max_samples: int = 1000
    batch_size: int = 8
    max_length: int = 256

    openai_api_key: str = ""
    hf_api_key: str = ""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
