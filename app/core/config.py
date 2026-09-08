from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Candidate checkpoint dirs relative to the project root (cwd when running uvicorn).
_DEFAULT_MODEL_CANDIDATES = (
    "./checkpoints/best-model",
    "./checkpoints/week3-distilbert/best-model",
    "./checkpoints/week2-distilbert/best-model",
)


def resolve_model_path(preferred: str | None = None) -> str:
    """Return the first existing checkpoint path, or the preferred default."""
    candidates: list[str] = []
    if preferred:
        candidates.append(preferred)
    for path in _DEFAULT_MODEL_CANDIDATES:
        if path not in candidates:
            candidates.append(path)
    for path in candidates:
        if Path(path).exists():
            return path
    return preferred or _DEFAULT_MODEL_CANDIDATES[0]


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
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    rag_collection: str = "product_reviews"
    rag_top_k: int = 5

    openai_api_key: str = ""
    openai_model: str = "gpt-3.5-turbo"
    hf_api_key: str = ""
    hf_llm_model: str = "mistralai/Mistral-7B-Instruct-v0.2"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @property
    def resolved_model_path(self) -> str:
        return resolve_model_path(self.model_path)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
