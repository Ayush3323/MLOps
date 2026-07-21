from app.core.config import Settings, get_settings
from src.inference.predict import SentimentPredictor


def get_app_settings() -> Settings:
    return get_settings()


def get_predictor() -> SentimentPredictor:
    settings = get_settings()
    if not hasattr(get_predictor, "_instance"):
        get_predictor._instance = SentimentPredictor(model_path=settings.model_path, device=settings.device)
    return get_predictor._instance
