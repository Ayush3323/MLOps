from app.core.config import Settings, get_settings
from src.inference.predict import SentimentPredictor
from src.rag.pipeline import RagPipeline


def get_app_settings() -> Settings:
    return get_settings()


def get_predictor() -> SentimentPredictor:
    settings = get_settings()
    if not hasattr(get_predictor, "_instance"):
        get_predictor._instance = SentimentPredictor(
            model_path=settings.resolved_model_path,
            device=settings.device,
        )
    return get_predictor._instance


def get_rag_pipeline() -> RagPipeline:
    if not hasattr(get_rag_pipeline, "_instance"):
        get_rag_pipeline._instance = RagPipeline(predictor=get_predictor())
    return get_rag_pipeline._instance
