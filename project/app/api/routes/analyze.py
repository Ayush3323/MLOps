from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.api.deps import get_predictor
from src.inference.predict import SentimentPredictor

router = APIRouter(prefix="/api", tags=["inference"])


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Input review text")


class AnalyzeResponse(BaseModel):
    label_id: int
    label: str
    confidence: float
    scores: dict[str, float]


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze(payload: AnalyzeRequest, predictor: SentimentPredictor = Depends(get_predictor)) -> AnalyzeResponse:
    try:
        result = predictor.predict(payload.text)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return AnalyzeResponse(
        label_id=result.label_id,
        label=result.label,
        confidence=result.confidence,
        scores=result.scores,
    )
