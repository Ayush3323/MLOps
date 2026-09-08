from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.api.deps import get_rag_pipeline
from src.rag.pipeline import RagPipeline

router = APIRouter(prefix="/api", tags=["rag"])


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3)
    sentiment_filter: str | None = Field(default=None, pattern="^(negative|neutral|positive)$")
    n_results: int | None = Field(default=None, ge=1, le=20)


class QueryResponse(BaseModel):
    status: str
    answer: str
    mode: str
    question: str
    sentiment_filter: str | None
    sources: list[dict]


@router.post("/query", response_model=QueryResponse)
def query_reviews(
    payload: QueryRequest,
    pipeline: RagPipeline = Depends(get_rag_pipeline),
) -> QueryResponse:
    try:
        result = pipeline.answer_query(
            question=payload.question,
            sentiment_filter=payload.sentiment_filter,
            n_results=payload.n_results,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}") from exc

    return QueryResponse(
        status="ok",
        answer=result.answer,
        mode=result.mode,
        question=result.question,
        sentiment_filter=result.sentiment_filter,
        sources=result.sources,
    )
