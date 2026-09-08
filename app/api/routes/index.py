from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.api.deps import get_rag_pipeline
from src.rag.pipeline import RagPipeline

router = APIRouter(prefix="/api", tags=["rag"])


class ReviewIn(BaseModel):
    text: str = Field(..., min_length=1)
    review_id: str | None = None
    rating: float | None = Field(default=None, ge=1, le=5)
    product_id: str | None = None


class IndexRequest(BaseModel):
    reviews: list[ReviewIn] = Field(..., min_length=1, max_length=500)


class IndexResponse(BaseModel):
    status: str
    indexed_count: int
    collection_size: int
    reviews: list[dict]


@router.post("/index", response_model=IndexResponse)
def index_reviews(
    payload: IndexRequest,
    pipeline: RagPipeline = Depends(get_rag_pipeline),
) -> IndexResponse:
    try:
        result = pipeline.index_reviews([r.model_dump() for r in payload.reviews])
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - surface embedding failures cleanly
        raise HTTPException(status_code=500, detail=f"Indexing failed: {exc}") from exc

    return IndexResponse(
        status="ok",
        indexed_count=result["indexed_count"],
        collection_size=result["collection_size"],
        reviews=result["reviews"],
    )
