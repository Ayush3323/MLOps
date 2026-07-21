from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api", tags=["rag"])


class IndexRequest(BaseModel):
    review_count: int = Field(..., ge=1, description="Number of reviews to index")


class IndexResponse(BaseModel):
    status: str
    message: str
    accepted_review_count: int


@router.post("/index", response_model=IndexResponse)
def index_reviews(payload: IndexRequest) -> IndexResponse:
    return IndexResponse(
        status="accepted",
        message="Index endpoint scaffold ready for Week 4 RAG ingestion.",
        accepted_review_count=payload.review_count,
    )
