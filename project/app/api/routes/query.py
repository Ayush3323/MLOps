from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api", tags=["rag"])


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3)
    sentiment_filter: str | None = Field(default=None, pattern="^(negative|neutral|positive)$")


class QueryResponse(BaseModel):
    status: str
    message: str
    question: str
    sentiment_filter: str | None


@router.post("/query", response_model=QueryResponse)
def query_reviews(payload: QueryRequest) -> QueryResponse:
    return QueryResponse(
        status="accepted",
        message="Query endpoint scaffold ready for Week 4 retrieval + generation.",
        question=payload.question,
        sentiment_filter=payload.sentiment_filter,
    )
