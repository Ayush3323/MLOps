"""RAG pipeline: sentiment-aware indexing + retrieval + answer generation."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any, Protocol
from uuid import uuid4

from app.core.config import get_settings
from src.inference.predict import SentimentPredictor
from src.rag.embeddings import RetrievedReview, ReviewEmbeddingStore


class PredictorLike(Protocol):
    def predict(self, text: str) -> Any: ...


@dataclass
class IndexedReview:
    review_id: str
    text: str
    sentiment: str
    confidence: float
    rating: float | None
    product_id: str | None


@dataclass
class QueryAnswer:
    answer: str
    mode: str  # extractive | openai | huggingface
    question: str
    sentiment_filter: str | None
    sources: list[dict[str, Any]]


class RagPipeline:
    def __init__(
        self,
        store: ReviewEmbeddingStore | None = None,
        predictor: PredictorLike | None = None,
    ):
        settings = get_settings()
        self.settings = settings
        self.store = store or ReviewEmbeddingStore(
            chroma_path=settings.chroma_path,
            collection_name=settings.rag_collection,
            embedding_model=settings.embedding_model,
        )
        self.predictor = predictor or SentimentPredictor(
            model_path=settings.resolved_model_path,
            device=settings.device,
        )

    def index_reviews(self, reviews: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Run sentiment inference, then embed + upsert into Chroma.

        Accepts items shaped like:
          {"text": "...", "review_id"?: "...", "rating"?: 4.0, "product_id"?: "..."}
        """
        if not reviews:
            raise ValueError("reviews must not be empty.")

        prepared: list[dict[str, Any]] = []
        indexed: list[IndexedReview] = []

        for item in reviews:
            text = str(item.get("text") or "").strip()
            if not text:
                raise ValueError("Each review requires non-empty text.")

            prediction = self.predictor.predict(text)
            review_id = str(item.get("review_id") or uuid4())
            rating = item.get("rating")
            product_id = item.get("product_id")

            prepared.append(
                {
                    "review_id": review_id,
                    "text": text,
                    "sentiment": prediction.label,
                    "rating": float(rating) if rating is not None else None,
                    "product_id": str(product_id) if product_id is not None else None,
                }
            )
            indexed.append(
                IndexedReview(
                    review_id=review_id,
                    text=text,
                    sentiment=prediction.label,
                    confidence=float(prediction.confidence),
                    rating=float(rating) if rating is not None else None,
                    product_id=str(product_id) if product_id is not None else None,
                )
            )

        count = self.store.index_reviews(prepared)
        return {
            "indexed_count": count,
            "collection_size": self.store.count,
            "reviews": [asdict(r) for r in indexed],
        }

    def answer_query(
        self,
        question: str,
        sentiment_filter: str | None = None,
        n_results: int | None = None,
    ) -> QueryAnswer:
        top_k = n_results or self.settings.rag_top_k
        sources = self.store.retrieve(
            question,
            n_results=top_k,
            sentiment_filter=sentiment_filter,
        )
        source_payload = [_source_dict(s) for s in sources]

        if not sources:
            return QueryAnswer(
                answer=(
                    "No indexed reviews matched this query. "
                    "Call POST /api/index with reviews first."
                ),
                mode="extractive",
                question=question,
                sentiment_filter=sentiment_filter,
                sources=[],
            )

        if self.settings.openai_api_key:
            answer = self._generate_openai(question, sources)
            mode = "openai"
        elif self.settings.hf_api_key:
            answer = self._generate_huggingface(question, sources)
            mode = "huggingface"
        else:
            answer = self._extractive_answer(question, sources)
            mode = "extractive"

        return QueryAnswer(
            answer=answer,
            mode=mode,
            question=question,
            sentiment_filter=sentiment_filter,
            sources=source_payload,
        )

    def _build_context(self, sources: list[RetrievedReview]) -> str:
        lines: list[str] = []
        for i, src in enumerate(sources, start=1):
            sentiment = src.sentiment or "unknown"
            lines.append(f"[{i}] (sentiment={sentiment}) {src.text}")
        return "\n\n".join(lines)

    def _extractive_answer(self, question: str, sources: list[RetrievedReview]) -> str:
        bullets = []
        for src in sources:
            snippet = src.text if len(src.text) <= 220 else src.text[:217] + "..."
            label = src.sentiment or "unknown"
            bullets.append(f"- ({label}) {snippet}")
        joined = "\n".join(bullets)
        return (
            f"Based on the closest indexed reviews for: {question}\n\n"
            f"{joined}\n\n"
            "Tip: set OPENAI_API_KEY or HF_API_KEY in .env for generated summaries."
        )

    def _generate_openai(self, question: str, sources: list[RetrievedReview]) -> str:
        context = self._build_context(sources)
        prompt = (
            "You are analyzing product reviews. Answer using only the reviews below.\n\n"
            f"Reviews:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer concisely based on the reviews provided."
        )
        payload = {
            "model": self.settings.openai_model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 300,
            "temperature": 0.2,
        }
        request = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.settings.openai_api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenAI API error: {exc.code} {detail}") from exc

        return body["choices"][0]["message"]["content"].strip()

    def _generate_huggingface(self, question: str, sources: list[RetrievedReview]) -> str:
        context = self._build_context(sources)
        prompt = (
            "<s>[INST] You are analyzing product reviews. "
            "Answer using only the reviews below.\n\n"
            f"Reviews:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer concisely. [/INST]"
        )
        model = self.settings.hf_llm_model
        url = f"https://api-inference.huggingface.co/models/{model}"
        payload = {
            "inputs": prompt,
            "parameters": {"max_new_tokens": 300, "temperature": 0.2, "return_full_text": False},
        }
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.settings.hf_api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HuggingFace Inference API error: {exc.code} {detail}") from exc

        if isinstance(body, list) and body:
            first = body[0]
            if isinstance(first, dict) and "generated_text" in first:
                return str(first["generated_text"]).strip()
        if isinstance(body, dict) and "generated_text" in body:
            return str(body["generated_text"]).strip()
        return str(body)


def _source_dict(src: RetrievedReview) -> dict[str, Any]:
    return {
        "review_id": src.review_id,
        "text": src.text,
        "sentiment": src.sentiment,
        "rating": src.rating,
        "product_id": src.product_id,
        "distance": src.distance,
    }
