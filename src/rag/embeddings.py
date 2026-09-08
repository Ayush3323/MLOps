"""ChromaDB + sentence-transformers embedding store for product reviews."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer

from app.core.config import get_settings


@dataclass
class RetrievedReview:
    text: str
    sentiment: str | None
    rating: float | None
    product_id: str | None
    review_id: str
    distance: float | None


class ReviewEmbeddingStore:
    """Persist and query review embeddings in a local Chroma collection."""

    def __init__(
        self,
        chroma_path: str | None = None,
        collection_name: str | None = None,
        embedding_model: str | None = None,
    ):
        settings = get_settings()
        self.chroma_path = chroma_path or settings.chroma_path
        self.collection_name = collection_name or settings.rag_collection
        self.embedding_model_name = embedding_model or settings.embedding_model
        self._embedder: SentenceTransformer | None = None
        self._client = chromadb.PersistentClient(
            path=self.chroma_path,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def _ensure_embedder(self) -> SentenceTransformer:
        if self._embedder is None:
            self._embedder = SentenceTransformer(self.embedding_model_name)
        return self._embedder

    @property
    def count(self) -> int:
        return int(self._collection.count())

    def index_reviews(self, reviews: list[dict[str, Any]], batch_size: int = 64) -> int:
        """
        Index reviews into Chroma.

        Each review dict must include:
          - text (str)
          - review_id (str)
        Optional:
          - sentiment, rating, product_id
        """
        if not reviews:
            return 0

        texts = [str(r["text"]).strip() for r in reviews]
        if any(not t for t in texts):
            raise ValueError("Every review must include non-empty text.")

        ids = [str(r["review_id"]) for r in reviews]
        # Chroma prefers consistent metadata keys across rows.
        metadatas: list[dict[str, Any]] = []
        for review in reviews:
            metadatas.append(
                {
                    "sentiment": str(review.get("sentiment") or "unknown"),
                    "rating": float(review["rating"]) if review.get("rating") is not None else -1.0,
                    "product_id": str(review.get("product_id") or ""),
                }
            )

        embedder = self._ensure_embedder()
        embeddings = embedder.encode(texts, batch_size=batch_size, show_progress_bar=False)

        # Upsert so re-indexing the same ids is safe.
        self._collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
        )
        return len(ids)

    def retrieve(
        self,
        query: str,
        n_results: int = 5,
        sentiment_filter: str | None = None,
    ) -> list[RetrievedReview]:
        cleaned = query.strip()
        if not cleaned:
            raise ValueError("Query must not be empty.")
        if self.count == 0:
            return []

        embedder = self._ensure_embedder()
        query_embedding = embedder.encode([cleaned]).tolist()
        where = {"sentiment": sentiment_filter} if sentiment_filter else None
        top_k = min(n_results, self.count)

        result = self._collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        ids = (result.get("ids") or [[]])[0]

        retrieved: list[RetrievedReview] = []
        for i, text in enumerate(documents):
            meta = metadatas[i] if i < len(metadatas) and metadatas[i] else {}
            retrieved.append(
                RetrievedReview(
                    text=text,
                    sentiment=meta.get("sentiment"),
                    rating=meta.get("rating"),
                    product_id=meta.get("product_id"),
                    review_id=ids[i] if i < len(ids) else str(i),
                    distance=distances[i] if i < len(distances) else None,
                )
            )
        return retrieved
