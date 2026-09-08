from src.rag.pipeline import RagPipeline


class _FakePredictor:
    def predict(self, text: str):
        class _Result:
            label = "negative" if "bad" in text.lower() else "positive"
            confidence = 0.88

        return _Result()


class _FakeStore:
    def __init__(self):
        self.docs = []

    @property
    def count(self) -> int:
        return len(self.docs)

    def index_reviews(self, reviews, batch_size: int = 64):
        self.docs.extend(reviews)
        return len(reviews)

    def retrieve(self, query, n_results=5, sentiment_filter=None):
        from src.rag.embeddings import RetrievedReview

        matches = self.docs
        if sentiment_filter:
            matches = [d for d in matches if d.get("sentiment") == sentiment_filter]
        return [
            RetrievedReview(
                text=d["text"],
                sentiment=d.get("sentiment"),
                rating=d.get("rating"),
                product_id=d.get("product_id"),
                review_id=d["review_id"],
                distance=0.1,
            )
            for d in matches[:n_results]
        ]


def test_rag_index_and_extractive_query() -> None:
    store = _FakeStore()
    pipeline = RagPipeline(store=store, predictor=_FakePredictor())

    indexed = pipeline.index_reviews(
        [
            {"text": "Bad battery life", "review_id": "1", "rating": 1},
            {"text": "Great camera quality", "review_id": "2", "rating": 5},
        ]
    )
    assert indexed["indexed_count"] == 2
    assert store.count == 2

    answer = pipeline.answer_query("battery problems", sentiment_filter="negative")
    assert answer.mode == "extractive"
    assert answer.sources
    assert answer.sources[0]["sentiment"] == "negative"
    assert "indexed reviews" in answer.answer.lower() or "battery" in answer.answer.lower()
