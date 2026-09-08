from fastapi.testclient import TestClient

from app.main import app


class _FakePredictor:
    def predict(self, text: str):
        class _Result:
            label_id = 2
            label = "positive"
            confidence = 0.91
            scores = {"negative": 0.03, "neutral": 0.06, "positive": 0.91}

        return _Result()


class _FakeRagPipeline:
    def index_reviews(self, reviews):
        return {
            "indexed_count": len(reviews),
            "collection_size": len(reviews),
            "reviews": [
                {
                    "review_id": r.get("review_id") or f"id-{i}",
                    "text": r["text"],
                    "sentiment": "positive",
                    "confidence": 0.91,
                    "rating": r.get("rating"),
                    "product_id": r.get("product_id"),
                }
                for i, r in enumerate(reviews)
            ],
        }

    def answer_query(self, question, sentiment_filter=None, n_results=None):
        from types import SimpleNamespace

        return SimpleNamespace(
            answer="Battery life is praised in matching reviews.",
            mode="extractive",
            question=question,
            sentiment_filter=sentiment_filter,
            sources=[
                {
                    "review_id": "r1",
                    "text": "Battery lasts two days.",
                    "sentiment": "positive",
                    "rating": 5.0,
                    "product_id": "p1",
                    "distance": 0.12,
                }
            ],
        )


def test_health_endpoint() -> None:
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "app" in body


def test_analyze_endpoint() -> None:
    from app.api.deps import get_predictor

    app.dependency_overrides[get_predictor] = lambda: _FakePredictor()
    client = TestClient(app)
    response = client.post("/api/analyze", json={"text": "Great battery and camera quality."})
    app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["label"] in {"negative", "neutral", "positive"}
    assert isinstance(body["confidence"], float)
    assert set(body["scores"].keys()) == {"negative", "neutral", "positive"}


def test_index_endpoint() -> None:
    from app.api.deps import get_rag_pipeline

    app.dependency_overrides[get_rag_pipeline] = lambda: _FakeRagPipeline()
    client = TestClient(app)
    response = client.post(
        "/api/index",
        json={
            "reviews": [
                {"text": "Amazing screen", "rating": 5, "product_id": "sku-1"},
                {"text": "Battery drains fast", "rating": 2},
            ]
        },
    )
    app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["indexed_count"] == 2
    assert len(body["reviews"]) == 2


def test_query_endpoint() -> None:
    from app.api.deps import get_rag_pipeline

    app.dependency_overrides[get_rag_pipeline] = lambda: _FakeRagPipeline()
    client = TestClient(app)
    response = client.post(
        "/api/query",
        json={"question": "What do people say about battery life?", "sentiment_filter": "positive"},
    )
    app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["mode"] == "extractive"
    assert body["sources"]
    assert "battery" in body["answer"].lower()
