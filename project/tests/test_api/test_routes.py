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
