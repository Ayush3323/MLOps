# API reference

Base URL (local): `http://127.0.0.1:8000`  
App: [`app/main.py`](../app/main.py) — FastAPI `Review Intelligence API` v0.1.0  
Interactive: `/docs` (Swagger), `/redoc`

Models load on **first use** of analyze/index (cold start). Health does **not** verify that the checkpoint or Chroma are ready.

---

## `GET /health`

Liveness / basic config echo.

**Response 200**

```json
{
  "status": "ok",
  "app": "Review Intelligence API",
  "env": "dev",
  "device": "cpu"
}
```

---

## `POST /api/analyze`

Predict sentiment for one text.

**Request**

| Field | Type | Constraints |
|-------|------|-------------|
| `text` | string | required, `min_length=1` (no max length enforced) |

```json
{ "text": "Battery life is excellent" }
```

**Response 200**

```json
{
  "label_id": 2,
  "label": "positive",
  "confidence": 0.91,
  "scores": {
    "negative": 0.02,
    "neutral": 0.07,
    "positive": 0.91
  }
}
```

**Errors**

| Status | When |
|--------|------|
| 422 | Empty / invalid body; empty text after strip |
| 503 | Checkpoint path missing (`FileNotFoundError`) |

---

## `POST /api/index`

Run sentiment + embed + upsert into Chroma.

**Request**

| Field | Type | Constraints |
|-------|------|-------------|
| `reviews` | array | length 1–500 |
| `reviews[].text` | string | required, min length 1 |
| `reviews[].review_id` | string | optional |
| `reviews[].rating` | float | optional, 1–5 |
| `reviews[].product_id` | string | optional |

```json
{
  "reviews": [
    { "text": "Battery lasts two days", "rating": 5, "product_id": "B00A" },
    { "text": "Screen cracked in a week", "rating": 1 }
  ]
}
```

**Response 200**

```json
{
  "status": "ok",
  "indexed_count": 2,
  "collection_size": 2,
  "reviews": [
    {
      "review_id": "...",
      "text": "Battery lasts two days",
      "sentiment": "positive",
      "confidence": 0.88,
      "rating": 5.0,
      "product_id": "B00A"
    }
  ]
}
```

**Errors**

| Status | When |
|--------|------|
| 422 | Validation / empty review text |
| 503 | Missing sentiment model |
| 500 | Embedding / Chroma failure (`Indexing failed: ...`) |

---

## `POST /api/query`

Retrieve reviews and answer a question.

**Request**

| Field | Type | Constraints |
|-------|------|-------------|
| `question` | string | min length 3 |
| `sentiment_filter` | string | optional; `negative` \| `neutral` \| `positive` |
| `n_results` | int | optional; 1–20; default from `RAG_TOP_K` |

```json
{
  "question": "What do customers complain about?",
  "sentiment_filter": "negative",
  "n_results": 5
}
```

**Response 200**

```json
{
  "status": "ok",
  "answer": "...",
  "mode": "extractive",
  "question": "What do customers complain about?",
  "sentiment_filter": "negative",
  "sources": []
}
```

`mode` is one of: `extractive`, `openai`, `huggingface` — see [RAG.md](RAG.md).

**Errors**

| Status | When |
|--------|------|
| 422 | Validation / bad filter |
| 502 | Upstream LLM `RuntimeError` |
| 500 | Unexpected failure (`Query failed: ...`) |

---

## Auth, CORS, limits

- No authentication or API keys on inbound requests
- No CORS middleware configured
- Analyze text length unbounded at schema level — very large payloads can OOM
- Index capped at 500 reviews per call

Next: [CONFIGURATION.md](CONFIGURATION.md).
