# Testing

## How to run

```bash
cd /path/to/ML
source venv/bin/activate
python -m pytest -q
```

Config in [`pyproject.toml`](../pyproject.toml):

- `pythonpath = ["."]`
- `testpaths = ["tests"]`

There is no coverage gate and no `conftest.py`.

## Inventory (11 tests)

| File | What it covers |
|------|----------------|
| [`tests/test_api/test_routes.py`](../tests/test_api/test_routes.py) | `/health`, `/api/analyze`, `/api/index`, `/api/query` via `TestClient` + `dependency_overrides` |
| [`tests/test_src/test_preprocess.py`](../tests/test_src/test_preprocess.py) | Rating→label mapping; stratified split proportions |
| [`tests/test_src/test_train_and_predict.py`](../tests/test_src/test_train_and_predict.py) | `compute_metrics`, profile defaults, empty-text rejection |
| [`tests/test_src/test_rag_pipeline.py`](../tests/test_src/test_rag_pipeline.py) | Index + extractive query with fake store/predictor |

## Strategy

- **Fast unit tests** — no real DistilBERT, MiniLM, or Chroma downloads in CI-friendly runs
- API tests **override** `get_predictor` / `get_rag_pipeline` with fakes
- RAG tests inject fake embedding store + predictor into `RagPipeline`

## Covered contracts

- Health payload shape
- Analyze / index / query happy paths with mocks
- Label mapping and split stratification math
- Macro F1 / accuracy metric helper
- Mid-end profile sample caps
- Empty text rejected before model load

## Not covered (gaps)

| Gap | Risk |
|-----|------|
| Real checkpoint load / GPU path | Serving regressions undetected |
| Chroma + SentenceTransformer integration | Index/query failures only at runtime |
| OpenAI / HF generation branches | Mode priority bugs |
| End-to-end preprocess → train → predict | Pipeline drift |
| DVC workflows | Ops-only |
| Quality gate F1 ≥ 0.90 | No `evaluate` module / CI |
| Load testing / OOM on huge analyze text | Production risk |

## Manual smoke (optional)

With a real checkpoint:

```bash
python -c "from src.inference.predict import predict_text; print(predict_text('Battery life is excellent'))"
uvicorn app.main:app --port 8000
# then curl /api/analyze, /api/index, /api/query
```

Next: [OPERATIONS.md](OPERATIONS.md).
