# Codebase guide

Working directory for imports and CLIs: repo root.

## Recommended reading order (source)

1. `app/main.py` — HTTP surface
2. `app/core/config.py` — settings + path resolution
3. `app/api/deps.py` — singletons
4. `app/api/routes/*.py` — contracts
5. `src/inference/predict.py` — classifier
6. `src/rag/embeddings.py` → `src/rag/pipeline.py` — RAG
7. `src/data/preprocess.py` → `src/training/train.py` — offline ML
8. `tests/` — expected contracts

## Tree (substantive)

```text
ML/                              # git + DVC root
├── app/
│   ├── main.py                 # FastAPI app, GET /health
│   ├── core/config.py          # Settings, resolve_model_path
│   ├── api/deps.py             # get_predictor, get_rag_pipeline
│   ├── api/routes/
│   │   ├── analyze.py          # POST /api/analyze
│   │   ├── index.py            # POST /api/index
│   │   └── query.py            # POST /api/query
│   └── ml/loader.py            # EMPTY placeholder
├── src/
│   ├── data/preprocess.py      # Stream → clean → split → parquet CLI
│   ├── training/train.py       # DistilBERT + MLflow CLI
│   ├── inference/predict.py    # SentimentPredictor, predict_text
│   └── rag/
│       ├── embeddings.py       # ReviewEmbeddingStore (Chroma + MiniLM)
│       └── pipeline.py         # RagPipeline index + answer
├── tests/
│   ├── test_api/test_routes.py
│   └── test_src/
│       ├── test_preprocess.py
│       ├── test_train_and_predict.py
│       └── test_rag_pipeline.py
├── notebooks/
│   ├── colab_train.ipynb
│   └── COLAB_EXPORT.md
├── data/raw/                   # empty placeholder (.gitkeep)
├── data/processed/             # parquet + *.dvc
├── docker/Dockerfile           # EMPTY placeholder
├── .env.example
├── requirements.txt
└── pyproject.toml              # pytest pythonpath=["."]
```

Runtime-only (gitignored): `checkpoints/`, `chroma_db/`, `experiments/`, `.env`.

## Module map

### HTTP — `app/`

| File | Responsibility | Key symbols |
|------|----------------|-------------|
| `app/main.py` | Create app; mount routers; health | `app`, `health` |
| `app/core/config.py` | Env settings; model path fallbacks | `Settings`, `get_settings`, `resolve_model_path` |
| `app/api/deps.py` | Lazy singletons for routes | `get_predictor`, `get_rag_pipeline` |
| `app/api/routes/analyze.py` | Sentiment endpoint | `AnalyzeRequest/Response` |
| `app/api/routes/index.py` | Batch index (max 500) | `IndexRequest/Response` |
| `app/api/routes/query.py` | RAG Q&A | `QueryRequest/Response` |
| `app/ml/loader.py` | **Unused empty file** | — |

### Offline data — `src/data/preprocess.py`

| Symbol | Role |
|--------|------|
| `ELECTRONICS_REVIEWS_URL` | Amazon 2023 Electronics JSONL.gz |
| `map_rating_to_label` | ≤2→0, 3→1, ≥4→2 |
| `load_and_clean` | Stream, coerce text, drop empty |
| `split_dataset` | Stratified 80/10/10, `seed=42` |
| `tokenize` | Optional DistilBERT tokenize (**incompatible with current train loader**) |
| `main` | CLI `--max-samples`, `--output-dir`, `--tokenize` |

### Training — `src/training/train.py`

| Symbol | Role |
|--------|------|
| `TrainConfig` / `apply_profile_defaults` | `mid-end` vs `high-end` caps |
| `load_splits` | Expects `text` + `label` parquet columns |
| `compute_metrics` | accuracy + `f1_macro` |
| `train_and_evaluate` | Trainer, early stopping, MLflow, `best-model/` |

Experiment name: `review-intelligence-week3`. Default output: `./checkpoints/week3-distilbert`.

### Inference — `src/inference/predict.py`

| Symbol | Role |
|--------|------|
| `LABEL_MAP` | 0/1/2 → negative/neutral/positive |
| `SentimentPredictor` | Lazy HF load; CUDA only if requested and available |
| `predict_text` | Convenience wrapper for smoke scripts |

### RAG — `src/rag/`

| File / symbol | Role |
|---------------|------|
| `ReviewEmbeddingStore` | Persistent Chroma; cosine HNSW; MiniLM encode |
| `RetrievedReview` | Retrieval hit dataclass |
| `RagPipeline.index_reviews` | Predict sentiment → upsert |
| `RagPipeline.answer_query` | Retrieve → generate |
| `_extractive_answer` / `_generate_openai` / `_generate_huggingface` | Answer backends |

### Tests — `tests/`

| File | Focus |
|------|-------|
| `test_api/test_routes.py` | Health + three POST routes with dependency overrides |
| `test_src/test_preprocess.py` | Label map + stratified split math |
| `test_src/test_train_and_predict.py` | Metrics, profiles, empty text |
| `test_src/test_rag_pipeline.py` | Index + extractive query with fakes |

### Notebooks

| Asset | Role |
|-------|------|
| `colab_train.ipynb` | GPU training; duplicates some preprocess logic |
| `COLAB_EXPORT.md` | Zip → `checkpoints/production-distilbert` → symlink |

## Import / coupling notes

- Routes depend on `deps` → `src.inference` / `src.rag`.
- `src.*` modules import `app.core.config.get_settings` (circular layering risk if you try to publish `src` as a standalone package).
- Pytest sets `pythonpath = ["."]` so `app` and `src` import without installing the package.

## What is intentionally absent

- No `scripts/` directory — CLIs are module `__main__` blocks
- No `conftest.py` — fixtures are local to test files
- No `src/training/evaluate.py` — referenced only in the old learning plan
- No frontend package

Next: [WORKFLOWS.md](WORKFLOWS.md).
