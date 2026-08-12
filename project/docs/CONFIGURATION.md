# Configuration

Settings live in [`app/core/config.py`](../app/core/config.py) via **pydantic-settings**.  
Load order: environment variables / `.env` (cwd = `project/`) with case-insensitive keys. Template: [`.env.example`](../.env.example).

```bash
cp -n .env.example .env
```

Never commit `.env` (gitignored).

## Environment variables

| Variable | Default | Used by | Notes |
|----------|---------|---------|-------|
| `APP_NAME` | Review Intelligence API | `/health` | Display only |
| `APP_ENV` | `dev` | `/health` | Display only |
| `APP_HOST` | `0.0.0.0` | **Not read by app code** | Documented for uvicorn CLI convenience |
| `APP_PORT` | `8000` | **Not read by app code** | Pass to uvicorn yourself |
| `MODEL_PATH` | `./checkpoints/best-model` | Inference, RAG sentiment | Resolved via fallbacks if missing |
| `CHROMA_PATH` | `./chroma_db` | RAG store | Local PersistentClient path |
| `MLFLOW_TRACKING_URI` | `./experiments/mlruns` | Training CLI | File-backed tracking |
| `PROCESSED_DATA_DIR` | `./data/processed` | Train CLI default; settings | Preprocess CLI has its own `--output-dir` |
| `DEVICE` | `cpu` | Predictor | `cuda` only honored if CUDA available |
| `MAX_SAMPLES` | `1000` | Settings default | Preprocess CLI default is hardcoded `1000` unless you change invocation |
| `BATCH_SIZE` | `8` | Train `--train-batch-size` default | |
| `MAX_LENGTH` | `256` | Train tokenization | |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Chroma embedder | HF Hub id |
| `RAG_COLLECTION` | `product_reviews` | Chroma collection name | |
| `RAG_TOP_K` | `5` | Default retrieval count | Overridable per query |
| `OPENAI_API_KEY` | empty | Query generation | Enables `openai` mode |
| `OPENAI_MODEL` | `gpt-3.5-turbo` | OpenAI calls | |
| `HF_API_KEY` | empty | Query generation | Enables `huggingface` if OpenAI unset |
| `HF_LLM_MODEL` | `mistralai/Mistral-7B-Instruct-v0.2` | HF Inference | |

## Model path resolution

`Settings.resolved_model_path` → `resolve_model_path(preferred)`:

1. Preferred `MODEL_PATH` if that path exists  
2. Else first existing among:
   - `./checkpoints/best-model`
   - `./checkpoints/week3-distilbert/best-model`
   - `./checkpoints/week2-distilbert/best-model`
3. Else return preferred / first candidate (may not exist → 503 on predict)

## Secrets handling

| Secret | Storage | Exposure |
|--------|---------|----------|
| OpenAI / HF keys | `.env` only | Server → provider over HTTPS |
| Model weights | Local disk | Not secrets, but large / gitignored |
| DVC remotes | Not configured | Add credentials outside Git if you add S3/etc. |

## Runtime coupling

`get_settings()` is `@lru_cache`d. Changing `.env` requires **process restart** to pick up new values. Swapping checkpoint files also requires restart if the predictor singleton already loaded weights.

## Uvicorn host/port

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

`APP_HOST` / `APP_PORT` in `.env` do not start the server by themselves.

Next: [TESTING.md](TESTING.md).
