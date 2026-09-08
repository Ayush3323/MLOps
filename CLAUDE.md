# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository layout

Git, DVC, and all Python code live in this single directory — there is no separate nested project folder.

```
ML/                          ← git + DVC root; cwd for all commands below
├── DVC.md                   ← DVC commands
├── .dvc/                    ← DVC metadata (remote not configured)
├── venv/                    ← local virtualenv (gitignored)
├── app/                     ← HTTP API (FastAPI)
├── src/                     ← preprocess / train / infer / RAG
├── tests/
├── notebooks/                ← Colab training notebook
├── data/processed/          ← parquet + *.dvc sidecars (gitignored data, tracked pointers)
├── checkpoints/              ← model weights (gitignored, local only)
├── chroma_db/                ← vector store (gitignored, local only)
├── experiments/mlruns/       ← MLflow file store (gitignored)
└── docs/                     ← full documentation suite, read in this order:
    README.md → OVERVIEW.md → ARCHITECTURE.md → CODEBASE_GUIDE.md →
    CONFIGURATION.md → DATA.md → WORKFLOWS.md → TESTING.md →
    MODEL_LIFECYCLE.md → RAG.md → API.md → OPERATIONS.md → TROUBLESHOOTING.md
```

`Project.md` is a historical/aspirational 6-week learning plan (Django, Docker Compose, GitHub Actions, Prometheus/Grafana, Celery, quantization) — none of that is implemented. Don't treat it as current-state documentation; treat `README.md` and `docs/` as ground truth.

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU-only; omit index-url for CUDA
pip install -r requirements.txt
cp -n .env.example .env
```

Torch is intentionally unpinned in `requirements.txt` so CPU vs CUDA is a manual choice. Parquet I/O needs `pyarrow` (usually pulled in transitively by `datasets`). Only Python 3.13 was available when this venv was last verified (no 3.11 on this machine) — `pyproject.toml` requires `>=3.11`, so 3.13 satisfies it, and a full install + `pytest -q` run passed cleanly on it.

If the venv ever misbehaves (activates but `which python`/`which pip` still resolve to `/usr/bin/python3`), the venv is corrupt or was never activated in that shell — delete `venv/` and recreate it with the commands above rather than trying to patch it.

## Common commands

Run from the repo root:

```bash
# Run the API (reload for dev)
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Full test suite
python -m pytest -q

# Single test file / test
python -m pytest tests/test_api/test_routes.py -q
python -m pytest tests/test_src/test_rag_pipeline.py::test_name -q

# Preprocess (streams Amazon Electronics JSONL; network required)
python -m src.data.preprocess --max-samples 1000 --output-dir data/processed

# Smoke train (CPU, caps train set at 2k)
python -m src.training.train --profile mid-end --data-dir data/processed \
  --output-dir checkpoints/week3-distilbert --train-batch-size 8 --eval-batch-size 16

# MLflow UI
mlflow ui --backend-store-uri ./experiments/mlruns --host 0.0.0.0 --port 5000

# DVC (same directory as everything else now)
dvc add data/processed/train.parquet
```

Do **not** pass `--tokenize` to `preprocess` if training afterward with `src.training.train` — the trainer re-tokenizes from raw `text` and the two are incompatible.

Real DistilBERT fine-tuning (~50k samples) should happen on Colab/GPU via `notebooks/colab_train.ipynb` + `notebooks/COLAB_EXPORT.md`, then the checkpoint is copied back and symlinked:

```bash
ln -sfn production-distilbert/best-model checkpoints/best-model
```

## Architecture

**Request flow:** `app/main.py` mounts three routers (`analyze`, `index`, `query`) that all go through `app/api/deps.py`, which lazily instantiates and caches a singleton `SentimentPredictor` (`src/inference/predict.py`) and `RagPipeline` (`src/rag/pipeline.py`) on first request — there is no FastAPI lifespan warmup, so the first call to each endpoint pays model-load latency.

- **`POST /api/analyze`** → `SentimentPredictor.predict` → lazy HF load of DistilBERT from the resolved checkpoint.
- **`POST /api/index`** → for each review: sentiment predict, then embed with MiniLM (`src/rag/embeddings.py: ReviewEmbeddingStore`) and upsert into a local Chroma `PersistentClient` with metadata (`sentiment`, `rating`, `product_id`, `review_id`).
- **`POST /api/query`** → embed question, retrieve top-k from Chroma (optional sentiment filter), then answer via a priority chain: `OPENAI_API_KEY` set → OpenAI chat completion; else `HF_API_KEY` set → HF Inference API; else an **extractive** bullet summary of retrieved snippets (the default with no keys configured).

**Config coupling:** `app/core/config.py` (`Settings`, pydantic-settings) is imported by both the API layer and the offline `src/*` CLIs (preprocess, train) — the ML layer currently depends on the API config module, which matters if `src` is ever split into a standalone package.

**Checkpoint resolution** (`Settings.resolved_model_path`) tries in order: `MODEL_PATH` env var → `./checkpoints/best-model` → `./checkpoints/week3-distilbert/best-model` → `./checkpoints/week2-distilbert/best-model`.

**Persistence boundaries** — all local disk, all gitignored, none of it a shared/server backend: model weights (`checkpoints/`), vectors (`chroma_db/`, a local `PersistentClient` not a Chroma server), experiments (`experiments/mlruns/`, file-backed MLflow), processed data (`data/processed/`, optionally DVC-tracked).

**Known placeholders — do not build on these as if they're active:** `app/ml/loader.py` is an empty file (singleton loading actually lives in `app/api/deps.py`); `docker/Dockerfile` is empty (no containerized deploy exists yet).

**`get_settings()` is `@lru_cache`d** — changing `.env` or swapping checkpoint files requires a process restart to take effect.

## Testing notes

11 tests, no coverage gate, no `conftest.py` (fixtures are local to test files). `pyproject.toml` sets `pythonpath = ["."]` so `app`/`src` import without installing the package — always run pytest from the repo root.

Tests are fast/mocked by design: API tests override `get_predictor`/`get_rag_pipeline` with fakes via FastAPI `dependency_overrides`; RAG tests inject fake embedding store + predictor into `RagPipeline`. There is no real DistilBERT/MiniLM/Chroma download in the test run. Known gaps (see `docs/TESTING.md`): real checkpoint/GPU load, live Chroma+SentenceTransformer integration, OpenAI/HF generation branches, end-to-end preprocess→train→predict, DVC workflows, and the F1 ≥ 0.90 quality gate (no `evaluate` module exists).

## Data

Amazon Electronics reviews are streamed and mapped to 3-class sentiment: rating ≤2 → negative (0), 3 → neutral (1), ≥4 → positive (2). Splits are stratified 80/10/10 with `seed=42`. Processed parquet lives in `data/processed/`, is gitignored, and is versioned with DVC in place (see `DVC.md`) — DVC has no remote configured yet, so `dvc push`/`pull` need one set up first before they're useful across machines.

Smoke/local training runs on small sample counts and can show low macro F1 — this is expected and not a regression; a serious fine-tune needs Colab/GPU at ~50k samples.
