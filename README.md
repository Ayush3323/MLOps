# Review Intelligence System

Sentiment-aware product review system using **FastAPI**, **DistilBERT**, **ChromaDB**, and MLflow.

- **Git + DVC root:** this directory
- **Full docs:** [`docs/`](docs/README.md) (high → low)

---

## Status checklist

| Area | Status |
|------|--------|
| Env + config (`.env`, FastAPI settings) | Done |
| Preprocess + stratified splits | Done (`src/data/preprocess.py`) |
| DistilBERT train + MLflow | Done (`src/training/train.py`) |
| Inference + `POST /api/analyze` | Done |
| RAG embeddings + `POST /api/index` / `POST /api/query` | Done (extractive by default) |
| Processed parquet + DVC sidecars | Present under `data/processed/` (DVC remote not configured) |
| Docker / GitHub Actions / monitoring | Not yet |

Typical local checkpoint symlink: `checkpoints/best-model` → `week3-distilbert/best-model`.
Smoke / tiny-data runs can show **low macro F1**. Use Colab/GPU for a serious train (see [`notebooks/`](notebooks/) and [`docs/MODEL_LIFECYCLE.md`](docs/MODEL_LIFECYCLE.md)).

---

## Setup (CPU / mid-end)

```bash
cd /path/to/ML
python3 -m venv venv && source venv/bin/activate   # create first if missing

# CPU-only Torch (~200MB) — avoid full CUDA wheel on laptops without NVIDIA
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

cp -n .env.example .env
```

Details: [`INSTALL.md`](INSTALL.md). First RAG call downloads **`sentence-transformers/all-MiniLM-L6-v2`** (~90MB) from Hugging Face once.

---

## Run the API

```bash
cd /path/to/ML
source venv/bin/activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

```bash
curl http://127.0.0.1:8000/health

curl -X POST http://127.0.0.1:8000/api/analyze \
  -H 'Content-Type: application/json' \
  -d '{"text":"Battery life is excellent"}'

curl -X POST http://127.0.0.1:8000/api/index \
  -H 'Content-Type: application/json' \
  -d '{"reviews":[{"text":"Battery lasts two days","rating":5},{"text":"Screen cracked in a week","rating":1}]}'

curl -X POST http://127.0.0.1:8000/api/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"What do customers complain about?","sentiment_filter":"negative"}'
```

Without `OPENAI_API_KEY` / `HF_API_KEY`, `/api/query` returns an **extractive** answer from retrieved reviews.
Full contract: [`docs/API.md`](docs/API.md). Interactive OpenAPI: `/docs` and `/redoc`.

---

## Data prep

Local processed splits are typically built at **50k** samples (40k / 5k / 5k). Schema and DVC: [`docs/DATA.md`](docs/DATA.md).

```bash
# Sanity / small run
python -m src.data.preprocess --max-samples 1000 --output-dir data/processed

# Scale-up (network required; streams Amazon Electronics JSONL)
python -m src.data.preprocess --max-samples 50000 --output-dir data/processed
```

Do **not** pass `--tokenize` if you plan to train with `src.training.train` — training re-tokenizes from raw `text`.

---

## Training — where to run what

| Task | Where | Notes |
|------|--------|-------|
| Smoke train | This machine | `python -m src.training.train --profile mid-end` (caps train at 2k) |
| Real DistilBERT fine-tune (~50k) | Colab T4 / GPU | [`notebooks/colab_train.ipynb`](notebooks/colab_train.ipynb) + [`notebooks/COLAB_EXPORT.md`](notebooks/COLAB_EXPORT.md) |
| Embed + Chroma index | This machine | Via `/api/index` |
| Optional OpenAI / HF answers | This machine | Set keys in `.env` |

```bash
python -m src.training.train \
  --profile mid-end \
  --data-dir data/processed \
  --output-dir checkpoints/week3-distilbert \
  --train-batch-size 8 \
  --eval-batch-size 16
```

After Colab/GPU training:

```bash
ln -sfn production-distilbert/best-model checkpoints/best-model
# or: MODEL_PATH=./checkpoints/production-distilbert/best-model in .env
```

MLflow UI:

```bash
mlflow ui --backend-store-uri ./experiments/mlruns --host 0.0.0.0 --port 5000
```

More: [`docs/MODEL_LIFECYCLE.md`](docs/MODEL_LIFECYCLE.md), [`docs/WORKFLOWS.md`](docs/WORKFLOWS.md).

---

## DVC

Data under `data/` is gitignored. Version processed parquet with DVC. See [`DVC.md`](DVC.md) and [`docs/DATA.md`](docs/DATA.md).

---

## Tests

```bash
python -m pytest -q
```

See [`docs/TESTING.md`](docs/TESTING.md).

---

## Hugging Face downloads (summary)

| Asset | Size | When |
|-------|------|------|
| `distilbert-base-uncased` | ~260MB | First train / tokenizer load |
| `all-MiniLM-L6-v2` | ~90MB | First `/api/index` or embed |
| Amazon Electronics JSONL | streamed | Preprocess |
| Mistral-7B weights | ~14GB+ | **Only** if you load locally — prefer HF Inference API |

---

## Next roadmap

1. Fine-tune on Colab at ~50k; promote checkpoint; aim for test **F1 macro ≥ 0.90**.
2. Index a real review batch; exercise filtered `/api/query`.
3. Commit DVC sidecars + configure a remote when sharing data.
4. Later: Docker Compose, GitHub Actions quality gate, monitoring.

Historical learning plan (Django-era aspirations, partly outdated): [`Project.md`](Project.md).
