# Review Intelligence System

Sentiment-aware product review system using **FastAPI**, **DistilBERT**, **ChromaDB**, and an MLOps-first workflow.

Repo path on this machine: `/home/hodorinfo/Desktop/ML`  
App code lives in: `project/`

---

## Status checklist

| Area | Status |
|------|--------|
| Env + config (`.env`, FastAPI settings) | Done |
| Preprocess + stratified splits | Done (`src/data/preprocess.py`) |
| DistilBERT train + MLflow | Done (`src/training/train.py`) |
| Inference + `POST /api/analyze` | Done |
| RAG embeddings + `POST /api/index` / `POST /api/query` | Done (extractive by default) |
| DVC initialized at repo root | Partial (`.dvc/` exists; track data next) |
| Docker / GitHub Actions / monitoring | Not yet |

Current local checkpoint: `checkpoints/best-model` → `week3-distilbert/best-model`  
Metrics on that smoke model are **low F1** (tiny data). Use Colab/GPU for a real train (see `notebooks/`).

---

## Setup (this machine — CPU / mid-end)

```bash
cd /home/hodorinfo/Desktop/ML/project
source ../venv/bin/activate   # create first if missing: python3 -m venv ../venv

# CPU-only Torch (~200MB) — avoid full CUDA wheel
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install pytest

cp .env.example .env   # already created if you cloned after this update
```

First RAG call downloads **`sentence-transformers/all-MiniLM-L6-v2`** (~90MB) from Hugging Face once.

---

## Run the API

```bash
cd /home/hodorinfo/Desktop/ML/project
source ../venv/bin/activate
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

Without `OPENAI_API_KEY` / `HF_API_KEY`, `/api/query` returns an **extractive** answer from retrieved reviews (no LLM download).

---

## Data prep (this machine)

Sanity (already ran ~1k rows locally):

```bash
cd /home/hodorinfo/Desktop/ML/project
source ../venv/bin/activate
python -m src.data.preprocess --max-samples 1000 --output-dir data/processed
```

Scale-up for real training (CPU OK, takes longer; network required):

```bash
python -m src.data.preprocess --max-samples 50000 --output-dir data/processed
```

---

## Training — where to run what

| Task | Where | Command / notes |
|------|--------|-----------------|
| Smoke train (tiny) | **This machine** | `python -m src.training.train --profile mid-end` |
| Real DistilBERT fine-tune (50k, 3 epochs) | **Colab T4 / other GPU** | See `notebooks/colab_train.ipynb` + `notebooks/COLAB_EXPORT.md` |
| Embed + Chroma index | **This machine** | Via `/api/index` (MiniLM is CPU-friendly) |
| Optional Mistral-7B **local** | **Other GPU 16GB+** | Not needed if you use HF Inference API key |
| Optional OpenAI answers | **This machine** | Set `OPENAI_API_KEY` in `.env` |

Mid-end smoke train:

```bash
python -m src.training.train \
  --profile mid-end \
  --data-dir data/processed \
  --output-dir checkpoints/week-local \
  --train-batch-size 8 \
  --eval-batch-size 16
```

After Colab/GPU training, copy `best-model/` into `project/checkpoints/production-distilbert/` then:

```bash
ln -sfn production-distilbert/best-model checkpoints/best-model
# or set MODEL_PATH=./checkpoints/production-distilbert/best-model in .env
```

MLflow UI:

```bash
mlflow ui --backend-store-uri ./experiments/mlruns --host 0.0.0.0 --port 5000
```

---

## DVC (repo root `/home/hodorinfo/Desktop/ML`)

Data files under `project/data/` are gitignored. Version them with DVC:

```bash
cd /home/hodorinfo/Desktop/ML
source venv/bin/activate
pip install dvc

# If .dvc/config is empty / incomplete:
dvc init --force   # only if needed; keep existing .dvc if already valid

dvc add project/data/processed/train.parquet
dvc add project/data/processed/validation.parquet
dvc add project/data/processed/test.parquet

git add project/data/processed/*.dvc project/data/processed/.gitignore .dvc .gitignore
git commit -m "Track processed Amazon review splits with DVC"
```

Push data to a remote when ready (`dvc remote add -d myremote ...` then `dvc push`). Until then, DVC still tracks local hashes.

---

## Tests

```bash
cd /home/hodorinfo/Desktop/ML/project
source ../venv/bin/activate
python -m pytest -q
```

---

## Hugging Face downloads (summary)

| Asset | Size | When |
|-------|------|------|
| `distilbert-base-uncased` | ~260MB | First train / tokenizer load |
| `all-MiniLM-L6-v2` | ~90MB | First `/api/index` or embed |
| Amazon Electronics JSONL | streamed | Preprocess |
| Mistral-7B weights | ~14GB+ | **Only** if you load the model locally — prefer HF API |

---

## Next roadmap

1. Scale preprocess to 50k on this machine (or upload parquet to Colab).
2. Fine-tune on Colab; download `best-model`; point `MODEL_PATH`.
3. Aim for test **F1 macro ≥ 0.90**; compare runs in MLflow.
4. Index a real review batch via `/api/index`; exercise `/api/query`.
5. Later: Docker Compose, GitHub Actions quality gate, monitoring.
