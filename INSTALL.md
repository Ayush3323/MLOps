# Installation Guide

Repo root: this directory (Git, DVC, and the Python package all live here)
Full docs: [`docs/`](docs/README.md)

## Why two Torch options?

Default `pip install torch` often pulls **CUDA** libraries (~2GB+). On a mid-end laptop without an NVIDIA GPU, install **CPU-only** Torch instead.

## Mid-end (16GB RAM, no GPU)

```bash
cd /path/to/ML
python3 -m venv venv
source venv/bin/activate

pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
cp -n .env.example .env
```

`pytest` and `httpx` are already listed in `requirements.txt`.

## High-end (NVIDIA GPU)

```bash
cd /path/to/ML
source venv/bin/activate
pip install torch
pip install -r requirements.txt
cp -n .env.example .env
```

Set `DEVICE=cuda` in `.env` when a GPU is available and the CUDA build of Torch is installed.

## What each package is for

| Package | Use |
|---------|-----|
| fastapi, uvicorn, pydantic-settings, python-dotenv | API |
| transformers, torch, accelerate | Fine-tune DistilBERT + inference |
| datasets | Stream / load review data and parquet |
| mlflow | Experiment tracking |
| chromadb, sentence-transformers | RAG embeddings + vector search |
| scikit-learn, pandas, numpy | Preprocess, metrics |
| dvc | Data versioning (run from repo root; see [`DVC.md`](DVC.md)) |
| pytest, httpx | Tests |

**Note:** Parquet I/O typically needs **pyarrow**, which often arrives transitively via `datasets`. If parquet load fails, `pip install pyarrow`. Torch is intentionally **not** pinned in `requirements.txt` so you choose CPU vs CUDA.

## Hugging Face cache

Models download once into `~/.cache/huggingface/`. First train and first RAG index need network access.

## Colab / other GPU machine

You do **not** need CUDA Torch on this laptop for production training. Use [`notebooks/colab_train.ipynb`](notebooks/colab_train.ipynb), then copy the checkpoint back ([`notebooks/COLAB_EXPORT.md`](notebooks/COLAB_EXPORT.md)).

## Next steps

1. [README.md](README.md) — run the API
2. [docs/WORKFLOWS.md](docs/WORKFLOWS.md) — full pipeline
3. [docs/CONFIGURATION.md](docs/CONFIGURATION.md) — `.env` reference
