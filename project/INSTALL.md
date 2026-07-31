# Installation Guide

Repo: `/home/hodorinfo/Desktop/ML`  
Project package: `project/`

## Why two options?

Default `pip install torch` pulls **CUDA** libraries (~2GB+). On a mid-end laptop without an NVIDIA GPU, install **CPU-only** Torch instead.

## Mid-end (16GB RAM, no GPU)

```bash
cd /home/hodorinfo/Desktop/ML/project
python3 -m venv ../venv
source ../venv/bin/activate

pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install pytest
cp -n .env.example .env
```

## High-end (NVIDIA GPU)

```bash
cd /home/hodorinfo/Desktop/ML/project
source ../venv/bin/activate
pip install torch
pip install -r requirements.txt
pip install pytest
cp -n .env.example .env
```

Set `DEVICE=cuda` in `.env` when a GPU is available.

## What each package is for

| Package | Use |
|---------|-----|
| fastapi, uvicorn, pydantic-settings, python-dotenv | API |
| transformers, torch, accelerate | Fine-tune DistilBERT + inference |
| datasets | Load / stream review data |
| mlflow | Experiment tracking |
| chromadb, sentence-transformers | RAG embeddings + vector search |
| scikit-learn, pandas, numpy | Preprocess, metrics |
| dvc | Data versioning (install also usable from repo root) |

## Hugging Face cache

Models download once into `~/.cache/huggingface/`. First train and first RAG index need network access.

## Colab / other GPU machine

Do **not** need CUDA Torch on this laptop for production training. Use `notebooks/colab_train.ipynb`, then copy the checkpoint back (see `notebooks/COLAB_EXPORT.md`).
