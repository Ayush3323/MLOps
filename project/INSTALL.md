# Installation Guide

## Why two options?

The default `pip install torch` pulls in **CUDA (GPU) libraries** (~2GB+): nvidia-cublas, nvidia-cudnn, etc. That is heavy and not needed on a mid-end system without an NVIDIA GPU.

All other packages (fastapi, transformers, datasets, mlflow, chromadb, …) are required for this project and stay the same.

## Mid-end (16GB RAM, 256GB SSD, no GPU or CPU-only)

Use **CPU-only PyTorch** to save disk and download time:

```bash
cd ~/Desktop/MLops/project    # or your project path
python3 -m venv ../venv       # or: venv in project/
source ../venv/bin/activate  # or: venv\Scripts\activate on Windows

# 1. Install PyTorch CPU-only first (~200MB instead of ~2GB)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 2. Install the rest (from project directory so requirements.txt is found)
pip install -r requirements.txt
```

## High-end (with NVIDIA GPU, or when you want CUDA)

```bash
source venv/bin/activate
pip install torch
pip install -r requirements.txt
```

## What each group is for

| Package | Use in project |
|---------|----------------|
| fastapi, uvicorn, pydantic-settings, python-dotenv | API (analyze, index, query) |
| transformers, torch | Fine-tuning DistilBERT, inference |
| datasets | Load Amazon Reviews from HuggingFace |
| mlflow | Experiment tracking |
| chromadb, sentence-transformers | RAG, embeddings, vector search |
| scikit-learn, pandas, numpy | Preprocessing, splits, metrics |
| dvc | Data versioning |

Nothing listed here is redundant; the only “heavy” part you can avoid on mid-end is **CUDA** by using the CPU-only torch index.
