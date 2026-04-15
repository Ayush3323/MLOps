# Review Intelligence System

Sentiment-aware product review system using FastAPI, DistilBERT, and an MLOps-first workflow.

## Week 1 status

- [x] Environment dependencies installed (CPU-friendly Torch path supported).
- [x] Project structure created.
- [x] `.env.example` and config scaffolding created.
- [x] Preprocessing pipeline implemented (`load -> label -> stratified split -> save`).
- [x] Unit tests added for label mapping and stratified split.
- [x] Sanity run completed with `max_samples=1000`.

## Run locally

```bash
cd ~/Desktop/MLops/project
source ../venv/bin/activate
uvicorn app.main:app --reload
```

Health endpoint:

```bash
curl http://127.0.0.1:8000/health
```

## Preprocessing commands

Run sanity data prep (used for Week 1):

```bash
python -m src.data.preprocess --max-samples 1000 --output-dir data/processed
```

Run tests:

```bash
python -m pytest -q tests/test_src/test_preprocess.py
```

## DVC quick start (repo root)

If your git repo root is `~/Desktop/MLops`:

```bash
cd ~/Desktop/MLops
dvc init
git add .dvc .gitignore
git commit -m "Add DVC"
git push
```

Then track first dataset artifact:

```bash
dvc add project/data/processed/train.parquet
git add project/data/processed/train.parquet.dvc .gitignore
git commit -m "Track processed train split with DVC"
git push
```

## Week 2 commands

### 1) DVC setup (repo root)

```bash
cd ~/Desktop/MLops
source venv/bin/activate
dvc --version
dvc init
git add .dvc .gitignore
git commit -m "Initialize DVC"
git push
```

### 2) Train baseline model (mid-end safe)

```bash
cd ~/Desktop/MLops/project
source ../venv/bin/activate
python -m src.training.train \
  --data-dir data/processed \
  --output-dir checkpoints/week2-distilbert \
  --num-epochs 2 \
  --train-batch-size 8 \
  --eval-batch-size 16 \
  --max-length 256 \
  --max-train-samples 400 \
  --max-eval-samples 100
```

### 3) MLflow UI

```bash
cd ~/Desktop/MLops/project
source ../venv/bin/activate
mlflow ui --backend-store-uri ./experiments/mlruns --host 0.0.0.0 --port 5000
```

### 4) Quick inference smoke test

```bash
cd ~/Desktop/MLops/project
source ../venv/bin/activate
python -c "from src.inference.predict import predict_text; print(predict_text('Battery life is excellent', model_path='checkpoints/week2-distilbert/best-model'))"
```

### 5) Run all tests

```bash
cd ~/Desktop/MLops/project
source ../venv/bin/activate
python -m pytest -q tests/test_src
```
