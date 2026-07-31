# Colab → local export checklist

Use this after finishing `colab_train.ipynb` (or an equivalent GPU machine run).

## On Colab / GPU machine

1. Confirm metrics look reasonable (target: **test F1 macro ≥ 0.90** once you use ~50k samples).
2. Zip the checkpoint folder:
   ```bash
   !cd /content/checkpoints && zip -r best-model.zip production-distilbert/best-model
   ```
3. Download `best-model.zip` from the Colab Files panel (or `files.download(...)`).
4. Optional: download `mlruns` / `test_predictions.json` for analysis.

## On this machine (`/home/hodorinfo/Desktop/ML`)

1. Unzip into the project checkpoints dir:
   ```bash
   cd /home/hodorinfo/Desktop/ML/project
   mkdir -p checkpoints/production-distilbert
   unzip ~/Downloads/best-model.zip -d checkpoints/production-distilbert
   # Ensure path is: checkpoints/production-distilbert/best-model/{config.json,model.safetensors,tokenizer*}
   ```
2. Point the API at the new weights:
   ```bash
   ln -sfn production-distilbert/best-model checkpoints/best-model
   # or edit .env: MODEL_PATH=./checkpoints/production-distilbert/best-model
   ```
3. Smoke test:
   ```bash
   source ../venv/bin/activate
   python -c "from src.inference.predict import predict_text; print(predict_text('Battery life is excellent'))"
   ```
4. Restart API:
   ```bash
   uvicorn app.main:app --reload --port 8000
   ```

## Do **not** commit large weights to Git

`checkpoints/` and `*.safetensors` are gitignored. Keep models on disk or in object storage / DVC remote.

## What stays on this machine vs Colab

| Artifact | Keep where |
|----------|------------|
| Code (`project/`) | This machine + Git |
| `data/processed/*.parquet` | This machine; track with DVC |
| Fine-tuned `best-model` | This machine after download; train on Colab |
| `chroma_db/` | This machine (rebuild via `/api/index`) |
| MLflow runs | Prefer this machine; Colab runs can be exported separately |

## Optional: upload parquet to Colab instead of re-downloading Amazon data

```bash
# On this machine after preprocess --max-samples 50000
# Upload train/validation/test.parquet to Colab /content/data/processed/
```

Then skip streaming in the notebook and call train with `--data-dir /content/data/processed`.
