# Colab → local export checklist

Use this after finishing `colab_train.ipynb` (or an equivalent GPU machine run).  
Related: [docs/MODEL_LIFECYCLE.md](../docs/MODEL_LIFECYCLE.md), [docs/WORKFLOWS.md](../docs/WORKFLOWS.md).

## On Colab / GPU machine

1. Confirm metrics look reasonable (target: **test F1 macro ≥ 0.90** once you use ~50k samples).
2. Zip the checkpoint folder:
   ```bash
   !cd /content/checkpoints && zip -r best-model.zip production-distilbert/best-model
   ```
3. Download `best-model.zip` from the Colab Files panel (or `files.download(...)`).
4. Optional: download `mlruns` / `test_predictions.json` for analysis.

## On this machine

1. Unzip into the checkpoints dir:
   ```bash
   cd /path/to/ML
   mkdir -p checkpoints/production-distilbert
   unzip ~/Downloads/best-model.zip -d checkpoints/production-distilbert
   # Ensure path is: checkpoints/production-distilbert/best-model/{config.json,model.safetensors,tokenizer*}
   ```
   If the zip already contains a `best-model/` folder at the top level, adjust so the final path matches above.
2. Point the API at the new weights:
   ```bash
   ln -sfn production-distilbert/best-model checkpoints/best-model
   # or edit .env: MODEL_PATH=./checkpoints/production-distilbert/best-model
   ```
3. Smoke test:
   ```bash
   source venv/bin/activate
   python -c "from src.inference.predict import predict_text; print(predict_text('Battery life is excellent'))"
   ```
4. Restart API (required if uvicorn already loaded an older singleton):
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

## Naming note

| Source | Typical directory |
|--------|-------------------|
| Local `train.py` default | `checkpoints/week3-distilbert/best-model` |
| Colab notebook | `checkpoints/production-distilbert/best-model` |
| API symlink | `checkpoints/best-model` → one of the above |

## Do **not** commit large weights to Git

`checkpoints/` and `*.safetensors` are gitignored. Keep models on disk or in object storage / a large DVC remote.

## What stays on this machine vs Colab

| Artifact | Keep where |
|----------|------------|
| Code | This machine + Git |
| `data/processed/*.parquet` | This machine; track with DVC ([DVC.md](../DVC.md)) |
| Fine-tuned `best-model` | This machine after download; train on Colab |
| `chroma_db/` | This machine (rebuild via `/api/index`) |
| MLflow runs | Prefer this machine; Colab experiment name differs (`review-intelligence-colab`) |

## Optional: upload parquet to Colab instead of re-downloading Amazon data

```bash
# On this machine after preprocess --max-samples 50000
# Upload train/validation/test.parquet to Colab /content/data/processed/
```

Then skip streaming in the notebook and train with `--data-dir /content/data/processed` (or the notebook’s equivalent cell).
