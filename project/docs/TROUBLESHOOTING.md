# Troubleshooting

## API returns 503 on `/api/analyze` or `/api/index`

**Cause:** Checkpoint path missing.

**Fix:**

1. Confirm `ls checkpoints/best-model` (or your `MODEL_PATH`) exists and contains model + tokenizer files.
2. Train locally or import from Colab ([COLAB_EXPORT.md](../notebooks/COLAB_EXPORT.md)).
3. `ln -sfn week3-distilbert/best-model checkpoints/best-model`
4. Restart uvicorn.

## Predictions are always “positive” / low F1

**Cause:** Smoke training on tiny or heavily skewed data; majority-class collapse.

**Fix:** Train with more data (Colab, `high-end` / uncapped), review `test_predictions.json` and MLflow. Macro F1 is the right success metric — see [MODEL_LIFECYCLE.md](MODEL_LIFECYCLE.md).

## First RAG call hangs or fails downloading

**Cause:** Hugging Face Hub download of MiniLM (~90MB) needs network; cache cold.

**Fix:** Ensure outbound HTTPS; wait for cache under `~/.cache/huggingface/`. Retry after network recovery.

## `/api/query` returns empty / “no reviews” style extractive answer

**Cause:** Chroma collection empty or filter too strict.

**Fix:** Call `/api/index` first; drop `sentiment_filter` or index more negatives; check `collection_size` from index response.

## `/api/query` returns 502

**Cause:** OpenAI or HF Inference call failed (bad key, quota, model id).

**Fix:** Unset keys to force extractive mode, or fix provider credentials / model names in `.env` and restart.

## CUDA requested but runs on CPU

**Cause:** `DEVICE=cuda` but `torch.cuda.is_available()` is false (CPU torch wheel or no GPU).

**Fix:** Install CUDA torch on a GPU machine, or leave `DEVICE=cpu`. Serving on CPU is supported.

## Training fails looking for `text` column

**Cause:** Parquet was written with `preprocess --tokenize`.

**Fix:** Re-run preprocess **without** `--tokenize`.

## `dvc pull` does nothing useful / no remote

**Cause:** DVC remote not configured.

**Fix:** Re-run preprocess locally, copy parquet, or `dvc remote add` + `dvc push`/`pull` — [DVC.md](../../DVC.md).

## Import errors for `app` or `src`

**Cause:** Wrong working directory.

**Fix:** Always run from `project/` (or rely on pytest `pythonpath`). Activate the repo `venv`.

## Torch install is multi-GB on a laptop

**Cause:** Default `pip install torch` pulls CUDA.

**Fix:** CPU wheel — [INSTALL.md](../INSTALL.md).

## Changed `.env` but behavior unchanged

**Cause:** Cached `get_settings()` and/or loaded model singleton.

**Fix:** Restart uvicorn.

## MLflow UI shows weird absolute paths

**Cause:** Runs recorded on another machine/path (historical).

**Fix:** Treat as metadata only; metrics remain useful. Prefer fresh runs on this machine.

## Empty files you might open by mistake

| Path | Note |
|------|------|
| `app/ml/loader.py` | Placeholder — use `app/api/deps.py` |
| `docker/Dockerfile` | Placeholder — no build yet |

Next: [GLOSSARY.md](GLOSSARY.md).
