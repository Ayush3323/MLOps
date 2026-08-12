# Model lifecycle

## Goal

Fine-tune **`distilbert-base-uncased`** for 3-class sentiment (`negative` / `neutral` / `positive`), track runs in MLflow, and serve the best checkpoint through FastAPI.

Target quality (roadmap / Colab scale): **test F1 macro ≥ 0.90**. Local smoke models often fall short.

## Training entrypoint

```bash
python -m src.training.train --profile mid-end|high-end [options]
```

Implementation: [`src/training/train.py`](../src/training/train.py).

### Profiles

| Profile | Epochs (default) | Train cap | Val/test cap |
|---------|------------------|-----------|--------------|
| `mid-end` | 2 | 2,000 | 400 each |
| `high-end` | 3 | none (full train split) | 2,000 each |

Other defaults: `learning_rate=2e-5`, `weight_decay=0.01`, `seed=42`, `max_length` from settings (256), early stopping patience 2 on `eval_f1_macro`.

CLI flags can override caps, batch sizes, output directory, etc. (`--help`).

### Metrics

`compute_metrics` returns:

- `accuracy`
- `f1_macro` (best-model selection: `metric_for_best_model="eval_f1_macro"`)

Macro F1 treats classes equally — important under positive skew ([DATA.md](DATA.md)).

### Artifacts written

| Artifact | Path |
|----------|------|
| Best weights + tokenizer | `{output_dir}/best-model/` |
| Predictions for analysis | `{output_dir}/test_predictions.json` |
| MLflow run | `experiments/mlruns/` (experiment `review-intelligence-week3`) |

Default `--output-dir`: `./checkpoints/week3-distilbert`.

Expected files inside `best-model/`: `config.json`, `model.safetensors` (or equivalent), tokenizer files (`tokenizer.json`, `tokenizer_config.json`, …).

## Checkpoint resolution for serving

[`app/core/config.py`](../app/core/config.py) `resolve_model_path`:

1. `MODEL_PATH` / `settings.model_path`
2. `./checkpoints/best-model`
3. `./checkpoints/week3-distilbert/best-model`
4. `./checkpoints/week2-distilbert/best-model`

Common promotion pattern:

```bash
ln -sfn week3-distilbert/best-model checkpoints/best-model
# or after Colab:
ln -sfn production-distilbert/best-model checkpoints/best-model
```

`SentimentPredictor` lazy-loads on first `predict()`. Restart uvicorn after swapping weights if a process already loaded the old model.

## Local vs Colab

| Path | Experiment name | Typical output |
|------|-----------------|----------------|
| Local `train.py` | `review-intelligence-week3` | `checkpoints/week3-distilbert/` |
| Colab notebook | `review-intelligence-colab` | `production-distilbert/best-model` then export |

Export steps: [COLAB_EXPORT.md](../notebooks/COLAB_EXPORT.md).

## Local observation (smoke quality)

Tiny or mid-end runs can **collapse** toward the majority “positive” class and report low macro F1 (e.g. ~0.27) even when accuracy looks acceptable. Treat that as a training-scale / imbalance issue, not an API bug. Prefer Colab with the full ~40k train split for resume-quality metrics.

## Inference contract

[`src/inference/predict.py`](../src/inference/predict.py):

- Device: CUDA only if `DEVICE=cuda` **and** `torch.cuda.is_available()`; else CPU
- Empty text → `ValueError` (API maps to 422)
- Missing path → `FileNotFoundError` (API maps to 503)
- Output: `label_id`, `label`, `confidence`, per-class `scores`

Smoke:

```bash
python -c "from src.inference.predict import predict_text; print(predict_text('Battery life is excellent'))"
```

## Reproducibility gaps (honest)

| Gap | Impact |
|-----|--------|
| Unpinned `requirements.txt` | Installs drift over time |
| Torch installed separately | Document which wheel you used |
| `pyarrow` not listed | Usually pulled by `datasets`; Colab installs explicitly |
| Colab duplicates preprocess | Drift risk vs `src/data/preprocess.py` |
| MLflow / checkpoints gitignored | Machine-local unless you export |
| No `evaluate.py` quality gate | CI cannot enforce F1 ≥ 0.90 yet |
| mid-end silent subsample | Easy to confuse with “full 50k train” |

## Related stubs

- `src.training.evaluate` — **does not exist** (mentioned in old plan only)
- Docker packaging of models — **not implemented**

Next: [RAG.md](RAG.md).
