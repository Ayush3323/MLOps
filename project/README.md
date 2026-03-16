# Review Intelligence System

Sentiment-aware product review system: fine-tuned DistilBERT, RAG + vector search, FastAPI. See [Project.md](Project.md) for the full plan.

## Quick start

- **Install:** See [INSTALL.md](INSTALL.md) (CPU-only vs GPU).
- **Run API (later):** `uvicorn app.main:app --reload` from project root.

---

## What to do next (Week 1)

Requirements are installed. Do these in order:

### 1. Initialize Git and DVC

```bash
cd ~/Desktop/MLops/project
git init
dvc init
```

Commit the initial structure (venv stays outside project or is gitignored):

```bash
git add .
git commit -m "Initial structure, gitignore, requirements"
```

### 2. Add config and env template

- **`app/core/config.py`** — Load settings from environment (e.g. with `pydantic-settings`): `MODEL_PATH`, `CHROMA_PATH`, `MLFLOW_URI`, optional `OPENAI_API_KEY` (for RAG), and hardware-related: `MAX_SAMPLES`, `BATCH_SIZE`, `DEVICE` (e.g. `cpu`).
- **`.env.example`** — List variable names and example values (no real secrets). Copy to `.env` and fill in locally; `.env` is gitignored.

### 3. Data pipeline (preprocessing)

- **`src/data/preprocess.py`** — Implement:
  - `load_and_clean(split, max_samples)` — Load `McAuley-Lab/Amazon-Reviews-2023`, subset `raw_review_Electronics`; map star rating to label: 1–2 → 0, 3 → 1, 4–5 → 2.
  - Train/val/test split: 80/10/10, **stratified by label** (before any other processing).
  - Optional: `tokenize(dataset, tokenizer, max_length=256)` with `DistilBertTokenizer`.
- **Sanity check:** Run with `max_samples=1000`, confirm label distribution and split proportions. Use `data/processed/` for saved splits; later you can track with DVC.

### 4. Optional but useful

- **EDA:** Small script or notebook that loads a sample and plots rating/label distribution and review length.
- **Tests:** `tests/test_src/data/test_preprocess.py` — test label mapping and stratification.

---

## Week 1 exit criteria

- [ ] Git and DVC inited, first commit done.
- [ ] `app/core/config.py` and `.env.example` in place.
- [ ] `src/data/preprocess.py` loads Electronics reviews, maps labels, produces stratified train/val/test (and optionally tokenizes).
- [ ] Sanity run with 1000 samples succeeds.

Then you’re set for Week 2: fine-tuning and MLflow.
