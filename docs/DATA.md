# Data

## Source

Preprocessing streams Amazon Reviews 2023 **Electronics** JSONL (gzip) from:

`https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Electronics.jsonl.gz`

Defined in [`src/data/preprocess.py`](../src/data/preprocess.py) as `ELECTRONICS_REVIEWS_URL`.  
This is a direct JSON stream via HuggingFace `datasets` (`load_dataset("json", ...)`), **not** the Hub config string described in the old learning plan.

`data/raw/` is empty (placeholder only). Raw snapshots are not DVC-tracked today.

## Cleaning rules

| Step | Behavior |
|------|----------|
| Text | Prefer `text`; fallback to `title`; strip whitespace |
| Drop | Rows with empty text after coercion |
| Rating → label | `≤2 → 0` (negative), `3 → 1` (neutral), `≥4 → 2` (positive) |
| Kept columns | `text`, `label`, `rating`, `parent_asin` |

## Splits

- Stratified **80% / 10% / 10%** train / validation / test
- `sklearn.model_selection.train_test_split` with `random_state=42`
- Written as parquet: `train.parquet`, `validation.parquet`, `test.parquet`

## Schema contract (required by training)

| Column | Type | Required for train? | Notes |
|--------|------|---------------------|-------|
| `text` | string | **Yes** | Re-tokenized in `train.py` |
| `label` | int `{0,1,2}` | **Yes** | Renamed to `labels` after tokenize |
| `rating` | float | No | Dropped before Trainer |
| `parent_asin` | string | No | Product id; dropped before Trainer |

Label names are consistent across preprocess, train (`LABEL_NAMES`), and inference (`LABEL_MAP`).

## Local observation (this workspace)

Verified on disk under `data/processed/` after a 50k preprocess:

| Split | Rows | Labels `{0,1,2}` |
|-------|------|------------------|
| train | 40,000 | `{5009, 2729, 32262}` |
| validation | 5,000 | `{626, 341, 4033}` |
| test | 5,000 | `{626, 342, 4032}` |

~80% positive — strong class imbalance. Macro F1 is the training selection metric for this reason. No class weights / resampling are implemented yet.

## CLI

```bash
python -m src.data.preprocess --max-samples 50000 --output-dir data/processed
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--max-samples` | 1000 | Cap streamed rows (`0`/negative treated as unlimited in code path via `None` only when falsy — prefer positive caps) |
| `--output-dir` | `data/processed` | Output directory |
| `--tokenize` | off | Pre-tokenize with DistilBERT |
| `--max-length` | 256 | Used only with `--tokenize` |

### Important caveat: `--tokenize`

`train.py` expects raw `text` and tokenizes itself. Parquet produced with `--tokenize` will **break** training unless you change the loader. Leave `--tokenize` off for the standard workflow.

## DVC status

| Item | Status |
|------|--------|
| DVC init | Repo root `.dvc/` exists |
| Remote | **Not configured** (`.dvc/config` empty / no remote) |
| Sidecars | `data/processed/*.parquet.dvc` present locally |
| Pipeline | No `dvc.yaml` — manual `dvc add` |
| Raw data | Not tracked |

Commands: [DVC.md](../DVC.md). After regenerating parquet, re-run `dvc add` on the three files and commit the `.dvc` pointers.

Collaborators without a remote must copy parquet (or re-run preprocess) manually.

## Regeneration checklist

1. Re-run preprocess with desired `--max-samples`
2. Confirm row counts / label distributions
3. Update DVC pointers from repo root
4. Re-train if the training set changed materially
5. Rebuild Chroma index if review corpus for RAG should match new data (indexing is independent API traffic, not automatic from parquet)

Next: [MODEL_LIFECYCLE.md](MODEL_LIFECYCLE.md).
