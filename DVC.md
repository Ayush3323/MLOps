# DVC quick reference

Working directory: the **repository root**, e.g. `/path/to/ML`.

`.dvc/` already exists. Processed parquet under `data/processed/` is gitignored — track it with DVC.
Deeper data docs: [`docs/DATA.md`](docs/DATA.md).

## Current state

| Item | Status |
|------|--------|
| DVC initialized | Yes (`.dvc/`) |
| Remote | **Not configured** — `dvc push`/`pull` need a remote first |
| Local sidecars | `data/processed/*.parquet.dvc` may already exist after `dvc add` |
| Pipeline file | No `dvc.yaml` yet — manual `dvc add` only |

## One-time / first track

```bash
cd /path/to/ML
source venv/bin/activate
pip install dvc

# Ensure DVC is initialized (safe if already done)
dvc status || dvc init

dvc add data/processed/train.parquet
dvc add data/processed/validation.parquet
dvc add data/processed/test.parquet

git add \
  data/processed/*.dvc \
  .dvc \
  .gitignore
# If dvc add created data/processed/.gitignore, add that too:
# git add data/processed/.gitignore

git commit -m "Track processed review splits with DVC"
```

## After regenerating data (e.g. 50k samples)

```bash
cd /path/to/ML
source venv/bin/activate
python -m src.data.preprocess --max-samples 50000 --output-dir data/processed

dvc add data/processed/train.parquet
dvc add data/processed/validation.parquet
dvc add data/processed/test.parquet
git add data/processed/*.dvc
git commit -m "Update DVC pointers for regenerated processed splits"
```

## Optional remote (another machine / cloud)

```bash
dvc remote add -d localremote /path/to/dvc-storage
# or: dvc remote add -d s3remote s3://your-bucket/ml-data
dvc push
# on another machine after git pull:
dvc pull
```

Without a remote, DVC still records local content hashes; collaborators must copy parquet or re-run preprocess.

Do **not** `dvc add` model checkpoints unless you have a large remote — prefer copying `best-model` manually from Colab (see [`notebooks/COLAB_EXPORT.md`](notebooks/COLAB_EXPORT.md)).

## Related

- [docs/DATA.md](docs/DATA.md) — schema, imbalance, tokenize caveat
- [docs/WORKFLOWS.md](docs/WORKFLOWS.md) — end-to-end commands
- [README.md](README.md) — repository landing page
