# DVC quick reference (repo root)

Working directory: `/home/hodorinfo/Desktop/ML`

`.dvc/` already exists. Processed parquet under `project/data/processed/` is gitignored — track it with DVC.

## One-time / first track

```bash
cd /home/hodorinfo/Desktop/ML
source venv/bin/activate
pip install dvc

# Ensure DVC is initialized (safe if already done)
dvc status || dvc init

cd /home/hodorinfo/Desktop/ML
dvc add project/data/processed/train.parquet
dvc add project/data/processed/validation.parquet
dvc add project/data/processed/test.parquet

git add \
  project/data/processed/train.parquet.dvc \
  project/data/processed/validation.parquet.dvc \
  project/data/processed/test.parquet.dvc \
  project/data/processed/.gitignore \
  .dvc \
  .gitignore
git commit -m "Track processed review splits with DVC"
```

## After regenerating data (e.g. 50k samples)

```bash
cd /home/hodorinfo/Desktop/ML/project
source ../venv/bin/activate
python -m src.data.preprocess --max-samples 50000 --output-dir data/processed

cd /home/hodorinfo/Desktop/ML
dvc add project/data/processed/train.parquet
dvc add project/data/processed/validation.parquet
dvc add project/data/processed/test.parquet
git add project/data/processed/*.dvc
git commit -m "Update DVC pointers for 50k processed splits"
```

## Optional remote (another machine / cloud)

```bash
dvc remote add -d localremote /path/to/dvc-storage
# or: dvc remote add -d s3remote s3://your-bucket/ml-data
dvc push
# on another machine after git pull:
dvc pull
```

Do **not** `dvc add` model checkpoints unless you have a large remote — prefer copying `best-model` manually from Colab (see `project/notebooks/COLAB_EXPORT.md`).
