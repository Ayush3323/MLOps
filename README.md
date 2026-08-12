# Review Intelligence System (repository root)

Sentiment-aware product review intelligence: fine-tune DistilBERT, serve predictions over FastAPI, and answer natural-language questions over reviews with RAG (ChromaDB + MiniLM embeddings).

## Layout


| Path                             | Role                                                       |
| -------------------------------- | ---------------------------------------------------------- |
| `[project/](project/)`           | Application code, tests, notebooks, local data/checkpoints |
| `[project/docs/](project/docs/)` | High-to-low documentation suite (start here for learning)  |
| `[DVC.md](DVC.md)`               | Data Version Control commands (run from this directory)    |
| `[.dvc/](.dvc/)`                 | DVC metadata (remote not configured yet)                   |
| `venv/`                          | Local Python virtualenv (gitignored)                       |


Git and DVC live at **this** directory. Day-to-day Python commands run with cwd = `[project/](project/)`.

```text
/home/.../ML/                 ← git + DVC root (you are here)
├── DVC.md
├── .dvc/
├── venv/                     ← optional local env
└── project/                  ← FastAPI + ML package
    ├── README.md             ← quickstart
    ├── docs/                 ← full documentation
    ├── app/                  ← HTTP API
    ├── src/                  ← preprocess / train / infer / RAG
    ├── tests/
    ├── notebooks/
    └── data/processed/       ← parquet (+ *.dvc sidecars)
```



## Start reading

1. [Project quickstart](project/README.md) — setup, run API, status
2. [Documentation index](project/docs/README.md) — guided high → low path
3. [Overview](project/docs/OVERVIEW.md) — what the system does
4. [Architecture](project/docs/ARCHITECTURE.md) — components and data flow



## Quick commands

```bash
# From repo root
cd project
source ../venv/bin/activate   # or create: python3 -m venv ../venv

# Install (CPU Torch) — see project/INSTALL.md
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Interactive API docs (when server is up): [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

## Current state (summary)


| Area                                                            | Status                                   |
| --------------------------------------------------------------- | ---------------------------------------- |
| Preprocess → train → infer → RAG API                            | Implemented                              |
| FastAPI (`/health`, `/api/analyze`, `/api/index`, `/api/query`) | Implemented                              |
| DVC pointers for processed parquet                              | Present locally; remote not configured   |
| Docker / CI / monitoring                                        | Not implemented (placeholders / roadmap) |


For the original 6-week learning plan (aspirational; partly outdated), see `[project/Project.md](project/Project.md)`.