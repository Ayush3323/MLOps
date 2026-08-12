# Overview

## What this project does

The **Review Intelligence System** turns product reviews into:

1. **Sentiment labels** — DistilBERT classifies text as negative / neutral / positive.
2. **Searchable memory** — reviews are embedded and stored in ChromaDB.
3. **Natural-language answers** — users ask questions; the system retrieves relevant reviews and returns an extractive or LLM-generated answer (RAG).

Example: *“What do customers complain about in negative reviews?”* → retrieve negative reviews about the topic → answer with citations to those reviews.

## Who it is for

| Audience | Use |
|----------|-----|
| You / learners | End-to-end MLOps practice: data → train → serve → RAG |
| API clients | HTTP consumers of `/api/analyze`, `/api/index`, `/api/query` |
| Future collaborators | Documented layout, schemas, and workflows |

There is **no web UI** in the current codebase — clients are curl, scripts, or any HTTP tool. Interactive exploration uses FastAPI’s `/docs`.

## Capabilities (implemented)

- Stream Amazon Electronics reviews → clean → stratified 80/10/10 parquet splits
- Fine-tune `distilbert-base-uncased` (3-class) with HuggingFace `Trainer` + MLflow
- Load a local checkpoint and predict sentiment
- Index reviews (sentiment + embeddings) into a local Chroma collection
- Query with optional sentiment filter; answer modes: **OpenAI → Hugging Face Inference → extractive**
- Unit/API tests with mocked model and vector store dependencies

## Non-goals / not implemented yet

- Django REST Framework (original plan); **FastAPI** is the real stack
- Production Docker Compose / Kubernetes
- Auth, CORS policy, rate limiting
- Prometheus / Grafana monitoring or drift detection
- Dedicated `src.training.evaluate` quality-gate module
- CI workflows (`.github/workflows`)
- Class rebalancing / weighted loss for the ~80% positive skew
- Official UI or admin dashboard

## Technology choices (current)

| Concern | Choice | Why |
|---------|--------|-----|
| API | FastAPI + Uvicorn | Lightweight async-capable HTTP, auto OpenAPI |
| Classifier | DistilBERT (HF Transformers) | Small, strong text baseline for fine-tuning |
| Experiment tracking | MLflow (local file store) | Compare training runs |
| Embeddings | `all-MiniLM-L6-v2` | Fast CPU-friendly semantic vectors |
| Vector DB | Chroma PersistentClient | Local disk, no separate server required |
| Data versioning | DVC at repo root | Keep large parquet out of Git |
| GPU training | Colab notebook | Mid-end laptops stay CPU-only for serving |

## Repository shape

- **Git/DVC root:** parent of `project/`
- **App package:** `project/` (`app/` HTTP, `src/` ML)
- **Docs:** [`README.md`](README.md) in this folder

Next: [ARCHITECTURE.md](ARCHITECTURE.md).
