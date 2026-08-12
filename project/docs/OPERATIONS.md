# Operations

## Current deployment model

**Host-run only:** activate venv, start Uvicorn on the laptop/server that has checkpoints and (optionally) `chroma_db/`.

Not implemented: Docker image (empty [`docker/Dockerfile`](../docker/Dockerfile)), Compose, K8s, reverse proxy TLS, process supervisors beyond what you add yourself.

## Artifact locations (cwd = `project/`)

| Artifact | Path | Git |
|----------|------|-----|
| Checkpoints | `checkpoints/` | Ignored |
| Active model symlink | `checkpoints/best-model` | Ignored |
| Vector DB | `chroma_db/` | Ignored |
| MLflow | `experiments/mlruns/` | Ignored (`experiments/`) |
| Processed data | `data/processed/*.parquet` | Ignored; track via `*.dvc` |
| Secrets | `.env` | Ignored |

## Day-2 tasks

### Swap model

```bash
# After placing new weights
ln -sfn production-distilbert/best-model checkpoints/best-model
# restart uvicorn — singleton may hold old weights
```

Or set `MODEL_PATH` in `.env` and restart.

### Rebuild Chroma

```bash
rm -rf chroma_db   # destructive
# restart API, re-POST /api/index with desired corpus
```

### Backup suggestions

- Copy `checkpoints/*/best-model/` and `data/processed/*.parquet` to external storage
- Export MLflow runs if you care about history across machines
- Do not rely on Git for weights or parquet

### Health checks

`GET /health` confirms the process is up and echoes `device` / env. It does **not** check:

- checkpoint existence
- successful model load
- Chroma readability
- LLM API keys

Add deeper checks later if you productionize.

## Security posture (dev)

| Control | Status |
|---------|--------|
| AuthN / AuthZ | None |
| CORS | Unconfigured |
| Rate limits | None |
| TLS | Bring your own reverse proxy |
| Error detail | 500 responses may include exception strings |
| Secrets | Stay in `.env`; never send client-side |

Assume the API is **trusted-network / localhost** until hardened.

## Observability

| Signal | Status |
|--------|--------|
| Structured request logs | Not implemented |
| Metrics / Prometheus | Not implemented |
| Tracing | Not implemented |
| Training metrics | MLflow only |

## Data sharing

Use DVC remotes when ready ([DVC.md](../../DVC.md)). Until then, share parquet out-of-band or regenerate with preprocess.

## Process notes

- First request after start may be slow (model + embedder load).
- `src` depends on `app.core.config` — keep cwd and `.env` consistent.
- Empty placeholders `app/ml/loader.py` and `docker/Dockerfile` are not operational entrypoints.

Next: [TROUBLESHOOTING.md](TROUBLESHOOTING.md).
