# Documentation index

Read top → bottom for a high-level-to-low-level understanding of the Review Intelligence System.

## Recommended path

| Step | Doc | What you learn |
|------|-----|----------------|
| 1 | [OVERVIEW.md](OVERVIEW.md) | Problem, capabilities, scope, tech choices |
| 2 | [ARCHITECTURE.md](ARCHITECTURE.md) | System diagrams, runtime vs offline paths |
| 3 | [CODEBASE_GUIDE.md](CODEBASE_GUIDE.md) | Where every important file lives |
| 4 | [WORKFLOWS.md](WORKFLOWS.md) | End-to-end commands from data → API |
| 5 | [DATA.md](DATA.md) | Dataset schema, splits, DVC, imbalance |
| 6 | [MODEL_LIFECYCLE.md](MODEL_LIFECYCLE.md) | Training profiles, metrics, checkpoints |
| 7 | [RAG.md](RAG.md) | Indexing, retrieval, answer modes |
| 8 | [API.md](API.md) | Request/response contracts |
| 9 | [CONFIGURATION.md](CONFIGURATION.md) | Environment variables |
| 10 | [TESTING.md](TESTING.md) | What tests cover (and what they do not) |
| 11 | [OPERATIONS.md](OPERATIONS.md) | Artifacts, model swap, ops limits |
| 12 | [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | Common failures |
| 13 | [GLOSSARY.md](GLOSSARY.md) | Shared vocabulary |

## Related operational docs (outside `docs/`)

| Doc | Role |
|-----|------|
| [../README.md](../README.md) | Project quickstart + status checklist |
| [../README.md](../README.md) | Repository root landing page |
| [../INSTALL.md](../INSTALL.md) | CPU vs GPU install |
| [../DVC.md](../DVC.md) | DVC track / update / remote |
| [../notebooks/COLAB_EXPORT.md](../notebooks/COLAB_EXPORT.md) | Colab checkpoint → local |
| [../Project.md](../Project.md) | Historical / aspirational learning plan |

## How this documentation is written

- **Current state** describes what the code does today (FastAPI stack).
- **Local observations** call out machine-specific artifacts (e.g. 50k parquet on disk).
- **Roadmap** items (Docker, CI quality gates, monitoring) are marked as not implemented.
- Prefer these docs over outdated Django / Compose sections in `Project.md`.
