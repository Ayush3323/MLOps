# Glossary

| Term | Meaning in this project |
|------|-------------------------|
| **DistilBERT** | Compact transformer used as the sentiment classifier base (`distilbert-base-uncased`) |
| **Fine-tuning** | Updating a pretrained model on labeled Amazon review sentiment |
| **F1 macro** | Average of per-class F1 scores; primary model selection metric |
| **Label map** | `0=negative`, `1=neutral`, `2=positive` from star ratings |
| **Checkpoint** | Saved model + tokenizer directory under `checkpoints/` |
| **MLflow** | Experiment tracking UI/store under `experiments/mlruns/` |
| **Embedding** | Dense vector representing review meaning (MiniLM) |
| **ChromaDB** | Local vector database storing embeddings + metadata |
| **Collection** | Named Chroma table (default `product_reviews`) |
| **RAG** | Retrieve relevant reviews, then answer (extractive or LLM) |
| **Extractive mode** | Answer built from retrieved snippets without an LLM |
| **Sentiment filter** | Metadata filter on `negative` / `neutral` / `positive` during query |
| **DVC** | Data Version Control — tracks large parquet via `.dvc` sidecars |
| **Sidecar** | Small `.dvc` file committed to Git pointing at cached data hashes |
| **Singleton DI** | One shared predictor/pipeline instance per process (`deps.py`) |
| **Cold start** | First request loads model/embedder into memory |
| **mid-end / high-end** | Training profiles controlling sample caps and epochs |
| **Parquet** | Columnar file format for train/val/test splits |
| **Stratified split** | Train/val/test preserve class proportions |
| **Imbalance** | Majority positive reviews (~80% in the 50k local set) |
| **Quality gate** | CI rule that fails if metrics drop below a threshold (planned, not built) |
| **HF Hub** | Hugging Face model/dataset hosting used for downloads |
| **Inference API** | Hosted LLM HTTP API (Hugging Face) used optionally for RAG answers |

Back to the [documentation index](README.md).
