"""Week 1 preprocessing pipeline for Amazon Reviews sentiment training."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer

ELECTRONICS_REVIEWS_URL = (
    "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/"
    "review_categories/Electronics.jsonl.gz"
)


def map_rating_to_label(rating: float) -> int:
    """Map rating to sentiment class: 0=negative, 1=neutral, 2=positive."""
    if rating <= 2:
        return 0
    if rating == 3:
        return 1
    return 2


def _coerce_text(example: dict[str, Any]) -> str:
    text = str(example.get("text") or "").strip()
    title = str(example.get("title") or "").strip()
    if text:
        return text
    return title


def load_and_clean(max_samples: int = 1000) -> Dataset:
    """Load Amazon electronics reviews and produce `text` + `label` columns."""
    stream = load_dataset(
        "json",
        data_files=ELECTRONICS_REVIEWS_URL,
        split="train",
        streaming=True,
    )

    rows: list[dict[str, Any]] = []
    limit = max_samples if max_samples and max_samples > 0 else None
    for i, record in enumerate(stream):
        rows.append(record)
        if limit is not None and i + 1 >= limit:
            break

    dataset = Dataset.from_list(rows)

    def transform(example: dict[str, Any]) -> dict[str, Any]:
        rating = float(example["rating"])
        text = _coerce_text(example)
        return {
            "text": text,
            "label": map_rating_to_label(rating),
            "rating": rating,
            "parent_asin": str(example.get("parent_asin", "")),
        }

    cleaned = dataset.map(transform)
    cleaned = cleaned.filter(lambda ex: len(ex["text"].strip()) > 0)
    keep_cols = ["text", "label", "rating", "parent_asin"]
    drop_cols = [c for c in cleaned.column_names if c not in keep_cols]
    if drop_cols:
        cleaned = cleaned.remove_columns(drop_cols)
    return cleaned


def split_dataset(dataset: Dataset, seed: int = 42) -> DatasetDict:
    """Create stratified train/val/test split (80/10/10)."""
    rows = dataset.to_list()
    labels = [row["label"] for row in rows]

    train_rows, temp_rows = train_test_split(
        rows,
        test_size=0.2,
        random_state=seed,
        stratify=labels,
    )

    temp_labels = [row["label"] for row in temp_rows]
    val_rows, test_rows = train_test_split(
        temp_rows,
        test_size=0.5,
        random_state=seed,
        stratify=temp_labels,
    )

    return DatasetDict(
        {
            "train": Dataset.from_list(train_rows),
            "validation": Dataset.from_list(val_rows),
            "test": Dataset.from_list(test_rows),
        }
    )


def tokenize(dataset: Dataset, max_length: int = 256) -> Dataset:
    """Tokenize text for DistilBERT."""
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    def _tokenize(batch: dict[str, list[Any]]) -> dict[str, Any]:
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    return dataset.map(_tokenize, batched=True)


def save_splits(splits: DatasetDict, output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for split_name, ds in splits.items():
        ds.to_parquet(output / f"{split_name}.parquet")


def label_distribution(dataset: Dataset) -> dict[int, int]:
    counts: dict[int, int] = {0: 0, 1: 0, 2: 0}
    for label in dataset["label"]:
        counts[int(label)] += 1
    return counts


def run_sanity(max_samples: int = 1000, output_dir: str = "data/processed") -> None:
    raw = load_and_clean(max_samples=max_samples)
    splits = split_dataset(raw)
    save_splits(splits, output_dir)

    print(f"Loaded {len(raw)} cleaned rows")
    for name in ["train", "validation", "test"]:
        ds = splits[name]
        print(f"{name}: {len(ds)} rows, label_distribution={label_distribution(ds)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare sentiment dataset splits")
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--output-dir", type=str, default="data/processed")
    parser.add_argument("--tokenize", action="store_true")
    parser.add_argument("--max-length", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw = load_and_clean(max_samples=args.max_samples)
    splits = split_dataset(raw)

    if args.tokenize:
        splits = DatasetDict(
            {k: tokenize(v, max_length=args.max_length) for k, v in splits.items()}
        )

    save_splits(splits, args.output_dir)

    print(f"Saved splits to {args.output_dir}")
    for split_name, ds in splits.items():
        print(f"{split_name}: {len(ds)} rows")


if __name__ == "__main__":
    main()
