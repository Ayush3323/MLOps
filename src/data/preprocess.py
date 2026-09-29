"""
Week 1 preprocessing pipeline for Amazon Reviews sentiment training.

This script loads a streaming dataset of Amazon Electronics reviews from a public URL,
cleans and transforms it, splits it into train/validation/test sets (stratified by label),
and optionally tokenizes the text for DistilBERT. The final splits are saved as Parquet files.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer

# Public URL for the Amazon Electronics reviews JSON Lines gzip file (2023 release)
ELECTRONICS_REVIEWS_URL = (
    "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/"
    "review_categories/Electronics.jsonl.gz"
)


def map_rating_to_label(rating: float) -> int:
    """
    Convert a numeric rating (1-5) into a ternary sentiment class.
    Args:
        rating: Star rating (float, typically 1.0 - 5.0).
    Returns:
        0 for negative (1-2 stars), 1 for neutral (3 stars), 2 for positive (4-5 stars).
    """
    if rating <= 2:
        return 0
    if rating == 3:
        return 1
    return 2


def _coerce_text(example: dict[str, Any]) -> str:
    """
    Extract a usable text field from a raw review record.
    The dataset may have 'text' (review body) and 'title' (review title).
    Prefer the body; if missing, fall back to the title. If both are missing,
    an empty string will be returned (filtered out later).

    Args:
        example: A single raw record from the dataset.

    Returns:
        A non‑empty string if possible, otherwise an empty string.
    """
    text = str(example.get("text") or "").strip()
    title = str(example.get("title") or "").strip()
    if text:
        return text
    return title


def load_and_clean(max_samples: int = 1000) -> Dataset:
    """
    Load a sample of the Amazon Electronics reviews and perform basic cleaning.

    Steps:
        1. Load the JSON Lines dataset in streaming mode to avoid downloading everything.
        2. Take up to `max_samples` records (or all if max_samples <= 0).
        3. Convert each record to a standard format with 'text', 'label', 'rating',
           and 'parent_asin' (product ID).
        4. Discard records with empty text after coercion.
        5. Drop all other columns to keep the dataset lean.

    Args:
        max_samples: Maximum number of records to load. Use None or <=0 for all.

    Returns:
        A Hugging Face Dataset with columns: text, label, rating, parent_asin.
    """
    # Stream the JSONL file – this returns an iterable of dictionaries
    stream = load_dataset(
        "json",
        data_files=ELECTRONICS_REVIEWS_URL,
        split="train",
        streaming=True,
    )

    # Materialize a limited number of rows into a list
    rows: list[dict[str, Any]] = []
    limit = max_samples if max_samples and max_samples > 0 else None
    for i, record in enumerate(stream):
        rows.append(record)
        if limit is not None and i + 1 >= limit:
            break

    # Create an in‑memory Dataset from the list
    dataset = Dataset.from_list(rows)

    def transform(example: dict[str, Any]) -> dict[str, Any]:
        """Apply rating→label mapping and text extraction to one record."""
        rating = float(example["rating"])
        text = _coerce_text(example)
        return {
            "text": text,
            "label": map_rating_to_label(rating),
            "rating": rating,
            "parent_asin": str(example.get("parent_asin", "")),
        }

    # Apply the transformation and remove rows with empty text
    cleaned = dataset.map(transform)
    cleaned = cleaned.filter(lambda ex: len(ex["text"].strip()) > 0)

    # Keep only the columns we actually need
    keep_cols = ["text", "label", "rating", "parent_asin"]
    drop_cols = [c for c in cleaned.column_names if c not in keep_cols]
    if drop_cols:
        cleaned = cleaned.remove_columns(drop_cols)
    return cleaned


def split_dataset(dataset: Dataset, seed: int = 42) -> DatasetDict:
    """
    Split a dataset into stratified train/validation/test sets.

    The split ratios are 80% train, 10% validation, 10% test.
    Stratification ensures that the class distribution (0/1/2) is preserved
    in each split.

    Args:
        dataset: A Hugging Face Dataset with a 'label' column.
        seed: Random seed for reproducibility.

    Returns:
        A DatasetDict with 'train', 'validation', and 'test' keys.
    """
    rows = dataset.to_list()
    labels = [row["label"] for row in rows]

    # First split: 80% train, 20% temporary (to be split into val/test)
    train_rows, temp_rows = train_test_split(
        rows,
        test_size=0.2,
        random_state=seed,
        stratify=labels,
    )

    # Second split: 50% of temp -> validation, 50% -> test
    temp_labels = [row["label"] for row in temp_rows]
    val_rows, test_rows = train_test_split(
        temp_rows,
        test_size=0.5,
        random_state=seed,
        stratify=temp_labels,
    )

    # Convert lists back to Datasets and wrap in a DatasetDict
    return DatasetDict(
        {
            "train": Dataset.from_list(train_rows),
            "validation": Dataset.from_list(val_rows),
            "test": Dataset.from_list(test_rows),
        }
    )


def tokenize(dataset: Dataset, max_length: int = 256) -> Dataset:
    """
    Tokenize the 'text' column using the DistilBERT tokenizer.

    The tokenizer truncates sequences to `max_length` and pads them to that length,
    producing input_ids and attention_mask ready for model input.

    Args:
        dataset: A Dataset with a 'text' column.
        max_length: Maximum token length for truncation/padding.

    Returns:
        The same dataset with added tokenizer output columns.
    """
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    def _tokenize(batch: dict[str, list[Any]]) -> dict[str, Any]:
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    # Batched mapping for efficiency
    return dataset.map(_tokenize, batched=True)


def save_splits(splits: DatasetDict, output_dir: str | Path) -> None:
    """
    Save each split in the DatasetDict as a Parquet file.

    The files are named '{split_name}.parquet' and placed under `output_dir`.

    Args:
        splits: DatasetDict containing 'train', 'validation', 'test'.
        output_dir: Directory where the Parquet files will be written.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for split_name, ds in splits.items():
        ds.to_parquet(output / f"{split_name}.parquet")


def label_distribution(dataset: Dataset) -> dict[int, int]:
    """
    Count the number of examples per class in a dataset.

    Useful for sanity checks to ensure stratification worked.

    Args:
        dataset: A Dataset with a 'label' column.

    Returns:
        A dict mapping label (0,1,2) to its frequency.
    """
    counts: dict[int, int] = {0: 0, 1: 0, 2: 0}
    for label in dataset["label"]:
        counts[int(label)] += 1
    return counts


def run_sanity(max_samples: int = 1000, output_dir: str = "data/processed") -> None:
    """
    Quick sanity run: load, split, save, and print summary statistics.

    This is a convenience function for testing the pipeline interactively.
    """
    raw = load_and_clean(max_samples=max_samples)
    splits = split_dataset(raw)
    save_splits(splits, output_dir)

    print(f"Loaded {len(raw)} cleaned rows")
    for name in ["train", "validation", "test"]:
        ds = splits[name]
        print(f"{name}: {len(ds)} rows, label_distribution={label_distribution(ds)}")


def parse_args() -> argparse.Namespace:
    """Set up command‑line argument parsing."""
    parser = argparse.ArgumentParser(description="Prepare sentiment dataset splits")
    parser.add_argument("--max-samples", type=int, default=1000,
                        help="Maximum number of records to load (default: 1000)")
    parser.add_argument("--output-dir", type=str, default="data/processed",
                        help="Directory to save the split Parquet files")
    parser.add_argument("--tokenize", action="store_true",
                        help="If set, apply DistilBERT tokenization before saving")
    parser.add_argument("--max-length", type=int, default=256,
                        help="Max token length for tokenization (default: 256)")
    return parser.parse_args()


def main() -> None:
    """
    Main entry point when the script is run directly.

    It loads data, creates splits, optionally tokenizes, and saves the results.
    """
    args = parse_args()

    # Load and clean the raw data
    raw = load_and_clean(max_samples=args.max_samples)
    # Perform stratified split
    splits = split_dataset(raw)

    # Optional tokenization
    if args.tokenize:
        splits = DatasetDict(
            {k: tokenize(v, max_length=args.max_length) for k, v in splits.items()}
        )

    # Save to disk
    save_splits(splits, args.output_dir)

    # Print summary
    print(f"Saved splits to {args.output_dir}")
    for split_name, ds in splits.items():
        print(f"{split_name}: {len(ds)} rows")


if __name__ == "__main__":
    main()