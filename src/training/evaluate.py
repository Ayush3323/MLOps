from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from app.core.config import get_settings
from src.training.train import compute_metrics


def parse_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description="Score a trained checkpoint against a processed data split"
    )
    parser.add_argument("--model", type=str, default=settings.resolved_model_path)
    parser.add_argument("--data-dir", type=str, default=settings.processed_data_dir)
    parser.add_argument("--split", type=str, default="test", choices=["validation", "test"])
    parser.add_argument("--max-length", type=int, default=settings.max_length)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Exit with status 1 if f1_macro on the split is below this value",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to write the metrics as JSON",
    )
    return parser.parse_args()


def load_split(data_dir: str, split: str) -> Dataset:
    path = Path(data_dir) / f"{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"No {split} split at {path}. Run `python -m src.data.preprocess` first."
        )
    return load_dataset("parquet", data_files={split: str(path)})[split]


def tokenize_split(dataset: Dataset, tokenizer: AutoTokenizer, max_length: int) -> Dataset:
    def _tokenize(batch: dict[str, list]) -> dict:
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    tokenized = dataset.map(_tokenize, batched=True)
    tokenized = tokenized.rename_column("label", "labels")
    keep_columns = {"input_ids", "attention_mask", "labels"}
    remove_cols = [c for c in tokenized.column_names if c not in keep_columns]
    if remove_cols:
        tokenized = tokenized.remove_columns(remove_cols)
    return tokenized


def evaluate(
    model_path: str,
    data_dir: str,
    split: str,
    max_length: int,
    batch_size: int,
) -> dict[str, float]:
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    dataset = tokenize_split(load_split(data_dir, split), tokenizer, max_length)

    with tempfile.TemporaryDirectory() as scratch_dir:
        trainer = Trainer(
            model=model,
            args=TrainingArguments(
                output_dir=scratch_dir,
                per_device_eval_batch_size=batch_size,
                report_to=[],
            ),
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
            compute_metrics=compute_metrics,
        )
        raw_metrics = trainer.evaluate(eval_dataset=dataset, metric_key_prefix=split)

    return {k: float(v) for k, v in raw_metrics.items()}


def main() -> None:
    args = parse_args()
    metrics = evaluate(args.model, args.data_dir, args.split, args.max_length, args.batch_size)

    print(f"Model: {args.model}")
    print(f"Split: {args.split} ({args.data_dir})")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    if args.output:
        Path(args.output).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"Wrote metrics to {args.output}")

    if args.threshold is not None:
        f1_key = f"{args.split}_f1_macro"
        f1 = metrics.get(f1_key)
        if f1 is None or f1 < args.threshold:
            print(f"FAIL: {f1_key}={f1} is below threshold {args.threshold}")
            raise SystemExit(1)
        print(f"PASS: {f1_key}={f1} >= threshold {args.threshold}")


if __name__ == "__main__":
    main()
