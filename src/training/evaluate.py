from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from datasets import Dataset, load_dataset
from sklearn.metrics import classification_report, confusion_matrix
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from app.core.config import get_settings
from src.training.train import LABEL_NAMES, compute_metrics


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
    parser.add_argument(
        "--per-class",
        action="store_true",
        help="Also report per-class precision/recall/F1 and a confusion matrix",
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
    per_class: bool = False,
) -> dict[str, Any]:
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
        # predict() runs the same single forward pass as evaluate() but also
        # returns raw logits/labels, so per-class stats don't need a second pass.
        output = trainer.predict(dataset, metric_key_prefix=split)

    metrics: dict[str, Any] = {k: float(v) for k, v in output.metrics.items()}

    if per_class:
        preds = np.argmax(output.predictions, axis=-1)
        labels = output.label_ids
        target_names = [LABEL_NAMES[i] for i in sorted(LABEL_NAMES)]
        report = classification_report(
            labels, preds, target_names=target_names, output_dict=True, zero_division=0
        )
        metrics["per_class_report"] = report
        metrics["confusion_matrix"] = {
            "labels": target_names,
            "matrix": confusion_matrix(labels, preds).tolist(),
        }

    return metrics


def main() -> None:
    args = parse_args()
    metrics = evaluate(
        args.model, args.data_dir, args.split, args.max_length, args.batch_size, args.per_class
    )

    print(f"Model: {args.model}")
    print(f"Split: {args.split} ({args.data_dir})")
    for key, value in metrics.items():
        if key in ("per_class_report", "confusion_matrix"):
            continue
        print(f"  {key}: {value:.4f}")

    if "per_class_report" in metrics:
        print("\nPer-class report:")
        for label, stats in metrics["per_class_report"].items():
            if not isinstance(stats, dict):
                continue
            print(
                f"  {label:12} precision={stats['precision']:.3f} "
                f"recall={stats['recall']:.3f} f1={stats['f1-score']:.3f} "
                f"support={int(stats['support'])}"
            )

        cm = metrics["confusion_matrix"]
        col_labels = cm["labels"]
        print("\nConfusion matrix (rows=true, cols=predicted):")
        print("              " + "".join(f"{l:>10}" for l in col_labels))
        for true_label, row in zip(col_labels, cm["matrix"]):
            print(f"  {true_label:10}" + "".join(f"{v:10d}" for v in row))

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
