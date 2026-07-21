from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
from datasets import DatasetDict, load_dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from app.core.config import get_settings


MODEL_NAME = "distilbert-base-uncased"
LABEL_NAMES = {0: "negative", 1: "neutral", 2: "positive"}


@dataclass
class TrainConfig:
    profile: str
    data_dir: str
    output_dir: str
    model_name: str
    max_length: int
    num_epochs: int
    train_batch_size: int
    eval_batch_size: int
    learning_rate: float
    weight_decay: float
    seed: int
    max_train_samples: int | None
    max_eval_samples: int | None
    gradient_accumulation_steps: int
    warmup_ratio: float
    early_stopping_patience: int | None


def parse_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Train DistilBERT sentiment model")
    parser.add_argument("--profile", type=str, default="mid-end", choices=["mid-end", "high-end"])
    parser.add_argument("--data-dir", type=str, default=settings.processed_data_dir)
    parser.add_argument("--output-dir", type=str, default="./checkpoints/week3-distilbert")
    parser.add_argument("--model-name", type=str, default=MODEL_NAME)
    parser.add_argument("--max-length", type=int, default=settings.max_length)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--train-batch-size", type=int, default=settings.batch_size)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    return parser.parse_args()


def apply_profile_defaults(args: argparse.Namespace) -> argparse.Namespace:
    if args.profile == "mid-end":
        if args.num_epochs is None:
            args.num_epochs = 2
        if args.max_train_samples is None:
            args.max_train_samples = 2000
        if args.max_eval_samples is None:
            args.max_eval_samples = 400
    else:
        if args.num_epochs is None:
            args.num_epochs = 3
        if args.max_train_samples is None:
            args.max_train_samples = None
        if args.max_eval_samples is None:
            args.max_eval_samples = 2000
    return args


def build_config(args: argparse.Namespace) -> TrainConfig:
    return TrainConfig(
        profile=args.profile,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_name=args.model_name,
        max_length=args.max_length,
        num_epochs=args.num_epochs,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        early_stopping_patience=args.early_stopping_patience,
    )


def load_splits(data_dir: str) -> DatasetDict:
    data_path = Path(data_dir)
    files = {
        "train": str(data_path / "train.parquet"),
        "validation": str(data_path / "validation.parquet"),
        "test": str(data_path / "test.parquet"),
    }
    return load_dataset("parquet", data_files=files)


def maybe_limit_samples(splits: DatasetDict, cfg: TrainConfig) -> DatasetDict:
    if cfg.max_train_samples:
        n = min(cfg.max_train_samples, len(splits["train"]))
        splits["train"] = splits["train"].select(range(n))
    if cfg.max_eval_samples:
        n_val = min(cfg.max_eval_samples, len(splits["validation"]))
        n_test = min(cfg.max_eval_samples, len(splits["test"]))
        splits["validation"] = splits["validation"].select(range(n_val))
        splits["test"] = splits["test"].select(range(n_test))
    return splits


def tokenize_splits(splits: DatasetDict, tokenizer: AutoTokenizer, max_length: int) -> DatasetDict:
    def _tokenize(batch: dict[str, list[Any]]) -> dict[str, Any]:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
        )

    tokenized = splits.map(_tokenize, batched=True)
    tokenized = tokenized.rename_column("label", "labels")
    keep_columns = {"input_ids", "attention_mask", "labels"}
    for split in tokenized.keys():
        remove_cols = [c for c in tokenized[split].column_names if c not in keep_columns]
        if remove_cols:
            tokenized[split] = tokenized[split].remove_columns(remove_cols)
    return tokenized


def compute_metrics(eval_pred: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1_macro": float(f1_score(labels, preds, average="macro")),
    }


def train_and_evaluate(cfg: TrainConfig) -> dict[str, Any]:
    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment("review-intelligence-week3")

    splits = maybe_limit_samples(load_splits(cfg.data_dir), cfg)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    tokenized = tokenize_splits(splits, tokenizer, cfg.max_length)

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=3,
        id2label=LABEL_NAMES,
        label2id={v: k for k, v in LABEL_NAMES.items()},
    )

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    args = TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=20,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.train_batch_size,
        per_device_eval_batch_size=cfg.eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        warmup_ratio=cfg.warmup_ratio,
        num_train_epochs=cfg.num_epochs,
        weight_decay=cfg.weight_decay,
        seed=cfg.seed,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1_macro",
        greater_is_better=True,
        report_to=[],
    )

    callbacks = []
    if cfg.early_stopping_patience is not None and cfg.early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience))

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    with mlflow.start_run():
        mlflow.log_params(asdict(cfg))
        mlflow.log_param("mlflow_tracking_uri", settings.mlflow_tracking_uri)

        trainer.train()
        val_metrics = trainer.evaluate(eval_dataset=tokenized["validation"])
        test_metrics = trainer.evaluate(eval_dataset=tokenized["test"], metric_key_prefix="test")
        mlflow.log_metrics({k: float(v) for k, v in {**val_metrics, **test_metrics}.items()})

        # Save confusion-matrix-ready predictions for later analysis.
        test_pred = trainer.predict(tokenized["test"])
        pred_labels = np.argmax(test_pred.predictions, axis=-1).tolist()
        true_labels = test_pred.label_ids.tolist()
        analysis_file = output_dir / "test_predictions.json"
        analysis_file.write_text(
            json.dumps({"pred_labels": pred_labels, "true_labels": true_labels}, indent=2),
            encoding="utf-8",
        )
        mlflow.log_artifact(str(analysis_file), artifact_path="analysis")

        best_model_dir = output_dir / "best-model"
        trainer.save_model(str(best_model_dir))
        tokenizer.save_pretrained(str(best_model_dir))
        mlflow.log_param("best_model_dir", str(best_model_dir))

    return {
        "validation": val_metrics,
        "test": test_metrics,
        "best_model_dir": str(best_model_dir),
    }


def main() -> None:
    args = apply_profile_defaults(parse_args())
    cfg = build_config(args)
    results = train_and_evaluate(cfg)
    print("Training complete.")
    print(f"Best model saved to: {results['best_model_dir']}")
    print(f"Validation metrics: {results['validation']}")
    print(f"Test metrics: {results['test']}")


if __name__ == "__main__":
    main()
