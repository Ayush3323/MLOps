from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.core.config import get_settings


LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}


def label_id_to_name(label_id: int) -> str:
    return LABEL_MAP[int(label_id)]


@dataclass
class PredictionResult:
    label_id: int
    label: str
    confidence: float
    scores: dict[str, float]


class SentimentPredictor:
    def __init__(self, model_path: str | None = None, device: str | None = None):
        settings = get_settings()
        self.model_path = model_path or settings.model_path
        self.device = "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        self._tokenizer: AutoTokenizer | None = None
        self._model: AutoModelForSequenceClassification | None = None

    def _ensure_loaded(self) -> None:
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if self._model is None:
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            self._model.to(self.device)
            self._model.eval()

    def predict(self, text: str) -> PredictionResult:
        self._ensure_loaded()
        assert self._tokenizer is not None
        assert self._model is not None

        encoded = self._tokenizer(text, truncation=True, return_tensors="pt")
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        with torch.no_grad():
            logits = self._model(**encoded).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

        label_id = int(probs.argmax())
        confidence = float(probs[label_id])
        scores = {label_id_to_name(i): float(score) for i, score in enumerate(probs)}
        return PredictionResult(
            label_id=label_id,
            label=label_id_to_name(label_id),
            confidence=confidence,
            scores=scores,
        )


def predict_text(text: str, model_path: str | None = None) -> dict[str, Any]:
    predictor = SentimentPredictor(model_path=model_path)
    result = predictor.predict(text)
    return {
        "label_id": result.label_id,
        "label": result.label,
        "confidence": result.confidence,
        "scores": result.scores,
    }
