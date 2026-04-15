import numpy as np

from src.inference.predict import LABEL_MAP, label_id_to_name
from src.training.train import compute_metrics


def test_compute_metrics_returns_expected_keys() -> None:
    logits = np.array(
        [
            [4.0, 1.0, 0.5],
            [0.2, 3.5, 0.3],
            [0.1, 0.2, 4.1],
            [3.0, 1.0, 0.9],
        ]
    )
    labels = np.array([0, 1, 2, 0])
    metrics = compute_metrics((logits, labels))

    assert "accuracy" in metrics
    assert "f1_macro" in metrics
    assert metrics["accuracy"] == 1.0
    assert metrics["f1_macro"] == 1.0


def test_label_mapping_contract() -> None:
    assert label_id_to_name(0) == "negative"
    assert label_id_to_name(1) == "neutral"
    assert label_id_to_name(2) == "positive"
    assert set(LABEL_MAP.keys()) == {0, 1, 2}
