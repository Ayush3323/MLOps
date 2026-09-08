from datasets import Dataset

from src.data.preprocess import map_rating_to_label, split_dataset


def test_map_rating_to_label() -> None:
    assert map_rating_to_label(1.0) == 0
    assert map_rating_to_label(2.0) == 0
    assert map_rating_to_label(3.0) == 1
    assert map_rating_to_label(4.0) == 2
    assert map_rating_to_label(5.0) == 2


def test_split_dataset_stratified() -> None:
    rows = []
    for _ in range(40):
        rows.append({"text": "neg", "label": 0, "rating": 1.0, "parent_asin": "a"})
    for _ in range(20):
        rows.append({"text": "neu", "label": 1, "rating": 3.0, "parent_asin": "b"})
    for _ in range(40):
        rows.append({"text": "pos", "label": 2, "rating": 5.0, "parent_asin": "c"})

    dataset = Dataset.from_list(rows)
    splits = split_dataset(dataset, seed=42)

    assert len(splits["train"]) == 80
    assert len(splits["validation"]) == 10
    assert len(splits["test"]) == 10

    def counts(split_name: str) -> dict[int, int]:
        labels = splits[split_name]["label"]
        return {k: labels.count(k) for k in [0, 1, 2]}

    assert counts("train") == {0: 32, 1: 16, 2: 32}
    assert counts("validation") == {0: 4, 1: 2, 2: 4}
    assert counts("test") == {0: 4, 1: 2, 2: 4}
