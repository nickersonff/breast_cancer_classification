from collections.abc import Iterator
from typing import Any

from sklearn.model_selection import StratifiedGroupKFold

DataItem = dict[str, str | int]


def iter_kfold_splits(
    data: list[DataItem],
    n_splits: int = 10,
    shuffle: bool = True,
    random_state: int = 42,
) -> Iterator[tuple[int, list[DataItem], list[DataItem]]]:
    """Yield patient-independent, approximately label-balanced K-Fold splits."""
    if n_splits < 2:
        raise ValueError("validation.n_splits must be at least 2")
    if len(data) < n_splits:
        raise ValueError(
            "validation.n_splits cannot be greater than the number of records"
        )

    if not data:
        return

    patient_key = "patient_id" if "patient_id" in data[0] else None

    label_key = "label" if "label" in data[0] else None
    if patient_key is None or label_key is None:
        raise ValueError(
            "K-Fold validation requires patient and label fields for grouped splitting"
        )
    if any(patient_key not in item or label_key not in item for item in data):
        raise ValueError("all records must contain the patient and label fields")

    groups = [item[patient_key] for item in data]
    labels = [item[label_key] for item in data]
    if len(set(groups)) < n_splits:
        raise ValueError(
            "validation.n_splits cannot be greater than the number of patients"
        )

    splitter = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=shuffle, random_state=random_state
    )
    for fold_index, (train_indices, valid_indices) in enumerate(
        splitter.split(data, labels, groups)
    ):
        yield (
            fold_index,
            [data[index] for index in train_indices],
            [data[index] for index in valid_indices],
        )


def validation_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return K-Fold settings with backwards-compatible defaults."""
    settings = config.get("validation", {})
    return {
        "n_splits": int(settings.get("n_splits", 10)),
        "shuffle": bool(settings.get("shuffle", True)),
        "random_state": int(settings.get("random_state", 42)),
        "data_list_key": str(settings.get("data_list_key", "train")),
        "enabled": bool(settings.get("enabled", False)),
        "run_prefix": str(settings.get("run_prefix", "kfold")),
    }
