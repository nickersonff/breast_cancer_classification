from collections.abc import Iterator
from typing import Any

from sklearn.model_selection import KFold

DataItem = dict[str, str | int]


def iter_kfold_splits(
    data: list[DataItem],
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: int | None = 42,
) -> Iterator[tuple[int, list[DataItem], list[DataItem]]]:
    """Yield independent train/validation records for each plain K-Fold split."""
    if n_splits < 2:
        raise ValueError("validation.n_splits must be at least 2")
    if len(data) < n_splits:
        raise ValueError(
            "validation.n_splits cannot be greater than the number of records"
        )
    if not shuffle:
        random_state = None

    splitter = KFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state,
    )
    for fold_index, (train_indices, valid_indices) in enumerate(splitter.split(data)):
        yield (
            fold_index,
            [data[index] for index in train_indices],
            [data[index] for index in valid_indices],
        )


def validation_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return K-Fold settings with backwards-compatible defaults."""
    settings = config.get("validation", {})
    return {
        "n_splits": int(settings.get("n_splits", 5)),
        "shuffle": bool(settings.get("shuffle", True)),
        "random_state": settings.get("random_state", 42),
        "data_list_key": str(settings.get("data_list_key", "train")),
        "enabled": bool(settings.get("enabled", False)),
        "run_prefix": str(settings.get("run_prefix", "kfold")),
    }