from typing import Any

import pytest

from pt.validation.kfold import iter_kfold_splits, validation_config


def test_kfold_covers_each_record_once() -> None:
    records: list[dict[str, str | int]] = [
        {
            "patient_id": f"patient-{index}",
            "image": f"image-{index}.npy",
            "label": index % 2,
        }
        for index in range(10)
    ]

    splits = list(iter_kfold_splits(records, n_splits=10))
    validation_images: list[str | int] = [
        item["image"]
        for _, _, validation_records in splits
        for item in validation_records
    ]

    assert len(splits) == 10
    assert sorted(validation_images) == sorted(item["image"] for item in records)
    for _, train_records, validation_records in splits:
        train_images: set[str | int] = {item["image"] for item in train_records}
        validation_images = list({item["image"] for item in validation_records})
        assert train_images.isdisjoint(validation_images)


def test_kfold_rejects_invalid_number_of_splits() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        list(iter_kfold_splits([], n_splits=1))


def test_validation_settings_are_configurable() -> None:
    settings: dict[str, Any] = validation_config(
        {
            "validation": {
                "enabled": True,
                "n_splits": 3,
                "shuffle": True,
                "random_state": 42,
                "data_list_key": "train",
                "run_prefix": "experiment",
            }
        }
    )

    assert settings == {
        "enabled": True,
        "n_splits": 3,
        "shuffle": True,
        "random_state": 42,
        "data_list_key": "train",
        "run_prefix": "experiment",
    }
