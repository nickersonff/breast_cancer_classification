from pathlib import Path
from typing import Any

import pytest

from pt import orchestrator


def test_run_train_uses_manifest_without_repartitioning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def fake_train(*args: Any, **kwargs: Any) -> None:
        calls.append((args, kwargs))

    monkeypatch.setattr(orchestrator, "_run_single_train", fake_train)

    manifest_path = tmp_path / "dataset_site-0_DDSM_KFOLD.json"
    orchestrator.run_train(
        str(tmp_path),
        str(manifest_path),
        {"hyperparameters": {}},
        batch=32,
        cnn="resnet",
    )

    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[:2] == (str(tmp_path), str(manifest_path))
    assert kwargs["batch"] == 32
    assert kwargs["cnn"] == "resnet"
    assert kwargs["run_name"].startswith("dataset_site-0_DDSM_KFOLD_run_")


def test_preprocessing_builds_variants_from_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    def fake_preprocess_json(**kwargs: Any) -> str:
        calls.append(("preprocess_json", kwargs))
        return str(tmp_path / f"{kwargs['size']}.json")

    def fake_run_train(*args: Any, **kwargs: Any) -> tuple[float, float, float, float]:
        calls.append(("run_train", {"args": args, "kwargs": kwargs}))
        return 0.8, 0.6, 0.9, 0.7

    monkeypatch.setattr(orchestrator, "preprocess_json", fake_preprocess_json)
    monkeypatch.setattr(orchestrator, "run_train", fake_run_train)

    manifest_path = tmp_path / "dataset_site-0_DDSM_KFOLD.json"
    orchestrator.preprocessing(
        {
            "io_dirs": {"preprocess_prefix": "data/preprocessed/"},
            "hyperparameters": {"batch_size": 32, "architecture": "resnet"},
        },
        str(manifest_path),
    )

    assert calls[0][0] == "preprocess_json"
    assert calls[1][0] == "run_train"
    assert calls[1][1]["args"][1] == str(manifest_path)
    assert len([call for call in calls if call[0] == "preprocess_json"]) == 16
    assert len([call for call in calls if call[0] == "run_train"]) == 16
