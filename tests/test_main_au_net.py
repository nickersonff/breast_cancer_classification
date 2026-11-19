from importlib.machinery import ModuleSpec
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from pt.learners import au_net_trainer

MAIN_PATH: Path = Path(__file__).resolve().parents[1] / "main.py"
MAIN_SPEC: ModuleSpec | None = spec_from_file_location("application_main", MAIN_PATH)
assert MAIN_SPEC is not None
assert MAIN_SPEC.loader is not None
application: ModuleType = module_from_spec(MAIN_SPEC)
MAIN_SPEC.loader.exec_module(application)


def test_main_dispatches_au_net_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config: dict[str, Any] = {
        "hyperparameters": {"task": "au_net", "architecture": "resnet"},
        "io_dirs": {},
        "au_net": {"annotations_dir": "annotations"},
    }
    calls: list[tuple[dict[str, Any], Path]] = []

    def fake_train(config: dict[str, Any], project_root: Path) -> None:
        calls.append((config, project_root))

    monkeypatch.setattr(application, "load_config", lambda: config)
    monkeypatch.setattr(
        application,
        "setup_logging",
        lambda *_args: ("test.log", application.getLogger("test")),
    )
    monkeypatch.setattr(au_net_trainer, "train_au_net_from_config", fake_train)

    application.main()

    assert calls == [(config["au_net"], Path(application.PROJECT_ROOT))]


def test_main_dispatches_separate_au_net_test_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config: dict[str, Any] = {
        "hyperparameters": {"task": "au_net_test", "architecture": "resnet"},
        "io_dirs": {},
        "au_net": {"checkpoint_path": "runs/au_net_pectoral/au_net_pectoral_best.pt"},
    }
    calls: list[tuple[dict[str, Any], Path]] = []

    def fake_test(config: dict[str, Any], project_root: Path) -> None:
        calls.append((config, project_root))

    monkeypatch.setattr(application, "load_config", lambda: config)
    monkeypatch.setattr(
        application,
        "setup_logging",
        lambda *_args: ("test.log", application.getLogger("test")),
    )
    monkeypatch.setattr(au_net_trainer, "test_au_net_from_config", fake_test)

    application.main()

    assert calls == [(config["au_net"], Path(application.PROJECT_ROOT))]


def test_au_net_config_resolves_paths_and_training_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, Any]] = []
    model: object = object()
    checkpoint: Path = tmp_path / "best.pt"
    results: dict[str, Any] = {"best_validation_dice": 0.8}

    def fake_train(**kwargs: Any) -> tuple[object, Path, dict[str, Any]]:
        calls.append(kwargs)
        return model, checkpoint, results

    monkeypatch.setattr(au_net_trainer, "train_au_net", fake_train)
    config: dict[str, Any] = {
        "annotations_dir": "labels",
        "metadata_dir": "metadata",
        "output_dir": "runs/segmentation",
        "image_size": 256,
        "batch_size": 2,
        "epochs": 4,
        "learning_rate": 0.0002,
    }

    returned = au_net_trainer.train_au_net_from_config(config, tmp_path)

    assert returned == (model, checkpoint, results)
    assert calls[0]["annotations_dir"] == tmp_path / "labels"
    assert calls[0]["metadata_dir"] == tmp_path / "metadata"
    assert calls[0]["output_dir"] == tmp_path / "runs/segmentation"
    assert calls[0]["image_size"] == 256
    assert calls[0]["batch_size"] == 2
    assert calls[0]["epochs"] == 4
    assert calls[0]["learning_rate"] == 0.0002


def test_au_net_test_config_uses_explicit_best_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, Any]] = []
    expected: dict[str, Any] = {"sample_count": 1}

    def fake_evaluate(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return expected

    monkeypatch.setattr(au_net_trainer, "evaluate_au_net_cbis_test", fake_evaluate)
    config: dict[str, Any] = {
        "annotations_dir": "labels",
        "metadata_dir": "metadata",
        "checkpoint_path": "runs/custom_best.pt",
        "test_output_dir": "runs/test-results",
        "batch_size": 2,
        "workers": 0,
    }

    result = au_net_trainer.test_au_net_from_config(config, tmp_path)

    assert result == expected
    assert calls[0]["annotations_dir"] == tmp_path / "labels"
    assert calls[0]["metadata_dir"] == tmp_path / "metadata"
    assert calls[0]["checkpoint_path"] == tmp_path / "runs/custom_best.pt"
    assert calls[0]["output_dir"] == tmp_path / "runs/test-results"
    assert calls[0]["batch_size"] == 2
    assert calls[0]["workers"] == 0
