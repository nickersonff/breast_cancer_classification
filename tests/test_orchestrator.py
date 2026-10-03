import json
from pathlib import Path
from typing import Any

import pytest

from pt import orchestrator


def _config(results_dir: Path) -> dict[str, dict[str, str | int | bool | float]]:
    return {
        "hyperparameters": {"aggregation_epochs": 1, "lr": 0.001},
        "validation": {
            "enabled": True,
            "n_splits": 3,
            "shuffle": True,
            "random_state": 42,
            "data_list_key": "train",
            "run_prefix": "experiment",
        },
        "io_dirs": {"results_dir": str(results_dir)},
    }


def test_run_kfold_trains_each_fold_and_writes_aggregate_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path: Path = tmp_path / "manifest.json"
    records: list[dict[str, str | int]] = [
        {
            "image": f"image-{index}.npy",
            "label": index % 2,
            "patient_id": f"patient-{index}",
        }
        for index in range(9)
    ]
    for record in records:
        (tmp_path / str(record["image"])).touch()
    manifest_path.write_text(json.dumps({"train": records, "test": []}))
    calls: list[dict[str, str | int | dict[str, dict[str, str | int]]]] = []

    def fake_train(
        *args: tuple[Any], **kwargs: dict[str, dict[str, str | int]]
    ) -> tuple[float, float, float]:
        calls.append(
            {
                "train_size": len(kwargs["train_datalist"]),
                "valid_size": len(kwargs["valid_datalist"]),
                "run_name": kwargs["run_name"],
            }
        )
        return 0.8, 0.6, 0.9

    monkeypatch.setattr(orchestrator, "_run_single_train", fake_train)

    orchestrator.run_kfold(
        str(tmp_path),
        str(manifest_path),
        _config(tmp_path / "results"),
        run_prefix="experiment",
    )

    result_path: Path = tmp_path / "results" / "experiment_metrics.json"
    plot_path: Path = tmp_path / "results" / "experiment_metrics_scatter.png"
    result: dict[str, dict[str, float]] = json.loads(result_path.read_text())
    assert len(calls) == 3
    assert all(call["train_size"] == 6 for call in calls)
    assert all(call["valid_size"] == 3 for call in calls)
    assert [call["run_name"] for call in calls] == [
        "experiment_fold_01",
        "experiment_fold_02",
        "experiment_fold_03",
    ]
    assert result["aggregate"] == {
        "accuracy_mean": 0.8,
        "accuracy_std": 0.0,
        "kappa_mean": 0.6,
        "kappa_std": 0.0,
        "roc_auc_mean": 0.9,
        "roc_auc_std": 0.0,
    }
    assert plot_path.exists()
    assert plot_path.stat().st_size > 0


def test_run_kfold_rejects_unknown_manifest_key(tmp_path: Path) -> None:
    manifest_path: Path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"train": [], "test": []}))

    try:
        orchestrator.run_kfold(
            str(tmp_path),
            str(manifest_path),
            _config(tmp_path / "results"),
        )
    except ValueError as error:
        assert "No records found" in str(error)
    else:
        raise AssertionError("run_kfold should reject an empty manifest split")
