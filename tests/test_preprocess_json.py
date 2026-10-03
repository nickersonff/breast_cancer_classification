import json
from pathlib import Path

import numpy as np
import pytest

from pt.preprocessing.preprocess_json import (
    load_datalist,
    preprocess_json,
    resolve_datalist,
)


def test_load_datalist_resolves_existing_images_and_skips_missing(
    tmp_path: Path,
) -> None:
    (tmp_path / "present.npy").write_bytes(b"")
    manifest_path: Path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "train": [
                    {"image": "present.npy", "label": 1},
                    {"image": "missing.npy", "label": 0},
                ]
            }
        )
    )

    records: list[dict[str, str | int]] = load_datalist(
        str(manifest_path), base_dir=str(tmp_path)
    )

    assert records == [{"image": str(tmp_path / "present.npy"), "label": 1}]


def test_resolve_datalist_does_not_mutate_selected_records(tmp_path: Path) -> None:
    (tmp_path / "image.npy").write_bytes(b"")
    selected: list[dict[str, str | int]] = [{"image": "image.npy", "label": 1}]

    resolved: list[dict[str, str | int]] = resolve_datalist(
        selected, base_dir=str(tmp_path)
    )

    assert resolved == [{"image": str(tmp_path / "image.npy"), "label": 1}]
    assert selected == [{"image": "image.npy", "label": 1}]


def test_preprocess_json_writes_variant_manifest_and_three_channel_images(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.npy"
    np.save(source, np.arange(16, dtype=np.float32).reshape(4, 4))
    manifest = tmp_path / "input.json"
    manifest.write_text(
        json.dumps(
            {
                "train": [{"image": source.name, "label": 1, "patient_id": "p1"}],
                "test": [],
            }
        )
    )

    result = preprocess_json(
        out_path=str(tmp_path / "processed"),
        datalist=str(manifest),
        config={"io_dirs": {"preprocess_prefix": ""}},
        norm="min-max",
        size=8,
    )

    result_manifest = json.loads(Path(result).read_text())
    output = tmp_path / "processed" / "min-max_none_8" / "source.npy"
    image = np.load(output)
    assert image.shape == (8, 8, 3)
    assert image.min() == pytest.approx(0.0)
    assert image.max() == pytest.approx(1.0)
    assert result_manifest["train"][0]["image"] == "min-max_none_8/source.npy"


def test_preprocess_json_reports_missing_source_image(tmp_path: Path) -> None:
    manifest = tmp_path / "input.json"
    manifest.write_text(
        json.dumps({"train": [{"image": "missing.npy", "label": 0}], "test": []})
    )

    with pytest.raises(FileNotFoundError, match="Source image not found"):
        preprocess_json(
            out_path=str(tmp_path / "processed"),
            datalist=str(manifest),
            config={"io_dirs": {"preprocess_prefix": ""}},
        )
