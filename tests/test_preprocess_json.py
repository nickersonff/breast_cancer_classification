import json
from pathlib import Path

from pt.preprocessing.preprocess_json import load_datalist, resolve_datalist


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
