from json import load
from os.path import exists, join
from typing import Any


def get_records(
    data_prefix: str, settings: dict[str, Any], dataset_root: str
) -> list[dict[str, str | int]]:
    with open(data_prefix) as manifest_file:
        manifest: dict[str, list[dict[str, str | int]]] = load(manifest_file)
    records = manifest.get(settings["data_list_key"], [])
    if not records:
        raise ValueError(
            f"No records found under manifest key '{settings['data_list_key']}'"
        )

    resolved_records: list[dict[str, str | int]] = []
    for record in records:
        image_path: str | None = record.get("image")
        if image_path is None and all(
            isinstance(record.get(key), str) for key in ("image_cc", "image_mlo")
        ):
            image_path = record["image_cc"]

        if image_path is None:
            continue
        view_paths: list[str | int | None] = [record.get("image")]
        if "image_cc" in record and "image_mlo" in record:
            view_paths = [record["image_cc"], record["image_mlo"]]
        if all(
            exists(path if str(path).startswith("/") else join(dataset_root, str(path)))
            for path in view_paths
        ):
            resolved_records.append(record)

    if not resolved_records:
        raise ValueError("No records with existing images remain after filtering")
    if settings["n_splits"] > len(resolved_records):
        raise ValueError(
            f"n_splits ({settings['n_splits']}) cannot exceed the number of "
            f"records with existing images ({len(resolved_records)})"
        )
    return resolved_records
