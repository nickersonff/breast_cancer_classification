from json import dump, load
from os import listdir, makedirs, remove
from os.path import exists, isdir, isfile, join
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from pt.preprocessing.preprocess_dicom import dicom_preprocess
from pt.utils.constants import Constants
from pt.utils.filters_utils import apply_filters


def load_datalist(
    filename: str, data_list_key: str = "train", base_dir: str = ""
) -> list[dict[str, str | int]]:
    with open(filename) as f:
        data: dict[str, list[dict[str, str | int]]] = load(f)

    data_list: list[dict[str, str | int]] = []
    missing_count: int = 0
    for item in data[data_list_key]:
        image_path: str = join(base_dir, str(item["image"]))
        if not isfile(image_path):
            missing_count += 1
            continue
        item: dict[str, str | int] = item.copy()
        item["image"] = image_path
        data_list.append(item)

    if missing_count:
        print(
            f"[!] Skipped {missing_count} missing image(s) while loading {filename} ({data_list_key})"
        )

    return data_list


def resolve_datalist(
    data: list[dict[str, str | int]], base_dir: str = ""
) -> list[dict[str, str | int]]:
    """Resolve image paths for records already selected by a validation split."""
    resolved: list[dict[str, str | int]] = []
    missing_count: int = 0
    for item in data:
        image_path: str = join(base_dir, str(item["image"]))
        if not isfile(image_path):
            missing_count += 1
            continue
        resolved_item = item.copy()
        resolved_item["image"] = image_path
        resolved.append(resolved_item)

    if missing_count:
        print(f"[!] Skipped {missing_count} missing image(s) from selected split")
    return resolved


def preprocess_json(
    out_path: str,
    config: dict[str, Any],
    norm: str = "",
    filter: str = "",
    size: int = 224,
    datalist: str = "",
) -> str:
    """Preprocess an existing NumPy-image manifest and return its new manifest."""
    if not datalist:
        raise ValueError("A datalist manifest is required")
    with open(datalist) as manifest_file:
        manifest = load(manifest_file)

    project_root = Constants.get_absolute_project_path()
    manifest_root = str(Path(datalist).resolve().parent)
    variant = "_".join((norm or "raw", filter or "none", str(size))).replace("/", "-")
    variant_dir = join(out_path, variant)
    makedirs(variant_dir, exist_ok=True)
    processed_manifest: dict[str, list[dict[str, Any]]] = {"train": [], "test": []}

    for split in processed_manifest:
        for item in manifest.get(split, []):
            source = str(item["image"])
            if isfile(source):
                source_path = source
            else:
                configured_root = join(
                    project_root,
                    config["io_dirs"].get("preprocess_prefix", ""),
                    source,
                )
                source_path = (
                    join(manifest_root, source)
                    if isfile(join(manifest_root, source))
                    else configured_root
                )
            if not isfile(source_path):
                raise FileNotFoundError(f"Source image not found: {source_path}")

            image = np.load(source_path).astype(np.float32)
            if image.ndim == 3:
                image = image[..., 0] if image.shape[-1] in (1, 3) else image[0]
            if image.ndim != 2:
                raise ValueError(f"Expected a 2-D mammogram image: {source_path}")

            if filter:
                image_min = float(image.min())
                image_range = float(image.max() - image_min)
                image = (
                    np.zeros_like(image)
                    if image_range == 0
                    else (image - image_min) / image_range * 255.0
                )
                image = apply_filters(image, filter, np.float32)
            if norm == "min-max":
                image_min = float(image.min())
                image_range = float(image.max() - image_min)
                image = (
                    np.zeros_like(image)
                    if image_range == 0
                    else (image - image_min) / image_range
                )
            elif norm == "z-score":
                image = (image - image.mean()) / (image.std() + 1e-8)

            image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
            output_name = f"{Path(source).stem}.npy"
            np.save(
                join(variant_dir, output_name), image[..., np.newaxis].repeat(3, axis=2)
            )
            processed_item = dict(item)
            processed_item["image"] = join(variant, output_name)
            processed_manifest[split].append(processed_item)

    manifest_path = join(out_path, f"{variant}.json")
    with open(manifest_path, "w") as manifest_file:
        dump(processed_manifest, manifest_file, indent=2)
    return manifest_path


def path_exists(path: str = "") -> bool:
    return exists(path) and isdir(path) and bool(listdir(path))


def clean_path(path: str):
    if exists(path) and isdir(path):
        path_dir: list[str] = listdir(path)
        for file in path_dir:
            remove(join(path, file))


def preprocess_db(
    out_path: str,
    datalist: list[dict[str, Any]],
    norm: str = "",
    filter: str = "",
    size: int = 224,
):

    # clean_path(out_path) # if want delete all files inside the path

    list_img: list[str] = []
    for i in datalist:
        dir_name: str = i["npy"].replace(".npy", "")
        img_file: str = i["dicom"]
        save_prefix: str = join(out_path, dir_name)

        dicom_preprocess(img_file, save_prefix, norm=norm, filter=filter, size=size)

        if isfile(save_prefix + ".npy"):
            list_img.append(save_prefix)

    print(f"Images transformed: {len(list_img)}")
