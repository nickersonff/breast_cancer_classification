from json import dump, load
from logging import info
from os import listdir, makedirs, remove
from os.path import exists, isdir, isfile, join
from pathlib import Path
from typing import Any

from cv2 import INTER_AREA, resize
from numpy import float32, load as numpy_load, newaxis, save, zeros_like

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
        keys = ("image_cc", "image_mlo") if "image_cc" in item else ("image",)
        image_paths = [join(base_dir, str(item[key])) for key in keys]
        if not all(isfile(path) for path in image_paths):
            missing_count += 1
            continue
        item: dict[str, str | int] = item.copy()
        for key, path in zip(keys, image_paths):
            item[key] = path
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
        keys = ("image_cc", "image_mlo") if "image_cc" in item else ("image",)
        image_paths = [join(base_dir, str(item[key])) for key in keys]
        if not all(isfile(path) for path in image_paths):
            missing_count += 1
            continue
        resolved_item = item.copy()
        for key, path in zip(keys, image_paths):
            resolved_item[key] = path
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

            image = numpy_load(source_path).astype(float32)
            if image.ndim == 3:
                image = image[..., 0] if image.shape[-1] in (1, 3) else image[0]
            if image.ndim != 2:
                raise ValueError(f"Expected a 2-D mammogram image: {source_path}")

            if filter:
                image_min = float(image.min())
                image_range = float(image.max() - image_min)
                image = (
                    zeros_like(image)
                    if image_range == 0
                    else (image - image_min) / image_range * 255.0
                )
                image = apply_filters(image, filter, float32)
            if norm == "min-max":
                image_min = float(image.min())
                image_range = float(image.max() - image_min)
                image = (
                    zeros_like(image)
                    if image_range == 0
                    else (image - image_min) / image_range
                )
            elif norm == "z-score":
                image = (image - image.mean()) / (image.std() + 1e-8)

            image = resize(image, (size, size), interpolation=INTER_AREA)
            output_name = f"{Path(source).stem}.npy"
            save(join(variant_dir, output_name), image[..., newaxis].repeat(3, axis=2))
            processed_item = dict(item)
            processed_item["image"] = join(variant, output_name)
            processed_manifest[split].append(processed_item)

    manifest_path = join(out_path, f"{variant}.json")
    with open(manifest_path, "w") as manifest_file:
        dump(processed_manifest, manifest_file, indent=2)
    return manifest_path


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
    list_img: list[str] = []
    for i in datalist:
        dir_name: str = i["npy"].replace(".npy", "")
        img_file: str = i["dicom"]
        save_prefix: str = join(out_path, dir_name)

        dicom_preprocess(img_file, save_prefix, norm=norm, filter=filter, size=size)

        if isfile(save_prefix + ".npy"):
            list_img.append(save_prefix)
        info(f"Images transformed: {len(list_img)}")


def preprocess_multiview_db(
    out_path: str,
    datalist: list[dict[str, Any]],
    norm: str = "",
    filter: str = "",
    size: int = 224,
) -> list[dict[str, Any]]:
    """Preprocess complete CC/MLO pairs and return paired manifest records."""
    grouped: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    missing = 0
    for item in datalist:
        view = str(item.get("view", "")).upper()
        if view not in {"CC", "MLO"}:
            continue
        if not isfile(str(item.get("dicom", ""))):
            missing += 1
            continue
        key = (str(item["patient_id"]), str(item.get("laterality", "")))
        grouped.setdefault(key, {})[view] = item

    paired: list[dict[str, Any]] = []
    failed_pairs = 0
    for (patient_id, laterality), views in grouped.items():
        if "CC" not in views or "MLO" not in views:
            continue
        output: dict[str, Any] = {
            "patient_id": patient_id,
            "laterality": laterality,
            "label": int(views["CC"]["label"]),
        }
        generated: list[str] = []
        try:
            for view in ("CC", "MLO"):
                item = views[view]
                stem = Path(str(item["npy"])).stem
                save_prefix = join(out_path, stem)
                dicom_preprocess(
                    str(item["dicom"]),
                    save_prefix,
                    norm=norm,
                    filter=filter,
                    size=size,
                )
                generated.append(save_prefix + ".npy")
                output[f"image_{view.lower()}"] = f"{stem}.npy"
        except (OSError, ValueError) as error:
            failed_pairs += 1
            print(
                f"[WARNING] Skipping invalid multi-view pair "
                f"{patient_id}/{laterality}: {error}"
            )
            for generated_file in generated:
                if isfile(generated_file):
                    remove(generated_file)
            continue
        paired.append(output)

    if missing:
        print(f"Skipped {missing} VinDr record(s) with missing DICOM files")
    if failed_pairs:
        print(f"Skipped {failed_pairs} multi-view pair(s) with invalid DICOM data")
    print(f"Multi-view pairs transformed: {len(paired)}")
    return paired
