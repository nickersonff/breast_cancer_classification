from glob import glob
from json import load
from os import listdir, remove
from os.path import exists, isdir, isfile, join
from typing import Any, cast

from pt.preprocessing.preprocess_dicom import dicom_preprocess


def load_datalist(
    filename: str, data_list_key: str = "train", base_dir: str = ""
) -> list[dict[str, str | int]]:
    with open(filename, "r") as f:
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


def path_exists(path: str = "") -> bool:
    return exists(path) and isdir(path) and bool(listdir(path))


def clean_path(path: str):
    if exists(path) and isdir(path):
        path_dir: list[str] = listdir(path)
        for file in path_dir:
            remove(join(path, file))


def preprocess_db(
    out_path: str,
    config: dict[str, Any],
    norm: str = "",
    filter: str = "",
    size: int = 224,
    datalist: str = "",
):

    # clean_path(out_path) # if want delete all files inside the path

    with open(datalist) as file:
        c = load(file)

    is_liga: bool = False
    image_file_path: list[str | dict[str, str]] = []

    if datalist.__contains__("LIGA"):
        image_file_path.extend(
            [{"image": l["image"], "dicom": l["dicom"]} for l in c["train"]]
        )
        image_file_path.extend(
            [{"image": l["image"], "dicom": l["dicom"]} for l in c["test"]]
        )
        is_liga = True
    else:
        image_file_path.extend([l["image"] for l in c["train"]])
        image_file_path.extend([l["image"] for l in c["test"]])

    print(f"Images found: {len(image_file_path)}")

    list_img: list[str] = []
    for i in image_file_path:
        i = cast(str, i)
        if is_liga:
            i = cast(dict[str, str], i)
            dicom_root: str = config["io_dirs"].get("dicom_root_LIGA")
            dir_name: str = i["image"].replace(".npy", "")
            img_file: list[str] = [i["dicom"]]
            save_prefix: str = join(out_path, dir_name)
        elif i.startswith(("Calc", "Mass")):
            dicom_root: str = config["io_dirs"].get("dicom_root_DDSM")
            dir_name: str = i.replace(".npy", "")
            img_file: list[str] = glob(
                join(dicom_root, dir_name, "**", "*.dcm"), recursive=True
            )
            save_prefix: str = join(out_path, dir_name)
        else:
            dicom_root: str = config["io_dirs"].get("dicom_root_VINDR")
            image_id: str = i.split("_")[0]
            img: str = i.split("_")[1].replace(".npy", "")
            img_file: list[str] = glob(
                join(dicom_root, image_id, img + "*.dicom"), recursive=True
            )
            save_prefix: str = join(out_path, image_id + "_" + img)

        if not img_file:
            print(
                f"[!] No source file found for {save_prefix} under {dicom_root}; skipping"
            )
            continue

        dicom_preprocess(img_file[0], save_prefix, norm=norm, filter=filter, size=size)

        if isfile(save_prefix + ".npy"):
            list_img.append(save_prefix)

    print(f"Images transformed: {len(list_img)}")
