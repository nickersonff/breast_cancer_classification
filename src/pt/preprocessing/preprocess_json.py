import glob
import json
import os

from src.pt.preprocessing.preprocess_dicom import dicom_preprocess


def load_datalist(filename, data_list_key="train", base_dir=""):
    with open(filename, "r") as f:
        data = json.load(f)

    data_list = data[data_list_key]
    for d in data_list:
        d["image"] = os.path.join(base_dir, d["image"])

    return data_list


def path_exists(caminho_da_pasta=""):

    if os.path.exists(caminho_da_pasta) and os.path.isdir(caminho_da_pasta):
        if os.listdir(caminho_da_pasta):
            return True
        else:
            return False
    else:
        return False


def clean_path(diretorio):
    path = diretorio
    if os.path.exists(path) and os.path.isdir(path):
        dir = os.listdir(path)
        for file in dir:
            os.remove(os.path.join(path, file))


def preprocess_db(
    out_path, norm="", filter="", size=224, datalist="", config: dict = None
):

    # clean_path(out_path) # if want delete all files inside the path

    with open(datalist) as file:
        c = json.load(file)

    ehLiga = False

    if datalist.__contains__("LIGA"):
        image_file_path = []
        image_file_path.extend(
            [{"image": l["image"], "dicom": l["dicom"]} for l in c["train"]]
        )
        image_file_path.extend(
            [{"image": l["image"], "dicom": l["dicom"]} for l in c["test"]]
        )
        ehLiga = True
    else:
        image_file_path = []
        image_file_path.extend([l["image"] for l in c["train"]])
        image_file_path.extend([l["image"] for l in c["test"]])

    print(f"Images found: {len(image_file_path)}")

    list_img = []
    for i in image_file_path:
        if ehLiga:
            dicom_root = config["io_dirs"].get("dicom_root_LIGA")
            dir_name = i["image"].replace(".npy", "")
            img_file = [i["dicom"]]
            save_prefix = os.path.join(out_path, dir_name)
        elif i.startswith("Calc") or i.startswith("Mass"):
            dicom_root = config["io_dirs"].get("dicom_root_DDSM")
            dir_name = i.replace(".npy", "")
            img_file = glob.glob(
                os.path.join(dicom_root, dir_name, "**", "*.dcm"), recursive=True
            )
            save_prefix = os.path.join(out_path, dir_name)
        else:
            dicom_root = config["io_dirs"].get("dicom_root_VINDR")
            id = i.split("_")[0]
            img = i.split("_")[1].replace(".npy", "")
            img_file = glob.glob(
                os.path.join(dicom_root, id, img + "*.dicom"), recursive=True
            )
            save_prefix = os.path.join(out_path, id + "_" + img)

        _success, _dc_tags = dicom_preprocess(
            img_file[0], save_prefix, norm=norm, filter=filter, size=size
        )

        if os.path.isfile(save_prefix + ".npy"):
            _success = True
            list_img.append(save_prefix)
        else:
            _success = False

    print(f"Images transformed: {len(list_img)}")
