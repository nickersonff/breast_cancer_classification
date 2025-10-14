
import glob
import numpy as np
import pandas as pd
import os
import json
from utils.preprocess_dicom import dicom_preprocess

def excluirRegistros(diretorio):
    path = diretorio
    dir = os.listdir(path)
    for file in dir:
        os.remove(os.path.join(path, file))

def preprocess(dicom_root, out_path, norm="", filter="", size=224, dataset='/home/nfferreira/data/dataset_site-1.json'):

    excluirRegistros(out_path)
    with open(dataset) as file:
        c = json.load(file)

    image_file_path = []
    image_file_path.extend([l['image'] for l in c['train']])
    image_file_path.extend([l['image'] for l in c['test']])

    print(f'Número de imagens encontradas: {len(image_file_path)}')
    list_img = []
    for i in image_file_path:
        dir_name = i.replace('.npy', '') 
        img_file = glob.glob(os.path.join(dicom_root, dir_name, "**", "*.dcm"), recursive=True)
        save_prefix = os.path.join(out_path, dir_name)
        for i in img_file:
            _success, _dc_tags = dicom_preprocess(i,  save_prefix, norm=norm, filter=filter, size=size)
    
        if os.path.isfile(save_prefix + ".npy"):
            _success = True
            list_img.append(save_prefix)
        else:
            _success = False
    print(f'Número de imagens transformadas: {len(list_img)}')

def preprocessMixedDB(out_path, norm="", filter="", size=224, datalist=''):

    excluirRegistros(out_path)
    
    with open(datalist) as file:
        c = json.load(file)

    image_file_path = []
    image_file_path.extend([l['image'] for l in c['train']])
    image_file_path.extend([l['image'] for l in c['test']])
    print(f'Número de imagens encontradas: {len(image_file_path)}')

    list_img = []
    for i in image_file_path:
        if i.startswith('Calc') or i.startswith('Mass'):
            dicom_root = "/home/nfferreira/DDSM/CBIS-DDSM"
            dir_name = i.replace('.npy', '')
            img_file = glob.glob(os.path.join(dicom_root, dir_name, "**", "*.dcm"), recursive=True)
            save_prefix = os.path.join(out_path, dir_name)
        else:
            dicom_root = "/home/nfferreira/VinDr/images"
            id = i.split("_")[0]
            img = i.split("_")[1].replace('.npy', '')
            img_file = glob.glob(os.path.join(dicom_root, id, img + "*.dicom"), recursive=True)
            save_prefix = os.path.join(out_path, id + "_" + img)
        
        _success, _dc_tags = dicom_preprocess(img_file[0],  save_prefix, norm=norm, filter=filter, size=size)
    
        if os.path.isfile(save_prefix + ".npy"):
            _success = True
            list_img.append(save_prefix)
        else:
            _success = False
    

    print(f'Número de imagens transformadas: {len(list_img)}')

if __name__ == "__main__":
    preprocess("/home/nfferreira/DDSM/CBIS-DDSM",
                "/home/nfferreira/data/preprocessed-2/ROI", norm="min-max",filter="CLAHE+WEINER", size=2048)
    

    