
import glob
import numpy as np
import pandas as pd
import os
import json
from utils.preprocess_dicom import dicom_preprocess



def excluirRegistros(diretorio):
    path = diretorio
    if os.path.exists(path) and os.path.isdir(path):
        dir = os.listdir(path)
        for file in dir:
            os.remove(os.path.join(path, file))

def preprocessMixedDB(out_path, norm="", filter="", size=224, datalist=''):

    excluirRegistros(out_path)
    
    with open(datalist) as file:
        c = json.load(file)

    ehLiga = False

    if datalist.__contains__('LIGA'):
        image_file_path = []
        image_file_path.extend([{"image": l['image'], "dicom": l['dicom']} for l in c['train']])
        image_file_path.extend([{"image": l['image'], "dicom": l['dicom']} for l in c['test']])
        ehLiga = True
    else: 
        image_file_path = []
        image_file_path.extend([l['image'] for l in c['train']])
        image_file_path.extend([l['image'] for l in c['test']])
    
    print(f'Número de imagens encontradas: {len(image_file_path)}')

    list_img = []
    for i in image_file_path:
        if ehLiga:
            dicom_root = "/home/nfferreira/LIGA/dicom" #change dicom path
            dir_name = i['image'].replace('.npy', '')
            #img_file = glob.glob(i['dicom'], recursive=True)
            img_file = [i['dicom']]
            save_prefix = os.path.join(out_path, dir_name)
        elif i.startswith('Calc') or i.startswith('Mass'):
            dicom_root = "/home/nfferreira/DDSM/CBIS-DDSM" #change dicom path
            dir_name = i.replace('.npy', '')
            img_file = glob.glob(os.path.join(dicom_root, dir_name, "**", "*.dcm"), recursive=True)
            save_prefix = os.path.join(out_path, dir_name)
        else:
            dicom_root = "/home/nfferreira/VinDr/images" #change dicom path
            id = i.split("_")[0]
            img = i.split("_")[1].replace('.npy', '')
            img_file = glob.glob(os.path.join(dicom_root, id, img + "*.dicom"), recursive=True)
            save_prefix = os.path.join(out_path, id + "_" + img)

        _success, _dc_tags = dicom_preprocess(img_file[0], save_prefix, norm=norm, filter=filter, size=size)
    
        if os.path.isfile(save_prefix + ".npy"):
            _success = True
            list_img.append(save_prefix)
        else:
            _success = False
    
    print(f'Número de imagens transformadas: {len(list_img)}')