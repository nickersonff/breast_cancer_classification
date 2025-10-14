# Copyright 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import cv2
import numpy as np
import pydicom
#import skimage.io
#import skimage.exposure
from utils.img_utils import *
from scipy import stats
from scipy.signal import wiener
from sklearn import preprocessing

def crop_image(image):
    
    im_clone = image.copy()
    im_clone = np.frombuffer(im_clone, np.uint8)
    # Aplica um limiar para separar a mama do fundo preto
    _, thresh = cv2.threshold(im_clone, 20, 255, cv2.THRESH_BINARY)

    # Encontra os contornos
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Encontra o maior contorno (assumindo que é a mama)
    largest_contour = max(contours, key=cv2.contourArea)

    # Encontra a caixa delimitadora do contorno
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Recorta a imagem para conter apenas a mama
    cropped_image = image[y:y+h, x:x+w]

    return cropped_image

def dicom_preprocess(dicom_file, save_prefix, norm="", filter="", size=224, man=""):
    try:
        # Read needed dicom tags
        ds = pydicom.dcmread(dicom_file)  # , stop_before_pixels=True)
        print(ds)
        try: # TENTA RECUPERAR UM METADADO DA LIGA
            code = ds.ViewCodeSequence[0].ViewModifierCodeSequence[0].CodeMeaning
        except BaseException:
            code = None
        
        try: # TENTA RECUPERAR UM METADADO DO DDSM - FULL IMAGE, CROPED OU ROI
            type_img = ds.SeriesDescription
        except:
            type_img = None

        # Filter image
        dc_tags = ""
        curr_img = ds.pixel_array.astype(np.float32)
        fabricante = True
        try:
            fabricante = man in ds.Manufacturer
        except BaseException:
            fabricante = True
        
        if filter != "":
            curr_img = (curr_img - np.min(curr_img))/ (np.max(curr_img) - np.min(curr_img))
            curr_img *= 255
        
        if (fabricante):
            
            if "MEDIAN" in filter:
                # Median Filter 3x3
                curr_img = cv2.medianBlur(np.array(curr_img, dtype=np.uint8), ksize=3)
            if "GAUSSIAN" in filter:
                # Gaussian filter 7x7
                curr_img = cv2.GaussianBlur(curr_img,ksize=(3,3),sigmaX=1, sigmaY=1)
            if "WIENER" in filter:
                # weiner filter 7x7
                curr_img = wiener(curr_img.astype(np.float32), (7, 7))
            if "BILATERAL" in filter:
                # Bilateral filter 5x5
                curr_img = np.array(curr_img, dtype=np.uint8)
                curr_img = cv2.bilateralFilter(curr_img, 5, 5 * 2, 5 / 2)
            if "CLAHE" in filter:
                curr_img = clahe(curr_img, size=32)
            #Normalização com Z-Score
            if norm == "z-score":
                img_flat = curr_img.flatten()
                norm_img = preprocessing.StandardScaler().fit_transform(img_flat.reshape(-1,1)).flatten()
                curr_img = norm_img.reshape(curr_img.shape)
            elif norm == "min-max":
                img_flat = curr_img.flatten()
                norm_img = preprocessing.MinMaxScaler().fit_transform(img_flat.reshape(-1,1)).flatten()
                curr_img = norm_img.reshape(curr_img.shape)
        else:
            print(dicom_file + " não é da marca informada " + man)
            return False, f"{dicom_file} failed"
        
        # Resize and replicate into 3 channels
        if size != None:
            curr_img = cv2.resize(curr_img, (size, size))      
        curr_img = np.concatenate(
            (
                curr_img[:, :, np.newaxis],
                curr_img[:, :, np.newaxis],
                curr_img[:, :, np.newaxis],
            ),
            axis=-1,
        )
        scaled_image = curr_img.copy()
        scaled_image = (scaled_image - np.min(scaled_image))/ (np.max(scaled_image) - np.min(scaled_image))
        scaled_image *= 255
        
        if type_img != None and 'cropped' in type_img:
            save_prefix += '_CROP'
        elif type_img == None and 'cropped' in dicom_file:
            save_prefix += '_CROP'
        elif type_img != None and 'ROI' in type_img:
            save_prefix += '_ROI'
        elif type_img == None and 'ROI' in dicom_file:
            save_prefix += '_ROI'
        cv2.imwrite(save_prefix + ".png", scaled_image.astype(np.uint8))
        # Save output file
        os.makedirs(os.path.dirname(save_prefix), exist_ok=True)
        np.save(save_prefix + ".npy", curr_img.astype(np.float32))
            
    except BaseException as e:
        print(f"[WARNING] Reading {dicom_file} failed with Exception: {e}")
        return False, f"{dicom_file} failed"

    return True, dc_tags

