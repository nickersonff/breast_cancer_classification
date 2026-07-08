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

def dicom_preprocess(dicom_file, save_prefix, norm="", filter="", size=224, man=""):
    try:
        # Read needed dicom tags
        ds = pydicom.dcmread(dicom_file)  # , stop_before_pixels=True)
        
        try: 
            code = ds.ViewCodeSequence[0].ViewModifierCodeSequence[0].CodeMeaning
        except BaseException:
            code = None
        
        try: 
            type_img = ds.SeriesDescription
        except:
            type_img = None

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
                # Median Filter

                curr_img = cv2.medianBlur(np.array(curr_img, dtype=np.uint8), ksize=3)

            if "GAUSSIAN" in filter:
                # Gaussian filter
                curr_img = cv2.GaussianBlur(curr_img,ksize=(3,3),sigmaX=1, sigmaY=1)

            if "WIENER" in filter:
                # weiner filter 7x7
                curr_img = wiener(curr_img.astype(np.float32), (7, 7))
                
            if "BILATERAL" in filter:
                # Bilateral filter
                curr_img = np.array(curr_img, dtype=np.uint8)
                curr_img = cv2.bilateralFilter(curr_img, 5, 5 * 2, 5 / 2)
                
            if "CLAHE" in filter:
                curr_img = clahe(curr_img, size=32, clip=3.0)
                
            if norm == "z-score":
                #print('Usou z-score para padronizar!')
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

        # Save output file        
        os.makedirs(os.path.dirname(save_prefix), exist_ok=True)        
        np.save(save_prefix + ".npy", curr_img.astype(np.float32))
            
    except BaseException as e:
        print(f"[WARNING] Reading {dicom_file} failed with Exception: {e}")
        return False, f"{dicom_file} failed"

    return True, dc_tags

