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
from typing import Any

import cv2
from numpy import concatenate, float32, ndarray, newaxis, save
from pydicom import FileDataset, dcmread
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from pt.utils.filters_utils import apply_filters


def dicom_preprocess(
    dicom_file: str,
    save_prefix: str,
    norm: str = "",
    filter: str = "",
    size: int = 224,
    man: str = "",
) -> None:
    try:
        # Read needed dicom tags
        ds: FileDataset = dcmread(dicom_file)  # , stop_before_pixels=True)

        type_image = float32

        curr_img: ndarray[type_image, Any] = ds.pixel_array.astype(type_image)
        img_flat: ndarray[type_image, Any]
        norm_img: ndarray[type_image, Any]
        fabricante: bool = True

        try:
            fabricante = man in ds.Manufacturer
        except AttributeError:
            fabricante = True

        if filter != "":
            image_min = curr_img.min()
            image_range = curr_img.max() - image_min
            if image_range == 0:
                curr_img.fill(0)
            else:
                curr_img = (curr_img - image_min) / image_range
                curr_img *= 255
        if fabricante:
            curr_img = apply_filters(curr_img, filter, type_image)

            img_flat = curr_img.flatten()
            norm_img = (
                (MinMaxScaler() if norm == "min-max" else StandardScaler())
                .fit_transform(img_flat.reshape(-1, 1))
                .flatten()
            )
            curr_img = norm_img.reshape(curr_img.shape)

        else:
            print(dicom_file + " não é da marca informada " + man)

        # Resize and replicate into 3 channels
        curr_img = cv2.resize(curr_img, (size, size))

        curr_img = concatenate(
            (
                curr_img[:, :, newaxis],
                curr_img[:, :, newaxis],
                curr_img[:, :, newaxis],
            ),
            axis=-1,
        )

        # Save output file
        os.makedirs(os.path.dirname(save_prefix), exist_ok=True)
        save(save_prefix + ".npy", curr_img.astype(type_image))

    except BaseException as e:
        print(f"[WARNING] Reading {dicom_file} failed with Exception: {e}")
        raise
