from typing import Any

from cv2 import GaussianBlur, bilateralFilter, medianBlur
from numpy import array, float32, ndarray, uint8
from scipy.signal import wiener

from pt.utils.img_utils import clahe


def apply_filters(
    curr_img: ndarray[float32, Any], filter: str, type_image: type
) -> ndarray[float32, Any]:
    if "MEDIAN" in filter:
        # Median Filter
        curr_img = medianBlur(array(curr_img, dtype=uint8), ksize=3)

    if "GAUSSIAN" in filter:
        # Gaussian filter
        curr_img = GaussianBlur(curr_img, ksize=(3, 3), sigmaX=1, sigmaY=1)

    if "WIENER" in filter:
        # weiner filter 7x7
        curr_img = wiener(curr_img.astype(type_image), (7, 7))

    if "BILATERAL" in filter:
        # Bilateral filter
        curr_img = array(curr_img, dtype=uint8)
        curr_img = bilateralFilter(curr_img, 5, 5 * 2, 5 / 2)

    if "CLAHE" in filter:
        curr_img = clahe(curr_img, size=32, clip=3.0)

    return curr_img
