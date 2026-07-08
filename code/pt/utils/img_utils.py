import cv2
import numpy as np
from matplotlib import pyplot as plt
from typing import List
import scipy.ndimage.filters as flt
import warnings

def clahe(img, clip=2.0, size=32):
    #contrast enhancement
    if size == None:
        clahe = cv2.createCLAHE(clipLimit=clip)
    else:
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(size,size)) # depois deixar esse
    
    cl = clahe.apply(np.array(img).astype(np.uint8))
    return cl
