from cv2 import createCLAHE
from cv2.typing import MatLike
from numpy import array, uint8


def clahe(img: MatLike, clip: float = 2.0, size: int = 32) -> MatLike:

    clahe = createCLAHE(clipLimit=clip, tileGridSize=(size, size))

    cl: MatLike = clahe.apply(array(img).astype(uint8))
    return cl
