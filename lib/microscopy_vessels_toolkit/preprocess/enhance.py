import cv2
import numpy as np


def enhance_microscopy(img, mask=None, gamma="auto", clahe=5, denoise=7):
    is_float = img.dtype == np.float32 or img.dtype == np.float64
    if is_float:
        img = (img * 255).astype(np.uint8)

    if gamma is not None:
        img = gamma_correction(img, mask, gamma)
    if clahe is not None:
        img = rgb_clahe(img, tile_size=clahe)
    if denoise:
        img = denoising(img, denoise)
    if mask is not None:
        img *= mask[:, :, None]

    if is_float:
        img = img.astype(np.float32) / 255

    return img


def gamma_correction(src, mask=None, gamma="auto"):
    if gamma == "auto":
        gamma_src = src
        if mask is not None:
            gamma_src = gamma_src[mask]
        gamma = np.log(gamma_src.mean()) / np.log(128)

    invGamma = 1 / gamma
    table = np.array([((i / 255) ** invGamma) * 255 for i in range(256)], dtype=np.uint8)

    return cv2.LUT(src, table)


def rgb_clahe(img, tile_size=5):
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(tile_size, tile_size))
    return cv2.merge([clahe.apply(_) for _ in cv2.split(img)])


def lab_clahe(img, tile_size=5, only_l=False):
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(tile_size, tile_size))

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    if only_l:
        l, a, b = cv2.split(lab)  # noqa: E741
        l = clahe.apply(l)  # noqa: E741
        lab = cv2.merge((l, a, b))
    else:
        lab = cv2.merge([clahe.apply(_) for _ in cv2.split(lab)])
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return img


def hsv_clahe(img, tile_size=5):
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(tile_size, tile_size))

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = clahe.apply(v)
    s = clahe.apply(s)
    hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return img


def denoising(img, size=7):
    return cv2.bilateralFilter(img, size, 50, 50)
