import pandas as pd
import numpy as np
import cv2
from scipy.ndimage.morphology import binary_fill_holes

def get_indices_pandas(data):
    d = data.ravel()
    f = lambda x: np.unravel_index(x.index, data.shape)
    return pd.Series(d).groupby(d).apply(f)


def normalize(arr):
    arr = (arr - arr.min())/(arr.max() - arr.min() + 1e-7)
    arr = arr*255
    return arr

def round_up_to_odd(f):
    return int(np.floor(f) // 2 * 2 + 1)

def get_contours(img):
    _, contours, hierarchy = cv2.findContours(img,cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
    mask = np.zeros_like(img)
    for cntrs in contours:
        mask = cv2.fillConvexPoly(mask, points = cntrs, color = 255)           
    mask = binary_fill_holes(mask)*1
    mask = mask.astype("uint8")
    return mask    