import numpy as np
import rasterio
from spectral.io import envi
import cv2
import sys
import os

def load_image_envi(header_path):

    ds = envi.open(header_path)
    profile = ds.metadata
    wavelengths = profile["wavelength"]
    wavelengths = np.array([float(i) for i in wavelengths])

    return ds.load(), profile, wavelengths

def save_image_envi(array, output_hdr_path, master_metadata):
    envi.save_image(output_hdr_path, array, metadata=master_metadata, force=True)

    print("image saved to: " + output_hdr_path)
