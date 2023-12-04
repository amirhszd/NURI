import numpy as np
import rasterio
from spectral.io import envi
import cv2
import sys
import os

def load_images(vnir_path, swir_path):
    with rasterio.open(vnir_path) as src:
        vnir_arr = src.read()
        vnir_profile = src.profile
        vnir_wavelengths = np.array([float(i.split(" ")[0]) for i in src.descriptions])
    with rasterio.open(swir_path) as src:
        swir_arr = src.read()
        swir_profile = src.profile
        swir_wavelengths = np.array([float(i.split(" ")[0]) for i in src.descriptions])

    return (vnir_arr, vnir_profile, vnir_wavelengths), (swir_arr, swir_profile, swir_wavelengths)



def load_images_envi(vnir_path, swir_path):

    vnir_ds = envi.open(vnir_path)
    vnir_profile = vnir_ds.metadata
    vnir_wavelengths = vnir_profile["wavelength"]
    vnir_wavelengths = np.array([float(i) for i in vnir_wavelengths])
    vnir_arr = np.transpose(vnir_ds.load(), [2,0,1])

    swir_ds = envi.open(swir_path)
    swir_profile = swir_ds.metadata
    swir_wavelengths = swir_profile["wavelength"]
    swir_wavelengths = np.array([float(i) for i in swir_wavelengths])
    swir_arr = np.transpose(swir_ds.load(), [2,0,1])

    return (vnir_arr, vnir_profile, vnir_wavelengths), (swir_arr, swir_profile, swir_wavelengths)




def save_image(swir_arr, swir_wavelengths, swir_path, vnir_arr, vnir_profile, M):
    swir_registered_bands = []
    for i in range(len(swir_wavelengths)):
        swir_registered_bands.append(
            cv2.warpPerspective(np.fliplr(swir_arr[i]), M, (vnir_arr.shape[2], vnir_arr.shape[1])))

    # save data
    import os
    # output_path = swir_path.replace(".tif", "_warped.tif")
    output_path = os.path.basename(swir_path.replace(".tif", "_warped.tif"))
    vnir_profile.update(count=len(swir_registered_bands))
    with rasterio.open(output_path, 'w', **vnir_profile) as dst:
        for i, band in enumerate(swir_registered_bands):
            dst.write_band(i + 1, band)

    from gdal_set_band_description import set_band_descriptions
    bands = [int(i) for i in range(1, len(swir_wavelengths) + 1)]
    names = swir_wavelengths.astype(str)
    band_desciptions = zip(bands, names)
    set_band_descriptions(output_path, band_desciptions)


    print("Registered Image Saved to " + output_path)
    sys.exit()

def save_image_envi(swir_arr, swir_wavelengths, swir_path, vnir_arr, vnir_profile, M):

    swir_registered_bands = []
    for i in range(len(swir_wavelengths)):
        swir_registered_bands.append(
            cv2.warpPerspective(np.fliplr(swir_arr[i]), M, (vnir_arr.shape[2], vnir_arr.shape[1])))


    par_dir = os.path.dirname(swir_path)

    # replicating vnir metadata except the bands and wavelength
    metadata = {}
    for k, v in vnir_profile.items():
        if (k != "bands") or (k != "wavelength"):
            metadata[k] = vnir_profile[k]
    metadata["bands"] = str(len(swir_wavelengths))
    metadata["wavelength"] = [str(i) for i in swir_wavelengths]

    swir_registered_bands = np.transpose(swir_registered_bands, [1,2,0])
    envi.save_image(swir_path.replace(".hdr", "_warped.hdr"), swir_registered_bands, metadata=metadata, force=True)

    print("image saved to: " + swir_path.replace(".hdr", "_warped.hdr"))
