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



def load_images_envi(mica_path, swir_path):

    mica_ds = envi.open(mica_path)
    mica_profile = mica_ds.metadata
    mica_wavelengths = [475,560,634,668,717]
    mica_arr = mica_ds.load()

    swir_ds = envi.open(swir_path)
    swir_profile = swir_ds.metadata
    swir_wavelengths = swir_profile["wavelength"]
    swir_wavelengths = np.array([float(i) for i in swir_wavelengths])
    swir_arr = swir_ds.load()

    return (mica_arr, mica_profile, mica_wavelengths ), (swir_arr, swir_profile, swir_wavelengths)


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

def save_crop_image_envi(swir_arr, swir_wavelengths, swir_path, vnir_arr, vnir_profile, M):

    # a sample warped band to get the extents of the image
    sample_warped_band = cv2.warpPerspective(swir_arr[..., 0], M, (vnir_arr.shape[1], vnir_arr.shape[0]))[..., None]

    # Extract non-zero data extents
    non_zero_indices = np.nonzero(sample_warped_band)
    xmin, xmax = min(non_zero_indices[1]), max(non_zero_indices[1])
    ymin, ymax = min(non_zero_indices[0]), max(non_zero_indices[0])

    # creating lat and lon rasters and then cropping them accordingly
    lon_values = np.linspace(float(vnir_profile["map info"][3]),
                             float(vnir_profile["map info"][3]) + (int(vnir_profile["samples"])) * float(vnir_profile["map info"][5]),
                             int(vnir_profile["samples"]))
    lat_values = np.linspace(float(vnir_profile["map info"][4]),
                             float(vnir_profile["map info"][4]) - (int(vnir_profile["lines"])) * float(vnir_profile["map info"][6]),
                             int(vnir_profile["lines"]))
    lon_raster, lat_raster = np.meshgrid(lon_values, lat_values)
    lon_raster_crop = lon_raster[ymin:ymax+1, xmin:xmax+1]
    lat_raster_crop = lat_raster[ymin:ymax+1, xmin:xmax+1]
    del lon_raster, lat_raster

    # warping and cropping the data to the extents
    swir_registered_bands = []
    for i in range(len(swir_wavelengths)):
        swir_registered_bands.append(
            cv2.warpPerspective(swir_arr[...,i], M, (vnir_arr.shape[1], vnir_arr.shape[0]))[ymin:ymax+1, xmin:xmax+1, None])

    # replicating vnir metadata except the bands and wavelength
    metadata = {}
    for k, v in vnir_profile.items():
        if (k != "bands") or (k != "wavelength"):
            metadata[k] = vnir_profile[k]
    metadata["bands"] = str(len(swir_wavelengths))
    metadata["wavelength"] = [str(i) for i in swir_wavelengths]
    metadata["map info"][3] = lon_raster_crop.min()
    metadata["map info"][4] = lat_raster_crop.max()
    metadata["description"] = swir_path.replace(".hdr", "_warped.hdr")

    swir_registered_bands = np.concatenate(swir_registered_bands, 2)
    envi.save_image(swir_path.replace(".hdr", "_warped.hdr"), swir_registered_bands, metadata=metadata, force=True)

    print("image saved to: " + swir_path.replace(".hdr", "_warped.hdr"))
