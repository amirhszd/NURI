import cv2
import tifffile
from PIL import Image
from osgeo import gdal
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import skimage

def main(warp_iamge, base_image):
    base_ds = gdal.Open(base_image)
    geotransform = base_ds.GetGeoTransform()
    x_res, y_res = geotransform[1], -geotransform[5]

    warp_options = gdal.WarpOptions(xRes=x_res, yRes=y_res)
    output_name = warp_image.replace(".", "_resampled.")
    gdal.Warp(output_name, warp_image, options=warp_options)


if __name__ == "__main__":

    warp_image = "/dirs/data/tirs/axhcis/Projects/NURI/Data/20210723_tait_labsphere_processed/1205/pair/raw_1504_nuc_rd_or_refl_swir_QUAD.img"
    base_image = "/dirs/data/tirs/axhcis/Projects/NURI/Data/20210723_tait_labsphere_processed/1205/pair/raw_1504_nuc_rd_or_refl_swir.img"

    main(warp_image, base_image)