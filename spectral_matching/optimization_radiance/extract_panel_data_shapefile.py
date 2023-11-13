import numpy as np
import geopandas as gpd
from utils.mask_from_shapefile import mask_from_shapefile
import tifffile
import os
from scipy.io import savemat, loadmat
def hdr2tif(hdr_filename):
    tif_filename = hdr_filename.replace(".hdr",".tif")
    bin_filename = hdr_filename.replace(".hdr","")
    os.system(f"gdal_translate -of GTiff {bin_filename} {tif_filename}")
    print("Converted binary image to TIF.")
    return tif_filename

def raster_mask_from_shapefile(raster_path, shapefile_path, column):

    # create mask from shapefile
    train_df = gpd.read_file(shapefile_path)
    masks_dict, wls = mask_from_shapefile(raster_path, shapefile_path, column)
    return masks_dict, wls


def main(filename, shapfile_path):

    # convert the data to tif
    if filename.endswith(".hdr"):
        try:
            filename = hdr2tif(filename)
        except:
            Exception("Failed to convert HDR to TIF.")

    tiff_image = tifffile.imread(filename)
    masks_dict_bool, wls = raster_mask_from_shapefile(filename, shapfile_path, column = "id")

    mat_file_dict = {}
    for k,v in masks_dict_bool.items():
        mat_file_dict[k.replace("group","panel")] = tiff_image[v]
    mat_file_dict["wls"] = wls

    savemat(filename.replace(".tif","_panels.mat"), mat_file_dict)
    print("saved mat file to: ", filename.replace(".tif","_panels.mat"))

if __name__ == "__main__":

    datas = ["/Volumes/T7/axhcis/Projects/NURI/data/uk_lab_data/cal_test/2023_10_12_10_56_30_VNIR/data.hdr",
             "/Volumes/T7/axhcis/Projects/NURI/data/uk_lab_data/cal_test/2023_10_12_11_15_28_SWIR/data.hdr",
             "/Volumes/T7/axhcis/Projects/NURI/data/uk_lab_data/cal_test/2023_10_12_11_21_02_SWIR/data.hdr",
             "/Volumes/T7/axhcis/Projects/NURI/data/uk_lab_data/cal_test/2023_10_12_11_29_13_VNIR/data.hdr"]

    shapefiles = ["/Volumes/T7/axhcis/Projects/NURI/data/uk_lab_data/cal_test/2023_10_12_10_56_30_VNIR/panels.geojson",
             "/Volumes/T7/axhcis/Projects/NURI/data/uk_lab_data/cal_test/2023_10_12_11_15_28_SWIR/panels.geojson",
             "/Volumes/T7/axhcis/Projects/NURI/data/uk_lab_data/cal_test/2023_10_12_11_21_02_SWIR/panels.geojson",
             "/Volumes/T7/axhcis/Projects/NURI/data/uk_lab_data/cal_test/2023_10_12_11_29_13_VNIR/panels.geojson"]

    for data, shapefile in zip(datas, shapefiles):
        main(data, shapefile)

