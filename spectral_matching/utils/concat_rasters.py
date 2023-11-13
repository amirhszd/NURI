# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 11:42:18 2021

@author: WKS202
"""

from osgeo import gdal
import rasterio
import numpy as np
import os
import glob
import shutil
import sys
import copy
def concat_rasters(filenames):
    
    main_dir = os.path.dirname(filenames[0])
        
    print("Resampling rasters...")
    resampled_filenames, resampled_dir = resample_rasters(filenames, main_dir)
            
    print("Splitting rasters...")    
    new_filenames, bands_dir, num_channels = split_rasters(resampled_filenames, main_dir)
    
    print("Merging rasters...")
    merge_raster_path = merge_rasters(new_filenames)
    
    print("Tidying up...")
    # moving original files to original_dir
    original_dir = os.path.join(os.path.dirname(filenames[0]), "original_files") 
    if not os.path.exists(original_dir):
        os.mkdir(original_dir)            
    for filename in filenames:
        shutil.move(filename, original_dir)    
    
    # removing resampled files
    shutil.rmtree(resampled_dir)
    
    # bringing back the merged file to initial dir
    merged_raster_path_moved = shutil.move(merge_raster_path, main_dir) 
    
    # removing bands folder
    shutil.rmtree(bands_dir)
    
    print(f"The new merged raster is written to {merged_raster_path_moved}")
    print(f"The original files are under {original_dir}")
    print("Done!")     

    return merged_raster_path_moved, num_channels    
    
    
def resample_rasters(filenames, main_dir):    
    
    resampled_dir = os.path.join(main_dir, "tmp_resampled")
    if not os.path.exists(resampled_dir):
        os.mkdir(resampled_dir)    
    
    all_width = []
    all_height = []
    # Load raster    
    for filename in filenames:
        with rasterio.open(filename, "r") as src:
            raster_meta = src.meta
            all_width.append(raster_meta["width"])
            all_height.append(raster_meta["height"])
            
    max_width = np.max(all_width)
    max_height = np.max(all_height)
    
    
    to_resample_ind = []
    for c, (width, heigth) in enumerate(zip(all_width, all_height)):
        if not (width== max_width and heigth== max_height):
            to_resample_ind.append(c)

    new_filenames = copy.copy(filenames)
    if len(to_resample_ind) == 0:
        print("No resampling needed. All rasters are the same size.")    
    else:
        print(f"Resampling {len(to_resample_ind)} raster(s).")        
        for index in to_resample_ind:
            basename = os.path.basename(filenames[index])
            new_filename = os.path.join(resampled_dir,
                                        basename.split(".")[0] + "_resampled." + basename.split(".")[1])
            
            options = gdal.TranslateOptions(format = 'GTiff',
                                            width = max_width,
                                            height = max_height,
                                            resampleAlg= "nearest")
            
            ds = gdal.Translate(new_filename,
                                filenames[index], options = options,
                                outputType = gdal.GDT_Float32)
            # closing the file
            ds = None
            
            new_filenames.pop(index)
            new_filenames.append(new_filename)
                        
        print("Resampling Done.")
    
    return new_filenames, resampled_dir
    
    

def split_rasters(filenames, main_dir):        
    # create a new directory
    bands_dir = os.path.join(main_dir,"tmp_bands")
    if not os.path.exists(bands_dir):
        os.mkdir(bands_dir)
        
    c = 1
    for filename in filenames:
        with rasterio.open(filename, "r") as src:
            raster_meta = src.meta
            raster_array = src.read()            
            if raster_array.shape[0] == 4 and raster_meta["dtype"] == "uint8": # if its rgb image basically
                raster_array = raster_array[:3]
            for band in raster_array:
                band = np.expand_dims(band,0)
                meta = raster_meta.copy()
                meta.update(count = 1)
                meta.update(dtype = "float32")
                band_dir = os.path.join(bands_dir, os.path.basename(filename).replace(".tif",f"_{c}.tif"))
                with rasterio.open(band_dir,'w', **meta) as dst:
                    dst.write(band.astype(rasterio.float32))   
                    sys.stdout.write("{} raster(s) written.".format(c))
                    sys.stdout.flush()
                    c = c + 1
                    
    new_filenames = glob.glob(bands_dir + "\**.tif*")
    print("Splitting done.")
    
    return new_filenames, bands_dir, c - 1

def merge_rasters(filenames):            
    dir_name = os.path.dirname(filenames[0])
    merged_vrt_path = os.path.join(dir_name,"merged.vrt")
    merged_tiff_path = os.path.join(dir_name,"merged.tiff")
    options = gdal.BuildVRTOptions(separate = True)
    vrt = gdal.BuildVRT(merged_vrt_path, filenames, options=options)
    gdal.Translate(merged_tiff_path, vrt)
    vrt = None
    
    print("Rasters merged.")
    return merged_tiff_path