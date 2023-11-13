# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 12:16:26 2021

@author: WKS202
"""

import os
import rasterio
from rasterio.plot import reshape_as_image
from rasterio.mask import mask
from rasterio.features import rasterize
import pandas as pd
import geopandas as gpd
from shapely.geometry import mapping, Point, Polygon
from shapely.ops import cascaded_union
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import geojson
import fiona
# from shapely.ops import cascaded_unionarray


def crop_from_shapefile(raster_path, shape_path, output_dir, nodata = 0):
    """
    Function generating cropping a raster using a shapefile boundary file.

    Parameters
    ----------
    raster_path : String
        Path for raster. Should be a .tiff file, and not a .tif file. Weird
        thing with rasterio.
    shape_path : String
        Geojson boundary file.

    Returns
    -------
    None
    """    
    
    # Load raster    

    with fiona.open(shape_path, "r") as shapefile:
        shapes = [feature["geometry"] for feature in shapefile]
    
    with rasterio.open(raster_path) as src:
        crop_raster, crop_transform = rasterio.mask.mask(src, shapes, nodata = nodata, crop =True)
        out_meta = src.meta
        
    out_meta.update({"driver": "GTiff",
                     "nodata": nodata,
                     "height": crop_raster.shape[1],
                     "width": crop_raster.shape[2],
                     "transform": crop_transform})
    
    crop_raster_path = os.path.join(output_dir,
                                    os.path.basename(raster_path).replace(".tiff",
                                                                     "_masked.tiff"))
    with rasterio.open(crop_raster_path, "w", **out_meta) as dest:
        dest.write(crop_raster)                
        
    return crop_raster_path, crop_raster
