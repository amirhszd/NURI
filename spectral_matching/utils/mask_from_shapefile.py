# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 11:11:17 2021

@author: WKS202
"""

import os
import rasterio
from rasterio.plot import reshape_as_image
import rasterio.mask
from rasterio.features import rasterize
import pandas as pd
import geopandas as gpd
from shapely.geometry import mapping, Point, Polygon
from shapely.ops import cascaded_union, unary_union
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from shapely.geometry import Polygon
from pyproj import Proj, Transformer
from shapely.ops import cascaded_union, unary_union
import numpy as np



def mask_from_shapefile(raster_path, shape_path):
    """
    Function generating masks from a vector file.

    Parameters
    ----------
    raster_path : String
        Path for raster. Should be a .tiff file, and not a .tif file. Weird
        thing with rasterio.
    shape_path : String
        Path for shapefile.

    Returns
    -------
    None
    """    
    
    # Load raster    
    with rasterio.open(raster_path, "r") as src:
        raster_meta = src.meta
    im_size = (raster_meta['height'], raster_meta['width'])        
    
    #load shapefile or GeoJson
    masks = []
    train_df = gpd.read_file(shape_path)

    # match the crs with raster crs
    # if train_df.crs != raster_meta["crs"]:
    train_df = train_df.to_crs(raster_meta["crs"])
    n_classes = int(train_df["class"].max())
    poly_dict = {"class_" + str(class_n): [] for class_n in range(1,n_classes+1)}

    for class_n in range(1, n_classes+1):
        sub_df = train_df[train_df["class"] == class_n]
        if sub_df.size == 0:
            raise Exception("All classes not included, double check your shapefile.")

        for num, row in sub_df.iterrows():
            p = row['geometry']
            try:
                poly = poly_from_utm(p, raster_meta['transform'])
                poly_dict["class_" + str(class_n)].append((poly,
                                                           row["class"]))
            except:
                continue
            # for p in row['geometry']:
            #     poly = poly_from_utm(p, raster_meta['transform'])
            #     poly_dict["class_" + str(class_n)].append((poly,
            #                                                row["class"]))
            
        mask = rasterize(shapes=poly_dict["class_" + str(class_n)], 
                         out_shape=im_size)
        masks.append(np.expand_dims(mask, 0))
    # created two layers of masks to cover overlapping areas
    masks = np.concatenate(masks,0)

    meta = src.meta.copy()
    meta.update(count = n_classes)
    meta.update(dtype = "uint8")
    meta.update(nodata = 0)
    mask_path = os.path.join(str(Path(shape_path).parent.absolute()), os.path.basename(shape_path).split(".")[0] + "_mask.tiff")
    with rasterio.open(mask_path,
                       'w', **meta) as dst:
        dst.write(masks.astype(rasterio.uint8))   
        
    return mask_path


def poly_from_utm(polygon, transform):
    """
    Auxilary function grabbing polygons and transforming from any coordinates to UTM coordinates
    to image coordinates.
    """
    poly_pts = []

    poly = unary_union(polygon)
    for i in np.array(poly.exterior.coords):
        poly_pts.append(~transform * tuple(i))

    new_poly = Polygon(poly_pts)
    return new_poly
