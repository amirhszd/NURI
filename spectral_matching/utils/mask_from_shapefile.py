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

column = "class"


def mask_from_shapefile(raster_path, shape_path, column):
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
        wls = [float(i.split(" ")[0]) for i in src.descriptions]
    im_size = (raster_meta['height'], raster_meta['width'])        
    
    #load shapefile or GeoJson
    masks = []
    train_df = gpd.read_file(shape_path)

    if raster_meta["crs"] is not None:
        train_df = train_df.to_crs(raster_meta["crs"])
    else:
        print("Data does not have CRS attached to it")
        # if the transform is all posititve but the geopandas dataframe shows
        # negative direction in the y direction
        if raster_meta['transform'][4] > 0 and (train_df.iloc[0]["geometry"].centroid.xy[1][0] < 0):
            from affine import Affine
            raster_meta['transform'] = Affine(raster_meta['transform'][0],
                                               raster_meta['transform'][1],
                                               raster_meta['transform'][2],
                                               raster_meta['transform'][3],
                                               -1*raster_meta['transform'][4],
                                               raster_meta['transform'][5])



    n_groups = len(np.unique(train_df[column]))
    poly_dict = {"group_" + str(id): [] for id in np.unique(train_df[column])}
    masks_dict = {"group_" + str(id): [] for id in np.unique(train_df[column])}

    for id in np.unique(train_df[column]):
        sub_df = train_df[train_df[column] == id]
        if sub_df.size == 0:
            raise Exception("All classes not included, double check your shapefile.")

        for num, row in sub_df.iterrows():
            p = row['geometry']
            try:
                poly = poly_from_utm(p, raster_meta['transform'])
                poly_dict["group_" + str(id)].append((poly,
                                                           row[column]))
            except:
                continue
            
        mask = rasterize(shapes=poly_dict["group_" + str(id)],
                         out_shape=im_size)
        masks_dict["group_" + str(id)] = mask > 0

    return masks_dict, wls


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
