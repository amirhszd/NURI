# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 10:30:04 2021

@author: WKS202
"""
import shapely
from rasterio import features
import numpy as np

def mask_to_polygons(mask, transform, connectivity):
    #shapely.speedups.disable()
    all_polygons = []
    features_ = features.shapes(mask.astype(np.int16), mask=(mask >0), connectivity = connectivity, transform=transform)
    
    for c, (shape, value) in enumerate(features_, start = 1):
        all_polygons.append(shapely.geometry.shape(shape))     
        print(f"{c} done.", end='\r')
        
    return all_polygons