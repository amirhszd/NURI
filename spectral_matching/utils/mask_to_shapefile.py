# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 11:31:15 2021

@author: WKS202
"""
import rasterio
import cv2
import numpy as np
import tqdm
import geopandas as gpd

def mask_to_shapefile(raster_path, y_pred_mask, proba_map, min_size = 100, max_size = 5000, ksize = 5):
    
    
    with rasterio.open(raster_path, "r") as src:
        transform = src.transform

    y_pred_mask_class = (y_pred_mask == 1)*1
    y_pred_mask_class = y_pred_mask_class.astype("uint8")
    
    y_pred_mask_blur = cv2.GaussianBlur(y_pred_mask_class,(ksize,ksize),0)
    # defining the kernel i.e. Structuring element
    kernel = np.ones((ksize, ksize), np.uint8)      
    # defining the opening function 
    # over the image and structuring element
    mask_opened = cv2.morphologyEx(y_pred_mask_blur, cv2.MORPH_OPEN, kernel)
    #mask_erode = cv2.erode(mask_opened,kernel, iterations  = 1)    
    mask_final = cv2.medianBlur(mask_opened, ksize)
    
    del mask_opened, kernel, y_pred_mask_blur
    
    # divide the mask into splits of 4000 pixels if its above 4000 pixels in height, otherwise rotate it
    division_n = 2500
    long_axis = np.argmax(mask_final.shape)
    # rolling axis to make sure the longer side is coming first    
    # if the mask's bigger axis is larger than 4000 chop it up
    if mask_final.shape[long_axis] > division_n:
        ndiv = int(mask_final.shape[long_axis]/division_n)
        divs = [(i+1)*division_n for i in range(ndiv)]
        mask_final_splits = np.split(mask_final,divs, long_axis)
                
        splits_indices = [[None,None] for i in mask_final_splits]
        stati_axis_range = [0,mask_final.shape[long_axis-1]]
        start_long_axis = 0
        for c,i in enumerate(splits_indices):
            i[long_axis] = [start_long_axis,start_long_axis + mask_final_splits[c].shape[long_axis]]
            i[long_axis - 1] = stati_axis_range
            start_long_axis = i[long_axis][1]            
        all_polygons = []
        probs_avg = []
        
        for j, mask_final_split in enumerate(mask_final_splits, start = 0):
            # run connected components              
            connectivity = 8 
            # Perform the operation
            (_, labels, stats, _) = cv2.connectedComponentsWithStats(mask_final_split, connectivity, cv2.CV_32S)
            to_keep = np.where((stats[:,4] > min_size) & (stats[:,4] < max_size))[0]  
            to_keep = [v for v in to_keep if v != 0]                        
            if len(to_keep) > 0:     
                new_label = np.zeros_like(mask_final).astype("uint32")
                ind = get_indices_pandas(labels)
                c = 1
                pbar = tqdm(total = len(to_keep), desc = f"Retouching labels part {j} of {len(mask_final_splits)}...")           
                for label_value in to_keep:
                    if label_value != 0:        
                        indices = ind[label_value]                        
                        new_label[splits_indices[j][0][0]:splits_indices[j][0][1],splits_indices[j][1][0]:splits_indices[j][1][1]][indices] = c
                        probs_avg.append(np.mean(proba_map[indices[0] + splits_indices[j][0][0],indices[1] + splits_indices[j][1][0]]))
                        c = c + 1       
                        pbar.update(1)
                # convert mask to polygons
                all_polygons.extend(mask_to_polygons(new_label, transform, connectivity)) 
    #otherwise just use as is                                  
    else:                            
        # run connected components              
        connectivity = 8 
        # Perform the operation
        (_, labels, stats, _) = cv2.connectedComponentsWithStats(mask_final, connectivity, cv2.CV_32S)
        to_keep = np.where((stats[:,4] > min_size) & (stats[:,4] < max_size))[0]  
        to_keep = [v for v in to_keep if v != 0]
                
        if len(to_keep) > 0:           
            ind = get_indices_pandas(labels)
            probs_avg = []
            new_label = np.zeros_like(labels)
            c = 1
            pbar = tqdm(total = len(to_keep), desc = "Retouching labels...")           
            for label_value in to_keep:
                if label_value != 0:        
                    indices = ind[label_value]
                    new_label[indices] = c
                    probs_avg.append(np.mean(proba_map[indices]))
                    c = c + 1       
                    pbar.update(1)
            # convert mask to polygons
            all_polygons = mask_to_polygons(new_label, transform, connectivity)                       
            
    df = {"id": [i for i in range(len(all_polygons))],
          "accuracy":probs_avg}
        
    gdf = gpd.GeoDataFrame(df, geometry=gpd.GeoSeries(all_polygons))
    gdf = gdf.round(3)
    #gdf = gdf.dissolve().explode()
    print("")
    print(f"{len(gdf)} predictions generated!")
    gdf.to_file(os.path.join(os.path.dirname(raster_path),"predictions.geojson"))     

def get_indices_pandas(data):
    d = data.ravel()
    f = lambda x: np.unravel_index(x.index, data.shape)
    return pd.Series(d).groupby(d).apply(f)        