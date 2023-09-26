from scipy.io import loadmat
from imresize import imresize
from fusion_algorithms import fusion_algorithms
import matplotlib
from hypersharpening_selected_bands import PAN_Select
from hypersharpening_synthesized_bands import PAN_HyperSharpening
matplotlib.use('Agg')
import numpy as np
import tifffile
from utils import plot_imagefile

if __name__ == "__main__":
    """
    Most of these algorithms are run by first upsampling an specific cube, then using another dataset with higher
    resolution to integrate spatial components into. For example, the pansharpening is done by first upsampling the 
    hyperspectral image using basic upsampling, then using the informaiton in pansharpened image to ingest that
    into the hyperspectral image.
     
    The steps for fusion are outlined as below:
    
    these would be our real-world data
    I_SWIR_LR = 25x25
    I_MS_LR = 150x150
    I_PAN = 600x600
    ratio_SWIR_MS = 6 (or 1/6 to be real)
    
    
    1. upsample I_SWIR_LR to MS res ==> I_SWIR_upsampled_MS_res
    
    ############# Fusion Scheme 1
        1. fuse I_MS(600x600) with I_PAN(600x600)
        2. fuse I_SWIR(600x600) with I_PAN(600x600)
        3. concatenate the two outputs
    
    ############# Fusion Scheme 2
        1. takes in the fused (PAN_MS) data and SWIR data and perform synthesized bands or selected bands approach on it (I_PAN_MS).
        2. fuses SWIR with I_PAN_MS (from hypershaprneing/band select)
        
        SYNTHESIZED BAND APPROACH:
        
        
        SELECTED BAND APPROACH:
    
    
    ############ Fusion technique: pansharpening and hypersharpening is the same if you take care of MTF stuff with the
    below approach
    the fusion technique has two steps:
        1. Calculate LP filtered PAN image
            1. caluclate MTF of the sensor (could be bypassed)
                1. compute cut off frequency
                2. calculate gaussian window function
                3. generate gaussian filter
                4. normalize gaussian filter
                5. generate gaussian windowed FIR filter
                6. apply FIR filter"
            2. downsample PAN_LP image
            3. upsample PAN_LP image
        2. Perform generalized laplacian pyramid content based decision algorithm (GLP-CBD) algorithm:
            for each band:
            1. calculate covariance between passed image and PAN_LP
            2. calcualte g, paramter that shows the amount of intensity modulation to high frequency component
            3. image + g* (PAN - PAN_LP); (PAN - PAN_LP) is the high frequency PAN image

    
    
    """
    # TODO: the available data at full resolution are the data that are already interpolated for the ease of the user
    # TODO: the optimal way to do this is to have it as a class with functions
    hypershapening = "syn" # or "sel"
    selected_algorithm = 5 # 5.Gauss-CBD
    data = loadmat("/dirs/data/tirs/axhcis/Projects/NURI/Data/Dataset_WV3_Sim_APEX.mat")
    data["I_PAN"] = data["I_PAN"][...,None]


    ################### FUSION SCHEME 1 ############################
    # Fusino Scheme: upsampled MS is pansharpened, upsampled SWIR is pansharpened, then the two are concatenated.
    # PAN - MS
    # pefrom pansharpeing on MS, we are using I-MS as the upsampled MS_LR image, and then we fuse it using I-PAN.
    I_Fus_PAN_MS = fusion_algorithms(data["I_MS"], # also at 600x600, but it has been upsampled prior from original size.
                                     data["I_PAN"], # at 600x600
                                     data["ratio_MS_PAN"], # original size to upsampled size, ratio = 6
                                     data["sensor"], # the worldview 3 sensor MTF.
                                     data["imtag"],
                                     5) # the output is the res of pan, using GLP-CBD algorithm.
    # PAN - SWIR
    # pefrom pansharpeing on SWIR, we are using I-SWIR as the upsampled SWIR_LR image, and then we fuse it using I-PAN.
    I_Fus_PAN_SWIR = fusion_algorithms(data["I_SWIR"],  # also at 600x600, but it has been upsampled prior from original size.
                                       data["I_PAN"], # at 600x600
                                       data["ratio_MS_PAN"] * data["ratio_SWIR_MS"], # SWIR To pan is much larger range 6x4
                                       data["sensor"],
                                       data["imtag"],
                                       5)
    # now concatenate the two cubes
    I_Fus_Final_1 = np.concatenate([I_Fus_PAN_MS, I_Fus_PAN_SWIR], 2)
    # write to raster using tifffile
    tifffile.imwrite("/dirs/data/tirs/axhcis/Projects/NURI/Results/wv3_data/fused_image_scheme1.tif", I_Fus_Final_1)
    plot_imagefile("/dirs/data/tirs/axhcis/Projects/NURI/Results/wv3_data/fused_image_scheme1.tif")


    ########################## FUSION SCHEME 2 #########################
    # using fusion scheme 2 (synthesized band and selected bands)
    # TODO: you do not necessarily have to pass pan sharpened image to the model, keep it at the scale that it is,
    # TODO: and pass MS at the interpolated resolution.
    if hypershapening == "syn": # sythetic band approach
        #TODO: the I_PAN_image acts as a Pan image that we got using hypershapening that we later on pass to fusion algorithm
        # to get high spatial swir imag
        # then we concatenate that with MS
        # using pansharpenned MS
        I_PAN_MS = PAN_HyperSharpening(I_Fus_PAN_MS, data["I_SWIR"], data["ratio_SWIR_MS"] * data["ratio_MS_PAN"], 5);
    else: # band select approach
        I_PAN_MS = PAN_Select(I_Fus_PAN_MS, data["I_SWIR"], data["ratio_SWIR_MS"] * data["ratio_MS_PAN"], data["sensor"], data["imtag"])

    I_Fus_PAN_SWIR = fusion_algorithms(data["I_SWIR"], I_PAN_MS, data["ratio_SWIR_MS"]*data["ratio_MS_PAN"],
                                       data["sensor"], data["imtag"], 5)
    I_Fus_Final_2 = np.concatenate([I_Fus_PAN_MS, I_Fus_PAN_SWIR], 2)
    tifffile.imwrite("/dirs/data/tirs/axhcis/Projects/NURI/Results/wv3_data/fused_image_scheme2.tif", I_Fus_Final_2)
    plot_imagefile("/dirs/data/tirs/axhcis/Projects/NURI/Results/wv3_data/fused_image_scheme2.tif")


    ################# FUSION SCHEME NEW ###############
    # fusion scheme NEW, but not with the pansharpenned stuff. For the data we have whcih does not include pansharpen band
    I_SWIR_at_MS_res = imresize(data["I_SWIR_LR"], data["ratio_SWIR_MS"], 'bilinear')  # ratio is 6
    if hypershapening == "syn": # sythetic band approach
        I_PAN_MS = PAN_HyperSharpening(data["I_MS_LR"], I_SWIR_at_MS_res, data["ratio_SWIR_MS"], 5);
    else: # band select approach
        I_PAN_MS = PAN_Select(data["I_MS_LR"], I_SWIR_at_MS_res, data["ratio_SWIR_MS"], data["sensor"], data["imtag"]);

    I_Fus_PAN_SWIR = fusion_algorithms(I_SWIR_at_MS_res, I_PAN_MS, data["ratio_SWIR_MS"],
                                       data["sensor"], data["imtag"], 5)
    I_Fus_Final_2 = np.concatenate([data["I_MS_LR"], I_Fus_PAN_SWIR], 2)
    tifffile.imwrite("/dirs/data/tirs/axhcis/Projects/NURI/Results/wv3_data/fused_image_scheme_panfree.tif", I_Fus_Final_2)
    plot_imagefile("/dirs/data/tirs/axhcis/Projects/NURI/Results/wv3_data/fused_image_scheme_panfree.tif")

    tifffile.imwrite("/dirs/data/tirs/axhcis/Projects/NURI/Results/wv3_data/GT.tif",
                     data["I_GT"])
    plot_imagefile("/dirs/data/tirs/axhcis/Projects/NURI/Results/wv3_data/GT.tif")



    # TODO: RUN