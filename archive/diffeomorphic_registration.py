from spectral.io import envi
import numpy as np
import numpy.ma as ma
import os
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import minmax_scale
import matplotlib.cm as cm
import matplotlib.widgets as widgets
from scipy import interpolate
import pyelastix
def load_image_envi(waterfall_path):
    vnir_ds = envi.open(waterfall_path)
    vnir_profile = vnir_ds.metadata
    vnir_arr = vnir_ds.load()

    return vnir_arr, vnir_profile


def main():

    # load the SWIR image and select band 38
    or_hdr = "/Volumes/T7/axhcis/Projects/NURI/data/20210723_tait_labsphere/1133/SWIR/raw_1504_nuc_or_plusindices3.hdr"
    swir_arr, swir_profile= load_image_envi(or_hdr)
    swir_patch = swir_arr[1064:1181, 493:668, 37]


    # load the micasense and select last band
    mica_hdr = "/Volumes/T7/axhcis/Projects/NURI/data/20210723_tait_labsphere/1133/Micasense/NURI_micasense_1133_transparent_mosaic_stacked_warped.hdr"
    mica_arr, mica_profile = load_image_envi(mica_hdr)
    mica_patch = mica_arr[2655:2772, 812:987, -1]

    # fig, ax = plt.subplots(1,2, figsize = (10,8))
    # ax[0].imshow(swir_patch)
    # ax[0].axis("off")
    # ax[1].imshow(mica_patch)
    # ax[1].axis("off")

    # Get params and change a few values
    params = pyelastix.get_default_params()
    params.MaximumNumberOfIterations = 200
    params.FinalGridSpacingInVoxels = 10

    # Apply the registration (im1 and im2 can be 2D or 3D)
    mica_patch_deformed, field = pyelastix.register(np.ascontiguousarray(mica_patch.astype('float32')), np.ascontiguousarray(swir_patch.astype('float32')), params)



if __name__ == "__main__":
    main()







