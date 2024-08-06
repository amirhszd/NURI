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
from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
from dipy.align.metrics import SSDMetric, CCMetric
import dipy.align.imwarp as imwarp
from dipy.data import get_fnames
from dipy.io.image import load_nifti_data
from dipy.segment.mask import median_otsu
from dipy.viz import regtools
from scipy.ndimage import gaussian_filter
import matplotlib
from scipy.stats import pearsonr
from sklearn.preprocessing import minmax_scale

def load_image_envi(waterfall_path):
    vnir_ds = envi.open(waterfall_path)
    vnir_profile = vnir_ds.metadata
    vnir_arr = vnir_ds.load()

    return vnir_arr, vnir_profile

def callback_CC(sdr, status):
    # Status indicates at which stage of the optimization we currently are
    # For now, we will only react at the end of each resolution of the scale
    # space
    if status == imwarp.RegistrationStages.SCALE_END:
        # get the current images from the metric
        wmoving = sdr.metric.moving_image
        wstatic = sdr.metric.static_image
        # draw the images on top of each other with different colors
        regtools.overlay_images(wmoving, wstatic, 'Warped moving', 'Overlay',
                                'Warped static')


def get_highest_corr_coeff(swir_patch, mica_patch):

    correllation_arr = np.zeros((swir_patch.shape[-1], mica_patch.shape[-1]))

    # Loop through each band in SWIR and Mica
    for i in range(swir_patch.shape[2]):  # Assuming the number of bands is in the third dimension
        for j in range(mica_patch.shape[2]):  # Assuming the number of bands is in the third dimension
            # Calculate correlation coefficient using sampled data
            correlation, _ = pearsonr(swir_patch[...,i].flatten(), mica_patch[...,j].flatten())
            correllation_arr[i,j] = correlation


    highest_corr_ind = np.unravel_index(np.argmax(correllation_arr), correllation_arr.shape)
    highest_corr = correllation_arr[highest_corr_ind]

    return list(highest_corr_ind), highest_corr


def main():

    # load the SWIR image and select band 38
    or_hdr = "/Volumes/T7/axhcis/Projects/NURI/data/20210723_tait_labsphere/1133/SWIR/raw_1504_nuc_or_plusindices3.hdr"
    swir_arr, swir_profile= load_image_envi(or_hdr)
    swir_patch = swir_arr[1064:1181, 493:668].squeeze()


    # load the micasense and select last band
    mica_hdr = "/Volumes/T7/axhcis/Projects/NURI/data/20210723_tait_labsphere/1133/Micasense/NURI_micasense_1133_transparent_mosaic_stacked_warped.hdr"
    mica_arr, mica_profile = load_image_envi(mica_hdr)
    mica_patch = mica_arr[2655:2772, 812:987].squeeze()

    # performing correllation coefficient between the two to pick the highest
    highest_corr_ind, highest_corr = get_highest_corr_coeff(swir_patch, mica_patch)
    swir_patch = swir_patch[..., highest_corr_ind[0]]
    swir_patch = minmax_scale(swir_patch)
    mica_patch = mica_patch[..., highest_corr_ind[1]]
    mica_patch = minmax_scale(mica_patch)
    print(f"SWIR B{highest_corr_ind[0]} with Mica B{highest_corr_ind[1]}. Pearson's correllation coefficient: {highest_corr:.4f}")

    # Running low pass filter on the swir patch to minimize the
    swir_patch_lpf = gaussian_filter(swir_patch, sigma=1)
    mica_patch_lpf = gaussian_filter(mica_patch, sigma=0)

    sigma_diff = 3.0
    radius = 2
    metric = CCMetric(2, sigma_diff, radius)
    sdr = SymmetricDiffeomorphicRegistration(metric=metric,
                                             step_length=1.0,
                                             level_iters=[100,50],
                                             inv_iter=50,
                                             ss_sigma_factor=0.1,
                                             opt_tol=1.e-1)

    sdr.callback = callback_CC


    mapping = sdr.optimize(mica_patch_lpf, swir_patch_lpf)
    # regtools.plot_2d_diffeomorphic_map(mapping, 4, 'diffeomorphic_map.png')
    swir_patch_warped = mapping.transform(swir_patch)
    fig, ax = plt.subplots(3,2, figsize = (10,8))
    ax = ax.flatten()
    ax[0].imshow(swir_patch)
    ax[0].axis("off")
    ax[0].set_title("SWIR")
    ax[1].imshow(swir_patch_lpf)
    ax[1].axis("off")
    ax[1].set_title("SWIR LP")
    ax[2].imshow(mica_patch)
    ax[2].axis("off")
    ax[2].set_title("Mica")
    ax[3].imshow(mica_patch_lpf)
    ax[3].axis("off")
    ax[3].set_title("Mica LP")
    ax[4].imshow(swir_patch_warped)
    ax[4].axis("off")
    ax[4].set_title("SWIR warped")
    ax[5].axis("off")
    plt.show()





if __name__ == "__main__":

    """
    The problem with this registration is that it requires a symmetric registration, this is mainly used in meidcal imaging
    
    
    """
    main()







