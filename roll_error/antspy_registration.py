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
import ants

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

def create_patch_indices(array1):
    while True:
        # Generate random 150x150 patch indices
        rand_x = np.random.randint(0, array1.shape[1] - 200)
        patch_x_indices = np.arange(rand_x, rand_x + 200)

        rand_y = np.random.randint(0, array1.shape[0] - 200)
        patch_y_indices = np.arange(rand_y, rand_y + 200)

        yy, xx = np.meshgrid(patch_y_indices, patch_x_indices)

        # Check if there are no zeros in the corresponding positions of array1
        if np.all(array1[yy, xx, 0] != 0):
            break  # Break the loop if condition is satisfied

    return [yy, xx]

def apply_transform(fixed, moving, transform):

    regis_bands = []
    for i in range(moving.shape[-1]):
        regis_bands.append(ants.apply_transforms(fixed=fixed, moving=moving,
                                              transformlist=transform['fwdtransforms'])[...,None])
    regis_bands = np.concatenate(regis_bands, 2)
    return regis_bands

def get_transform(fixed, moving):
    transform = ants.registration(fixed=fixed, moving=moving, type_of_transform='Elastic')
    return transform

def main(swir_hdr, mica_hdr):

    # load the SWIR image and select band 38
    swir_arr, swir_profile= load_image_envi(swir_hdr)
    mica_arr, mica_profile = load_image_envi(mica_hdr)

    # grab a patch to calculate the highest coefficient bands
    patch_indices = create_patch_indices(swir_arr)
    swir_patch = swir_arr[patch_indices].squeeze()
    mica_patch = mica_arr[patch_indices].squeeze()

    # performing correllation coefficient between the two to pick the highest
    highest_corr_ind, highest_corr = get_highest_corr_coeff(swir_patch, mica_patch)
    swir_image = swir_arr[..., highest_corr_ind[0]].squeeze()
    mica_image = mica_arr[..., highest_corr_ind[1]].squeeze()
    mask = swir_image == 0
    print(f"SWIR B{highest_corr_ind[0]} with Mica B{highest_corr_ind[1]}. Pearson's correllation coefficient: {highest_corr:.4f}")

    # to_uint8 = lambda x: ((x - x.min()) / (x.max() - x.min()) * 255)
    # mica_image_uint8 = to_uint8(mica_image)
    # mica_image_uint8[mask] = 0
    # swir_image_uint8 = to_uint8(swir_image)
    # swir_image_uint8[mask] = 0

    swir_image_uint8_ants = ants.from_numpy(swir_image, is_rgb=False, has_components=False)
    mica_image_uint8_ants = ants.from_numpy(mica_image, is_rgb=False, has_components=False)
    mytx = ants.registration(fixed=mica_image_uint8_ants,
                             moving = swir_image_uint8_ants,
                             type_of_transform='Elastic')

    # lets plot 3 bands
    swir_rgb_bands_warped = []
    for i in [10,40,90]:
        swir_band_ants = ants.from_numpy(np.array(swir_arr[...,i].squeeze()), is_rgb=False, has_components=False)
        swir_rgb_bands_warped.append(ants.apply_transforms(fixed=mica_image_uint8_ants, moving=swir_band_ants,
                                                       transformlist=mytx['fwdtransforms']).numpy()[...,None])
    swir_rgb_bands_warped = np.concatenate(swir_rgb_bands_warped,2)





    # performing correllation coefficient between the two to pick the highest
    highest_corr_ind, highest_corr = get_highest_corr_coeff(swir_patch, mica_patch)
    swir_patch = swir_arr[..., highest_corr_ind[0]].squeeze()
    mica_patch = mica_arr[..., highest_corr_ind[1]].squeeze()
    print(f"SWIR B{highest_corr_ind[0]} with Mica B{highest_corr_ind[1]}. Pearson's correllation coefficient: {highest_corr:.4f}")

    # Running low pass filter on the swir patch to minimize the
    swir_patch_lpf = gaussian_filter(swir_patch, sigma=2)
    mica_patch_lpf = gaussian_filter(mica_patch, sigma=1)

    # convert to ants
    swir_patch_ants_lpf = ants.from_numpy(swir_patch_lpf.T, has_components=False)
    swir_patch_ants = ants.from_numpy(swir_patch.T, has_components=False)
    mica_patch_ants_lpf = ants.from_numpy(mica_patch_lpf.T, has_components=False)
    mica_patch_ants = ants.from_numpy(mica_patch.T, has_components=False)

    # perfrom registration
    # for visual purposes grab the visual appealing bands
    mica_image = mica_arr[..., 0:3]
    swir_image = swir_arr[..., np.array([10,40,90])]
    to_uint8 = lambda x: ((x - x.min()) / (x.max() - x.min()) * 255).astype(np.uint8)
    mica_image_uint8 = to_uint8(mica_image)
    mica_image_uint8_ants = ants.from_numpy(mica_image_uint8, has_components=True)
    swir_image_uint8 = to_uint8(swir_image)
    swir_image_uint8_ants = ants.from_numpy(swir_image_uint8, has_components=True)

    transform = get_transform(fixed=mica_image_uint8_ants, moving=swir_image_uint8_ants)
    apply_transform()
    mytx = ants.registration(fixed=mica_patch_ants_lpf, moving=swir_patch_ants_lpf, type_of_transform='Elastic')

    for i in range(3):
        swir_patch_warped_ants = ants.apply_transforms(fixed=mica_image_uint8_ants, moving=swir_image_uint8_ants,
                                              transformlist=mytx['fwdtransforms'])
    swir_patch_warped = swir_patch_warped_ants.numpy().T
    swir_patch_warped[swir_patch_warped == 0] = np.nan

    fig, ax = plt.subplots(1,3, figsize = (10,3))
    ax[0].imshow(swir_patch)
    ax[0].axis("off")
    ax[0].set_title("SWIR")
    ax[1].imshow(mica_patch)
    ax[1].axis("off")
    ax[1].set_title("Mica")
    ax[2].imshow(swir_patch_warped)
    ax[2].axis("off")
    ax[2].set_title("SWIR warped")
    plt.show()





if __name__ == "__main__":

    """
    antspy has many more algorithms that the other ones.
    """
    mica_hdr = "/Volumes/T7/axhcis/Projects/NURI/data/20210723_tait_labsphere/1133/Micasense/NURI_micasense_1133_transparent_mosaic_stacked_warped_cropped.hdr"
    swir_hdr = "/Volumes/T7/axhcis/Projects/NURI/data/20210723_tait_labsphere/1133/SWIR/raw_1504_nuc_or_plusindices3_warped.hdr"
    main(swir_hdr, mica_hdr)







