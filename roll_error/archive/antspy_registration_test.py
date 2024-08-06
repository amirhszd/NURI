from spectral.io import envi
import numpy as np
import matplotlib.pyplot as plt
import dipy.align.imwarp as imwarp
from dipy.viz import regtools
from scipy.ndimage import gaussian_filter
from scipy.stats import pearsonr
import ants
from skimage.transform import rescale
import itk
from scipy.ndimage import binary_dilation, binary_closing
from scipy.signal import medfilt2d


def load_image_envi(waterfall_path):
    vnir_ds = envi.open(waterfall_path)
    vnir_profile = vnir_ds.metadata
    vnir_arr = np.array(vnir_ds.load())

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
def main(hs_hdr, mica_hdr, qa_hdr, hs_type = "swir"):

    # load the SWIR and mica
    hs_arr, hs_profile= load_image_envi(hs_hdr)
    mica_arr, mica_profile = load_image_envi(mica_hdr)
    qa_arr, qa_profile = load_image_envi(qa_hdr)

    to_uint8 = lambda x: ((x - x.min()) / (x.max() - x.min()) * 255).astype(np.uint8)
    qa_arr = to_uint8(qa_arr)
    mask = (qa_arr >= 9).astype(int)
    for i in range(5):
        mask = medfilt2d(mask.squeeze(), 3)
        plt.imshow(mask)
        mask = binary_dilation(mask, iterations=1)
        plt.imshow(mask)
        mask = binary_closing(mask, iterations=1)
        plt.imshow(mask)
        mask = mask.astype(int)

    # grabbing the bands necessary to perform mutual information on
    if hs_type == "swir":
        hs_bands = np.arange(0, 12)
    elif hs_type == "vnir":
        #TODO: define what bands vnir is
        hs_bands = np.arange(0, 12)

    hs_image = np.average(hs_arr[..., hs_bands], axis=2)
    mica_image = mica_arr[..., -1].squeeze() # grabbing the last band in mica

    to_uint8 = lambda x: ((x - np.min(x[np.nonzero(x)])) / (np.max(x[np.nonzero(x)]) - np.min(x[np.nonzero(x)])) * 255).astype(
            np.uint8)

    # lets run a low pass filter
    mica_image_uint8 = to_uint8(mica_image)
    hs_image_uint8 = to_uint8(hs_image)
    from cv2 import bilateralFilter
    mica_image_uint8 = bilateralFilter(mica_image_uint8, 20 ,15, 20)

    hs_image_ants = ants.from_numpy(hs_image_uint8, is_rgb=False, has_components=False)
    mica_image_ants = ants.from_numpy(mica_image_uint8, is_rgb=False, has_components=False)

    methods = ["Elastic", "SyN"]
    fig, ax = plt.subplots(len(methods), 9, figsize = (15,len(methods)*2))
    patches = [[(400,700),(300,600)],
               [(250,450),(175,375)],
               [(800,1100),(300,650)]]

    for i, method in enumerate(methods):

        mtx = ants.registration(fixed=mica_image_ants,
                                moving=hs_image_ants,
                                type_of_transform=method)
        hs_image_ants_warped = ants.apply_transforms(fixed=mica_image_ants, moving=hs_image_ants,
                                                      transformlist=mtx['fwdtransforms']).numpy()
        ax[i, 4].set_title(f"{method}")

        for j, patch in enumerate(patches):

            mica_patch = mica_image_ants[patch[0][0]:patch[0][1], patch[1][0]:patch[1][1]]
            hs_patch = hs_image_ants[patch[0][0]:patch[0][1], patch[1][0]:patch[1][1]]
            hs_warp_patch = hs_image_ants_warped[patch[0][0]:patch[0][1], patch[1][0]:patch[1][1]]

            ax[i, 3*j].imshow(mica_patch)
            ax[i, 3 * j].axis('off')
            ax[i, 3*j + 1].imshow(hs_patch)
            ax[i, 3*j + 1].axis('off')
            ax[i, 3*j + 2].imshow(hs_warp_patch)
            ax[i, 3*j + 2].axis('off')
    plt.tight_layout()
    plt.savefig("antspy_methods_bilateral_20_15_20.pdf")










if __name__ == "__main__":

    """
    antspy has many more algorithms that the other ones.
    """
    mica_hdr = "/Volumes/T7/axhcis/Projects/NURI/data/20210723_tait_labsphere/1133/Micasense/NURI_micasense_1133_transparent_mosaic_stacked_warped_cropped.hdr"
    hs_hdr = "/Volumes/T7/axhcis/Projects/NURI/raw_1504_nuc_or_plusindices3_warped_SS.hdr"
    change_hdr = "/Volumes/T7/axhcis/Projects/NURI/raw_1504_nuc_or_plusindices3_warped_QA.hdr"
    main(hs_hdr, mica_hdr, change_hdr)







