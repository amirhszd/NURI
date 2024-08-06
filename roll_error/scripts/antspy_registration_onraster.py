from cv2 import bilateralFilter
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
from tqdm import tqdm
import copy
from multiprocessing import Pool
from itertools import repeat
from .shape_shifter import load_image_envi_fast, load_image_envi, save_image_envi
import sys
import argparse


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


def apply_registration_to_band(hs_band, mica_image_uint8, transformlist):

    # some of the bands have all zero values
    nonzeros = np.nonzero(hs_band)
    if len(np.nonzero(hs_band)[0]) == 0:
        return hs_band[...,None]

    min = np.min(hs_band[nonzeros])
    max = hs_band.max()
    # convert to 0-1
    to_uint8_simple = lambda x: (
            (x - np.min(x[np.nonzero(x)])) / (np.max(x[np.nonzero(x)]) - np.min(x[np.nonzero(x)])) * 255)
    band_uint8 = to_uint8_simple(hs_band)
    band_uint8[hs_band == 0] = 0
    # residual to add later
    band_uint8_res = band_uint8 - band_uint8.astype(np.uint8)
    # process image
    hs_band_ants = ants.from_numpy(band_uint8.astype(np.uint8), is_rgb=False, has_components=False)
    mica_image_ants = ants.from_numpy(mica_image_uint8, is_rgb=False, has_components=False)
    hs_band_ants_warped = ants.apply_transforms(fixed=mica_image_ants, moving=hs_band_ants,
                                                transformlist=transformlist).numpy()
    # adding the residual
    hs_band_ants_warped = hs_band_ants_warped + band_uint8_res
    # adding the min and max to the entire array
    hs_band_ants_warped_final = hs_band_ants_warped * (max - min) / 255 + min
    # apply the mask back to it
    hs_band_ants_warped_final[hs_band == 0] = 0
    hs_band_ants_warped_final = hs_band_ants_warped_final.astype(int)
    return hs_band_ants_warped_final[...,None]


def main(hs_hdr, mica_hdr,
         hs_bands,
         mica_band,
         kernel = 20,
         sigma_color = 15,
         sigma_space = 20):

    # Function to convert images to uint8
    to_uint8 = lambda x: ((x - np.min(x[np.nonzero(x)])) / (np.max(x[np.nonzero(x)]) - np.min(x[np.nonzero(x)])) * 255).astype(
            np.uint8)

    # Load the hyperspectral and Mica images
    hs_arr, hs_profile, _ = load_image_envi(hs_hdr)
    mica_arr, mica_profile, _ = load_image_envi(mica_hdr)

    # Calculate the mean of the specified hyperspectral bands
    hs_image = np.mean(hs_arr[..., hs_bands], axis=2)
    mica_image = mica_arr[..., mica_band].squeeze()  # grabbing the last band in mica

    # Apply a low-pass filter using a bilateral filter
    mica_image_uint8 = to_uint8(mica_image)
    hs_image_uint8 = to_uint8(hs_image)

    mica_image_uint8 = bilateralFilter(mica_image_uint8, kernel , sigma_color, sigma_space)

    # Convert images to ANTs format
    hs_image_ants = ants.from_numpy(hs_image_uint8, is_rgb=False, has_components=False)
    mica_image_ants = ants.from_numpy(mica_image_uint8, is_rgb=False, has_components=False)

    # Calculate the transformation matrix using ANTs registration
    print("Calculating the transformation!")
    mtx = ants.registration(fixed=mica_image_ants,
                            moving=hs_image_ants,
                            type_of_transform="SyN")

    # # applying the transform on all channels
    # bands_warped = []
    # pbar = tqdm(total = hs_arr.shape[-1],
    #             position=0,
    #             leave=True,
    #             desc = "Performing non rigid transformation on bands")
    #
    # for i in range(hs_arr.shape[-1]):
    #     band = apply_registration_to_band(hs_arr, i, mica_image_ants, mtx)
    #     bands_warped.append(band[...,None])
    #     pbar.update(1)
    #
    # # saving the image
    # bands_warped = np.concatenate(bands_warped, 2)
    # output_hdr_filename = hs_hdr.replace(".hdr", "_ereg.hdr")
    # save_image_envi(bands_warped, hs_profile, output_hdr_filename)

    # Prepare to apply the transformation on all channels
    num_bands = hs_arr.shape[-1]
    # num_bands = 20
    pbar = tqdm(total = num_bands,
                position=0,
                leave=True,
                desc = "Performing non rigid transformation on bands")

    # Define the step size and generate index chunks for parallel processing
    def generate_index_chunks(total_length, chunk_size):
        return [np.arange(i, min(i + chunk_size, total_length)) for i in range(0, total_length, chunk_size)]

    step = 8
    indices_chunks = generate_index_chunks(num_bands, step)
    bands_warped = []

    # Process the bands in chunks using multiprocessing
    for indices_chunk in indices_chunks:
        hs_bands = []
        for index in indices_chunk:
            hs_bands.append(hs_arr[...,index])

        args_list = list(zip(hs_bands, repeat(mica_image_uint8), repeat(mtx['fwdtransforms'])))
        # we wont do more than 8 at a time due to memory
        with Pool(8) as pool:
            bands_warped.extend(pool.starmap(apply_registration_to_band, args_list))

        pbar.update(8)

    bands_warped = np.concatenate(bands_warped, axis=2)

    # Saving the image
    output_hdr_filename = hs_hdr.replace(".hdr", "_ereg.hdr")

    # TODO: USE THE FUNCTION FROM OUTSIDE WE HAVE MANY MULTIPLES OF THIS
    save_image_envi(bands_warped, hs_profile, output_hdr_filename, dtype = "uint16", ext= "")

    return output_hdr_filename


if __name__ == "__main__":
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(description='Antspy elastic registration technique.')
        parser.add_argument('--hs_hdr', type=str, help='Hyperspectral HDR filename.')
        parser.add_argument('--mica_hdr', type=str, help='Mica HDR filename.')
        parser.add_argument('--hs_bands', type=int, nargs='+', help='List of hyperspectral band indices to use.')
        parser.add_argument('--mica_band', type=int, help='Specific band index of the Mica image to use.')
        parser.add_argument('--kernel', type=int, default=20, help='Kernel size for the bilateral filter.')
        parser.add_argument('--sigma_color', type=int, default=15, help='Sigma color for the bilateral filter.')
        parser.add_argument('--sigma_space', type=int, default=20, help='Sigma space for the bilateral filter.')
        args = parser.parse_args()

        hs_hdr = args.hs_hdr
        mica_hdr = args.mica_hdr
        hs_bands = args.hs_bands
        mica_band = args.mica_band
        kernel = args.kernel
        sigma_color = args.sigma_color
        sigma_space = args.sigma_space

        main(hs_hdr, mica_hdr, hs_bands, mica_band, kernel, sigma_color, sigma_space)
    else:
        # debug
        hs_hdr = "/Volumes/T7/axhcis/Projects/NURI/raw_1504_nuc_or_plusindices3_warped_ss.hdr"
        mica_hdr = "/Volumes/T7/axhcis/Projects/NURI/data/20210723_tait_labsphere/1133/Micasense/NURI_micasense_1133_transparent_mosaic_stacked_warped_cropped.hdr"
        hs_bands = [1, 2, 3]  # replace with actual bands
        mica_band = 0  # replace with actual band

        main(hs_hdr, mica_hdr, hs_bands, mica_band)






