import rasterio
import numpy as np
import numexpr as ne
from tqdm import tqdm


def compute_correlation(vnir_band_flat, swir_band_flat, vnir_mean, swir_mean, vnir_std, swir_std):
    # Use numexpr for the correlation calculation
    correlation = ne.evaluate(
        'sum((vnir_band_flat - vnir_mean) * (swir_band_flat - swir_mean)) / (vnir_std * swir_std * len(vnir_band_flat))')
    return correlation


def load_image(image_path):
    with rasterio.open(image_path) as vnir_src:
        vnir_bands = vnir_src.read()  # Read all VNIR bands into a numpy array
        vnir_meta = vnir_src.meta  # Copy the metadata from VNIR image
    return vnir_bands, vnir_meta

def selected_band(vnir_bands, swir_bands, output_image_path):

    # Precompute the mean and standard deviation for all VNIR and SWIR bands
    vnir_stats = [(vnir_band.flatten(), vnir_band.mean(), vnir_band.std()) for vnir_band in vnir_bands]
    swir_stats = [(swir_band.flatten(), swir_band.mean(), swir_band.std()) for swir_band in swir_bands]

    # Prepare an empty array for the output image
    selected_bands = np.zeros(swir_bands.shape, dtype=vnir_bands.dtype)

    # Iterate over each band in the SWIR image using tqdm for progress tracking
    for swir_band_idx, (swir_band_flat, swir_mean, swir_std) in tqdm(enumerate(swir_stats), total=len(swir_stats),
                                                                     desc="Processing SWIR Bands"):
        max_correlation = -1
        selected_vnir_band = None

        # Compare with each VNIR band
        for vnir_band_flat, vnir_mean, vnir_std in vnir_stats:
            correlation = compute_correlation(vnir_band_flat, swir_band_flat, vnir_mean, swir_mean, vnir_std, swir_std)

            if correlation > max_correlation:
                max_correlation = correlation
                selected_vnir_band = vnir_band_flat.reshape(vnir_bands.shape[1:])

        # Assign the selected VNIR band to the output image
        selected_bands[swir_band_idx] = selected_vnir_band

    return selected_bands

def write_band(selected_bands, swir_meta, output_image_path):
    # Write the output image
    with rasterio.open(output_image_path, 'w', **swir_meta) as dst:
        dst.write(selected_bands)

def load_wv3(path):
    from scipy.io import loadmat
    data = loadmat(path)
    swir_bands = data["I_MS"]
    vnir_bands = data["I_SWIR"]
    return vnir_bands, swir_bands

if __name__ == "__main__":
    # vnir_image_path = "path/to/vnir_image.tif"
    # swir_image_path = "path/to/swir_image.tif"
    # output_image_path = "selected_band.tif"
    # vnir_bands, vnir_meta = load_image(vnir_image_path)
    # swir_bands, swir_meta = load_image(swir_image_path)

    vnir_bands, swir_bands = load_wv3("/dirs/data/tirs/axhcis/Projects/NURI/Data/Dataset_WV3_Sim_APEX.mat")

    selected_bands = selected_band(vnir_bands, swir_bands)

    # write_band(selected_bands, swir_meta, output_image_path)