
import os
from itertools import repeat
import copy
from spectral.io import envi
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.linear_model import LinearRegression, RANSACRegressor
from skimage.metrics import normalized_mutual_information
from scipy.interpolate import griddata
from tqdm import tqdm
from sklearn.decomposition import PCA
from skimage.draw import polygon
from scipy.spatial import ConvexHull
from matplotlib.path import Path
from scipy.ndimage import binary_dilation
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
from scipy.signal import medfilt2d
import psutil
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
import rasterio
import os
import torch.nn.functional as F
import torch
from scipy.ndimage import median_filter


def load_image_envi(waterfall_path, bands = None, hs = True):
    """
    Load an image from an ENVI file.

    Parameters:
    - waterfall_path (str): Path to the ENVI file.

    Returns:
    - vnir_arr (numpy.ndarray): Loaded image data as a NumPy array.
    - vnir_profile (dict): Metadata/profile of the ENVI file.
    """
    vnir_ds = envi.open(waterfall_path)  # Open the ENVI file
    vnir_profile = vnir_ds.metadata  # Extract metadata
    if bands is None:
        vnir_arr = np.array(vnir_ds.load())  # Load image data into a NumPy array
    else:
        # envi takes indices
        bands = list(bands)
        if hs:
            bands.append(int(vnir_profile['bands']) - 1)
        vnir_arr = vnir_ds.read_bands(bands)

    return vnir_arr, vnir_profile

def load_image_envi_rasterio(waterfall_path, bands=None, hs = False):
    """
    Load an image from an ENVI file using rasterio. THIS IS MUCH FASTER THAN SPECTRAL ENVI.

    Parameters:
    - waterfall_path (str): Path to the ENVI file.

    Returns:
    - vnir_arr (numpy.ndarray): Loaded image data as a NumPy array.
    - vnir_profile (dict): Metadata/profile of the ENVI file.
    """
    vnir_ds = envi.open(waterfall_path)  # Open the ENVI file
    vnir_profile = vnir_ds.metadata

    from time import time
    img_path = waterfall_path.split(".hdr")[0] + '.img' if os.path.isfile(waterfall_path.split(".hdr")[0] + '.img') else waterfall_path.split(".hdr")[0]
    with rasterio.open(img_path) as src:
        if bands is None:
            hs_arr = np.moveaxis(src.read(),0,-1)
        else:
            # rasterio takes band numbers to gotta pass +1
            bands = bands + 1
            bands = list(bands)
            if hs:
                bands.append(int(vnir_profile['bands']))
            hs_arr = np.moveaxis(src.read(bands), 0, -1)
    return hs_arr, vnir_profile


def save_image_envi(new_hs_arr, old_hs_profile, hs_hdr_path, dtype=None, ext = None):
    """
    Save a hyperspectral image to an ENVI file.

    Parameters:
    - new_hs_arr (numpy.ndarray): New hyperspectral image data.
    - old_hs_profile (dict): Metadata/profile of the original hyperspectral image.
    - hs_hdr_path (str): Path to save the ENVI file.
    - dtype (dtype, optional): Data type of the image data.

    Returns:
    - None
    """

    if ext == None:
        ext = ".img"
    else:
        ext = ""


    # Update the description in the metadata
    old_hs_profile["description"] = hs_hdr_path.replace(".hdr","")

    # Save the new hyperspectral image to the ENVI file
    envi.save_image(hs_hdr_path, new_hs_arr, metadata=old_hs_profile, force=True, dtype=dtype, ext = ext)

    # Print a message indicating the successful saving of the image
    print("Image saved to:", hs_hdr_path)

# def get_mask_from_convex_hull(image, y_new, x_new):
#     """
#     Generate a mask based on convex hull from given image and new points.
#
#     Parameters:
#     - image (numpy.ndarray): Image data.
#     - y_new (array-like): Y-coordinates of new points.
#     - x_new (array-like): X-coordinates of new points.
#
#     Returns:
#     - mask (numpy.ndarray): Binary mask indicating points within the convex hull.
#     """
#
#     if len(image.shape) == 3:
#         # Find NaN indices in case of a multi-channel image
#         nan_indices = np.where(np.isnan(image[...,0]))
#         image_shape = image.shape[:2]
#     else:
#         # Find NaN indices for a single-channel image
#         nan_indices = np.where(np.isnan(image))
#         image_shape = image.shape
#
#     # Exclude new points from the NaN indices
#     points_old = zip(nan_indices[0],nan_indices[1])
#     points_new = zip(y_new, x_new)
#     points_final = [i for i in points_old if i not in points_new]
#     points_final.extend(points_new)
#     points_final = [list(i) for i in points_final]
#     y_final, x_final = [list(t) for t in zip(*points_final)]
#     y_final, x_final = np.array(y_final).squeeze(), np.array(x_final).squeeze()
#     points_arr = np.concatenate([np.array(x_final)[..., None], np.array(y_final)[..., None]], 1)
#
#     # Compute convex hull
#     hull = ConvexHull(points_arr)
#
#     # Get points within the convex hull
#     points_within_hull = points_arr[hull.vertices]
#
#     # Create a path from the points within the hull
#     path = Path(points_within_hull)
#
#     # Create meshgrid of image coordinates
#     x, y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
#     points = np.vstack((x.flatten(), y.flatten())).T
#
#     # Check which points are within the convex hull
#     mask = path.contains_points(points).reshape(image_shape)
#
#     # Dilate the mask slightly
#     mask = binary_dilation(mask, iterations=5)
#
#     if len(image.shape) == 3:
#         # If the image is multi-channel, extend the mask to cover all channels
#         mask = np.tile(mask[..., None], image.shape[2])
#
#     return mask
#
# def get_mask_from_row(y_new, x_new, swir_image_copy):
#     """
#     Generate a mask based on row projection from given points.
#
#     Parameters:
#     - y_new (array-like): Y-coordinates of new points.
#     - x_new (array-like): X-coordinates of new points.
#     - swir_image_copy (numpy.ndarray): Copy of the SWIR image.
#
#     Returns:
#     - mask (numpy.ndarray): Binary mask indicating the row projection.
#     """
#     mask = np.zeros_like(swir_image_copy)
#     pca = PCA()  # Number of principal components to keep
#     data = np.vstack((y_new, x_new.squeeze())).T
#     pca.fit(data)
#     projected_data = pca.transform(data)
#     # Find the minimum and maximum values along each axis
#     x_min = np.min(projected_data[:, 1])
#     x_max = np.max(projected_data[:, 1])
#     y_min = np.min(projected_data[:, 0])
#     y_max = np.max(projected_data[:, 0])
#
#     proj_polygon_v = np.array([[y_max, x_min], [y_max, x_max], [y_min, x_max], [y_min, x_min]])
#     polygon_v = np.round(pca.inverse_transform(proj_polygon_v))
#     rr, cc = polygon(polygon_v[:, 0], polygon_v[:, 1], mask.shape)
#     mask[rr, cc] = 1
#
#     mask = mask.astype(bool)
#
#     return mask

def to_uint8(x):
    return ((x - np.min(x[np.nonzero(x)])) / (np.max(x[np.nonzero(x)]) - np.min(x[np.nonzero(x)])) * 255).astype(np.uint8)
def shift_rows_from_model(hs_image_copy, model, y_old, x_old, n):
    """
    Shift rows of a hyperspectral image based on a linear regression model.

    Parameters:
    - hs_image_copy (numpy.ndarray): Copy of the original hyperspectral image.
    - model (sklearn.linear_model): Trained linear regression model.
    - x_old (array-like): Old x-coordinates of points to be shifted.
    - y_old (array-like): Old y-coordinates of points to be shifted.
    - n (int): Number of pixels to shift.
    - quality_raster (numpy.ndarray, optional): Quality raster for storing shift quality metrics.

    Returns:
    - hs_image_copy (numpy.ndarray): Shifted hyperspectral image.
    - x_new (numpy.ndarray): New x-coordinates of shifted points.
    - y_new (numpy.ndarray): New y-coordinates of shifted points.
    - quality_raster (numpy.ndarray): Updated quality raster.
    """

    pixel_values = hs_image_copy[y_old, x_old]

    # Get residuals from old values
    y_old_model = model.predict(x_old.reshape(-1, 1))
    y_res = y_old_model - y_old

    # Add residuals to the points to calculate new coordinates
    x_new = x_old.reshape(-1, 1) + n
    y_new_model = model.predict(x_new)
    y_new = np.round(y_new_model - y_res).astype(int)

    if len(y_new) < 10:
        return None, None, None

    # Check if new coordinates exceed image dimensions
    if (x_new.max() >= hs_image_copy.shape[1]) or (y_new.max() >= hs_image_copy.shape[0]):
        x_ins = np.where((x_new < hs_image_copy.shape[1]) & (x_new > 0))[0]
        y_ins = np.where((y_new < hs_image_copy.shape[0]) & (y_new > 0))[0]
        ins_indices = np.array(list(set(x_ins) & set(y_ins)))
        y_new = y_new[ins_indices]
        x_new = x_new.squeeze()[ins_indices]
        hs_image_copy[y_new, x_new] = pixel_values[ins_indices]
    else:
        hs_image_copy[y_new, x_new.squeeze()] = pixel_values

    return hs_image_copy, x_new, y_new

# def calculate_mi(hs_image_copy, mica_image, min_row, max_row, min_col, max_col):
#     """
#     Calculate mutual information between a hyperspectral image and a Mica image patch.
#
#     Parameters:
#     - hs_patch (numpy.ndarray): Copy of the hyperspectral image.
#     - mica_patch (numpy.ndarray): Mica image.
#     - min_row (int): Minimum row index of the image patch.
#     - max_row (int): Maximum row index of the image patch.
#     - min_col (int): Minimum column index of the image patch.
#     - max_col (int): Maximum column index of the image patch.
#
#     Returns:
#     - mi (float): Mutual information value.
#     """
#
#     hs_patch = hs_image_copy[min_row:max_row + 1, min_col:max_col + 1]
#     mica_patch = mica_image[min_row:max_row + 1, min_col:max_col + 1]
#
#
#     # Convert patches to uint8 and then to int for compatibility with mutual information calculation
#     hs_patch = to_uint8(hs_patch).astype(int)
#     mica_patch = to_uint8(mica_patch).astype(int)
#
#     # Calculate mutual information between the patches
#     mi = normalized_mutual_information(hs_patch, mica_patch)
#     return mi

def calculate_mi_patch(hs_patch, mica_patch):
    """
    Calculate mutual information between a hyperspectral image and a Mica image patch.

    Parameters:
    - hs_patch (numpy.ndarray): Copy of the hyperspectral image.
    - mica_patch (numpy.ndarray): Mica image.
    - min_row (int): Minimum row index of the image patch.
    - max_row (int): Maximum row index of the image patch.
    - min_col (int): Minimum column index of the image patch.
    - max_col (int): Maximum column index of the image patch.

    Returns:
    - mi (float): Mutual information value.
    """

    if hs_patch is None:
        return np.nan

    # Convert patches to uint8 and then to int for compatibility with mutual information calculation
    hs_patch = to_uint8(hs_patch).astype(int)
    mica_patch = to_uint8(mica_patch).astype(int)

    # Calculate mutual information between the patches
    mi = normalized_mutual_information(hs_patch, mica_patch)
    return mi



# def get_shift_from_mi(hs_image, mica_image, y_old, x_old, min_row, max_row, min_col, max_col, n=3):
#     """
#     Determine pixel shift from mutual information (MI) between a hyperspectral image and a Mica image.
#
#     Parameters:
#     - hs_image (numpy.ndarray): Hyperspectral image.
#     - mica_image (numpy.ndarray): Mica image.
#     - y_old (array-like): Old y-coordinates of points.
#     - x_old (array-like): Old x-coordinates of points.
#     - min_row (int): Minimum row index of the image patch.
#     - max_row (int): Maximum row index of the image patch.
#     - min_col (int): Minimum column index of the image patch.
#     - max_col (int): Maximum column index of the image patch.
#     - n (int, optional): Number of pixel shifts to consider.
#
#     Returns:
#     - model (sklearn.linear_model): Best linear regression model.
#     - best_shift (int): Best pixel shift.
#     - max_mi_quality (float): Maximum mutual information quality metric.
#     """
#     mi_quality_metrics = []  # List to store MI quality metrics for different shifts
#     linear_models = []  # List to store linear regression models for different shifts
#
#     # Check if there are enough points for regression
#     if (len(x_old) < 2) or (len(y_old) < 2):
#         return None, np.nan, np.nan
#
#     # Fit RANSAC regression model to find the best linear fit
#     model = RANSACRegressor(LinearRegression(), max_trials=20, residual_threshold=30).fit(x_old.reshape(-1, 1), y_old)
#
#     # Calculate MI before shifting
#     mi_before = calculate_mi(hs_image, mica_image, min_row, max_row, min_col, max_col)
#
#     # Generate pixel shifts
#     pixel_shifts = np.arange(-n, +n + 1)
#
#     # Iterate over pixel shifts
#     for shift_value in pixel_shifts:
#         hs_image_copy = copy.copy(hs_image)
#
#         # Shift rows based on the regression model
#         hs_image_copy, x_new, y_new, _ = shift_rows_from_model(hs_image_copy, model, x_old, y_old, shift_value)
#
#         if (hs_image_copy is None) or (x_new is None) or (y_new is None):
#             linear_models.append(None)
#             mi_quality_metrics.append(np.nan)
#             continue
#
#         # Calculate MI after shifting
#         mi_after = calculate_mi(hs_image_copy, mica_image, min_row, max_row, min_col, max_col)
#
#         # Calculate MI quality metric
#         mi_quality_metric = (mi_after - mi_before) / mi_before
#         mi_quality_metrics.append(mi_quality_metric)
#         linear_models.append(model)
#
#     # Find the shift with the maximum MI quality metric
#     if len(linear_models) > 0:
#         argmax = np.argmax(mi_quality_metrics)
#         return linear_models[argmax], pixel_shifts[argmax], mi_quality_metrics[argmax]
#     else:
#         return None, np.nan, np.nan


def get_shift_from_mi_patch(hs_image, mica_image, y_old, x_old, n=3):
    """

    Determine pixel shift from mutual information (MI) between a hyperspectral image and a Mica image.

    Parameters:
    - hs_image (numpy.ndarray): Hyperspectral image.
    - mica_image (numpy.ndarray): Mica image.
    - y_old (array-like): Old y-coordinates of points offseted
    - x_old (array-like): Old x-coordinates of points offseted.
    - min_row (int): Minimum row index of the image patch.
    - max_row (int): Maximum row index of the image patch.
    - min_col (int): Minimum column index of the image patch.
    - max_col (int): Maximum column index of the image patch.
    - n (int, optional): Number of pixel shifts to consider.

    Returns:
    - model (sklearn.linear_model): Best linear regression model.
    - best_shift (int): Best pixel shift.
    - max_mi_quality (float): Maximum mutual information quality metric.
    """
    mi_quality_metrics = []  # List to store MI quality metrics for different shifts
    linear_models = []  # List to store linear regression models for different shifts

    # Check if there are enough points for regression
    if (len(x_old) < 2) or (len(y_old) < 2):
        return None, np.nan, np.nan


    # Fit RANSAC regression model to find the best linear fit
    model = RANSACRegressor(LinearRegression(), max_trials=20, residual_threshold=30).fit(x_old.reshape(-1, 1), y_old)

    # Calculate MI before shifting
    mi_before = calculate_mi_patch(hs_image, mica_image)

    # Generate pixel shifts
    pixel_shifts = np.arange(-n, +n + 1)

    # Iterate over pixel shifts
    for shift_value in pixel_shifts:
        hs_image_copy = copy.copy(hs_image)

        # Shift rows based on the regression model
        hs_image_copy, x_new, y_new = shift_rows_from_model(hs_image_copy, model, y_old, x_old, shift_value)

        if (hs_image_copy is None) or (x_new is None) or (y_new is None):
            linear_models.append(None)
            mi_quality_metrics.append(np.nan)
            continue

        # Calculate MI after shifting
        mi_after = calculate_mi_patch(hs_image_copy, mica_image)

        # Calculate MI quality metric
        mi_quality_metric = (mi_after - mi_before) / mi_before
        mi_quality_metrics.append(mi_quality_metric)
        linear_models.append(model)

    # Find the shift with the maximum MI quality metric
    if len(linear_models) > 0:
        argmax = np.argmax(mi_quality_metrics)
        return linear_models[argmax], pixel_shifts[argmax], mi_quality_metrics[argmax]
    else:
        return None, np.nan, np.nan


# def get_shift_from_mi_mpp(hs_image, mica_image, y_old, x_old, min_row, max_row, min_col, max_col, n=3):
#     """
#     Determine pixel shift from mutual information (MI) between a hyperspectral image and a Mica image.
#
#     Parameters:
#     - hs_image (numpy.ndarray): Hyperspectral image.
#     - mica_image (numpy.ndarray): Mica image.
#     - y_old (array-like): Old y-coordinates of points.
#     - x_old (array-like): Old x-coordinates of points.
#     - min_row (int): Minimum row index of the image patch.
#     - max_row (int): Maximum row index of the image patch.
#     - min_col (int): Minimum column index of the image patch.
#     - max_col (int): Maximum column index of the image patch.
#     - n (int, optional): Number of pixel shifts to consider.
#
#     Returns:
#     - model (sklearn.linear_model): Best linear regression model.
#     - best_shift (int): Best pixel shift.
#     - max_mi_quality (float): Maximum mutual information quality metric.
#     """
#     mi_quality_metrics = []  # List to store MI quality metrics for different shifts
#     linear_models = []
#     hs_image_copies = []# List to store linear regression models for different shifts
#
#     # Check if there are enough points for regression
#     if (len(x_old) < 2) or (len(y_old) < 2):
#         return None, np.nan, np.nan
#
#     # Fit RANSAC regression model to find the best linear fit
#     model = RANSACRegressor(LinearRegression(), max_trials=100, residual_threshold=30).fit(x_old.reshape(-1, 1), y_old)
#
#     # Calculate MI before shifting
#     mi_before = calculate_mi(hs_image, mica_image, min_row, max_row, min_col, max_col)
#
#     # Generate pixel shifts
#     pixel_shifts = np.arange(-n, +n + 1)
#
#     # setting up the arguments to process shift_rows_from_model
#     args_list = [(copy.copy(hs_image), model, x_old, y_old, shift_value) for shift_value in pixel_shifts]
#     # with ProcessPoolExecutor() as executor:
#     #     results = executor.map(lambda args: shift_rows_from_model(*args), args_list)
#     with Pool() as pool:
#         results = pool.starmap(shift_rows_from_model, args_list)
#     for result in results:
#         hs_image_copy, x_new, y_new, _ = result
#         hs_image_copies.append(hs_image_copy)
#         linear_models.append(model)
#
#     #grabbing the patches Extract patches from the hyperspectral and Mica images
#     hs_patches = [hs_image_copy[min_row:max_row + 1, min_col:max_col + 1] if hs_image_copy is not None else np.nan for hs_image_copy in hs_image_copies]
#     mica_patch = mica_image[min_row:max_row + 1, min_col:max_col + 1]
#
#     # setting up the arguments to process mica_patch_mi
#     mis_after = []
#     args_list = [(hs_patches[c], mica_patch) for c, shift_value in enumerate(pixel_shifts)]
#     with Pool() as pool:
#         results = pool.starmap(calculate_mi_patch, args_list)
#     for result in results:
#         mis_after.append(result)
#
#     # Calculate MI after shifting
#     mi_quality_metrics = [(mi_after - mi_before) / mi_before for mi_after in mis_after]
#
#     # Find the shift with the maximum MI quality metric
#     if len(linear_models) > 0:
#         argmax = np.nanargmax(mi_quality_metrics)
#         return linear_models[argmax], pixel_shifts[argmax], mi_quality_metrics[argmax]
#     else:
#         return None, np.nan, np.nan


def medfilt3d(hs_arr_copy, kernel_size=3, use_torch = True):
    hs_arr_final = []
    # we are not messing with the waterfall band

    for band in tqdm(range(hs_arr_copy.shape[2] -  1), desc="performing band wise filtering",
                     position=0,
                     leave=True,):
        # using GPU
        if use_torch:
            hs_arr_final.append(medfilt2d_gpu(hs_arr_copy[..., band], kernel_size=kernel_size)[..., None])
        else:
            hs_arr_final.append(median_filter(hs_arr_copy[..., band], size=kernel_size)[...,None])

    # not messing with the last layer just because
    hs_arr_final.append(hs_arr_copy[...,-1][...,None])
    hs_arr_final = np.concatenate(hs_arr_final, 2)
    return hs_arr_final


def medfilt2d_gpu(image, kernel_size=3):
    """
    Apply a median filter to a 2D image using PyTorch.

    Parameters:
    - image (torch.Tensor): The input image tensor of shape (H, W) or (1, H, W).
    - kernel_size (int): The size of the median filter kernel (default: 3).

    Returns:
    - torch.Tensor: The filtered image tensor.
    """
    assert kernel_size % 2 == 1, "Kernel size must be odd."

    # Ensure the image has the shape (1, H, W)
    assert image.ndim == 2, "Image size 2d"

    # converting to torch.Tensor
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if device == "cpu":
        print("Did not find cuda compatible GPU, running median filtering using torch with CPU. Still faster!")

    image = torch.from_numpy(image).float().to(device)
    image = image[None,]

    # Pad the image to handle borders
    pad_size = kernel_size // 2
    padded_image = F.pad(image, (pad_size, pad_size, pad_size, pad_size), mode='reflect')

    # Get all sliding windows of the image
    windows = padded_image.unfold(1, kernel_size, 1).unfold(2, kernel_size, 1)
    windows = windows.contiguous().view(-1, kernel_size * kernel_size)

    # Compute the median for each window
    medians = windows.median(dim=1).values

    # Reshape the result back to the image shape
    filtered_image = medians.view(image.shape[1], image.shape[2]).cpu().numpy()

    return filtered_image

# def process_band(band_data):
#     """
#     Function to apply median filter to a single band of the hyperspectral image.
#     `band_data` is a tuple (band_array, kernel_size).
#     """
#     band_array, kernel_size = band_data
#     filtered_band = medfilt2d(band_array, kernel_size=kernel_size)
#     return filtered_band[..., None]  # Add a new axis for concatenation later

# def medfilt3d_mpp(hs_arr_copy, kernel_size=3):
#
#
#     num_bands = hs_arr_copy.shape[2] - 1  # Exclude the last "waterfall" band
#     band_data_list = [(hs_arr_copy[..., i], kernel_size) for i in range(num_bands)]
#     last_band = hs_arr_copy[..., -1]
#     del hs_arr_copy
#
#     # Create a multiprocessing pool and map process_band to each band data
#     with Pool(2) as pool:
#         hs_arr_final = list(tqdm(pool.imap(process_band, band_data_list),
#                             total=num_bands, desc="Performing band wise filtering"))
#
#     # Handle the last band (waterfall band) without filtering
#     hs_arr_final.append(last_band[..., None])
#
#     # Concatenate all bands along the third dimension
#     hs_arr_final = np.concatenate(hs_arr_final, axis=2)

# def shape_shift(hs_filename, mica_filename,
#                 hs_bands,
#                 mica_band,
#                 pixel_shift = 3, kernel_size = 3,
#                 hs_waterfall_rows_band_number = -1,
#                 use_torch = True):
#
#     # load the hyperspectral and mica
#     hs_arr, hs_profile = load_image_envi_rasterio(hs_filename)
#     mica_arr, mica_profile = load_image_envi_rasterio(mica_filename)
#
#     # grabbing georectified rows and the original array
#     waterfall_rows = hs_arr[..., hs_waterfall_rows_band_number].squeeze()
#     hs_arr_copy = copy.copy(hs_arr)
#
#     # grabbing the correspondiong image
#     hs_image = np.mean(hs_arr[..., hs_bands], axis=2)
#     mica_image = mica_arr[..., mica_band].squeeze() # grabbing the last band in mica
#
#     # setting up lists to save out the linear models for each row and the shift amount
#     quality_raster = np.zeros_like(hs_arr_copy)
#     quality_metrics = []
#     pbar = tqdm(total = np.max(waterfall_rows),
#                 position=0,
#                 leave=True,
#                 desc = "Performing shape shifter on waterfall rows.")
#
#     # for row_value in range(int(np.max(waterfall_rows))):
#     for row_value in np.arange(200,210):
#         # grab the position of the pixels
#         y_old, x_old = np.where(waterfall_rows == row_value)
#         mask_bool = waterfall_rows == row_value
#         rows, cols = np.where(mask_bool)
#
#         if (len(cols) == 0) & (len(rows) == 0):
#             pbar.update(1)
#             continue
#
#         # grab the boundary box
#         min_row, min_col = np.min(rows), np.min(cols)
#         max_row, max_col = np.max(rows), np.max(cols)
#
#         # shifting within the boundary box
#         linear_model, shift, quality_metric = get_shift_from_mi(hs_image,
#                                                                 mica_image,
#                                                                 y_old, x_old,
#                                                                 min_row, max_row,
#                                                                 min_col, max_col,
#                                                                 pixel_shift)
#         quality_metrics.append(quality_metric)
#
#         if linear_model is None:
#             pbar.update(1)
#             continue
#
#         # suggested method at this stage is cubic
#         hs_arr_copy, _, _, quality_raster = shift_rows_from_model(hs_arr_copy,
#                                                   linear_model,
#                                                   x_old,
#                                                   y_old,
#                                                   shift,
#                                                   quality_raster)
#
#         pbar.update(1)
#
#     print(f"Overall relative improvement: {np.nansum(quality_metrics):.5f}.")
#
#     # pefroming median filtering to smooth out the old values
#     hs_arr_copy = medfilt3d(hs_arr_copy, kernel_size=kernel_size, torch = use_torch)
#
#     ss_qa_filename = hs_filename.replace(".hdr", "_ss_qa.hdr")
#     ss_filename = hs_filename.replace(".hdr", "_ss.hdr")
#
#     # need to change the metadata to fit the quality raster information and then save it
#     quality_raster_metadata = copy.copy(hs_profile)
#     quality_raster_metadata["bands"] = "1"
#     quality_raster_metadata["band names"] = "Error Band"
#     del quality_raster_metadata["wavelength"]
#     quality_raster = np.mean(quality_raster, axis = 2)
#     save_image_envi(quality_raster, quality_raster_metadata, ss_qa_filename, ext="")
#     del quality_raster
#
#     # saving the image
#     save_image_envi(hs_arr_copy, hs_profile, ss_filename, ext="")
#
#     return ss_filename, ss_qa_filename



def shape_shift_mpp(hs_filename, mica_filename,
         hs_bands,
         mica_band,
         pixel_shift = 3, kernel_size = 3,
         hs_waterfall_rows_band_number = -1,
         use_torch = True,
         num_threads = None):

    if num_threads is None:
        num_threads = os.cpu_count()

    # Load the hyperspectral and Mica images
    hs_arr, hs_profile = load_image_envi_rasterio(hs_filename)
    mica_arr, mica_profile = load_image_envi_rasterio(mica_filename)

    # Convert arrays to float16 to speed up processing
    hs_arr = hs_arr.astype(np.float16)
    mica_arr = mica_arr.astype(np.float16)

    # Extract georectified rows and the original array
    waterfall_rows = hs_arr[..., hs_waterfall_rows_band_number].squeeze()
    hs_arr_shapeshifted = copy.copy(hs_arr)

    # Generate image for hyperspectral bands
    hs_image = np.mean(hs_arr[..., hs_bands], axis=2)
    mica_image = mica_arr[..., mica_band].squeeze()  # Grabbing the last band in Mica

    # List to save quality metrics
    quality_metrics = []

    def generate_index_chunks(total_length, chunk_size):
        return [np.arange(i, min(i + chunk_size, total_length)) for i in range(0, total_length, chunk_size)]

    row_value_chunks = generate_index_chunks(int(np.max(waterfall_rows)) + 1, num_threads)

    pbar = tqdm(total = np.max(waterfall_rows),
                position=0,
                leave=True,
                desc = "Finding SS transformations...")
    results = []
    y_olds_all = []
    x_olds_all = []

    # Process each chunk of rows
    for row_value_chunk in row_value_chunks:
        hs_images, mica_images, y_olds_offset, x_olds_offset = [], [], [], []
        for row_value in row_value_chunk:

            y_old, x_old = np.where(waterfall_rows == row_value)

            # Skip rows with less than 10 points or nodata values
            if (row_value == 0) or (len(y_old) < 10) & (len(x_old) < 10):
                continue

            # grab the boundary box
            min_row, min_col = np.min(y_old), np.min(x_old)
            max_row, max_col = np.max(y_old), np.max(x_old)

            hs_images.append(hs_image[min_row:max_row + 1, min_col:max_col+1])
            mica_images.append(mica_image[min_row:max_row + 1, min_col:max_col+1])
            y_olds_offset.append(y_old - y_old.min())
            x_olds_offset.append(x_old - x_old.min())
            y_olds_all.append(y_old)
            x_olds_all.append(x_old)


        if len(hs_images) == 0:
            continue

        args_list = list(zip(hs_images, mica_images, y_olds_offset, x_olds_offset, repeat(pixel_shift)))
        with Pool(num_threads) as pool:
            results.extend(pool.starmap(get_shift_from_mi_patch, args_list))

        pbar.update(num_threads)

    pbar = tqdm(total = np.max(waterfall_rows),
                position=0,
                leave=True,
                desc = "Applying SS transformation to all bands.")
    for c, result in enumerate(results):
        linear_model, shift, quality_metric = result
        if linear_model is not None:
            hs_arr_shapeshifted, _, _ = shift_rows_from_model(hs_arr_shapeshifted,
                                                                      linear_model,
                                                                      y_olds_all[c],
                                                                      x_olds_all[c],
                                                                      shift)
        pbar.update(1)

    print(f"Overall relative improvement: {np.nansum(quality_metrics):.5f}.")

    # pefroming median filtering to smooth out the old values
    hs_arr_shapeshifted = medfilt3d(hs_arr_shapeshifted, kernel_size=kernel_size, use_torch = use_torch )

    ss_qa_filename = hs_filename.replace(".hdr", "_ss_qa.hdr")
    ss_filename = hs_filename.replace(".hdr", "_ss.hdr")

    # Save the quality assurance image
    quality_raster_metadata = copy.copy(hs_profile)
    quality_raster_metadata["bands"] = "1"
    quality_raster_metadata["band names"] = "Error Band"
    del quality_raster_metadata["wavelength"]
    quality_raster = np.mean(np.abs(hs_arr_shapeshifted - hs_arr), axis = 2)
    save_image_envi(quality_raster, quality_raster_metadata, ss_qa_filename, ext="")
    del quality_raster

    # Save the SS image
    save_image_envi(hs_arr_shapeshifted, hs_profile, ss_filename, ext="")

    return ss_filename, ss_qa_filename



if __name__ == "__main__":
    """
    use environment py37
    """
    # parser = argparse.ArgumentParser(description='Process some integers.')
    # parser.add_argument('hs_filename', type=str, help='hyperspectral hdr filename, the hyperspectral data MUST include georectified rows and indices using headwalls propietry software on last two bands!')
    # parser.add_argument('mica_filename', type=str, help='Mica hdr filename, mica data MUST be downsamples and coregistrered using the coregister.py code, otherwise this code will fail.')
    # parser.add_argument('--pixel_shift', type=int, default=3, help='Number of pixels to shift')
    # parser.add_argument('--kernel_size', type=int, default=3, help='Size of the kernel')
    #
    # args = parser.parse_args()
    # main(args.swir_filename, args.mica_filename, pixel_shift=args.pixel_shift, kernel_size=args.kernel_size)

    # debug
    mica_filename = "/Volumes/T7/axhcis/Projects/NURI/NURI_micasense_1133_transparent_mosaic_stacked_warped_cropped.hdr"
    swir_filename = "/Volumes/T7/axhcis/Projects/NURI/raw_1504_nuc_or_plusindices3_warped.hdr"
    shape_shift(swir_filename, mica_filename, pixel_shift = 3, kernel_size=3)