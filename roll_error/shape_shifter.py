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


def load_image_envi(waterfall_path):
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
    vnir_arr = np.array(vnir_ds.load())  # Load image data into a NumPy array
    return vnir_arr, vnir_profile


def save_image_envi(new_hs_arr, old_hs_profile, hs_hdr_path, dtype=None):
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
    # Update the description in the metadata
    old_hs_profile["description"] = hs_hdr_path

    # Save the new hyperspectral image to the ENVI file
    envi.save_image(hs_hdr_path, new_hs_arr, metadata=old_hs_profile, force=True, dtype=dtype)

    # Print a message indicating the successful saving of the image
    print("Image saved to:", hs_hdr_path)

def get_mask_from_convex_hull(image, y_new, x_new):
    """
    Generate a mask based on convex hull from given image and new points.

    Parameters:
    - image (numpy.ndarray): Image data.
    - y_new (array-like): Y-coordinates of new points.
    - x_new (array-like): X-coordinates of new points.

    Returns:
    - mask (numpy.ndarray): Binary mask indicating points within the convex hull.
    """

    if len(image.shape) == 3:
        # Find NaN indices in case of a multi-channel image
        nan_indices = np.where(np.isnan(image[...,0]))
        image_shape = image.shape[:2]
    else:
        # Find NaN indices for a single-channel image
        nan_indices = np.where(np.isnan(image))
        image_shape = image.shape

    # Exclude new points from the NaN indices
    points_old = zip(nan_indices[0],nan_indices[1])
    points_new = zip(y_new, x_new)
    points_final = [i for i in points_old if i not in points_new]
    points_final.extend(points_new)
    points_final = [list(i) for i in points_final]
    y_final, x_final = [list(t) for t in zip(*points_final)]
    y_final, x_final = np.array(y_final).squeeze(), np.array(x_final).squeeze()
    points_arr = np.concatenate([np.array(x_final)[..., None], np.array(y_final)[..., None]], 1)

    # Compute convex hull
    hull = ConvexHull(points_arr)

    # Get points within the convex hull
    points_within_hull = points_arr[hull.vertices]

    # Create a path from the points within the hull
    path = Path(points_within_hull)

    # Create meshgrid of image coordinates
    x, y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
    points = np.vstack((x.flatten(), y.flatten())).T

    # Check which points are within the convex hull
    mask = path.contains_points(points).reshape(image_shape)

    # Dilate the mask slightly
    mask = binary_dilation(mask, iterations=5)

    if len(image.shape) == 3:
        # If the image is multi-channel, extend the mask to cover all channels
        mask = np.tile(mask[..., None], image.shape[2])

    return mask

def get_mask_from_row(y_new, x_new, swir_image_copy):
    """
    Generate a mask based on row projection from given points.

    Parameters:
    - y_new (array-like): Y-coordinates of new points.
    - x_new (array-like): X-coordinates of new points.
    - swir_image_copy (numpy.ndarray): Copy of the SWIR image.

    Returns:
    - mask (numpy.ndarray): Binary mask indicating the row projection.
    """
    mask = np.zeros_like(swir_image_copy)
    pca = PCA()  # Number of principal components to keep
    data = np.vstack((y_new, x_new.squeeze())).T
    pca.fit(data)
    projected_data = pca.transform(data)
    # Find the minimum and maximum values along each axis
    x_min = np.min(projected_data[:, 1])
    x_max = np.max(projected_data[:, 1])
    y_min = np.min(projected_data[:, 0])
    y_max = np.max(projected_data[:, 0])

    proj_polygon_v = np.array([[y_max, x_min], [y_max, x_max], [y_min, x_max], [y_min, x_min]])
    polygon_v = np.round(pca.inverse_transform(proj_polygon_v))
    rr, cc = polygon(polygon_v[:, 0], polygon_v[:, 1], mask.shape)
    mask[rr, cc] = 1

    mask = mask.astype(bool)

    return mask

def to_uint8(x):
    return ((x - np.min(x[np.nonzero(x)])) / (np.max(x[np.nonzero(x)]) - np.min(x[np.nonzero(x)])) * 255).astype(np.uint8)
def shift_rows_from_model(hs_image_copy, model, x_old, y_old, n, quality_raster=None):
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
        return None, None, None, None

    # Check if new coordinates exceed image dimensions
    if (x_new.max() >= hs_image_copy.shape[1]) or (y_new.max() >= hs_image_copy.shape[0]):
        x_ins = np.where((x_new < hs_image_copy.shape[1]) & (x_new > 0))[0]
        y_ins = np.where((y_new < hs_image_copy.shape[0]) & (y_new > 0))[0]
        ins_indices = np.array(list(set(x_ins) & set(y_ins)))
        y_new = y_new[ins_indices]
        x_new = x_new.squeeze()[ins_indices]
        if quality_raster is not None:
            quality_raster[y_new, x_new] = np.abs(pixel_values[ins_indices] - hs_image_copy[y_new, x_new])
        hs_image_copy[y_new, x_new] = pixel_values[ins_indices]
    else:
        if quality_raster is not None:
            quality_raster[y_new, x_new.squeeze()] = np.abs(pixel_values - hs_image_copy[y_new, x_new.squeeze()])
        hs_image_copy[y_new, x_new.squeeze()] = pixel_values

    return hs_image_copy, x_new, y_new, quality_raster

def calculate_mi(hs_image_copy, mica_image, min_row, max_row, min_col, max_col):
    """
    Calculate mutual information between a hyperspectral image and a Mica image patch.

    Parameters:
    - hs_image_copy (numpy.ndarray): Copy of the hyperspectral image.
    - mica_image (numpy.ndarray): Mica image.
    - min_row (int): Minimum row index of the image patch.
    - max_row (int): Maximum row index of the image patch.
    - min_col (int): Minimum column index of the image patch.
    - max_col (int): Maximum column index of the image patch.

    Returns:
    - mi (float): Mutual information value.
    """
    # Extract patches from the hyperspectral and Mica images
    hs_patch = hs_image_copy[min_row:max_row + 1, min_col:max_col + 1]
    mica_patch = mica_image[min_row:max_row + 1, min_col:max_col + 1]

    # Convert patches to uint8 and then to int for compatibility with mutual information calculation
    hs_patch = to_uint8(hs_patch).astype(int)
    mica_patch = to_uint8(mica_patch).astype(int)

    # Calculate mutual information between the patches
    mi = normalized_mutual_information(hs_patch, mica_patch)
    return mi


def get_shift_from_mi(hs_image, mica_image, y_old, x_old, min_row, max_row, min_col, max_col, n=3):
    """
    Determine pixel shift from mutual information (MI) between a hyperspectral image and a Mica image.

    Parameters:
    - hs_image (numpy.ndarray): Hyperspectral image.
    - mica_image (numpy.ndarray): Mica image.
    - y_old (array-like): Old y-coordinates of points.
    - x_old (array-like): Old x-coordinates of points.
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
    model = RANSACRegressor(LinearRegression(), max_trials=1000, residual_threshold=30).fit(x_old.reshape(-1, 1), y_old)

    # Calculate MI before shifting
    mi_before = calculate_mi(hs_image, mica_image, min_row, max_row, min_col, max_col)

    # Generate pixel shifts
    pixel_shifts = np.arange(-n, +n + 1)

    # Iterate over pixel shifts
    for shift_value in pixel_shifts:
        hs_image_copy = copy.copy(hs_image)

        # Shift rows based on the regression model
        hs_image_copy, x_new, y_new, _ = shift_rows_from_model(hs_image_copy, model, x_old, y_old, shift_value)

        if (hs_image_copy is None) or (x_new is None) or (y_new is None):
            linear_models.append(None)
            mi_quality_metrics.append(np.nan)
            continue

        # Calculate MI after shifting
        mi_after = calculate_mi(hs_image_copy, mica_image, min_row, max_row, min_col, max_col)

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

def medfilt3d(hs_arr_copy, kernel_size=3):
    hs_arr_final = []
    for band in range(hs_arr_copy.shape[2]):
        hs_arr_final.append(medfilt2d(hs_arr_copy[...,band], kernel_size=kernel_size)[...,None])
    hs_arr_final = np.concatenate(hs_arr_final, 2)
    return hs_arr_final

def main(hs_filename, mica_filename, pixel_shift = 3, kernel_size = 3, hs_type = "swir"):

    # load the hyperspectral and mica
    hs_arr, hs_profile = load_image_envi(hs_filename)
    mica_arr, mica_profile = load_image_envi(mica_filename)

    # grabbing georectified rows and the original array
    waterfall_rows = hs_arr[..., -2].squeeze()
    hs_arr_copy = copy.copy(hs_arr)

    # grabbing the bands necessary to perform mutual information on
    if hs_type == "swir":
        hs_bands = np.arange(0,12)
    elif hs_type == "vnir":
        #TODO: define what bands vnir is
        hs_bands = np.arange(0, 12)

    # grabbing the correspondiong image
    hs_image = np.mean(hs_arr[..., hs_bands], axis=2)
    mica_image = mica_arr[..., -1].squeeze() # grabbing the last band in mica

    # setting up lists to save out the linear models for each row and the shift amount
    quality_raster = np.zeros_like(hs_arr_copy)
    quality_metrics = []
    rows_values = np.arange(1, np.max(waterfall_rows) + 1)
    pbar = tqdm(total = np.max(waterfall_rows), position=0, leave=True)
    for row_value in rows_values:

        y_old, x_old = np.where(waterfall_rows == row_value)
        mask_bool = waterfall_rows == row_value
        rows, cols = np.where(mask_bool)
        min_row, min_col = np.min(rows), np.min(cols)
        max_row, max_col = np.max(rows), np.max(cols)

        linear_model, shift, quality_metric = get_shift_from_mi(hs_image,
                                                                mica_image,
                                                                y_old, x_old,
                                                                min_row, max_row,
                                                                min_col, max_col,
                                                                pixel_shift)
        quality_metrics.append(quality_metric)

        if linear_model is None:
            continue

        # suggested method at this stage is cubic
        hs_arr_copy, _, _, quality_raster = shift_rows_from_model(hs_arr_copy,
                                                  linear_model,
                                                  x_old,
                                                  y_old,
                                                  shift,
                                                  quality_raster)

        pbar.update(1)

    print(f"Overall relative improvement: {np.nansum(quality_metrics):.5f}.")

    # pefroming median filtering to smooth out the old values
    hs_arr_copy = medfilt3d(hs_arr_copy, kernel_size=kernel_size)

    # need to change the metadata to fit the quality raster information and then save it
    quality_raster_metadata = copy.copy(hs_profile)
    quality_raster_metadata["bands"] = "1"
    quality_raster_metadata["band names"] = "Error Band"
    del quality_raster_metadata["wavelength"]
    quality_raster = np.mean(quality_raster, axis = 2)
    save_image_envi(quality_raster, quality_raster_metadata, hs_filename.replace(".hdr", "_QA.hdr"))

    # saving the image
    save_image_envi(hs_arr_copy, hs_profile, hs_filename.replace(".hdr", "_SS.hdr"))


if __name__ == "__main__":
    """
    use environment py37
    """
    # parser = argparse.ArgumentParser(description='Process some integers.')
    # parser.add_argument('hs_filename', type=str, help='hyperspectral hdr filename, the hyperspectral data MUST include georectified rows and indices using headwalls propietry software on last two bands!')
    # parser.add_argument('mica_filename', type=str, help='Mica hdr filename, mica data MUST be downsamples and coregistrered using the coregister_controlpoints_gui.py code, otherwise this code will fail.')
    # parser.add_argument('--pixel_shift', type=int, default=3, help='Number of pixels to shift')
    # parser.add_argument('--kernel_size', type=int, default=3, help='Size of the kernel')
    #
    # args = parser.parse_args()
    # main(args.swir_filename, args.mica_filename, pixel_shift=args.pixel_shift, kernel_size=args.kernel_size)

    # debug
    mica_filename = "/dirs/data/tirs/axhcis/Projects/NURI/Data/labspehre_data/1133/Micasense/NURI_micasense_1133_transparent_mosaic_stacked_warped_cropped.hdr"
    swir_filename = "/dirs/data/tirs/axhcis/Projects/NURI/Data/labspehre_data/1133/SWIR/raw_1504_nuc_or_plusindices3_warped.hdr"
    main(swir_filename, mica_filename, pixel_shift = 3, kernel_size=3)