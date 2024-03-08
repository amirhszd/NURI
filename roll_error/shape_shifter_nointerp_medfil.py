import copy
from scipy.signal import medfilt2d
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
from sklearn.linear_model import LinearRegression, RANSACRegressor
from skimage.metrics import normalized_mutual_information
from scipy.interpolate import griddata
from tqdm import tqdm
from sklearn.decomposition import PCA
from skimage.draw import polygon
from scipy.interpolate import RectBivariateSpline
from scipy.spatial import ConvexHull
from matplotlib.path import Path
from scipy.ndimage import binary_dilation
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
import multiprocessing


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

def get_mask_from_convex_hull(image, y_new, x_new):

    if len(image.shape) == 3:
        nan_indices = np.where(np.isnan(image[...,0]))
        image_shape = image.shape[:2]
    else:
        nan_indices = np.where(np.isnan(image))
        image_shape = image.shape

    points_old = zip(nan_indices[0],nan_indices[1])
    points_new = zip(y_new, x_new)
    points_final = [i for i in points_old if i not in points_new]
    points_final.extend(points_new)
    points_final = [list(i) for i in points_final]
    y_final, x_final = [list(t) for t in zip(*points_final)]
    y_final, x_final = np.array(y_final).squeeze(), np.array(x_final).squeeze()
    points_arr = np.concatenate([np.array(x_final)[..., None], np.array(y_final)[..., None]], 1)

    hull = ConvexHull(points_arr)

    points_within_hull = points_arr[hull.vertices]

    # Create a path from the points within the hull
    path = Path(points_within_hull)

    # Create meshgrid of image coordinates
    x, y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
    points = np.vstack((x.flatten(), y.flatten())).T

    # Check which points are within the convex hull
    mask = path.contains_points(points).reshape(image_shape)

    # dilate a mask a little bit
    mask = binary_dilation(mask, iterations=5)

    if len(image.shape) == 3:
        mask = np.tile(mask[..., None], image.shape[2])

    return mask

def get_mask_from_row(y_new, x_new, swir_image_copy):

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

def interpolate_image(image, valid_mask, method, i):
    rows, cols, channels = np.indices(image.shape)
    points = np.column_stack((cols[valid_mask], rows[valid_mask]))
    indices = np.where(valid_mask)
    values = image[indices[0], indices[1], i]
    grid_x, grid_y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
    return i, griddata(points, values, (grid_x, grid_y), method=method)

def replace_nan_values(image, min_row, max_row, min_col, max_col, y_new, x_new, method="nearest"):

    # Get boolean mask representing the convex hull around new indices
    row_bool_mask = get_mask_from_convex_hull(image, y_new, x_new)

    # Crop masks and image to specified region
    row_bool_mask = row_bool_mask[min_row:max_row+1, min_col:max_col+1]
    image = image[min_row:max_row+1, min_col:max_col+1]

    # Create masks for NaN values and valid points within the convex hull
    nan_mask = np.isnan(image)
    valid_mask = (row_bool_mask) & (~nan_mask)

    # Generate coordinates and values for valid points
    if len(image.shape) == 3:
        start = time.time()
        interpolated_values = np.zeros_like(image)
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(interpolate_image, image, valid_mask, method, i) for i in range(image.shape[2])]
            for future in as_completed(futures):
                index, values = future.result()
                interpolated_values[...,index] = values
        print(f"total time: {time.time() - start}")

    else:
        rows, cols = np.indices(image.shape)
        points = np.column_stack((cols[valid_mask], rows[valid_mask]))
        values = image[valid_mask]
        grid_x, grid_y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
        interpolated_values = griddata(points, values, (grid_x, grid_y), method=method)

    # Replace NaN values with interpolated values
    interpolated_image = np.where(nan_mask, interpolated_values, image)

    return interpolated_image


def to_uint8(x):
    return ((x - np.min(x[np.nonzero(x)])) / (np.max(x[np.nonzero(x)]) - np.min(x[np.nonzero(x)])) * 255).astype(np.uint8)

def shift_rows_from_model(hs_image_copy, model, x_old, y_old, min_row, max_row, min_col, max_col, n, method = "nearest"):

    pixel_values = hs_image_copy[y_old, x_old]
    # hs_image_copy[y_old, x_old] = np.nan

    # lets get residuals from old values
    y_old_model = model.predict(x_old.reshape(-1, 1))
    y_res = y_old_model - y_old

    # and adding that residual to the points
    x_new = x_old.reshape(-1, 1) + n
    y_new_model = model.predict(x_new)
    y_new = np.round(y_new_model - y_res).astype(int)

    if len(y_new) < 10:
        return None, None, None

    if (x_new.max() >= hs_image_copy.shape[1]) or (y_new.max() >= hs_image_copy.shape[0]):
        x_ins = np.where((x_new < hs_image_copy.shape[1]) & (x_new > 0))[0]
        y_ins = np.where((y_new < hs_image_copy.shape[0]) & (y_new > 0))[0]
        ins_indices = np.array(list(set(x_ins) & set(y_ins)))
        y_new = y_new[ins_indices]
        x_new = x_new.squeeze()[ins_indices]
        hs_image_copy[y_new, x_new] = pixel_values[ins_indices]
    else:
        hs_image_copy[y_new, x_new.squeeze()] = pixel_values

    # # perform a quick interpolation over nan values for swir
    # hs_image_copy[min_row:max_row + 1, min_col:max_col + 1] = replace_nan_values(hs_image_copy, min_row, max_row, min_col,
    #                                                                              max_col, y_new, x_new, method)

    return hs_image_copy, x_new, y_new


def calculate_mi(hs_image_copy,
                             mica_image,
                             min_row,
                             max_row,
                             min_col,
                             max_col):
    # calculate the quality metric
    hs_patch = hs_image_copy[min_row:max_row + 1, min_col:max_col + 1]
    hs_patch = to_uint8(hs_patch).astype(int)
    mica_patch = mica_image[min_row:max_row + 1, min_col:max_col + 1]
    mica_patch = to_uint8(mica_patch).astype(int)
    mi = normalized_mutual_information(hs_patch, mica_patch)
    # mi = np.correlate(hs_patch.flatten(), mica_patch.flatten())[0]
    return mi

def save_image_envi(new_hs_arr, old_hs_profile, hs_hdr_path):
    hs_hdr_path_new = hs_hdr_path.replace(".hdr", "_ss.hdr")

    # replicating vnir metadata except the bands and wavelength
    old_hs_profile["description"] = hs_hdr_path_new
    envi.save_image(hs_hdr_path_new, new_hs_arr, metadata=old_hs_profile, force=True)

    print("image saved to: " + hs_hdr_path_new)

def get_shift_from_mi(hs_image,
                        mica_image,
                        y_old, x_old,
                        min_row, max_row,
                        min_col, max_col,
                        n = 3,
                        method = "nearest"):

    # run linear regression, passing Y and X
    mi_quality_metrics = []
    linear_models = []

    # we might have points that are crap
    if (len(x_old) < 2) or \
            (len(y_old) < 2):
        return None, np.nan, np.nan

    model = RANSACRegressor(LinearRegression(), max_trials=1000, residual_threshold=30).fit(x_old.reshape(-1, 1), y_old)
    mi_before = calculate_mi(hs_image, mica_image, min_row, max_row, min_col, max_col)
    pixel_shifts = np.arange(-n, +n + 1)
    for c, i in enumerate(pixel_shifts):
        #taking out the pixels and
        hs_image_copy = copy.copy(hs_image)

        hs_image_copy, x_new, y_new = shift_rows_from_model(hs_image_copy,
                                                            model,
                                                            x_old,
                                                            y_old,
                                                            min_row,
                                                            max_row,
                                                            min_col,
                                                            max_col,
                                                            i,
                                                            method)

        if (hs_image_copy is None) or \
                (x_new is None) or\
                (y_new is None):
            linear_models.append(None)
            mi_quality_metrics.append(np.nan)
            continue

        mi_after = calculate_mi(hs_image_copy,
                                mica_image,
                                min_row,
                                max_row,
                                min_col,
                                max_col)
        mi_quality_metric = (mi_after - mi_before) / (mi_before)
        mi_quality_metrics.append(mi_quality_metric)
        linear_models.append(model)

    if len(linear_models) > 0:
        argmax = np.argmax(mi_quality_metrics)
        print(f"shift: {pixel_shifts[argmax]} pixels, score: {mi_quality_metrics[argmax]:.4f}")
        return linear_models[argmax], pixel_shifts[argmax], mi_quality_metrics[argmax]
    else:
        return None, np.nan, np.nan

def main(hs_hdr, mica_hdr, method = "nearest", hs_type = "swir", ):

    # load the hyperspectral and mica
    hs_arr, hs_profile = load_image_envi(hs_hdr)
    # hs_arr = hs_arr[...,:5]
    mica_arr, mica_profile = load_image_envi(mica_hdr)
    waterfall_rows = hs_arr[..., -2].squeeze()
    hs_arr_copy = copy.copy(hs_arr)

    # grab a patch to calculate the highest coefficient bands
    if hs_type == "swir":
        hs_bands = np.arange(0,12)
    elif hs_type == "vnir":
        #TODO: define what bands vnir is
        hs_bands = np.arange(0, 12)

    # grabbing the correspondiong image
    hs_image = np.mean(hs_arr[..., hs_bands], axis=2)
    mica_image = mica_arr[..., -1].squeeze() # grabbing the last band in mica

    # setting up lists to save out the linear models for each row and the shift amount
    quality_metrics = []

    rows_values = np.arange(1, np.max(waterfall_rows) + 1)
    pbar = tqdm(total = np.max(waterfall_rows), position=0, leave=True)
    rows_values = rows_values[850:900]
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
                                                                3,
                                                                method)
        quality_metrics.append(quality_metric)

        if linear_model is None:
            continue

        # suggested method at this stage is cubic
        hs_arr_copy, _, _ = shift_rows_from_model(hs_arr_copy,
                                                  linear_model,
                                                  x_old,
                                                  y_old,
                                                  min_row,
                                                  max_row,
                                                  min_col,
                                                  max_col,
                                                  shift,
                                                  method)

        pbar.update(1)

    print(f"Overall relative improvement: {np.nansum(quality_metrics):.5f}.")
    save_image_envi(hs_arr, hs_profile, hs_hdr)


def main_debug(hs_filename, mica_filename, method = "nearest", hs_type = "swir", ):

    # # downsample both mica
    # hs_filename = warp_to_scale(hs_filename, 0.5 )
    # mica_filename =warp_to_scale(mica_filename, 0.5 )
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    # load the hyperspectral and mica
    hs_arr, hs_profile = load_image_envi(hs_filename)
    # hs_arr = hs_arr[...,:5]
    mica_arr, mica_profile = load_image_envi(mica_filename)
    waterfall_rows = hs_arr[..., -2].squeeze()
    hs_arr_copy = copy.copy(hs_arr)
    hs_arr_copy = np.mean(hs_arr_copy[..., np.arange(0,12)], axis=2)

    # grab a patch to calculate the highest coefficient bands
    if hs_type == "swir":
        hs_bands = np.arange(0,12)
    elif hs_type == "vnir":
        #TODO: define what bands vnir is
        hs_bands = np.arange(0, 12)

    # grabbing the correspondiong image
    hs_image = np.mean(hs_arr[..., hs_bands], axis=2)
    mica_image = mica_arr[..., -1].squeeze() # grabbing the last band in mica

    # setting up lists to save out the linear models for each row and the shift amount
    quality_metrics = []

    axs[0].imshow(hs_arr_copy, cmap='gray')
    axs[0].set_title('Original')
    im = axs[1].imshow(hs_arr_copy, cmap='gray')
    axs[1].set_title('Modified')
    axs[2].imshow(mica_image, cmap='gray')
    axs[2].set_title('Original')

    rows_values = np.arange(1, np.max(waterfall_rows) + 1)
    pbar = tqdm(total = np.max(waterfall_rows), position=0, leave=True)
    # rows_values = rows_values[800:1200]
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
                                                                3,
                                                                method)
        quality_metrics.append(quality_metric)

        if linear_model is None:
            continue

        # suggested method at this stage is cubic
        hs_arr_copy, _, _ = shift_rows_from_model(hs_arr_copy,
                                                  linear_model,
                                                  x_old,
                                                  y_old,
                                                  min_row,
                                                  max_row,
                                                  min_col,
                                                  max_col,
                                                  shift,
                                                  method)
        # plt.pause(0.5)

        pbar.update(1)
    hs_arr_copy = medfilt2d(hs_arr_copy, kernel_size=3)
    im.set_data(hs_arr_copy)

    print(f"Overall relative improvement: {np.nansum(quality_metrics):.5f}.")
    plt.show()
    save_image_envi(hs_arr, hs_profile, hs_filename, method)



if __name__ == "__main__":

    # todo: run ransac on each line which is not currently implemented
    # todo: run the shift in pixels on the entire hyperspectral array
    # todo: show the before and after in color
    # todo: os picking correllation band problematic? SWIR 113 best with Mica 0
    # todo: maybe 3 pixels left and right is not doing much


    mica_hdr = "/Volumes/T7/axhcis/Projects/NURI/data/20210723_tait_labsphere/1133/Micasense/NURI_micasense_1133_transparent_mosaic_stacked_warped_cropped.hdr"
    swir_hdr = "/Volumes/T7/axhcis/Projects/NURI/data/20210723_tait_labsphere/1133/SWIR/raw_1504_nuc_or_plusindices3_warped.hdr"
    main_debug(swir_hdr, mica_hdr, method = "nearest")


