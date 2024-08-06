from spectral.io import envi
import numpy as np
import numpy.ma as ma
import os
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import minmax_scale
import matplotlib.cm as cm
import matplotlib.widgets as widgets
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from scipy.interpolate import griddata, interp1d
import copy
from tqdm import tqdm

DEBUG = False
def load_image_envi(waterfall_path):
    vnir_ds = envi.open(waterfall_path)
    vnir_profile = vnir_ds.metadata
    vnir_wavelengths = vnir_profile["wavelength"]
    vnir_wavelengths = np.array([float(i) for i in vnir_wavelengths])
    vnir_arr = vnir_ds.load()

    return vnir_arr, vnir_profile, vnir_wavelengths


def detrend_signal(x, y):
    fit = np.polyval(np.polyfit(x, y, deg=1), x)
    return x - x.min(), y - fit

def fix_row_outliers(waterfall_rows, rows_unique, arr, threshold_pixels = 30):
    # FINDING BIZZARE PIXELS IN THE ROW DIRECTION

    if DEBUG:
        fig, ax = plt.subplots(1,1)
        fig2, ax2 = plt.subplots(1, 1)
    waterfall_rows_copy = copy.copy(waterfall_rows)
    outliers_prior = []
    outliers_fixed = []
    for i in rows_unique:
        if i == 0:
            continue

        row_indices_x, row_indices_y = np.where(waterfall_rows == i)
        # if the number of pixels is below 5% of the number of columns, consider them inlier
        if len(row_indices_x) < 0.05*waterfall_rows.shape[1]:
            continue

        reg = linear_model.RANSACRegressor(linear_model.LinearRegression(), max_trials=1000, residual_threshold=threshold_pixels)
        reg.fit(row_indices_x.reshape(-1, 1), row_indices_y)
        inliers = reg.inlier_mask_
        outliers = np.logical_not(inliers)

        # if there is some outliers put them back where they belong
        if sum(outliers.astype(int)) > 0:

            print(f"Found {sum(outliers.astype(int))} outliers; row {i}", end = "\r")
            # predicting the outliers using the model that we fit
            out_pred_y = np.array(list(map(round,reg.predict(row_indices_x[outliers].reshape(-1,1)))))

            # make sure the prediction is within the limits of the image
            out_pred_y[np.where(out_pred_y >= waterfall_rows.shape[1])] = waterfall_rows.shape[1] - 1
            out_pred_y[np.where(out_pred_y < 0)] = 0

            # appending prior and fixed to both
            outliers_prior.append([row_indices_x[outliers], row_indices_y[outliers]])
            outliers_fixed.append([row_indices_x[outliers], out_pred_y])

            # leave the old value of indices as nan and fix/update the waterfall values
            waterfall_rows_copy[row_indices_x[outliers], row_indices_y[outliers]] = np.repeat(np.nan, len(out_pred_y))
            waterfall_rows_copy[row_indices_x[outliers], out_pred_y] = i

            if DEBUG:
                ax.clear()
                ax.set_title(f"row = {i} \n"
                             f"outliers {np.sum(outliers.astype(int))} of {len(row_indices_y)}")
                ax.scatter(row_indices_x[inliers],  row_indices_y[inliers], label="inliers")
                ax.scatter(row_indices_x[outliers], row_indices_y[outliers], label="outliers")
                ax.scatter(row_indices_x[outliers], out_pred_y, label="outliers")
                ax2.clear()
                ax2.imshow(arr)
                ax2.scatter(row_indices_y[outliers],row_indices_x[outliers], c= "r")
                ax2.scatter(out_pred_y, row_indices_x[outliers], c="b")
                plt.pause(0.25)

    outliers_prior_x = [i[0] for i in outliers_prior]
    outliers_prior_y = [i[1] for i in outliers_prior]
    outliers_fixed_x = [i[0] for i in outliers_fixed]
    outliers_fixed_y = [i[1] for i in outliers_fixed]

    print(f"Total number of outliers found and replaced: {len(outliers_fixed_y)}")

    return outliers_prior_x, outliers_prior_y, outliers_fixed_x, outliers_fixed_y, waterfall_rows_copy



def move_indices_nan_values(vnir_arr, outliers_prior_x, outliers_prior_y, outliers_fixed_x, outliers_fixed_y):
    vnir_arr_copy = copy.copy(vnir_arr)

    for i in tqdm(range(len(outliers_prior_x))):
        # replace it with the new value
        vnir_arr_copy[outliers_fixed_x[i], outliers_fixed_y[i]] = vnir_arr[outliers_prior_x[i], outliers_prior_y[i]]

        # set old value to nan
        vnir_arr_copy[outliers_prior_x[i], outliers_prior_y[i]] = np.nan

    return vnir_arr_copy



def main(or_hdr):
    global c
    vnir_arr, vnir_profile, vnir_wavelengths = load_image_envi(or_hdr)

    # grab cols and row bands and get the unique values
    waterfall_rows = vnir_arr[..., -2].squeeze()
    waterfall_cols = vnir_arr[..., -1].squeeze()
    rows_unique = np.unique(waterfall_rows)
    cols_unique = np.unique(waterfall_cols)

    fig, ax = plt.subplots(1,3)
    ax[0].imshow(vnir_arr[...,40])
    ax[0].set_title("Raw")

    # grab bizzare pixels in rows
    outliers_prior_x,     \
        outliers_prior_y, \
        outliers_fixed_x, \
        outliers_fixed_y, \
        waterfall_rows_new = fix_row_outliers(waterfall_rows,
                                                        rows_unique,
                                                        vnir_arr[..., 40],
                                                        threshold_pixels= 35)

    # move the values of indices along all the channels and let the old values set to zero
    vnir_arr_new = move_indices_nan_values(vnir_arr, outliers_prior_x, outliers_prior_y, outliers_fixed_x, outliers_fixed_y)

    # setting x and y as internal temperature and bb since those are fixed for all four bands, and then interpolate voltage
    x = np.arange(0, vnir_arr.shape[1])
    y = np.arange(0, vnir_arr.shape[0])
    # mask invalid values
    array = np.ma.masked_invalid(vnir_arr[...,40]).squeeze()
    xx, yy = np.meshgrid(x, y)
    x1 = xx[~array.mask]
    y1 = yy[~array.mask]
    newarr = array[~array.mask]

    interp = griddata((x1, y1),
                   newarr.ravel(),
                   (xx, yy),
                   method='linear')

    ax[1].imshow(vnir_arr[...,40])
    ax[1].set_title("Raw with Pixels")
    ax[2].imshow(interp)
    ax[2].set_title("Interpolated")
    plt.suptitle("Close figure to save file image.")
    plt.show()



if __name__ == "__main__":
    or_hdr = "/Volumes/T7/axhcis/Projects/NURI/data/20210723_tait_labsphere/1133/SWIR/raw_1504_nuc_or_plusindices3.hdr"
    main(or_hdr)