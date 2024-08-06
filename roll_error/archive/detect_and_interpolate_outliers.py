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

def get_row_outliers(waterfall_rows, rows_unique, threshold_pixels = 30):
    # FINDING BIZZARE PIXELS IN THE ROW DIRECTION
    outliers_x_rows = []
    outliers_y_rows = []
    for i in rows_unique:
        if i == 0:
            continue

        row_indices = np.where(waterfall_rows == i)

        # if the number of pixels is below 5% of the number of columns, consider them inlier
        if len(row_indices[0]) < 0.05*waterfall_rows.shape[1]:
            continue

        reg = linear_model.RANSACRegressor(linear_model.LinearRegression(), max_trials=1000, residual_threshold=threshold_pixels)
        reg.fit(row_indices[0].reshape(-1, 1), row_indices[1])
        # xi = np.linspace(min(row_indices[0].reshape(-1, 1)), max(row_indices[0].reshape(-1, 1)), 500).reshape((-1, 1))
        # yi = reg.predict(xi)
        inliers = reg.inlier_mask_
        outliers = np.logical_not(inliers)

        if sum(outliers.astype(int)) > 0:
            outliers_x_rows.append(row_indices[0][outliers])
            outliers_y_rows.append(row_indices[1][outliers])


    outliers_x = []
    outliers_y = []
    for i in range(len(outliers_x_rows)):
        outlier_x = outliers_x_rows[i]
        outlier_y = outliers_y_rows[i]
        for j in range(len(outlier_x)):
            outliers_x.append(outlier_x[j])
            outliers_y.append(outlier_y[j])

    return outliers_x, outliers_y



def get_col_outliers(waterfall_cols, cols_unique, threshold_perc = 0.75):
    # FINDING BIZZARE PIXELS IN THE COLUMN DIRECTION
    outliers_x_cols = []
    outliers_y_cols = []
    columns_indices = [np.where(waterfall_cols == i) for i in cols_unique if i > 0]
    columns_indices_detrended = []
    for x, y in columns_indices:
        columns_indices_detrended.append(detrend_signal(x, y))

    # lets calcualte the mean column
    x_max_range = columns_indices_detrended[np.argmax([len(i[0]) for i in columns_indices_detrended])][0]
    columns_indices_detrended_interp = []
    for x, y in columns_indices_detrended:
        # Interpolate y values based on the largest_x
        f = interp1d(x, y, kind='linear', fill_value='extrapolate')
        interpolated_y = f(x_max_range)
        columns_indices_detrended_interp.append(interpolated_y[None, :])
    columns_indices_detrended_interp = np.concatenate(columns_indices_detrended_interp, 0)
    column_indices_mean = np.mean(columns_indices_detrended_interp, 0)
    column_indices_mean = np.nan_to_num(column_indices_mean, nan=0, posinf=0, neginf=0)

    # calculate residuals to see where the big changes are
    for c, (x, y) in enumerate(columns_indices_detrended):
        # Interpolate y values based on the largest_x
        f = interp1d(x_max_range, column_indices_mean, kind='linear', fill_value='extrapolate')
        mean_interpolated_y = f(x)
        x_res, y_res = x, np.abs(y - mean_interpolated_y)
        y_res = np.nan_to_num(y_res, nan=0, posinf=0, neginf=0)
        outliers = y_res > np.nanmax(y_res) * threshold_perc

        col_indices = np.where(waterfall_cols == c + 1)
        if sum(outliers.astype(int)) > 0:
            outliers_x_cols.append(col_indices[0][outliers])
            outliers_y_cols.append(col_indices[1][outliers])

    outliers_x = []
    outliers_y = []
    for i in range(len(outliers_x_cols)):
        outlier_x = outliers_x_cols[i]
        outlier_y = outliers_y_cols[i]
        for j in range(len(outlier_x)):
            outliers_x.append(outlier_x[j])
            outliers_y.append(outlier_y[j])

    return outliers_x, outliers_y

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
    outliers_x_rows, outliers_y_rows = get_row_outliers(waterfall_rows, rows_unique, threshold_pixels= 35)

    # grab bizzare pixels in cols
    outliers_x_cols, outliers_y_cols = get_col_outliers(waterfall_cols, cols_unique, threshold_perc= 0.95)

    outliers_x = np.concatenate([outliers_x_cols, outliers_x_rows])
    outliers_y = np.concatenate([outliers_y_cols, outliers_y_rows])
    vnir_arr[outliers_x, outliers_y] = np.nan

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