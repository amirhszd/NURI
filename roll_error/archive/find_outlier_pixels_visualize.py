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


def load_image_envi(waterfall_path):
    vnir_ds = envi.open(waterfall_path)
    vnir_profile = vnir_ds.metadata
    vnir_wavelengths = vnir_profile["wavelength"]
    vnir_wavelengths = np.array([float(i) for i in vnir_wavelengths])
    vnir_arr = vnir_ds.load()

    return vnir_arr, vnir_profile, vnir_wavelengths


def detrend_signal(x, y):
    fit = np.polyval(np.polyfit(x, y, deg=3), x)
    return x - x.min(), y - fit

def main(or_hdr):
    global c
    vnir_arr, vnir_profile, vnir_wavelengths = load_image_envi(or_hdr)

    # grab the rows two bands
    waterfall_rows = vnir_arr[..., -2].squeeze()
    rep_band = (minmax_scale(vnir_arr[..., np.array([40])].squeeze()) * 255).astype(np.uint8)

    rows_unique = np.unique(waterfall_rows)
    # grab all the indices of rows and columns

    outliers = []
    rows_indices_detrended = []
    fig, axd = plt.subplots(1, 2 ,figsize=(8, 8))

    for i in rows_unique:
        if i == 0:
            continue

        row_indices = np.where(waterfall_rows == i)
        reg = linear_model.RANSACRegressor(linear_model.LinearRegression(), max_trials=1000, residual_threshold=25)
        reg.fit(row_indices[0].reshape(-1, 1), row_indices[1])
        xi = np.linspace(min(row_indices[0].reshape(-1, 1)), max(row_indices[0].reshape(-1, 1)), 500).reshape((-1, 1))
        yi = reg.predict(xi)
        inliers = reg.inlier_mask_
        outliers = np.logical_not(inliers)

        axd[1].scatter(row_indices[0][inliers], row_indices[1][inliers], c='k', label='inliers')
        axd[1].scatter(row_indices[0][outliers], row_indices[1][outliers], c='r', label='outliers')
        axd[1].plot(xi, yi)
        axd[0].set_title(f"col = {i}")
        mx_col = ma.masked_array(rep_band, waterfall_rows == i)
        axd[0].imshow(mx_col)
        plt.autoscale()
        plt.pause(0.5)


if __name__ == "__main__":
    or_hdr = "/Volumes/T7/axhcis/Projects/NURI/data/20210723_tait_labsphere/1133/SWIR/raw_1504_nuc_or_plusindices3.hdr"
    main(or_hdr)