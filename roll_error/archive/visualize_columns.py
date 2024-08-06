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


def main():
    global c
    or_hdr = "/Volumes/T7/axhcis/Projects/NURI/data/20210723_tait_labsphere/1133/SWIR/raw_1504_nuc_or_plusindices3.hdr"
    vnir_arr, vnir_profile, vnir_wavelengths = load_image_envi(or_hdr)

    # grab the rows two bands
    waterfall_rows = vnir_arr[..., -2].squeeze()
    waterfall_columns = vnir_arr[..., -1].squeeze()
    rep_band = minmax_scale(vnir_arr[..., 80].squeeze()) * 256

    cols_unique = np.unique(waterfall_columns)
    # grab all the indices of rows and columns
    columns_indices = [np.where(waterfall_columns == i) for i in cols_unique if i > 0]
    columns_indices_detrended = []
    for x, y in columns_indices:
        columns_indices_detrended.append(detrend_signal(x,y))


    # lets calcualte the std
    x_max_range = columns_indices_detrended[np.argmax([len(i[0]) for i in columns_indices_detrended])][0]
    columns_indices_detrended_interp = []
    for x, y in columns_indices_detrended:
        # Interpolate y values based on the largest_x
        f = interpolate.interp1d(x, y, kind='linear', fill_value='extrapolate')
        interpolated_y = f(x_max_range)
        columns_indices_detrended_interp.append(interpolated_y[None,:])
    columns_indices_detrended_interp = np.concatenate(columns_indices_detrended_interp, 0)
    column_indices_std = np.nanstd(columns_indices_detrended_interp,0)
    column_indices_mean = np.mean(columns_indices_detrended_interp,0)

    # calculate residuals to see where the big changes are
    columns_indices_detrended_residuals = []
    for x, y in columns_indices_detrended:
        # Interpolate y values based on the largest_x
        f = interpolate.interp1d(x_max_range, column_indices_mean, kind='linear', fill_value='extrapolate')
        mean_interpolated_y = f(x)
        columns_indices_detrended_residuals.append((x, np.abs(y - mean_interpolated_y)))

    # plotting out the variables
    fig, axd = plt.subplot_mosaic([['left','detrended'], ['left','histx'], ["left", "histy"], ["left", "res"]], figsize = (10,10))
    x_det, y_det = detrend_signal(columns_indices[0][0], columns_indices[0][1])
    scatter2 = axd["detrended"].scatter(x_det, y_det,  s = 8)
    axd["detrended"].set_ylabel(f"Detrended Indices")
    im3 = axd["left"].imshow(waterfall_columns)
    scatter3 = axd["res"].scatter(columns_indices_detrended_residuals[0][0], columns_indices_detrended_residuals[0][1], s=8)
    axd["res"].set_ylabel(f"Detrended residuals")

    axd["histx"].hist(x_det, bins=50)
    axd["histx"].set_ylabel("X histogram")
    axd["histy"].hist(y_det, bins=50)
    axd["histy"].set_ylabel("Y histogram")


    plt.suptitle("use slider to change column number.")
    plt.autoscale()


    c = 1

    def on_changed(val):
        c = int(val)
        update(c)

    # Create a slider
    ax_slider = plt.axes([0.1, 0.05, 0.3, 0.03], facecolor='lightgoldenrodyellow')
    slider = widgets.Slider(ax_slider, 'Column', 1, len(cols_unique) - 2, valinit=c, valstep=1)
    slider.on_changed(on_changed)

    # Function to update plots based on slider value
    def update(c):
        # axd["original"].set_title(f"Indices Scatter plot, column = {cols_unique[c]}")
        # orig_data = np.concatenate([columns_indices[c][0][:, None], columns_indices[c][1][:, None]], 1)
        # scatter1.set_offsets(orig_data)
        # axd["original"].set_xlim([orig_data[:, 0].min() - 50, orig_data[:, 0].max() + 50])
        # axd["original"].set_ylim([orig_data[:, 1].min() - 50, orig_data[:, 1].max() + 50])

        x_det, y_det = detrend_signal(columns_indices[c][0], columns_indices[c][1])
        det_data = np.concatenate([x_det[:,None], y_det[:,None]],1)
        scatter2.set_offsets(det_data)

        axd["histx"].cla()
        axd["histy"].cla()
        axd["histx"].hist(det_data[:, 0], 50)
        axd["histy"].hist(det_data[:, 1], 50)
        axd["histx"].set_ylabel("X histogram")
        axd["histy"].set_ylabel("Y histogram")

        mx_col = ma.masked_array(rep_band, waterfall_columns == cols_unique[c])
        axd["left"].set_title(f"col = {cols_unique[c]}")
        im3.set_data(mx_col)

        res_data = np.concatenate([columns_indices_detrended_residuals[c][0][:,None], columns_indices_detrended_residuals[c][1][:,None]],1)
        scatter3.set_offsets(res_data)

        fig.canvas.draw_idle()

    update(c)  # Display initial plot


    plt.show()
    print("ok")


if __name__ == "__main__":
    main()