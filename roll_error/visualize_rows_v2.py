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
    rep_band = (minmax_scale(vnir_arr[..., 80].squeeze()) * 255).astype(np.uint8)

    del vnir_arr

    rows_unique = np.unique(waterfall_rows)
    # grab all the indices of rows and columns
    rows_indices = [np.where(waterfall_rows == i) for i in rows_unique if i > 0]
    rows_indices_detrended = []
    for x, y in rows_indices:
        rows_indices_detrended.append(detrend_signal(x,y))


    # plotting out the variables
    fig, axd = plt.subplot_mosaic([['left','detrended', 'undetrended'], ["left","histx", "histx_orig"],["left", "histy", "histy_orig"]], figsize = (15,8))
    # scatter1 = axd["original"].scatter(rows_indices[0][0], rows_indices[0][1],  s = 8)
    x_det, y_det = detrend_signal(rows_indices[0][0], rows_indices[0][1])
    scatter2 = axd["detrended"].scatter(x_det, y_det,  s = 8)
    scatter3 = axd["undetrended"].scatter(rows_indices[0][0], rows_indices[0][1], s=8)
    axd["detrended"].set_ylabel(f"Detrended Indices")
    axd["undetrended"].set_ylabel(f"Undetrended Indices")
    im3 = axd["left"].imshow(waterfall_rows)    # Draw the scatter plot and marginals.
    axd["histx"].hist(x_det, bins=50)
    axd["histx"].set_ylabel("X histogram")
    axd["histy"].hist(y_det, bins=50)
    axd["histy"].set_ylabel("Y histogram")
    axd["histx_orig"].hist(rows_indices[0][0], bins=50)
    axd["histx_orig"].set_ylabel("X Orig histogram")
    axd["histy_orig"].hist(rows_indices[0][1], bins=50)
    axd["histy_orig"].set_ylabel("Y Orig histogram")
    plt.suptitle("use slider to change row number.")
    plt.autoscale()

    c = 1
    def on_changed(val):
        c = int(val)
        update(c)

    # Create a slider
    ax_slider = plt.axes([0.1, 0.025, 0.8, 0.03], facecolor='lightgoldenrodyellow')
    slider = widgets.Slider(ax_slider, 'Row', 1, len(rows_unique) - 2, valinit=c, valstep=1)
    slider.on_changed(on_changed)

    # Function to update plots based on slider value
    def update(c):

        x_det, y_det = detrend_signal(rows_indices[c][0], rows_indices[c][1])
        det_data = np.concatenate([x_det[:,None], y_det[:,None]],1)
        scatter2.set_offsets(det_data)
        axd["detrended"].set_xlim([det_data[:, 0].min() - 10, det_data[:, 0].max() + 10])
        axd["detrended"].set_ylim([det_data[:, 1].min() - 10, det_data[:, 1].max() + 10])

        data = np.concatenate([rows_indices[c][0][:,None], rows_indices[c][1][:,None]],1)
        scatter3.set_offsets(data)
        axd["undetrended"].set_xlim([data[:, 0].min() - 10, data[:, 0].max() + 10])
        axd["undetrended"].set_ylim([data[:, 1].min() - 10, data[:, 1].max() + 10])

        axd["histx"].cla()
        axd["histy"].cla()
        axd["histx"].hist(det_data[:, 0], 50)
        axd["histy"].hist(det_data[:, 1], 50)
        axd["histx"].set_ylabel("X det histogram")
        axd["histy"].set_ylabel("Y det histogram")
        axd["histx_orig"].cla()
        axd["histy_orig"].cla()
        axd["histx_orig"].hist(data[:, 0], 50)
        axd["histy_orig"].hist(data[:, 1], 50)
        axd["histx_orig"].set_ylabel("X histogram")
        axd["histy_orig"].set_ylabel("Y histogram")

        mx_col = ma.masked_array(rep_band, waterfall_rows == rows_unique[c])
        axd["left"].set_title(f"col = {rows_unique[c]}")
        im3.set_data(mx_col)


        fig.canvas.draw_idle()

    update(c)  # Display initial plot


    plt.show()
    print("ok")


if __name__ == "__main__":
    main()