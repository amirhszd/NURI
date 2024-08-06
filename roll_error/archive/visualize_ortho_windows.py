from spectral.io import envi
import numpy as np
import numpy.ma as ma
import os
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import minmax_scale
import matplotlib.cm as cm
import matplotlib.widgets as widgets
def load_image_envi(waterfall_path):
    vnir_ds = envi.open(waterfall_path)
    vnir_profile = vnir_ds.metadata
    vnir_wavelengths = vnir_profile["wavelength"]
    vnir_wavelengths = np.array([float(i) for i in vnir_wavelengths])
    vnir_arr = vnir_ds.load()

    return vnir_arr, vnir_profile, vnir_wavelengths


def main():
    global c, k
    or_hdr = "/Volumes/T7/axhcis/Projects/NURI/data/20210723_tait_labsphere/1133/SWIR/raw_1504_nuc_or_plusindices3.hdr"
    vnir_arr, vnir_profile, vnir_wavelengths = load_image_envi(or_hdr)

    # grab the rows two bands
    waterfall_rows = vnir_arr[..., -2].squeeze()
    waterfall_columns = vnir_arr[..., -1].squeeze()
    rep_band = minmax_scale(vnir_arr[..., 80].squeeze()) * 256

    fig, axd = plt.subplot_mosaic([['top_left', 'top_right'], ['bottom_left', 'bottom_right']],
                                  constrained_layout=True)

    im1 = axd["top_left"].imshow(waterfall_rows)
    im2 = axd["top_right"].imshow(rep_band)
    im3 = axd["bottom_left"].imshow(waterfall_columns)
    im4 = axd["bottom_right"].imshow(rep_band)

    plt.suptitle("use up and down keys to change row number.")

    rows_unique = np.unique(waterfall_rows)
    cols_unique = np.unique(waterfall_columns)
    c = 0
    k = 0
    # Function to handle key presses
    def on_key(event):
        global c, k
        if event.key == 'up':
            c += 1
        elif event.key == 'down':
            c -= 1
        if event.key == 'right':
            k += 1
        elif event.key == 'left':
            k -= 1
        update(c,k)

    # Function to update plots based on slider value
    def update(c, k):

        mx_row = ma.masked_array(rep_band, waterfall_rows == rows_unique[c])
        axd["top_left"].set_title(f"row = {rows_unique[c]}")
        im1.set_data(mx_row)

        indices = np.where(waterfall_rows == rows_unique[c])
        rows_min, rows_max = np.min(indices[0]), np.max(indices[0])
        cols_min, cols_max = np.min(indices[1]), np.max(indices[1])
        rep_band_window = rep_band[rows_min:rows_max + 1, cols_min:cols_max + 1]
        row_window = waterfall_rows[rows_min:rows_max + 1, cols_min:cols_max + 1]
        mx_rep_band_row = ma.masked_array(rep_band_window, row_window == rows_unique[c])
        axd["top_right"].set_title(f"row = {rows_unique[c]}")
        im2.set_data(mx_rep_band_row)


        mx_col = ma.masked_array(rep_band, waterfall_columns == cols_unique[k])

        axd["bottom_left"].set_title(f"col = {cols_unique[k]}")
        im3.set_data(mx_col)

        indices = np.where(waterfall_columns == cols_unique[k])
        rows_min, rows_max = np.min(indices[0]), np.max(indices[0])
        cols_min, cols_max = np.min(indices[1]), np.max(indices[1])
        rep_band_window = rep_band[rows_min:rows_max + 1, cols_min:cols_max + 1]
        row_window = waterfall_columns[rows_min:rows_max + 1, cols_min:cols_max + 1]
        mx_rep_band_col = ma.masked_array(rep_band_window, row_window == cols_unique[k])
        axd["bottom_right"].set_title(f"col = {cols_unique[k]}")
        im4.set_data(mx_rep_band_col)

        fig.canvas.draw_idle()

    update(c, k)  # Display initial plot
    fig.canvas.mpl_connect('key_press_event', on_key)

    plt.show()
    print("ok")

if __name__ == "__main__":
    main()