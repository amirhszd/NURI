import json
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
from image_utils import load_images_envi, save_crop_image_envi
from scipy.stats import pearsonr
import cmocean
import json
from tqdm import tqdm


# def get_highest_corr_coeff(swir_patch, mica_patch):
#
#     correllation_arr = []
#     mica_patch = np.mean(mica_patch[...,0:3], 2)
#     min_samples = 10000
#
#     # Loop through each band in SWIR and Mica
#     for i in tqdm(0, range(swir_patch.shape[2], 5), desc = "Finding highly correllated SWIR-Mica bands"):  # Assuming the number of bands is in the third dimension
#         # Calculate correlation coefficient using sampled data
#         sampled_swir = np.random.choice(swir_patch[...,i].flatten(), min_samples, replace=False)
#         sampled_mica = np.random.choice(mica_patch.flatten(), min_samples, replace=False)
#         correlation, _ = pearsonr(sampled_swir, sampled_mica)
#         correllation_arr.append(correlation)
#
#
#     highest_corr_ind = np.unravel_index(np.argmax(correllation_arr), correllation_arr.shape)
#     highest_corr = correllation_arr[highest_corr_ind]
#
#     return list(highest_corr_ind), highest_corr


def init_figs(mica_arr,
              swir_arr):

    # picking a band close to
    mica_image = mica_arr[..., 0:3]
    swir_image = swir_arr[..., np.array([10,40,90])]

    # Create a figure with two subplots in a single row
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # convert that to uint8 for cv2
    plt.suptitle("Use the right mouse button to pick points; at least 4. \n"
                 "Close the figure when finished.")
    to_uint8 = lambda x: ((x - x.min()) / (x.max() - x.min()) * 255).astype(np.uint8)
    mica_image_uint8 = to_uint8(mica_image)
    swir_image_uint8 = to_uint8(swir_image)

    # Display the VNIR image on the left subplot
    ax1.imshow(mica_image_uint8, cmap=cmocean.cm.thermal)
    ax1.set_title('Mica Image')

    # Display the SWIR image on the right subplot
    ax2.imshow(swir_image_uint8, cmap=cmocean.cm.thermal)
    ax2.set_title('SWIR Image')

    return fig, ax1, ax2, mica_image, mica_image_uint8, swir_image, swir_image_uint8


def calculate_homography(mica_arr, swir_arr):
    global not_satisfied

    not_satisfied = True
    while not_satisfied:
        fig, ax1, ax2, mica_image, mica_image_uint8, swir_image, swir_image_uint8 = init_figs(mica_arr,
                                                                                              swir_arr)
        mica_points = []
        swir_points = []

        def on_click_mica(event):
            if event.inaxes == ax1 and event.button == 3:  # Left mouse button clicked in VNIR subplot
                mica_points.append((event.xdata, event.ydata))
                ax1.plot(event.xdata, event.ydata, 'ro')  # Plot a red dot at the clicked point
                ax1.figure.canvas.draw_idle()

        def on_click_swir(event):
            if event.inaxes == ax2 and event.button == 3:  # Left mouse button clicked in SWIR subplot
                swir_points.append((event.xdata, event.ydata))
                ax2.plot(event.xdata, event.ydata, 'ro')  # Plot a red dot at the clicked point
                ax2.figure.canvas.draw_idle()

        # Connect the mouse click events to the respective axes
        fig.canvas.mpl_connect('button_press_event', on_click_mica)
        fig.canvas.mpl_connect('button_press_event', on_click_swir)
        plt.show()


        print(f"Found point are:\n VNIR: {mica_points}\n SWIR: {swir_points}")
        # calculate homorgraphy based on points found
        # point passed to homography should be x, y order
        mica_points = np.array(mica_points)
        swir_points = np.array(swir_points)
        M, mask = cv2.findHomography(swir_points, mica_points, cv2.RANSAC, 5)


        # show the result and see if the use is satisfied
        fig, ax = plt.subplots(1,1, figsize=(6,6))
        def on_key(event):
            global not_satisfied
            if event.key == 'escape':  # Close figure if Escape key is pressed
                not_satisfied = False;
                plt.close(fig)
        ax.imshow(mica_image_uint8, alpha=0.5)
        ax.imshow(cv2.warpPerspective(swir_image_uint8, M, (mica_image.shape[1], mica_image.shape[0])),
                  alpha=0.5)
        ax.set_title('Overlay of Coregistered Image \n'
                     'if satisfied press Escape to save image\n'
                     'if NOT satisfied close the figure to restart.')
        fig.canvas.mpl_connect('key_press_event', on_key)
        plt.show()

    return M

def main(mica_path, swir_path, use_available_homography = True):

    # homography file name
    homography_path = os.path.join(os.path.dirname(swir_path), "homography_matrix.npy")

    # load images envi
    (mica_arr, mica_profile, mica_wavelengths),\
        (swir_arr, swir_profile, swir_wavelengths) = load_images_envi(mica_path, swir_path)

    if not use_available_homography:
        M = calculate_homography(mica_arr, swir_arr)
        if os.path.exists(homography_path):
            print("Overwriting the homography file.")
            np.save(homography_path, M)

    else:
        # M = np.array([[1.05359863e+00, 1.88936633e-01, 2.75214709e+02],
        #               [-3.32346837e-02, 1.51736943e+00, 1.54957014e+03],
        #               [-4.29539345e-06, 1.89811047e-04, 1.00000000e+00]])
        M = np.load(homography_path)

    save_crop_image_envi(swir_arr, swir_wavelengths, swir_path, mica_arr, mica_profile, M)

if __name__ == "__main__":
    """
    IMPORORTANT: I AM PASSING FINDING THE HIGHLY CORRELLATED BAND OVER A SMALL AREA OF THE IMAGE, INSTEAD OF THE ENTIRE
    THING, BE USEFUL TO JUST DO IT IN ANOTHER FASHION?
    
    This code is now doing neares neighbour interpolation!
    
    """


    mica_path = "/Volumes/T7/axhcis/Projects/NURI/data/20210723_tait_labsphere/1133/Micasense/NURI_micasense_1133_transparent_mosaic_stacked_warped.hdr"
    swir_path = "/Volumes/T7/axhcis/Projects/NURI/data/20210723_tait_labsphere/1133/SWIR/raw_1504_nuc_or_plusindices3.hdr"
    main(mica_path,swir_path, True)

