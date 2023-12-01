import cv2
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from image_utils import load_images_envi, save_image_envi
from matplotlib.widgets import Slider, Button
import sys

# def visualize_matches(image1, keypoints1, image2, keypoints2, matches):
#     # Draw matches on a new image
#     matched_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
#     plt.imshow(matched_image)
#     plt.show()

def init_figs():
    # Add sliders
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, hspace=0.3, wspace=0.4)

    descriptor_threshold_slider_ax = fig.add_axes([0.3, 0.01, 0.4, 0.02])
    descriptor_threshold_slider = Slider(descriptor_threshold_slider_ax, 'Descriptor Threshold', 0.1, 0.99,
                                         valinit=0.75)

    ransac_threshold_slider_ax = fig.add_axes([0.3, 0.05, 0.4, 0.02])
    ransac_threshold_slider = Slider(ransac_threshold_slider_ax, 'RANSAC Threshold', 1, 5, valinit=5)

    window_size_slider_ax = fig.add_axes([0.3, 0.09, 0.4, 0.02])
    window_size_slider = Slider(window_size_slider_ax, 'Window Size (nm)', 6, 60, valinit=25)

    return fig, ax1, ax2, descriptor_threshold_slider, ransac_threshold_slider, window_size_slider

def main(vnir_path,swir_path):
    global M
    M = None

    # initialize the figuer
    fig, ax1, ax2, descriptor_threshold_slider, ransac_threshold_slider, window_size_slider = init_figs()

    # load images
    (vnir_arr, vnir_profile, vnir_wavelengths),\
        (swir_arr, swir_profile, swir_wavelengths) = load_images_envi(vnir_path, swir_path)

    def update(val):
        global M
        # Update your variables here based on slider values
        descriptor_threshold = descriptor_threshold_slider.val
        ransac_threshold = ransac_threshold_slider.val
        window_size = int(window_size_slider.val)

        # pick the band closest to 950 nm
        vnir_pair_index = np.argmin(abs((vnir_wavelengths - 950)))
        swir_pair_index = np.argmin(abs((swir_wavelengths - 950)))

        # picking the at 950, and get an average of window size before and after to reduce noise
        res_vnir = 1.6
        window_size_vnir = int((window_size / res_vnir) / 2)
        res_swir = 6
        window_size_swir = int((window_size / res_swir) / 2)
        vnir_image = np.mean(vnir_arr[vnir_pair_index - window_size_vnir:vnir_pair_index + window_size_vnir],
                             0)  # spectral res: 1.6 nm
        swir_image = np.mean(swir_arr[swir_pair_index - window_size_swir:swir_pair_index + window_size_swir],
                             0)  # spectral res: 6 nm

        # convert that to uint8 for cv2
        to_uint8 = lambda x: ((x - x.min()) / (x.max() - x.min()) * 255).astype(np.uint8)
        vnir_image_uint8 = to_uint8(vnir_image)
        swir_image_uint8 = np.fliplr(to_uint8(swir_image))

        # Initialize SIFT detector
        sift = cv2.SIFT_create()

        # Detect keypoints and compute descriptors
        keypoints1, descriptors1 = sift.detectAndCompute(vnir_image_uint8, None)
        keypoints2, descriptors2 = sift.detectAndCompute(swir_image_uint8, None)

        # Initialize FLANN matcher
        flann = cv2.FlannBasedMatcher(dict(algorithm=0, trees=5), {})

        # Match keypoints
        matches = flann.knnMatch(descriptors1, descriptors2, k=2)

        # Apply ratio test to filter good matches

        good_matches = []
        for m, n in matches:
            if m.distance < descriptor_threshold * n.distance:
                good_matches.append(m)

        # draw and visualize the matches
        ax1.clear()
        matched_image = cv2.drawMatches(vnir_image_uint8, keypoints1, swir_image_uint8, keypoints2, good_matches, None,
                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        ax1.imshow(matched_image)
        ax1.set_title('Matches from SIFT')

        # Get corresponding points
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Compute transformation matrix using RANSAC
        M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, ransac_threshold)

        ax2.clear()
        ax2.imshow(cv2.warpPerspective(np.fliplr(swir_arr[0]), M, (vnir_image.shape[1], vnir_image.shape[0])),
                   alpha=0.5)
        ax2.imshow(vnir_image, alpha=0.5)
        ax2.set_title('Overlay of Coregistered Image')
        fig.canvas.draw_idle()

        def on_key(event):
            global not_satisfied
            if event.key == 'escape':  # Close figure if Escape key is pressed
                not_satisfied = False;
                plt.close(fig)

        plt.suptitle('When satisfied press ESCAPE to save image.')
        fig.canvas.mpl_connect('key_press_event', on_key)


    # Attach the update function to sliders
    descriptor_threshold_slider.on_changed(update)
    ransac_threshold_slider.on_changed(update)
    window_size_slider.on_changed(update)

    # Initial visualization
    update(None)
    plt.show()

    save_image_envi(swir_arr, swir_wavelengths, swir_path, vnir_arr, vnir_profile, M)



if __name__ == "__main__":
    # vnir_path = "/Volumes/T7/axhcis/Projects/NURI/data/uk_lab_data/cal_test/2023_10_12_10_56_30_VNIR/data.tif"
    # swir_path = "/Volumes/T7/axhcis/Projects/NURI/data/uk_lab_data/cal_test/2023_10_12_11_15_28_SWIR/data.tif"
    vnir_path = "/Users/amirhassanzadeh/Downloads/data_vnir.hdr"
    swir_path = "/Users/amirhassanzadeh/Downloads/data_swir.hdr"
    main(vnir_path,swir_path)

