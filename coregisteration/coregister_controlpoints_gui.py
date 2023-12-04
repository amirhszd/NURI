import cv2
import numpy as np
import matplotlib.pyplot as plt
from image_utils import load_images_envi, save_image_envi

# def visualize_matches(image1, keypoints1, image2, keypoints2, matches):
#     # Draw matches on a new image
#     matched_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
#     plt.imshow(matched_image)
#     plt.show()



def init_figs(vnir_arr,
              vnir_wavelengths,
              swir_arr,
              swir_wavelengths):

    # picking a band close to
    vnir_pair_index = np.argmin(abs((vnir_wavelengths - 950)))
    swir_pair_index = np.argmin(abs((swir_wavelengths - 950)))

    # picking the at 950, and get an average of window size before and after to reduce noise
    window_size = 25;
    res_vnir = 1.6
    window_size_vnir = int((window_size / res_vnir) / 2)
    res_swir = 6
    window_size_swir = int((window_size / res_swir) / 2)
    vnir_image = np.mean(vnir_arr[vnir_pair_index - window_size_vnir:vnir_pair_index + window_size_vnir],
                         0)  # spectral res: 1.6 nm
    swir_image = np.mean(swir_arr[swir_pair_index - window_size_swir:swir_pair_index + window_size_swir],
                         0)  # spectral res: 6 nm

    # Create a figure with two subplots in a single row
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # convert that to uint8 for cv2
    plt.suptitle("Use the right mouse button to pick points; at least 4. \n"
                 "Close the figure when finished.")
    to_uint8 = lambda x: ((x - x.min()) / (x.max() - x.min()) * 255).astype(np.uint8)
    vnir_image_uint8 = to_uint8(vnir_image)
    swir_image_uint8 = np.fliplr(to_uint8(swir_image))

    # Display the VNIR image on the left subplot
    ax1.imshow(vnir_image_uint8)
    ax1.set_title('VNIR Image')

    # Display the SWIR image on the right subplot
    ax2.imshow(swir_image_uint8)
    ax2.set_title('SWIR Image')

    return fig, ax1, ax2, vnir_image, vnir_image_uint8, swir_image, swir_image_uint8


def main(vnir_path,swir_path):
    global not_satisfied

    # load images envi
    (vnir_arr, vnir_profile, vnir_wavelengths),\
        (swir_arr, swir_profile, swir_wavelengths) = load_images_envi(vnir_path, swir_path)


    not_satisfied = True
    while not_satisfied:
        fig, ax1, ax2, vnir_image, vnir_image_uint8, swir_image, swir_image_uint8 = init_figs(vnir_arr,
                  vnir_wavelengths,
                  swir_arr,
                  swir_wavelengths)

        vnir_points = []
        swir_points = []

        def on_click_vnir(event):
            if event.inaxes == ax1 and event.button == 3:  # Left mouse button clicked in VNIR subplot
                vnir_points.append((event.xdata, event.ydata))
                ax1.plot(event.xdata, event.ydata, 'ro')  # Plot a red dot at the clicked point
                ax1.figure.canvas.draw_idle()

        def on_click_swir(event):
            if event.inaxes == ax2 and event.button == 3:  # Left mouse button clicked in SWIR subplot
                swir_points.append((event.xdata, event.ydata))
                ax2.plot(event.xdata, event.ydata, 'ro')  # Plot a red dot at the clicked point
                ax2.figure.canvas.draw_idle()

        # Connect the mouse click events to the respective axes
        fig.canvas.mpl_connect('button_press_event', on_click_vnir)
        fig.canvas.mpl_connect('button_press_event', on_click_swir)
        plt.show()


        print(f"Found point are:\n VNIR: {vnir_points}\n SWIR: {swir_points}")
        # calculate homorgraphy based on points found
        # point passed to homography should be x, y order
        vnir_points = np.array(vnir_points)
        swir_points = np.array(swir_points)
        M, mask = cv2.findHomography(swir_points, vnir_points, cv2.RANSAC, 5)


        # show the result and see if the use is satisfied
        fig, ax = plt.subplots(1,1, figsize=(6,6))
        def on_key(event):
            global not_satisfied
            if event.key == 'escape':  # Close figure if Escape key is pressed
                not_satisfied = False;
                plt.close(fig)
        ax.imshow(cv2.warpPerspective(np.fliplr(swir_arr[0]), M, (vnir_image.shape[1], vnir_image.shape[0])),
                   alpha=0.5)
        ax.imshow(vnir_image, alpha=0.5)
        ax.set_title('Overlay of Coregistered Image \n'
                     'if satisfied press Escape to save image\n'
                     'if NOT satisfied close the figure to restart.')
        fig.canvas.mpl_connect('key_press_event', on_key)
        plt.show()

    # save image at last
    save_image_envi(swir_arr, swir_wavelengths, swir_path, vnir_arr, vnir_profile, M)

if __name__ == "__main__":
    # vnir_path = "/Volumes/T7/axhcis/Projects/NURI/data/uk_lab_data/cal_test/2023_10_12_10_56_30_VNIR/data.tif"
    # swir_path = "/Volumes/T7/axhcis/Projects/NURI/data/uk_lab_data/cal_test/2023_10_12_11_15_28_SWIR/data.tif"
    vnir_path = "/Users/amirhassanzadeh/Downloads/data_vnir.hdr"
    swir_path = "/Users/amirhassanzadeh/Downloads/data_swir.hdr"
    main(vnir_path,swir_path)

