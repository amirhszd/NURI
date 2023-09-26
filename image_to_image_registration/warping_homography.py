
import spectral.io.envi as envi
import cv2
import tifffile
from PIL import Image
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import skimage

def get_rgb_image(image_ds):
    # grab blue, green and red band
    blue = np.ma.array(image_ds.read(109),mask=image_ds.read(109)==0)
    blue_lin = (blue - float(np.min(blue))) / (float(np.max(blue)) - float(np.min(blue)))
    green = np.ma.array(image_ds.read(69), mask=image_ds.read(69) == 0)
    green_lin = (green - float(np.min(green))) / (float(np.max(green)) - float(np.min(green)))
    red = np.ma.array(image_ds.read(29), mask=image_ds.read(29) == 0)
    red_lin = (red - float(np.min(red))) / (float(np.max(red)) - float(np.min(red)))

    rgb_image = np.stack([blue_lin,
                             green_lin,
                             red_lin], axis=2)

    # Contrast stretching
    p2 = np.percentile(rgb_image, 2)
    p98 = np.percentile(rgb_image, 98)
    rgb_img = skimage.exposure.rescale_intensity(rgb_image, in_range=(p2, p98))

    return rgb_img



def get_coord_pairs(nano_rgb, swir_rgb):
    global processing
    points_nano = []
    points_swir = []

    c = 0
    while True:
        fig1, ax1 = plt.subplots(1, 1, figsize=(8, 8))
        # plot the two image
        plt.suptitle('Click on a point in Nano and press "Enter", "Escape" to exit')
        ax1.imshow(nano_rgb)
        ax1.axis('off')
        ax1.set_title('Nano', fontsize=12)
        point_nano = plt.ginput(-1, timeout=-1)[-1]
        # get the last point
        ax1.scatter(point_nano[0], point_nano[1], marker="+", color= "red", s=32)
        ax1.figure.canvas.draw()

        fig2, ax2 = plt.subplots(1, 1, figsize=(8, 8))
        plt.suptitle('Click on a point in SWIR and press "Enter", "Escape" to exit')
        ax2.imshow(swir_rgb)
        ax2.axis('off')
        ax2.set_title('SWIR', fontsize=12)
        point_swir = plt.ginput(-1, timeout=-1)[-1]
        # get the last point
        ax2.scatter(point_swir[0], point_swir[1], marker="+", color= "red", s=32)
        ax2.figure.canvas.draw()
        points_nano.append(point_nano)  # [x,y]
        points_swir.append(point_swir)  # [x,y]
        plt.close("all")
        c += 1
        if c > 4 :
            response = input("Add more points? (y/n): ").lower()
            if response == "n":
                break


    return np.array(points_nano), np.array(points_swir)

def apply_warp(swir_ds,  points_nano, points_swir):

    swir_arr = swir_ds.read()

    # calculate homography
    homography, status = cv2.findHomography(points_swir, points_nano)

    # warp the array
    swir_arr_reshaped = swir_arr.reshape(swir_arr.shape[1], swir_arr.shape[2],-1)

    # do it channel by channel?


    swir_array_warped = skimage.transform.warp(swir_arr_reshaped,
                                               np.array(homography),
                                               output_shape=swir_arr_reshaped.shape)

    swir_array_warped = swir_array_warped.reshape(swir_arr.shape)

    # write out the raster
    with rasterio.open('temp.tif', 'w', **swir_ds.profile) as dst:
        dst.write(swir_array_warped)


def main(nano_fn, swir_fn):

    # load two images and grab the rgb image
    nano_ds = rasterio.open(nano_fn)
    nano_rgb = get_rgb_image(nano_ds)
    swir_ds = rasterio.open(swir_fn)
    swir_rgb = get_rgb_image(swir_ds)

    # getting pairs of points
    points_nano, points_swir = get_coord_pairs(nano_rgb, swir_rgb)

    # warping the swir image
    apply_warp(swir_ds, points_nano, points_swir)

    # write a piece of code that can pick up key pairs and save the coordinates

    # apply the key pairs from image1 and image2 and warp swir

    # save both out as a tiff, geolocation does not matter at this point.


if __name__ == "__main__":
    nano_fn = "/dirs/data/tirs/axhcis/Projects/NURI/Data/20210723_tait_labsphere_processed/1205/pair/raw_0_rd_rf_or_nano.img"
    swir_fn = "/dirs/data/tirs/axhcis/Projects/NURI/Data/20210723_tait_labsphere_processed/1205/pair/raw_1504_nuc_rd_or_refl_swir.img"
    main(nano_fn, swir_fn)