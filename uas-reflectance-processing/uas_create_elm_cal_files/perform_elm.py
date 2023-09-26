"""
    @author: Eon Rehman
"""
import numpy as np
import spectral.io.envi as envi
import matplotlib.pyplot as plt
import scipy.interpolate
from skimage import exposure
from uas_create_elm_cal_files.envi_header import *

def get_svc_data(panel_path,wavelength_uas):
    
    """
    Get the reflectance data from the panel interpolated to the UAS-data
    
    @param panel_path: the path to the panel spectroscopy data
    @param wavelength_uas: the wavelength of the UAS-sensor
    @return refl_interp: the reflectance of the panel interpolated to the UAS wavelength
    """

    with open(panel_path) as f:
        rows = [rows.strip() for rows in f]

    head = rows.index('data=') + 1
    raw_data = rows[head:]
    data_to_list = [point.split() for point in raw_data]
    data_to_float = np.asarray(data_to_list).astype(float)
    wavelength = data_to_float[:,0]
    refl = data_to_float[:,3]/100

    interp = scipy.interpolate.PchipInterpolator(wavelength,refl)
    refl_interp = interp(wavelength_uas)

    return refl_interp

def Get_RGB_Image(image_file):
    
    """
    Get RGB image from the UAS data
    
    @param image_file: the hsi image from the UAS
    @return RGB_img: the rgb image from the UAS
    """

    blue = np.ma.array(image_file[:, :, 109],mask=image_file[:, :, 109]==0)
    blue_lin = (blue - float(np.min(blue))) / (float(np.max(blue)) - float(np.min(blue)))
    green = np.ma.array(image_file[:, :, 69], mask=image_file[:, :, 69] == 0)
    green_lin = (green - float(np.min(green))) / (float(np.max(green)) - float(np.min(green)))
    red = np.ma.array(image_file[:, :, 29], mask=image_file[:, :, 29] == 0)
    red_lin = (red - float(np.min(red))) / (float(np.max(red)) - float(np.min(red)))

    rgb_image = np.stack([blue_lin,
                         green_lin,
                         red_lin], axis=2)

    # Contrast stretching
    p2 = np.percentile(rgb_image, 2)
    p98 = np.percentile(rgb_image, 98)
    RGB_img = exposure.rescale_intensity(rgb_image, in_range=(p2, p98))

    return RGB_img


def get_panel_raw_data(hdr_path,file_path,which_roi):
    
    """
    @param hdr_path: the path to the header file 
    @param file_path: the path to the image file
    @param which_roi: the roi over the chosen panel
    @return wavelengths: the wavelength of the UAS sensor
    @return raw_dn_data_2D_avg: the raw dn over the chosen panel in the image
    """

    # Get wavelengths and convert to NumPy array
    in_header = find_hdr_file(hdr_path)
    header_data = read_hdr_file(in_header)
    wavelengths = header_data['wavelength'].split(',')[0:]
    wavelengths = [float(w) for w in wavelengths]
    wavelengths = numpy.array(wavelengths)

    img = envi.open(hdr_path,file_path)
    img = img.open_memmap()

    row = img.shape[0]
    column = img.shape[1]
    bands = img.shape[2]

    RGB_img = Get_RGB_Image(img)

    fig = plt.figure(figsize=(20, 10))
    plt.imshow(RGB_img)
    plt.axis('off')
    plt.title(which_roi + ': click on the two corners of the area to enlarge, and press "Enter"', fontsize=12)
    zoom = plt.ginput(2, timeout=-1)
    temps_zoom = RGB_img[int(zoom[0][1]):int(zoom[1][1]), int(zoom[0][0]):int(zoom[1][0]), :]
    zoom_ok = plt.waitforbuttonpress()
    plt.close()

    fig = plt.figure(figsize=(20, 10))
    plt.axis('off')
    plt.title('Choose ROI', fontsize=20)
    plt.imshow(temps_zoom)
    x = plt.ginput(2)
    plt.close()

    coords = [[x[0][0] + zoom[0][0],x[0][1] + zoom[0][1]],
                  [x[1][0] + zoom[0][0],x[1][1] + zoom[0][1]]]
    coords = np.round(coords)

    raw_dn_data = img[int(coords[0][1]):int(coords[1][1]), int(coords[0][0]):int(coords[1][0]), :]
    raw_dn_data_2D = np.reshape(raw_dn_data,(raw_dn_data.shape[0]*raw_dn_data.shape[1],raw_dn_data.shape[2]))
    raw_dn_data_2D_avg = np.mean(raw_dn_data_2D,axis=0)

    return wavelengths,raw_dn_data_2D_avg
