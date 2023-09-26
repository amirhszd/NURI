"""
    @author: Eon Rehman
"""
import numpy as np
import spectral.io.envi as envi
import datetime
import csv
import os
from uas_dn_to_refl.envi_header import find_hdr_file,read_hdr_file,write_envi_header
import pdb

def get_cal_data(cal_data):
    
    """
    Read in the gain/bias data to convert raw dn to reflectance
    @param cal_data: path to the bias/gain data for elm
    @return slope_data
    @return intercept_data
    """

    rd = open(cal_data, 'rU')
    csv_reader = csv.reader(rd)
    data = list(csv_reader)
    data = np.asarray(data)

    wavelength = data[1:, 0]
    wavelength = wavelength.astype(np.float)
    wavelength = wavelength.flatten()

    slope = data[1:, 1]
    slope = slope.astype(np.float)
    slope_data = slope.flatten()

    intercept = data[1:, 2]
    intercept = intercept.astype(np.float)
    intercept_data = intercept.flatten()

    return slope_data,intercept_data

def perform_elm(slope_data,intercept_data,image_file,image_hdr_file):
    
    """
    Read in gain/bias and image file to convert raw dn to refl
    @param slope_data
    @param intercept_data
    @param image_file
    @param image_hdr_file
    @return the saved reflectance images 
    """

    # Get wavelengths and convert to NumPy array
    in_header = find_hdr_file(image_file)
    header_data = read_hdr_file(in_header)
    wavelengths = header_data['wavelength'].split(',')[0:]
    wavelengths = [float(w) for w in wavelengths]
    wavelengths = np.array(wavelengths)

    img = envi.open(image_hdr_file,image_file)
    img = img.open_memmap()
    img = np.ma.array(img,mask=img==0)

    row = img.shape[0]
    column = img.shape[1]
    bands = img.shape[2]

    print('Making the matrix for the slope data: {:%Y-%m-%d_%H:%M:%S}'.format(datetime.datetime.now()))
    slope_data_repmat = np.kron(np.ones((row,1,1)), slope_data)
    print('Making the matrix for the intercept data: {:%Y-%m-%d_%H:%M:%S}'.format(datetime.datetime.now()))
    int_data_repmat = np.kron(np.ones((row,1,1)), intercept_data)
    print('Starting the conversion from DN to Reflectance: {:%Y-%m-%d_%H:%M:%S}'.format(datetime.datetime.now()))
    refl_conv = img*slope_data_repmat + int_data_repmat
    print('Starting to save the data: {:%Y-%m-%d_%H:%M:%S}'.format(datetime.datetime.now()))
    refl_conv = np.ma.array(refl_conv,fill_value=0).filled()
    
    output_image_file = os.path.split(os.path.split(image_file)[0])[0] + '/processed_refl'
    #output_image_file = 'processed_refl'
    if not os.path.exists(output_image_file):
        os.makedirs(output_image_file)

    refl_file = output_image_file + '/' + image_hdr_file.rsplit('/')[-1].rsplit('.')[0] + '_refl.hdr'    
    envi.save_image(refl_file,refl_conv, force=True, dtype=np.float32)

    output_header_dict = read_hdr_file(find_hdr_file(image_hdr_file))
    output_header_dict['description'] = 'Hedwall Nano Refl by Eon R.'
    output_header_dict['lines'] = str(refl_conv.shape[0])
    output_header_dict['samples'] = str(refl_conv.shape[1])
    output_header_dict['data type'] = str(4)
    output_header_dict['interleave'] = 'bip'
    write_envi_header(refl_file,output_header_dict)
    
    return
