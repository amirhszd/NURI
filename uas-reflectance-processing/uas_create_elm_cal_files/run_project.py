"""
    @author: Eon Rehman
"""

from scipy.stats import linregress
import csv
import numpy as np
import json
from uas_create_elm_cal_files.perform_elm import get_panel_raw_data
from uas_create_elm_cal_files.perform_elm import get_svc_data
import pdb

def create_elm(filename):
    
    """
    Get bias/gain to convert dn to reflectance
    
    @param filenme:The .json file with the inputs
    @return save gain/bias to convert dn to reflectance
    """
    
    # ---------------------------------------------- Read in the Parameter Files ------------------------------------------#
    
    with open(filename, 'r') as openfile:
        # Reading from json file into dictionary
        variables = json.load(openfile)
        
    num_panels = variables['num_panels']
    black_panel = variables['black_panel']
    light_gray_panel = variables['light_gray_panel']
    dark_gray_panel = variables['dark_gray_panel']
    image_file = variables['image_file']
    image_hdr_file = variables['image_hdr_file']
    output_csv_cal_file = 'uas_create_elm_cal_files/cal_files/' + filename.rsplit('.')[0] + '.csv'
    
    # ---------------------------------------------- Read in the Parameter Files ------------------------------------------#
    
    if num_panels == 3: #If the image has 3 panels 
    # ---------------------------------------------- Read in Image and Get Raw-DN from Panels -----------------------------#
        wavelength_uas,black_panel_raw_dn_data = get_panel_raw_data(image_hdr_file,image_file,'Black Panel')
        _,light_gray_panel_raw_dn_data = get_panel_raw_data(image_hdr_file,image_file,'Light Gray Panel')
        _,dark_gray_panel_raw_dn_data = get_panel_raw_data(image_hdr_file,image_file,'Dark Gray Panel')
    # ---------------------------------------------- Read in Image and Get Raw-DN from Panels -----------------------------#
    
    
    # ---------------------------------------------- Read in SVC and Get Reflectance from Panels --------------------------#
        black_panel_refl = get_svc_data(black_panel,wavelength_uas)
        light_gray_panel_refl = get_svc_data(light_gray_panel,wavelength_uas)
        dark_gray_panel_refl = get_svc_data(dark_gray_panel,wavelength_uas)
    # ---------------------------------------------- Read in SVC and Get Reflectance from Panels --------------------------#
    
    
    # ---------------------------------------------- Create the slope and intercept files for the ELM ---------------------#
        svc_refl_data = np.stack([black_panel_refl,dark_gray_panel_refl,light_gray_panel_refl],axis=1)
        uas_raw_data = np.stack([black_panel_raw_dn_data,dark_gray_panel_raw_dn_data,light_gray_panel_raw_dn_data],axis=1)
    
        slope = np.zeros(len(black_panel_refl))
        intercept = np.zeros(len(black_panel_refl))
    
        for i in range(0,len(black_panel_refl)):
            slope[i], intercept[i], _, _, _ = linregress(uas_raw_data[i,:],svc_refl_data[i,:])
    # ---------------------------------------------- Create the slope and intercept files for the ELM ---------------------#
    
    elif num_panels == 2:  #If the image has 2 panels 
    # ---------------------------------------------- Read in Image and Get Raw-DN from Panels -----------------------------#
        wavelength_uas, black_panel_raw_dn_data = get_panel_raw_data(image_hdr_file, image_file, 'Black Panel')
        _, light_gray_panel_raw_dn_data = get_panel_raw_data(image_hdr_file, image_file, 'Light Gray Panel')
    # ---------------------------------------------- Read in Image and Get Raw-DN from Panels -----------------------------#
    
    
    # ---------------------------------------------- Read in SVC and Get Reflectance from Panels --------------------------#
        black_panel_refl = get_svc_data(black_panel, wavelength_uas)
        light_gray_panel_refl = get_svc_data(light_gray_panel, wavelength_uas)
    # ---------------------------------------------- Read in SVC and Get Reflectance from Panels --------------------------#
    
    
    # ---------------------------------------------- Create the slope and intercept files for the ELM ---------------------#
        svc_refl_data = np.stack([black_panel_refl, light_gray_panel_refl], axis=1)
        uas_raw_data = np.stack([black_panel_raw_dn_data, light_gray_panel_raw_dn_data],axis=1)
    
        slope = np.zeros(len(black_panel_refl))
        intercept = np.zeros(len(black_panel_refl))
    
        for i in range(0, len(black_panel_refl)):
            slope[i], intercept[i], _, _, _ = linregress(uas_raw_data[i, :], svc_refl_data[i, :])
    # ---------------------------------------------- Create the slope and intercept files for the ELM ---------------------#
    
    # ---------------------------------------------- Save the slope and intercept files for the ELM -----------------------#
    flname = open(output_csv_cal_file, 'wt')
    try:
        writer = csv.writer(flname)
        writer.writerow(('Wavelength', 'slope', 'intercept'))
        for i in range(len(black_panel_refl)):
            writer.writerow((wavelength_uas[i],slope[i],intercept[i]))
    finally:
        flname.close()
    # ---------------------------------------------- Save the slope and intercept files for the ELM -----------------------#
    
    print("")
    print('Finished ELM to convert DN to Refl')
    print('Gain/Bias to convert image file saved')
    print('------------------------------------------------------------------')
    
    return 0
