#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 17:20:26 2022

@author: eoncis

"""

"""This is the primary code to process any imagery collected by the UAS-sensor 
by RIT. This code process the geo-rectified raw DN images from the drone to 
reflectance using ELM. The code can process images from both the VNIR and SWIR
sensor. 

Inputs for the code are specified in the .json file
    1. "num_panels": number of panels used to perform ELM (can support 2 or 3)
    2. "black_panel": path to spectrometer data of the "darkest" panel 
    (currently supports spectrometer data from SVC)
    3. "light_gray_panel": path to spectrometer data of the second panel 
    4. "dark_gray_panel": path to spectrometer data of the third panel (can 
    leave empty if there are only two panels used to perform ELM)
    5. "image_file": path to the image file, which has all the panels within 
    the scene
    6. "image_hdr_file": path to the header file for the image
    7. "output_csv_cal_file": path of the bias/gain saved to convert raw dn to 
    reflectance for the uas-based HSI images

This code will first calculate the bias/gain to relate DN and 
reflectance using ground reflectance measurements of the calirated panels. The 
gain/bias will then be applied to all the HSI imagery collected during the 
flight.

NOTE: This code is currently set-up to read reflectance data of the 
panels collected from the SVC spectrometer. Future things to change code, read
data from ASD as well. 
"""
from uas_create_elm_cal_files.run_project import create_elm
from uas_dn_to_refl.run_project import create_dn_to_refl

filename = input("input json file: ") #The input .json file

create_elm(filename) #creates the bias/gain for elm
create_dn_to_refl(filename) #applies bias/gain to convert dn to reflectance to all HSI images
