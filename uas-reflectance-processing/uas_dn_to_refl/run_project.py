"""
    @author: Eon Rehman
"""
from uas_dn_to_refl.convert_raw_dn_to_reflectance import get_cal_data
from uas_dn_to_refl.convert_raw_dn_to_reflectance import perform_elm
import json
import pdb
from os.path import isfile, join
from os import listdir

def create_dn_to_refl(filename):
    
    """
    Convert raw dn to reflectance, all images from the specific UAS flight
    
    @param filenme:The .json file with the inputs
    @return save reflectance images 
    """
    
    # ---------------------------------------------- Read in the Parameter Files ------------------------------------------#
    
    with open(filename, 'r') as openfile:
        # Reading from json file into dictionary
        variables = json.load(openfile)
    
    
    cal_data = 'uas_create_elm_cal_files/cal_files/' + filename.rsplit('.')[0] + '.csv'
    image_path = variables['image_file'].split(variables['image_file'].split('/')[-1])[0]
    
    onlyfiles = [f for f in listdir(image_path) if isfile(join(image_path, f))]
    img_files = [s for s in onlyfiles if s.endswith('.hdr')]
    # ---------------------------------------------- Read in the Parameter Files ------------------------------------------#
    
    # ---------------------------------------------- Convert DN to REFL ------------------------------------------#
    for i in range(0,len(img_files)):
        
        print("")
        print('Processing Image # = ',str(i+1), ' of ', str(len(img_files)))
        print('------------------------------------------------------------------')
        #image_file = image_path + img_files[i].rsplit('.')[0] + '.img'
        image_file = image_path + img_files[i].rsplit('.')[0]
        image_hdr_file = image_path + img_files[i]
    # ---------------------------------------------- Read in the slope and intercept files for the ELM --------------------#
        slope_data,intercept_data = get_cal_data(cal_data)
    # ---------------------------------------------- Read in the slope and intercept files for the ELM --------------------#
    # ---------------------------------------------- Perform ELM and save the output reflectance image --------------------#
        perform_elm(slope_data,intercept_data,image_file,image_hdr_file)
    # ---------------------------------------------- Perform ELM and save the output reflectance image --------------------#
    
    # ---------------------------------------------- Convert DN to REFL ------------------------------------------#
    
    print("")
    print('----------- Processing Finished -----------')
    return 0
