import json
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(__file__))
from weighted_fusion.fuse_weighted_average import weighted_average
import rasterio
class fuse_coefficients_linear(weighted_average):

    def get_coeffs(self):
        with open('coeffs_linear.json', 'r') as file:
            # Load the JSON data
            data = json.load(file)
        return data

    def apply_coefficients(self, swir_overlap_interp,
                             swir_overlap_interp_wavelengths,
                             vnir_arr,
                             vnir_wavelengths,
                             coeffs_dict):

        # getting the overlap indices for vnir and swir
        vnir_overlap_indicies = [c for c, i in enumerate(vnir_wavelengths) if i in swir_overlap_interp_wavelengths]

        # extracting the overlap data from vnir
        vnir_overlap_data = vnir_arr[vnir_overlap_indicies]

        fuse_overlap_interp_applied = np.zeros_like(swir_overlap_interp)
        for c, wl in enumerate(swir_overlap_interp_wavelengths):
            fuse_overlap_interp_applied[c] = swir_overlap_interp[c]*coeffs_dict[str(wl)][1] + coeffs_dict[str(wl)][0]

        # setting the default vnir and swir values for places we dont have swir or vnir
        fuse_overlap_interp_applied[swir_overlap_interp == 0] = vnir_overlap_data[swir_overlap_interp == 0]
        fuse_overlap_interp_applied[vnir_overlap_data == 0] = swir_overlap_interp[vnir_overlap_data == 0]

        return fuse_overlap_interp_applied

    def run(self, vnir_path,
             swir_path, output_path):

        # load images
        (vnir_arr, vnir_profile, vnir_wavelengths), \
            (swir_arr, swir_profile, swir_wavelengths) = self.load_images_envi(vnir_path, swir_path)


        # swir_arr_interp is the array witht the overlap part interpolated to VNIR
        swir_overlap_interp, swir_overlap_interp_wavelengths = self.interpolate_swir_to_vnir(swir_arr, swir_wavelengths, vnir_wavelengths)


        fuse_overlap_interp_applied = self.apply_coefficients(swir_overlap_interp,
                                                              swir_overlap_interp_wavelengths,
                                                              vnir_arr,
                                                              vnir_wavelengths,
                                                              self.get_coeffs())

        # fusing the three pieces together
        vnir_swir_fused_data, vnir_swir_fused_wavelengths = self.fuse_data(vnir_arr,
                  vnir_wavelengths,
                  fuse_overlap_interp_applied,
                  swir_overlap_interp_wavelengths,
                  swir_arr,
                  swir_wavelengths)

        # save out the data
        self.save_image_envi(vnir_swir_fused_data, vnir_swir_fused_wavelengths, output_path, vnir_profile)

if __name__ == "__main__":
    vnir_path = "/Users/amirhassanzadeh/Downloads/data_vnir.hdr"
    swir_path = "/Users/amirhassanzadeh/Downloads/data_swir_warped.hdr"
    output_path = os.path.join("/Volumes/T7/axhcis", "vnir_swir_fused.hdr")
    averager = fuse_coefficients_linear(vnir_path,swir_path, output_path)

