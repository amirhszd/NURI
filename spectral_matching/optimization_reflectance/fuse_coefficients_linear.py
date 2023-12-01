import json
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(__file__))
from weighted_fusion.fuse_weighted_average import weighted_average
import rasterio
class fuse_coefficients(weighted_average):

    @staticmethod
    def get_coeffs(self):
        coeffs_dict = json.load("coeffs.json")
        return coeffs_dict

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
            fuse_overlap_interp_applied[c] = swir_overlap_interp[c]*coeffs_dict[wl][1] + coeffs_dict[wl][0]

        # setting the default vnir and swir values for places we dont have swir or vnir
        fuse_overlap_interp_applied[swir_overlap_interp == 0] = vnir_overlap_data[swir_overlap_interp == 0]
        fuse_overlap_interp_applied[vnir_overlap_data == 0] = swir_overlap_interp[vnir_overlap_data == 0]

        return fuse_overlap_interp_applied

    def run(self, vnir_path,
             swir_path,
             output_path):

        # load images
        (vnir_arr, vnir_profile, vnir_wavelengths), \
            (swir_arr, swir_profile, swir_wavelengths) = self.load_images(vnir_path, swir_path)


        # swir_arr_interp is the array witht the overlap part interpolated to VNIR
        swir_overlap_interp, swir_overlap_interp_wavelengths = self.interpolate_swir_to_vnir(swir_arr, swir_wavelengths, vnir_wavelengths)


        fuse_overlap_interp_applied = self.apply_coefficients(swir_overlap_interp, swir_overlap_interp_wavelengths, self.get_coeffs)

        # fusing the three pieces together
        vnir_swir_fused_data, vnir_swir_fused_wavelengths = self.fuse_data(vnir_arr,
                  vnir_wavelengths,
                  fuse_overlap_interp_applied,
                  swir_overlap_interp_wavelengths,
                  swir_arr,
                  swir_wavelengths)

        # save out the data
        vnir_profile.update(count=len(vnir_swir_fused_wavelengths))
        with rasterio.open(output_path, 'w', **vnir_profile) as dst:
            for i, band in enumerate(vnir_swir_fused_data):
                dst.write_band(i + 1, band)

        from gdal_set_band_description import set_band_descriptions
        bands = [int(i) for i in range(1, len(vnir_swir_fused_wavelengths) + 1)]
        names = vnir_swir_fused_wavelengths.astype(str)
        band_desciptions = zip(bands, names)
        set_band_descriptions(output_path, band_desciptions)
        print("Fused Image Saved to " + output_path)

if __name__ == "__main__":
    vnir_path = "/Volumes/T7/axhcis/Projects/NURI/data/uk_lab_data/cal_test/2023_10_12_10_56_30_VNIR/data.tif"
    swir_path = "/Volumes/T7/axhcis/Projects/NURI/data/uk_lab_data/cal_test/2023_10_12_11_15_28_SWIR/data_warped.tif"
    output_path = "/Volumes/Work/Projects/NURI/NURI/spectral_matching/weighted_average/fused_data_coefficients.tif"
    averager = fuse_coefficients(vnir_path,swir_path, output_path)

