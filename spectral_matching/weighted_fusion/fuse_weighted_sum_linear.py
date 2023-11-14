import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import os
import rasterio
from scipy.interpolate import interp1d
import os
from fuse_weighted_average import weighted_average


class weighted_average_linear(weighted_average):


    def weighted_linear(self, datasets, weights):

        # performing element wise multiplication on each of the data based on weights
        for c, data in enumerate(datasets):
            for q, weight in enumerate(weights[c]):
                datasets[c][q,...] = np.multiply(weight, data[q,...])

        return np.sum(datasets, axis=0)


    def band_average_overlap_linear(self, swir_overlap_interp,
                             swir_overlap_interp_wavelengths,
                             vnir_arr,
                             vnir_wavelengths):

        # getting the overlap indices for vnir and swir
        vnir_overlap_indicies = [c for c, i in enumerate(vnir_wavelengths) if i in swir_overlap_interp_wavelengths]

        # extracting the overlap data from vnir
        vnir_overlap_data = vnir_arr[vnir_overlap_indicies]

        # obtaining linear weights in the overlap region
        # the left most vnir band is the most reliable one and the right most is the least reliable
        # the left most swir band is the least reliable one and the right most is the most reliable
        vnir_overlap_weights = np.linspace(0.99,0.01, len(swir_overlap_interp_wavelengths))
        swir_overlap_weights = vnir_overlap_weights[::-1]

        # performing averaging of the overlap region
        print(f"Performing linear weighted summation of swir and vnir.")
        average_overlap_data = self.weighted_linear([swir_overlap_interp,
                                                    vnir_overlap_data],
                                                    weights=[swir_overlap_weights,
                                                             vnir_overlap_weights])

        # setting the default vnir and swir values for places we dont have swir or vnir
        average_overlap_data[swir_overlap_interp == 0] = vnir_overlap_data[swir_overlap_interp == 0]
        average_overlap_data[vnir_overlap_data == 0] = swir_overlap_interp[vnir_overlap_data == 0]

        return average_overlap_data, swir_overlap_interp_wavelengths

    def run(self, vnir_path,
             swir_path,
             output_path):

        # load images
        (vnir_arr, vnir_profile, vnir_wavelengths), \
            (swir_arr, swir_profile, swir_wavelengths) = self.load_images(vnir_path, swir_path)


        # band average
        # swir_arr_interp is the array witht the overlap part interpolated to VNIR
        swir_overlap_interp, swir_overlap_interp_wavelengths = self.interpolate_swir_to_vnir(swir_arr, swir_wavelengths, vnir_wavelengths)


        average_overlap_data, average_overlap_data_wavelengths = self.band_average_overlap_linear(swir_overlap_interp,
                                                                                     swir_overlap_interp_wavelengths,
                                                                                     vnir_arr,
                                                                                     vnir_wavelengths)

        # fusing the three pieces together
        vnir_swir_fused_data, vnir_swir_fused_wavelengths = self.fuse_data(vnir_arr,
                  vnir_wavelengths,
                  average_overlap_data,
                  average_overlap_data_wavelengths,
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
    output_path = "/Volumes/Work/Projects/NURI/NURI/spectral_matching/weighted_average/fused_data_linear_vnir_denoised.tif"
    averager = weighted_average_linear(vnir_path,swir_path, output_path)

