
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import os
import rasterio
from scipy.interpolate import interp1d
import os


class weighted_average():

    def __init__(self, vnir_path, swir_path, output_path):
        self.run(vnir_path, swir_path, output_path)

    def load_images(self, vnir_path,
                    swir_path):
        with rasterio.open(vnir_path) as src:
            vnir_arr = src.read()
            vnir_profile = src.profile
            vnir_wavelengths = np.array([float(i.split(" ")[0]) for i in src.descriptions])
        with rasterio.open(swir_path) as src:
            swir_arr = src.read()
            swir_profile = src.profile
            swir_wavelengths = np.array([float(i.split(" ")[0]) for i in src.descriptions])

        return (vnir_arr, vnir_profile, vnir_wavelengths), (swir_arr, swir_profile, swir_wavelengths)


    def interpolate_swir_to_vnir(self, swir_arr,
                                 swir_wavelengths,
                                 vnir_wavelengths):

        swir_indices = np.where((swir_wavelengths < 1000))
        vnir_indices = np.where((vnir_wavelengths > 900) & (vnir_wavelengths < 1000))

        swir_overlap_bands = swir_wavelengths[swir_indices]
        vnir_overlap_bands = vnir_wavelengths[vnir_indices]

        # upsample the swir to match vnir
        swir_overlap_data = np.transpose(swir_arr[swir_indices], [1,2,0])
        f_swir = interp1d(swir_overlap_bands,
                          swir_overlap_data.reshape(-1, swir_overlap_data.shape[-1]),
                          fill_value = "extrapolate") #x, y

        swir_overlap_vnir_res = f_swir(vnir_overlap_bands)

        swir_overlap_vnir_res =np.transpose(swir_overlap_vnir_res, [1, 0]).reshape(len(vnir_indices[0]),
                                                                             swir_arr.shape[1],
                                                                             swir_arr.shape[2])

        return swir_overlap_vnir_res, vnir_overlap_bands

    def write_to_file(self, master_path,
                      master_profile,
                      raster,
                      raster_wavelengths,
                      name="OverlapInterp"):

        output_path = os.path.basename(master_path.replace(".tif", f"_{name}.tif"))
        master_profile.update(count=len(raster_wavelengths))
        with rasterio.open(output_path, 'w', **master_profile) as dst:
            for i, band in enumerate(raster):
                dst.write_band(i + 1, band)
                dst.set_band_description(i + 1, str(raster_wavelengths[i]) + " nm")

        return output_path


    def band_average_overlap(self, swir_overlap_interp,
                             swir_overlap_interp_wavelengths,
                             vnir_arr,
                             vnir_wavelengths,
                             weights=(2,1)):

        # getting the overlap indices for vnir and swir
        vnir_overlap_indicies = [c for c, i in enumerate(vnir_wavelengths) if i in swir_overlap_interp_wavelengths]

        # extracting the overlap data from vnir
        vnir_overlap_data = vnir_arr[vnir_overlap_indicies]

        # performing averaging of the overlap region
        print(f"Performing band averaging with weights {weights} SWIR:VNIR.")
        average_overlap_data = np.average([swir_overlap_interp,
                                           vnir_overlap_data],
                                           axis = 0,
                                           weights = weights)

        # setting the default vnir and swir values for places we dont have swir or vnir
        average_overlap_data[swir_overlap_interp == 0] = vnir_overlap_data[swir_overlap_interp == 0]
        average_overlap_data[vnir_overlap_data == 0] = swir_overlap_interp[vnir_overlap_data == 0]

        return average_overlap_data, swir_overlap_interp_wavelengths

    def fuse_data(self, vnir_arr,
                  vnir_wavelengths,
                  average_overlap_data,
                  average_overlap_data_wavelengths,
                  swir_arr,
                  swir_wavelengths):


        # the final data is going to be vnir_w/o_overlap + overlap + swir_w/o_overlap
        vnir_non_overlap_indices = [c for c, i in enumerate(vnir_wavelengths) if i not in average_overlap_data_wavelengths]
        vnir_non_overlap_wavelength = vnir_wavelengths[vnir_non_overlap_indices]
        vnir_non_overlap = vnir_arr[vnir_non_overlap_indices]
        swir_non_overlap_indices = swir_wavelengths > 1000
        swir_non_overlap_wavelength = swir_wavelengths[swir_non_overlap_indices]
        swir_non_overlap = swir_arr[swir_non_overlap_indices]

        # concatenating the three together
        vnir_swir_fused_data = np.concatenate([vnir_non_overlap,
                                               average_overlap_data,
                                               swir_non_overlap], 0)
        vnir_swir_fused_wavelengths = np.concatenate([vnir_non_overlap_wavelength,
                                                      average_overlap_data_wavelengths,
                                                      swir_non_overlap_wavelength], 0)

        return vnir_swir_fused_data, vnir_swir_fused_wavelengths



    def run(self, vnir_path,
             swir_path,
             output_path):

        # load images
        (vnir_arr, vnir_profile, vnir_wavelengths), \
            (swir_arr, swir_profile, swir_wavelengths) = self.load_images(vnir_path, swir_path)


        # band average
        # swir_arr_interp is the array witht the overlap part interpolated to VNIR
        swir_overlap_interp, swir_overlap_interp_wavelengths = self.interpolate_swir_to_vnir(swir_arr, swir_wavelengths, vnir_wavelengths)


        average_overlap_data, average_overlap_data_wavelengths = self.band_average_overlap(swir_overlap_interp,
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
    output_path = "/Volumes/Work/Projects/NURI/NURI/spectral_matching/weighted_average/fused_data.tif"
    averager = weighted_average(vnir_path,swir_path, output_path)
