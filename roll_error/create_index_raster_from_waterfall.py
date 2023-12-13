from spectral.io import envi
import numpy as np
import os

def load_image_envi(waterfall_path):
    vnir_ds = envi.open(waterfall_path)
    vnir_profile = vnir_ds.metadata
    vnir_wavelengths = vnir_profile["wavelength"]
    vnir_wavelengths = np.array([float(i) for i in vnir_wavelengths])
    vnir_arr = vnir_ds.load()

    return vnir_arr, vnir_profile, vnir_wavelengths

def save_image_envi(new_arr, new_path, old_profile):
    # replicating vnir metadata except the bands and wavelength
    metadata = {}
    for k, v in old_profile.items():
        if (k != "bands") or (k != "wavelength"):
            metadata[k] = old_profile[k]
    metadata["bands"] = str(int(old_profile["bands"]) + 2)
    old_profile["wavelength"].extend(["2510.24123456", "2512.24123456"])
    metadata["wavelength"] = old_profile["wavelength"]

    # # channels are last for envi, first in rasterio
    envi.save_image(new_path, new_arr, metadata=metadata, force=True,
                    interleave= old_profile["interleave"],
                    dtype = np.uint16)

    print("image saved to: " + new_path)


def main(waterfall_hdr, output_path):
    vnir_arr, vnir_profile, vnir_wavelengths = load_image_envi(waterfall_hdr)

    # creating a mesh grid
    rows_vector = np.arange(1, vnir_arr.shape[0] + 1)
    columns_vector = np.arange(1, vnir_arr.shape[1] + 1)
    rows, columns = np.meshgrid(rows_vector, columns_vector, indexing = "ij")
    rows = rows[...,None]
    columns = columns[..., None]

    rows_cols_ind_raster = np.concatenate([vnir_arr, rows, columns], 2)

    save_image_envi(rows_cols_ind_raster, output_path, vnir_profile)


if __name__ == "__main__":
    waterfall_hdr = "/dirs/data/tirs/axhcis/Projects/NURI/Data/20210723_tait_labsphere_processed/1133/swir/raw_1504_nuc.hdr"
    output_path = "/dirs/data/tirs/axhcis/Projects/NURI/Data/20210723_tait_labsphere_processed/1133/swir/raw_1504_plusindices3.hdr"
    main(waterfall_hdr, output_path)