import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.io import loadmat, savemat
from scipy.interpolate import interp1d
from plot_regress import plot_regress
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def load_data(vnir_mat_path, swir_mat_path):
    # loading data for the specific integration time we have
    vnir_data = {k: v for k, v in loadmat(vnir_mat_path).items() if not k.startswith("__")}
    swir_data = {k: v for k, v in loadmat(swir_mat_path).items() if not k.startswith("__")}

    return vnir_data, swir_data

def interpolate_swir_to_vnir(swir_arr,
                             swir_wavelengths,
                             vnir_wavelengths):

    swir_indices = np.where((swir_wavelengths < 1000))
    vnir_indices = np.where((vnir_wavelengths > 900) & (vnir_wavelengths < 1000))

    swir_overlap_bands = swir_wavelengths[swir_indices]
    vnir_overlap_bands = vnir_wavelengths[vnir_indices]

    # upsample the swir to match vnir
    swir_overlap_data = np.transpose(swir_arr[swir_indices])
    f_swir = interp1d(swir_overlap_bands,
                      swir_overlap_data.reshape(-1, swir_overlap_data.shape[-1]),
                      fill_value = "extrapolate") #x, y

    swir_overlap_vnir_res = f_swir(vnir_overlap_bands)

    swir_overlap_vnir_res =np.transpose(swir_overlap_vnir_res)

    return swir_overlap_vnir_res, vnir_overlap_bands, vnir_indices


def calculate_coefficients(vnir_data, swir_data):

    y = vnir_data

    x = []
    # b0 - biases
    b0 = np.ones(swir_data.shape)
    x.append(b0)

    # b1
    x.append(swir_data)

    x = np.array(x).T

    coeff, residuals, rank, s = np.linalg.lstsq(x, y, rcond=None)

    x = x.T
    y_pred = coeff[0] + coeff[1] * x[1, :]

    mae = mean_absolute_error(y_pred, y)
    rmse = mean_squared_error(y_pred, y, squared=False)
    r2 = r2_score(y_pred, y)
    std = np.std(y_pred - y)

    print(coeff)
    print("MAE:", mae)
    print("RMSE:", rmse)
    print("R2:", r2)
    print("std:", std)

    return coeff, mae, rmse, r2, std


def main(vnir_mat_path, swir_mat_path):

    # load the data
    vnir_data, swir_data = load_data(vnir_mat_path, swir_mat_path)

    # intepolating swir to vnir
    swir_data_interp = {}
    for k,v in swir_data.items():
        if k.startswith("p"):
            a,  swir_data_interp_wls, vnir_overlap_indices = interpolate_swir_to_vnir(v.T,
                                                                swir_data["wls"].squeeze(),
                                                                vnir_data["wls"].squeeze())
            swir_data_interp[k] = np.mean(a.T,0)
    swir_data_interp["wl"] = swir_data_interp_wls

    # grab the overlapping region in the vnir
    for k,v in vnir_data.items():
            vnir_data[k] = np.mean(v.T[vnir_overlap_indices].T, 0)

    # calculate coefficients
    vnir_data_array = []
    keys = sorted(list(vnir_data.keys()))
    for key in keys:
        if key.startswith("p"):
            vnir_data_array.append(vnir_data[key][None])
    vnir_data_array = np.concatenate(vnir_data_array, 0)

    swir_data_interp_array = []
    keys = sorted(list(swir_data_interp.keys()))
    for key in keys:
        if key.startswith("p"):
            swir_data_interp_array.append(swir_data_interp[key][None])
    swir_data_interp_array = np.concatenate(swir_data_interp_array, 0)


    coeffs_per_band = np.zeros((62,2))
    mae_per_band = np.zeros((62,))
    rmse_per_band = np.zeros((62,))
    r2_per_band = np.zeros((62,))
    std_per_band = np.zeros((62,))

    for band_n in range(swir_data_interp_array.shape[1]):
        coeff, mae, rmse, r2, std = calculate_coefficients(vnir_data_array[:, band_n], swir_data_interp_array[:, band_n])
        coeffs_per_band[band_n, :] = coeff
        mae_per_band[band_n] = mae
        rmse_per_band[band_n] = rmse
        r2_per_band[band_n] = r2
        std_per_band[band_n] = std

    fig, ax = plt.subplots(2,2, figsize=(10,10))
    ax = ax.flatten()
    ax[0].plot(swir_data_interp["wl"], mae_per_band)
    ax[0].set_xlabel("Wavelength (nm)")
    ax[0].set_ylabel("MAE")

    ax[1].plot(swir_data_interp["wl"], r2_per_band)
    ax[1].set_xlabel("Wavelength (nm)")
    ax[1].set_ylabel("R2")

    ax[2].plot(swir_data_interp["wl"], coeffs_per_band[:,0])
    ax[2].set_xlabel("Wavelength (nm)")
    ax[2].set_ylabel("Slope")

    ax[3].plot(swir_data_interp["wl"], coeffs_per_band[:,1])
    ax[3].set_xlabel("Wavelength (nm)")
    ax[3].set_ylabel("Intercept")

    plt.savefig("optimization_results.pdf")

    saving_dict = {}
    for c, wl in enumerate(swir_data_interp["wl"]):
        saving_dict[wl] = list(coeffs_per_band[c])

    import json
    with open("coeffs.json", "w") as file:
        json.dump(saving_dict, file)





if __name__ == "__main__":
    vnir_mat_path = "/Volumes/T7/axhcis/Projects/NURI/data/" \
                                          "uk_lab_data/cal_test/" \
                                          "2023_10_12_10_56_30_VNIR/data_panels.mat"
    swir_mat_path = "/Volumes/T7/axhcis/Projects/NURI/data/" \
                                          "uk_lab_data/cal_test/" \
                                          "2023_10_12_11_15_28_SWIR/data_panels.mat"
    main(vnir_mat_path, swir_mat_path)




