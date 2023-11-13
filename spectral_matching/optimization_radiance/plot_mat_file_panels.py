import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import os

def main(mat_path, outfolder):
    panel_matfile = loadmat(mat_path)
    wls = panel_matfile["wls"].squeeze()
    for c, (k, v) in enumerate(panel_matfile.items()):
        if not ((k == "wls") or (k.startswith("__"))):
            fig, ax = plt.subplots(1,2, figsize = (8,4))
            for count, signal in enumerate(v):
                ax[0].plot(wls, signal)
            ax[0].set_xlabel("Wavelengths (nm)")
            ax[0].set_ylabel("Reflectance")
            ax[0].set_title(k + "%" + f" N = {count + 1}")


            mean_spectra = np.mean(v,0)
            std_spectra = np.std(v,0)
            ax[1].plot(wls, mean_spectra, label = "Mean spectrum")
            ax[1].plot(wls, mean_spectra + std_spectra, label="Mean + Std spectrum")
            ax[1].plot(wls, mean_spectra - std_spectra, label="Mean - Std spectrum")
            ax[1].set_xlabel("Wavelengths (nm)")
            ax[1].set_ylabel("Reflectance")
            ax[1].set_title(k + "%")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(outfolder, k + ".png"), dpi = 200)

if __name__ == "__main__":

    mat_files = ["/Volumes/T7/axhcis/Projects/NURI/data/uk_lab_data/cal_test/2023_10_12_10_56_30_VNIR/data_panels.mat",
            "/Volumes/T7/axhcis/Projects/NURI/data/uk_lab_data/cal_test/2023_10_12_11_15_28_SWIR/data_panels.mat",
            "/Volumes/T7/axhcis/Projects/NURI/data/uk_lab_data/cal_test/2023_10_12_11_21_02_SWIR/data_panels.mat",
            "/Volumes/T7/axhcis/Projects/NURI/data/uk_lab_data/cal_test/2023_10_12_11_29_13_VNIR/data_panels.mat"]

    outfolders = ["/Volumes/Work/Projects/NURI/results/UK_lab/calibration_tests/panels_spectra/2023_10_12_10_56_30_VNIR",
                "/Volumes/Work/Projects/NURI/results/UK_lab/calibration_tests/panels_spectra/2023_10_12_11_15_28_SWIR",
                "/Volumes/Work/Projects/NURI/results/UK_lab/calibration_tests/panels_spectra/2023_10_12_11_21_02_SWIR",
                "/Volumes/Work/Projects/NURI/results/UK_lab/calibration_tests/panels_spectra/2023_10_12_11_29_13_VNIR"]

    for mat_file, outfolder in zip(mat_files, outfolders):
        main(mat_file, outfolder)
