# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 17:40:15 2020

@author: Amirh
"""
import multiprocessing

import matplotlib.pyplot as plt
import numpy as np
import pickle
import tqdm
from sklearn.decomposition import PCA
import dtcwt
import os
import copy
from scipy.stats import median_absolute_deviation
from scipy.stats import norm
import rpy2
from rpy2.robjects.packages import importr
import array
from scipy.io import loadmat, savemat
import rpy2.robjects as ro
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
import sklearn.preprocessing
from multiprocessing import Process
from functools import partial



def array_splitter_rec(array, size, chunks):
#     the base case
    if (len(array) < size) and (len(array != 0)):
        chunks.append(array)
        return chunks
    elif len(array) == 0:
        return chunks

    chunks.append(array[:size]) # first
    array_splitter_rec(array[size:], size, chunks) # rest
    return chunks


def denoise_dtcwt_chunks(spectrums):
    result = []
    for spectrum in spectrums:
        result.append(dtcwtDenoise._denoise_dtcwt(spectrum))

    return result

def denoise_dtcwt_shared(spectrums_input, spectrum_output, start, end):
    for spectrum in spectrums_input[start, end]:
        spectrum_output[start, end] = dtcwtDenoise._denoise_dtcwt(spectrum)
    spectrum_output.shm.close()
    spectrums_input.shm.close()



class dtcwtDenoise():
    def __init__(self, window_size=3, nlevels=5):
        """
        denoise class
        Parameters:
            window_size: default is set to 3, have to be an odd number
            nlevels: number of levels in denoising complex waveform, 5 is chosen
                as a compromise between speed

        """
        self.window_size = window_size  # window_size of 3 is what is recommended in the paper
        self.nlevels = nlevels

    def denoise_warmup(self, x_pca):
        """
        Parameters:
            x = the data needed to be denoise (m,n) where m is number
            of samples and n is the number of bands

            x_pca = data used to determine the best number of principal components
            to retain
        """

        print("Using separate data for pca and variability assessment")
        self.x_pca = x_pca

        if self.x_pca.shape[0] < self.x_pca.shape[1]:
            # number of samples need to be more than number of features
            raise ValueError("Number of samples need to be more than number of features")

        while True:
            try:
                pca = PCA()
                _ = pca.fit_transform(sklearn.preprocessing.StandardScaler().fit_transform(self.x_pca))
                self.exp_var = pca.explained_variance_ratio_
                self.eigenvalues = pca.explained_variance_
                break
            except:
                pass

    def denoise(self, x, method):
        """
        Parameters:
            x = the data needed to be denoise (m,n) where m is number
            of samples and n is the number of bands

            method: method argument can be either string, float or int.
                if method is str:
                    using methods argument one can choose between two approaches in the literature
                    MAP: minimum average partial by Celicer et al.
                    Kaiser: kaiser's method which retains components with eigenvalue above 1

                if method is float:
                    a threshold value is used on the explained variability to retain components.
                    a threshold value of 0.90 will retail components that fall under cumulative
                    will explained variability of 0.9

                if method is int:
                    number of prinicpal components to retail is chose, a value of 5 keeps the first
                    five principal components and carries out denoising on rest.
        """
        # validating arguments
        self.method = method
        self.x = x
        if len(self.x.shape) < 2 or len(self.x_pca.shape) < 2:
            raise ValueError(
                "the input data should be 2 dimensiona (mxn) where m is number of samples and n is number of bands")

        if type(self.method) is str:
            self.method = self.method.lower()
            if not ((self.method == 'kaiser') or (self.method == 'map')):
                raise ValueError("Valid methods are either MAP or Kaiser")

        elif type(self.method) is int:
            print("Interger passed, using first k principal components to retain.")

        elif type(self.method) is float:
            if (self.method >= 1) or (self.method <= 0):
                raise ValueError("if using explaine variability trheshold, it should be between 0 and 1.")
            else:
                print("Float passed, using first k principal components to retain.")

                # determining number of components
        if self.method == "kaiser":
            k = self._KAISER()
            print("Retaining first {} components and performing denoising on rest".format(k))
            keep_indices = np.arange(k)
            denoise_indices = [i for i in range(len(self.exp_var)) if i not in keep_indices]

        elif self.method == "map":
            k = self._MAP()
            print("Retaining first {} components and performing denoising on rest".format(k))
            keep_indices = np.arange(k)
            denoise_indices = [i for i in range(len(self.exp_var)) if i not in keep_indices]

        elif type(self.method) is int:
            print("Retaining first {} components and performing denoising on rest".format(self.method))
            if self.method >= self.x.shape[1]:
                raise ValueError("number of K components should be smaller than number of features")

            keep_indices = np.arange(self.method)
            denoise_indices = [i for i in range(len(self.exp_var)) if i not in keep_indices]

        elif type(self.method) is float:
            print("Retaining components with cumulative explained variability below {}".format(self.method))

            keep_indices = np.where(np.cumsum(self.exp_var) < self.method)[0]
            k = np.max(keep_indices) + 1
            denoise_indices = [i for i in range(len(self.exp_var)) if i not in keep_indices]

        # if number of bands is not dividable by two, add one band to it because of the algorithm
        if len(denoise_indices) % 2 != 0:
            keep_indices = np.arange(keep_indices[-1] + 2)
            k = k + 1
            denoise_indices = [i for i in range(len(self.exp_var)) if i not in keep_indices]

        pca = PCA()
        x_pca_space = pca.fit_transform(self.x)

        bool_indices = np.zeros((len(self.exp_var)), dtype=bool)
        bool_indices[denoise_indices] = True

        below_thresh_bool = bool_indices
        above_thresh_bool = np.invert(bool_indices)

        # separating between pcs that need to be denoised and those that remain unchanged
        keep_pcs = x_pca_space[:, above_thresh_bool]
        denoise_pcs = x_pca_space[:, below_thresh_bool]

        # data_chunks = np.array_split(denoise_pcs, np.ceil(len(denoise_pcs) / 1e+3))
        #
        # # # Split the data into chunks
        # # data_chunks = [];
        # # div, mod = divmod(len(denoise_pcs), 5e+3)
        # # for i in range()
        # # data_chunks = array_splitter_rec(denoise_pcs, int(5e+3), [])
        #
        # # Create a ThreadPoolExecutor
        # print("Performing denoising...")
        # with Process(max_workers=4) as executor:
        #     # Use tqdm for progress bar
        #     results = list(tqdm.tqdm(executor.map(denoise_dtcwt_chunks, data_chunks), total=len(data_chunks)))
        # denoised_pcs = np.concatenate(results, axis=0)

        input_array, shm_input, output_array, shm_output = dtcwtDenoise.create_shared_memory(denoise_pcs)
        starts = np.arange(0, len(denoise_pcs),
                            int(len(denoise_pcs)/multiprocessing.cpu_count())).tolist()
        ends = np.arange(0, len(denoise_pcs),
                            int(len(denoise_pcs)/multiprocessing.cpu_count())).tolist()
        ends.pop(0); ends.append(len(denoise_pcs) - 1)

        processes = []
        for i in range(multiprocessing.cpu_count()):
            _process = Process(target=denoise_dtcwt_chunks, args=(shm_input.name, shm_output.name, starts[i],ends[i]))
            processes.append(_process)
            _process.start()

        for _process in processes:
            _process.join()

        shared_mem.unlink()

        # pbar = tqdm.tqdm(total=self.x.shape[0])
        # for denoise_pc in denoise_pcs:
        #     denoised_pcs.append(np.expand_dims(self._denoise_dtcwt(denoise_pc), 0))
        #     pbar.update(1)
        # denoised_pcs = np.concatenate(denoised_pcs, axis=0)

        x_pca_space_denoised = np.zeros(self.x.shape)
        x_pca_space_denoised[:, above_thresh_bool] = keep_pcs
        x_pca_space_denoised[:, below_thresh_bool] = denoised_pcs

        self.denoised_x = pca.inverse_transform(x_pca_space_denoised)

        return self.denoised_x
    @staticmethod
    def create_shared_memory(array):
        from multiprocessing.shared_memory import SharedMemory
        shm_input = SharedMemory(create=True, size=array.nbytes)
        input_array = np.ndarray(array.shape, dtype=array.dtype, buffer=shm_input.buf)
        input_array[:] = array[:]

        shm_output = SharedMemory(create=True, size=array.nbytes)
        output_array = np.ndarray(array.shape, dtype=array.dtype, buffer=shm_output.buf)
        output_array[:] = np.zeros_like(array)

        return input_array, shm_input, output_array, shm_output



    @staticmethod
    def _denoise_dtcwt(spectrum):

        w = dtcwt.Transform1d()
        eps = 1e-7  # adding an epsilon for numerical stability

        # forward pass on the transform with the number of levels selected
        coeffs = w.forward(spectrum, nlevels=5, include_scale=True)

        # grabbing the coefficients
        highpass_coeffs = coeffs.highpasses
        highpass_coeffs_new = copy.copy(highpass_coeffs)

        # calculating the parameters
        # noise_est is median absolute devian of coefficinet at finest level
        # based on Donoho et al. this is just an estimate.
        noise_est = median_absolute_deviation(highpass_coeffs[0]) / 0.6745  # noise std
        n = len(spectrum)  # signal length
        univ_thresh = noise_est * np.sqrt(2 * np.log(n))  # universal threshold

        window_size = 3

        for c in range(len(highpass_coeffs)):
            coeff = highpass_coeffs[c].squeeze()

            coeff_mat = np.array([coeff, ] * coeff.shape[0]).transpose().conj()
            diag_coeff_mat = np.diag(np.diag(coeff_mat))  # rows and columns
            # shift matrix up and down to get the window effect
            x = diag_coeff_mat
            bool_mat = np.invert(x == 0)
            for i in range(int(window_size / 2)):
                bool_dwn = np.tril(np.invert(np.roll(x, i + 1, axis=0) == 0))  # shift down
                bool_up = np.triu(np.invert(np.roll(x, i + 1, axis=-1) == 0))  # shift up

                bool_mat = np.logical_or(np.logical_or(bool_mat, bool_up), bool_dwn)

            # applying the mask on the array
            coeff_mat[np.invert(bool_mat)] = 0
            # average matrix
            averagor = np.ones(coeff.shape[0]) / np.sum(bool_mat, 1)
            # applying the average matrix and grabbing s2
            # the imaginary part after matrix multiplication is very close to zero
            s2 = averagor * (np.real(np.matmul(coeff.transpose(), coeff_mat)))

            # calculating the threshold coefficient that will be multiplied by high pass coeffs
            mult = np.maximum((1 - ((univ_thresh ** 2) / (s2 + eps))), np.zeros(s2.shape[0]))

            highpass_coeffs_new[c][:] = np.expand_dims(np.multiply(coeff, mult), 1)

        coeffs.highpasses = highpass_coeffs_new
        # transofrmign signal back
        spectrum_rec = w.inverse(coeffs)
        return spectrum_rec

    def snr(self, x):
        # calculating snr in a data set of samples and features
        gaus = np.zeros((2, x.shape[-1]))
        # calculate mean and std of each band across all samples
        for band in range(x.shape[-1]):
            mu = np.mean(x[:, band])
            sigma = np.std(x[:, band])
            gaus[:, band] = np.array([mu, sigma])

        mean = gaus[0, :]
        noise = gaus[1, :]
        SNR = (mean) / noise  # signal to noise is simply mean over std
        weights = SNR / sum(SNR)
        SNR_all = np.average(SNR, weights=weights)
        return SNR, SNR_all

    def _MAP(self):
        utils = importr('utils')
        efa = importr("EFA.dimensions")

        nr, nc = self.x_pca.shape
        x_r = ro.r.matrix(self.x_pca, nrow=nr, ncol=nc)

        results = efa.MAP(x_r, 'pearson', verbose=False)
        k = results[-2][0]

        return int(k)

    def _KAISER(self):
        indices = np.where(self.eigenvalues > 1.0)
        k = np.max(indices[0]) + 1

        return k

if __name__ == "__main__":
    import spectral.io.envi as envi
    import shutil
    hdr_filename = "/Volumes/Work/Projects/NURI/DATA/uk_lab_data/VIS-NIR_cube/data.hdr"
    # denoised_filename = hdr_filename.replace("data.hdr", "data_denoised.hdr")
    # shutil.copy(hdr_filename, denoised_filename)

    img  = envi.open(hdr_filename)
    arr = img.load()
    x = arr.reshape(-1, img.shape[-1])[:1000]

    denoiser = dtcwtDenoise()
    denoiser.denoise_warmup(x)
    x_denoised = denoiser.denoise(x, "kaiser")
    snr_x = denoiser.snr(x)
    snr_x_denoised = denoiser.snr(x_denoised)

    envi.save_image(hdr_filename.replace("data.hdr", "data_denoised.hdr"), x_denoised, metadata = img.metadata, force = True)



# %%
# plot_names = ["L1_1","L2_1","F1_1","F2_1","W1_1","W2_1",
#               "F2_2","W1_2","L1_2","W2_2","L2_2","F1_2",
#               "F1_3","W2_3","L2_3","F2_3","L1_3","W1_3",
#               "L2_4","L1_4","W1_4","W2_4","F1_4","F2_4"]
# all_data = []
# folder  = r'F:\SnapbeanSummer2019\Data'
# subfolders = [os.path.join(folder,subfolder) for subfolder in os.listdir(folder) if os.path.isdir(os.path.join(folder,subfolder))][:1]
# for subfolder in subfolders:
#     data = pickle.load(open(os.path.join(subfolder,"vegetation_gh_.pkl") ,'rb'))
#     for plot_name in plot_names:
#         plot = data[plot_name].copy()
#         indices = np.arange(plot.shape[0])
#         np.random.shuffle(indices)
#         all_data.append(plot[indices])

# all_data = np.concatenate(all_data,axis=0)
# x_pca = all_data
# x = plot[indices]

# import random
# denoiser = dtcwtDenoise(method=0.99)
# x_denoised = denoiser.denoise(x, x_pca)
# snr_x = denoiser.snr(x)
# snr_x_denoised = denoiser.snr(x_denoised)
# %%









