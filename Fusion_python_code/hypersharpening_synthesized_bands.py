import numpy as np
import numpy as np
from imresize import imresize
from sklearn.linear_model import LinearRegression


# weight estimation
def weightEstimation(I_MS, I_SWIR, ratio_SWIR_MS):
    # Downsample the MS image to obtain a low-resolution version
    I_MS_LR = imresize(I_MS, 1./ratio_SWIR_MS, 'bilinear')
    # upsample the MS image to obtain a low-resolution version
    I_MS_LP = imresize(I_MS_LR, ratio_SWIR_MS, 'bilinear')

    I_MS_LP_C = I_MS_LP.reshape(-1, I_MS_LP.shape[2])

    I_SWIR_C = I_SWIR.reshape(-1, I_SWIR.shape[2])

    w = np.zeros((I_MS_LP_C.shape[1] + 1, I_SWIR_C.shape[1]))
    w2 = np.zeros((I_MS_LP_C.shape[1], I_SWIR_C.shape[1]))
    r = np.zeros(I_SWIR_C.shape[1])
    r2 = np.zeros(I_SWIR_C.shape[1])

    # for each band of SWIR, find weights that can be used to estimate SWIR using low filtered MS
    for ii in range(I_SWIR_C.shape[1]):
        regressor = LinearRegression()
        regressor.fit(np.hstack((I_MS_LP_C, np.ones((I_MS_LP_C.shape[0], 1)))), I_SWIR_C[:, ii])
        w[:, ii] = regressor.coef_
        r[ii] = regressor.score(np.hstack((I_MS_LP_C, np.ones((I_MS_LP_C.shape[0], 1)))), I_SWIR_C[:, ii])

        # _, _, _, _, residuals = np.linalg.lstsq(np.hstack((I_MS_LP_C, np.ones((I_MS_LP_C.shape[0], 1)))),
        #                                         I_SWIR_C[:, ii], rcond=None)
        # r[ii] = np.sqrt(residuals[0] / I_SWIR_C.shape[0])

        regressor2 = LinearRegression()
        regressor2.fit(I_MS_LP_C, I_SWIR_C[:, ii])
        w2[:, ii] = regressor2.coef_
        r2[ii] = regressor2.score(I_MS_LP_C, I_SWIR_C[:, ii])
        # _, _, _, _, residuals2 = np.linalg.lstsq(I_MS_LP_C, I_SWIR_C[:, ii], rcond=None)
        # r2[ii] = np.sqrt(residuals2[0] / I_SWIR_C.shape[0])

    return w, w2, r, r2

# synthesizing
def syntheticHRImage(I_MS_LR, w):
    P = np.zeros_like(I_MS_LR[:, :, 0])
    for ii in range(I_MS_LR.shape[2]):
        P += w[ii] * I_MS_LR[:, :, ii]
    return P


def PAN_HyperSharpening(I_MS_LR, I_SWIR_MS, ratio_SWIR_MS, flagSpectrometer=3):

    #TODO : vommenting it out we want all bands
    # if flagSpectrometer == 1:
    #     bands = [0, 3, 5, 7]  # Indexing starts from 0 in Python
    # elif flagSpectrometer == 2:
    #     bands = [1, 2, 4, 6]  # Indexing starts from 0 in Python
    # elif flagSpectrometer == 3:
    #     bands = list(range(8))
    # else:
    #     bands = []

    w, w2, r, r2 = weightEstimation(I_MS_LR, I_SWIR_MS, ratio_SWIR_MS)

    Psyn_band = np.zeros_like(I_SWIR_MS)
    for ii in range(I_SWIR_MS.shape[2]):
        # Psyn_band[:, :, ii] = syntheticHRImage(I_MS_LR[:, :, bands], w2[:, ii])
        Psyn_band[:, :, ii] = syntheticHRImage(I_MS_LR, w2[:, ii])

    return Psyn_band

