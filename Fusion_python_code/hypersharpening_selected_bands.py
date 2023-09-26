import numpy as np

def maxCorrelationHRImage(I_MS_LR, I_SWIR_band, ratio_SWIR_MS, sensor, imtag, flagfilter):
    I_MS_LP = np.zeros_like(I_MS_LR)
    for ii in range(I_MS_LP.shape[2]):
        I_MS_LP[:, :, ii] = P_LP(I_MS_LR[:, :, ii], sensor, imtag, ratio_SWIR_MS, ii, flagfilter)

    r = np.zeros(I_MS_LR.shape[2])
    for ii in range(I_MS_LR.shape[2]):
        I_MS_LP_band = I_MS_LP[:, :, ii]
        C = np.corrcoef(I_SWIR_band.flatten(), I_MS_LP_band.flatten())
        r[ii] = C[0, 1]

    index_band_max = np.argmax(r)

    P = I_MS_LR[:, :, index_band_max]

    return P, r, index_band_max

def PAN_Select(I_MS_LR, I_SWIR_MS, ratio_SWIR_MS, sensor, imtag):
    Psel_band = np.zeros_like(I_SWIR_MS)
    for ii in range(I_SWIR_MS.shape[2]):
        Psel_band[:, :, ii] = maxCorrelationHRImage(I_MS_LR, I_SWIR_MS[:, :, ii], ratio_SWIR_MS, sensor, imtag, '')
    return Psel_band

# correllation function


