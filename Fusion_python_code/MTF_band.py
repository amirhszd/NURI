import numpy as np
from scipy.signal import kaiser, firwin, fftconvolve, convolve2d
from skimage.filters import gaussian

def gaussian(size,sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    shape = [size, size]
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def MTF_band(I_MS_band, sensor, tag, ratio, band):
    """
    Calculates the modulation transfer function (MTF) filtering for a specific band of a multispectral image.

    Inputs:
        I_MS_band: numpy array, input multispectral image band (not used).
        sensor: str, name of the sensor.
        tag: int or list of ints, tag or tags specifying specific bands (optional).
        ratio: float, scaling ratio (not used).
        band: int, band index.

    Outputs:
        I_Filtered: numpy array, filtered image band.

    MTF_band calculates the MTF filtering for a specific band of a multispectral image based on the sensor and band information.
    The MTF represents the ability of the imaging system to faithfully reproduce spatial details.

    The function first defines the MTF values for different bands based on the sensor.
    It then selects the appropriate MTF value(s) based on the provided tag or uses the default MTF values for all bands.

    Next, the function calculates the MTF filtering by applying a low-pass filter to the input image band.
    The steps involved in the calculation are as follows:

    1. Determines the length of the MTF filter kernel as N = 41.
    2. Computes the cutoff frequency (fcut) based on the scaling ratio (ratio).
    3. Calculates the parameter alpha for the Gaussian window function based on the MTF value for the specified band.
    4. Generates a Gaussian filter (H) using the skimage.filters.gaussian function with the specified parameters.
    5. Normalizes the Gaussian filter (Hd) by dividing it by its maximum value.
    6. Generates a Kaiser windowed FIR filter (h) using the scipy.signal.firwin function with the specified parameters.
    7. Applies the FIR filter to the input I_MS_band using the scipy.signal.fftconvolve function to obtain the low-pass filtered image, I_MS_LP.

    Finally, the filtered image band, I_MS_LP, is returned as the output of the function.
    """
    # Define MTF values based on sensor
    if sensor == 'QB':
        MTF_MS = np.array([0.34, 0.32, 0.30, 0.22])  # Band Order: B, G, R, NIR
    elif sensor == 'IKONOS':
        MTF_MS = np.array([0.26, 0.28, 0.29, 0.28])  # Band Order: B, G, R, NIR
    elif sensor == 'All_03':
        MTF_MS = 0.3
    elif sensor == 'MS_029_PAN_015':
        MTF_MS = 0.29
    elif sensor == 'GeoEye1':
        MTF_MS = np.array([0.23, 0.23, 0.23, 0.23])  # Band Order: B, G, R, NIR
    elif sensor == 'WV2':
        MTF_MS = np.array([0.35] * 7 + [0.27])
    elif sensor in ['WV3', 'WV3_4bands']:
        MTF_MS = np.array([0.325, 0.355, 0.360, 0.350, 0.365, 0.360, 0.335, 0.315])
        if sensor == 'WV3_4bands':
            tag = [2, 3, 5, 7]
    elif sensor in ['HYP', 'HYP_14_33', 'HYP_16_31']:
        MTF_MS = np.zeros(242)
        MTF_MS[:21] = 0.27  # VNIR
        MTF_MS[21:41] = 0.28
        MTF_MS[41:49] = 0.26
        MTF_MS[49:70] = 0.26
        MTF_MS[70:100] = 0.30  # SWIR
        MTF_MS[100:130] = 0.30
        MTF_MS[130:177] = 0.27
        MTF_MS[177:242] = 0.27
        if sensor == 'HYP_14_33':
            tag = list(range(14, 34))
        elif sensor == 'HYP_16_31':
            tag = list(range(16, 32))
    elif sensor == 'Ali_MS':
        MTF_MS = np.array([0.29, 0.30, 0.28, 0.29, 0.28, 0.29, 0.25, 0.25, 0.25])
    elif sensor == 'none':
        MTF_MS = 0.29

    if tag and isinstance(tag, int):
        GNyq = MTF_MS[tag]
    else:
        GNyq = MTF_MS

    # MTF Calculation
    N = 41
    fcut = 1 / float(ratio)
    alpha = np.sqrt((N * (fcut / 2))**2 / (-2 * np.log(GNyq[band])))
    H = gaussian(N, sigma=alpha)
    Hd = H / np.max(H)
    h = firwin(N, cutoff=fcut, window=('kaiser', alpha))
    # I_MS_LP = fftconvolve(I_MS_band, h[np.newaxis, :], mode='same', axes=0)
    I_MS_LP = convolve2d(I_MS_band, h[np.newaxis, :], mode='same')

    I_Filtered = I_MS_LP.astype(float)
    return I_Filtered