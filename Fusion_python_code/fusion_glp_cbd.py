import numpy as np
from P_LP import P_LP

def GLP_CBD_band(I_MS, I_PAN, I_PAN_LP):
    """

    Performs generalized laplacian pyramid content based decisio fusion for a single band of a multispectral image.

    Inputs:
        I_MS: numpy array, multispectral image band.
        I_PAN: numpy array, panchromatic image.
        I_PAN_LP: numpy array, low-pass filtered version of the panchromatic image.

    Outputs:
        I_Fus_band: numpy array, fused image band.

    GLP_CBD_band applies the generalized laplacian pyramid content based decision algorithm to
    fuse a multispectral image band (I_MS) with a panchromatic image (I_PAN) using a low-pass filtered version of the panchromatic image (I_PAN_LP).

    The function first calculates the covariance matrix (C) between the flattened versions of I_MS and I_PAN_LP.
    The covariance matrix represents the statistical relationship between the two images.

    Next, it computes the parameter 'g' as the covariance between I_MS and I_PAN_LP divided by the variance of
    I_PAN_LP. This parameter represents the amount of intensity modulation applied to the high-frequency component.

    The function then performs the band-wise fusion using the GLP_CBD algorithm. It subtracts the low-pass
    filtered version of the panchromatic image (I_PAN_LP) from the panchromatic image (I_PAN) to obtain the high-frequency component. The high-frequency component is then scaled by 'g'. Finally, the scaled high-frequency component is added to the multispectral image (I_MS) to obtain the fused image band (I_Fus_band).

    The resulting fused image band, I_Fus_band, is returned as the output of the function.

    Please note that this function is designed to perform the fusion for a single band. To obtain the complete
    fused image, this function should be applied to each band of the multispectral image separately.
    """
    C = np.cov(I_MS.flatten(), I_PAN_LP.flatten())
    g = C[0, 1] / C[1, 1]
    I_Fus_band = I_MS + g * (I_PAN - I_PAN_LP)
    return I_Fus_band

def fusion_GLP_CBD(I_MS, I_PAN, ratio, sensor, imtag, flagfilter):
    # replicates the channel dimension for PAN
    if I_PAN.shape[2] == 1:
        I_PAN = np.tile(I_PAN, (1, 1, I_MS.shape[2]))

    I_Fus = np.zeros_like(I_MS)
    for ii in range(I_MS.shape[2]):
        I_PAN_LP = P_LP(I_PAN[:, :, ii], sensor, imtag, ratio, ii, flagfilter)
        I_Fus[:, :, ii] = GLP_CBD_band(I_MS[:, :, ii], I_PAN[:, :, ii], I_PAN_LP)

    return I_Fus