from imresize import imresize
from MTF_band import MTF_band



def P_LP(I_PAN_band, sensor, imtag, ratio, band, flag):
    """
    Applies low-pass filtering to the panchromatic (PAN) image.

    Inputs:
        I_PAN_band: numpy array, PAN image band to be filtered.
        sensor: str, sensor name.
        imtag: str, tag.
        ratio: float, scaling ratio.
        band: int, band index.
        flag: str, flag indicating the type of filtering to be performed.

    Outputs:
        I_PAN_LP: numpy array, filtered PAN image band.

    P_LP applies low-pass filtering to the PAN image based on the sensor and flag information.
    The function is used to enhance the spatial resolution of the PAN image before performing image fusion with the multispectral (MS) image.

    The function first checks the value of the flag parameter to determine the type of filtering to be applied:
    - If the flag is 'Selva', it uses the MTF_band function to calculate a filter based on the sensor and band information.
    The filter is then applied to the PAN image band using the skimage.transform.resize function to downsample and upsample the image.
    - If the flag is not 'Selva', it directly applies a resizing operation to the PAN image band using the imresize function.

    The resulting filtered PAN image band, I_PAN_LP, is returned as the output of the function.
    """

    if flag == 'Selva':
        I_PAN_LP = MTF_band(I_PAN_band, sensor, imtag, ratio, band)
        I_PAN_LP = imresize(I_PAN_LP, 1./ratio, 'bilinear')
        I_PAN_LP = imresize(I_PAN_LP, ratio, 'bicubic')
    else:
        I_PAN_LP = imresize(imresize(I_PAN_band, 1. / ratio, 'bilinear'), ratio, 'bilinear')

    return I_PAN_LP