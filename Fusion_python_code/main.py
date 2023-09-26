from scipy.io import loadmat
from imresize import imresize
from fusion_algorithms import fusion_algorithms
import matplotlib
from hypersharpening_selected_bands import PAN_Select
from hypersharpening_synthesized_bands import PAN_HyperSharpening
matplotlib.use('Agg')
import numpy as np
import tifffile

if __name__ == "__main__":
    hypershapening = "syn" # or "sel"
    selected_algorithm = 5 # 5.Gauss-CBD
    data = loadmat("/dirs/data/tirs/axhcis/Projects/NURI/Data/Dataset_WV3_Sim_APEX.mat")
    data["I_PAN"] = data["I_PAN"][...,None]

    # upsampling the SWIR_LS using bilinear, nothing fancy
    I_SWIR_MS = imresize(data["I_SWIR_LR"],data["ratio_SWIR_MS"],'bilinear')


    # Fusino Scheme: 1
    # PAN - MS
    I_Fus_PAN_MS = fusion_algorithms(data["I_MS"], data["I_PAN"], data["ratio_MS_PAN"], data["sensor"], data["imtag"], 5);
    # PAN - SWIR
    I_Fus_PAN_SWIR = fusion_algorithms(data["I_SWIR"], data["I_PAN"], data["ratio_MS_PAN"] * data["ratio_SWIR_MS"], data["sensor"], data["imtag"], 5);
    I_Fus_Final_1 = np.concatenate([I_Fus_PAN_MS, I_Fus_PAN_SWIR], 2)


    # write to raster using tifffile
    tifffile.imwrite("scheme_1_fused_image.tif", I_Fus_Final_1)


    # usign fusion scheme 2 (synthesized band and selected bands)
    # since there is not PAN image we will be using the MS itself, we want to go to MS resolution

    if hypershapening == "syn":
        I_PAN_MS = PAN_HyperSharpening(I_Fus_PAN_MS, data["I_SWIR"], data["ratio_SWIR_MS"] * data["ratio_MS_PAN"], 5);
    else:
        I_PAN_MS = PAN_Select(I_Fus_PAN_MS, data["I_SWIR"], data["ratio_SWIR_MS"] * data["ratio_MS_PAN"], data["sensor"], data["imtag"]);

    I_Fus_PAN_SWIR = fusion_algorithms(data["I_SWIR"], I_PAN_MS, data["ratio_SWIR_MS"]*data["ratio_MS_PAN"],
                                       data["sensor"], data["imtag"], 5)
    I_Fus_Final_2 = np.concatenate([I_Fus_PAN_MS, I_Fus_PAN_SWIR], 2)

    tifffile.imwrite("scheme_2_fused_image.tif", I_Fus_Final_2)