import numpy as np
from fusion_glp_cbd import fusion_GLP_CBD
def fusion_algorithms(I_MS, I_PAN, ratio, sensor, imtag, selected_algorithm):
    if selected_algorithm == 5:
        I_Fus = fusion_GLP_CBD(I_MS, I_PAN, ratio, sensor, imtag, 'Selva')
    else:
        I_Fus = np.zeros(I_MS.shape)

    return I_Fus