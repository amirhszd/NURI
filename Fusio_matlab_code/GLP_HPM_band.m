function I_Fus_band = GLP_HPM_band(I_MS,I_PAN,I_PAN_LP) 
    I_Fus_band = I_MS .* (I_PAN ./ (I_PAN_LP + eps)); 
end