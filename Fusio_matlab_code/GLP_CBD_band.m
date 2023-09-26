function I_Fus_band = GLP_CBD_band(I_MS,I_PAN,I_PAN_LP) 
    C = cov(I_MS(:), I_PAN_LP(:));
    g = C(1,2)./C(2,2);
    I_Fus_band = I_MS + g .* (I_PAN - I_PAN_LP);
end