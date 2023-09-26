function I_Fus_band = GLP_HPF_band(I_MS,I_PAN,I_PAN_LP) 
      g = std(I_MS(:))./std(I_PAN_LP(:));
      I_Fus_band = I_MS + g .* (I_PAN - I_PAN_LP);
end