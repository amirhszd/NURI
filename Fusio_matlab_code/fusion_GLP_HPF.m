function I_Fus = fusion_GLP_HPF(I_MS, I_PAN, ratio, sensor, imtag, flagfilter)
    
    if size(I_PAN,3) == 1
       I_PAN = repmat(I_PAN, [1 1 size(I_MS,3)]); 
    end
       
    I_Fus = zeros(size(I_MS));
    for ii = 1 : size(I_MS,3)
        I_PAN_LP = P_LP(I_PAN(:,:,ii), sensor, imtag, ratio, ii, flagfilter);
        I_Fus(:,:,ii) = GLP_HPF_band(I_MS(:,:,ii),I_PAN(:,:,ii),I_PAN_LP);
    end
    
end