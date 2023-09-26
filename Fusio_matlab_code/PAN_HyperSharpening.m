%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Description: 
%           Hypersharpening approach as in [1]. 
% 
% Reference:
%       [1] G. Vivone and J. Chanussot, "Fusion of short-wave infrared and visible near-infrared WorldView-3 data", Information Fusion, vol. 61, pp. 71-83, 2020.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Psyn_band, wsyn2] = PAN_HyperSharpening(I_MS_LR, I_SWIR_MS, ratio_SWIR_MS, flagSpectrometer)
    if flagSpectrometer == 1
        bands = [1 4 6 8];
    elseif flagSpectrometer == 2
        bands = [2 3 5 7];
    elseif flagSpectrometer == 3
        bands = 1 : 8;
    else
        bands = [];
    end
    
    [~, wsyn2] = weightEstimation(I_MS_LR(:,:,bands), I_SWIR_MS, ratio_SWIR_MS);

    Psyn_band = zeros(size(I_SWIR_MS));
    for ii = 1 : size(I_SWIR_MS,3)
        Psyn_band(:,:,ii) = syntheticHRImage2(I_MS_LR(:,:,bands), wsyn2(:,ii));
    end 
end