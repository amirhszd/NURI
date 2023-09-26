%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Description: 
%           Selection band approach as in [1]. 
% 
% Reference:
%       [1] G. Vivone and J. Chanussot, "Fusion of short-wave infrared and visible near-infrared WorldView-3 data", Information Fusion, vol. 61, pp. 71-83, 2020.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Psel_band = PAN_Select(I_MS_LR, I_SWIR_MS, ratio_SWIR_MS, sensor, imtag)
    Psel_band = zeros(size(I_SWIR_MS));
    for ii = 1 : size(I_SWIR_MS,3)   
        Psel_band(:,:,ii) = maxCorrelationHRImage(I_MS_LR, I_SWIR_MS(:,:,ii), ratio_SWIR_MS, sensor, imtag, '');
    end
end