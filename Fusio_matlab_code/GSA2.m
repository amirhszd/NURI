%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Description: 
%           GSA.
% 
% Interface:
%           I_Fus_GSA = GSA2(I_MS,I_PAN)
%
% Inputs:
%           I_MS:       MS image upsampled at PAN scale;
%           I_PAN:      PAN image;
%           ratio:      scaling ratio.
%
% Outputs:
%           I_Fus_GSA:   GS pasharpened image.
% 
% References:
%           [Vivone14]  G. Vivone, L. Alparone, J. Chanussot, M. Dalla Mura, A. Garzelli, G. Licciardi, R. Restaino, and L. Wald, “A Critical Comparison Among Pansharpening Algorithms”, 
%                       IEEE Transaction on Geoscience and Remote Sensing, 2014. (Accepted)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function I_Fus_GSA = GSA2(I_MS,I_PAN,ratio)

%%% Intensity
I_PAN_LR = imresize(imresize(I_PAN, 1./ratio,'bilinear'),ratio,'bilinear');
alpha(1,1,:) = estimation_alpha(cat(3,I_MS,ones(size(I_MS,1),size(I_MS,2))),I_PAN_LR,'global');
I = sum(cat(3,I_MS,ones(size(I_MS,1),size(I_MS,2))) .* repmat(alpha,[size(I_MS,1) size(I_MS,2) 1]),3); 

%%% Coefficients
g = ones(1,1,size(I_MS,3)+1);
for ii = 1 : size(I_MS,3)
    h = I_MS(:,:,ii);
    c = cov(I(:),h(:));
    g(1,1,ii+1) = c(1,2)/var(I(:));
end

%%% Detail Extraction
delta = I_PAN - I;
deltam = repmat(delta(:),[1 size(I_MS,3)+1]);

%%% Fusion
V = I(:);
for ii = 1 : size(I_MS,3)
    h = I_MS(:,:,ii);
    V = cat(2,V,h(:));
end

gm = zeros(size(V));
for ii = 1 : size(g,3)
    gm(:,ii) = squeeze(g(1,1,ii)) .* ones(size(I_MS,1).*size(I_MS,2),1);
end

V_hat = V + deltam .* gm;

I_Fus_GSA = reshape(V_hat(:,2:end),[size(I_MS,1) size(I_MS,2) size(I_MS,3)]);

end