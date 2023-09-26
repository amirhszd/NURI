function P = syntheticHRImage2(I_MS_LR, w)
    P = zeros(size(I_MS_LR,1),size(I_MS_LR,2));
    for ii = 1 : size(I_MS_LR,3)
        P = P + w(ii) .* I_MS_LR(:,:,ii);     
    end
end