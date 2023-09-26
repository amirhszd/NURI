function [w,w2,r,r2] = weightEstimation(I_MS, I_SWIR, ratio_SWIR_MS)
    I_MS_LR = imresize(I_MS,1./ratio_SWIR_MS,'bilinear');
    I_MS_LP = imresize(I_MS_LR, ratio_SWIR_MS,'bilinear');
    I_MS_LP_C = reshape(I_MS_LP,[size(I_MS_LP,1)*size(I_MS_LP,2),size(I_MS_LP,3)]);
    I_SWIR_C = reshape(I_SWIR,[size(I_SWIR,1)*size(I_SWIR,2),size(I_SWIR,3)]);

    w = zeros(size(I_MS_LP_C,2)+1,size(I_SWIR_C,2));
    w2 = zeros(size(I_MS_LP_C,2),size(I_SWIR_C,2));
    r = zeros(1,size(I_SWIR_C,2));
    r2 = zeros(1,size(I_SWIR_C,2));
    for ii = 1 : size(I_SWIR_C,2)
        [w(:,ii),~,~,~,STATS] = regress(I_SWIR_C(:,ii),[I_MS_LP_C ones(size(I_MS_LP_C,1),1)]);
        [w2(:,ii),~,~,~,STATS2] = regress(I_SWIR_C(:,ii),I_MS_LP_C);
        r(ii) = sqrt(STATS(1));
        r2(ii) = sqrt(STATS2(1));
    end
    
end