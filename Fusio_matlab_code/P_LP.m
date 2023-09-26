function I_PAN_LP = P_LP(I_PAN_band,sensor,imtag,ratio,band,flag)

    if strcmp(flag,'Selva')
        I_PAN_LP = MTF_band(I_PAN_band,sensor,imtag,ratio,band);
        I_PAN_LP = imresize(I_PAN_LP, 1./ratio, 'nearest');
        I_PAN_LP = imresize(I_PAN_LP, ratio, 'bicubic');
    else
        I_PAN_LP = imresize(imresize(I_PAN_band, 1./ratio, 'bilinear'), ratio, 'bilinear');
    end
    
end